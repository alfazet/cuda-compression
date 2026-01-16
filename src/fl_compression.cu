#include "fl.cuh"
#include "timer.cuh"

void flCompressionCPU(Fl& fl, const std::vector<byte>& batch)
{
    for (u64 i = 0; i < fl.nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl.batchSize)
        {
            // the last chunk might be shorter than CHUNK_SIZE bytes
            len = fl.batchSize % CHUNK_SIZE;
        }

        // the bit depth of a chunk = the max. MSb of all bytes contained in it
        // (with one edge case: when all bytes are 0x00, the bitDepth should be 1, not 0)
        u8 bitDepth = 1;
        for (u64 j = 0; j < len; j++)
        {
            byte curByte = batch[i * CHUNK_SIZE + j];
            u8 bitDepthHere = 0;
            for (int b = 7; b >= 0; b--)
            {
                if ((0b1 << b) & curByte)
                {
                    bitDepthHere = b + 1;
                    break;
                }
            }
            bitDepth = max(bitDepth, bitDepthHere);
        }
        fl.bitDepth[i] = bitDepth;
        for (u64 j = 0; j < len; j++)
        {
            byte curByte = batch[i * CHUNK_SIZE + j] << (8 - bitDepth);
            u64 bitLoc = bitDepth * j;
            u64 byteLoc = bitLoc >> 3;
            u64 bitOffset = bitLoc & 0b111;
            // place the relevant bits of this byte
            // starting at the (7 - bitOffset)-th bit of the byteLoc-th byte
            fl.chunks[i][byteLoc] |= curByte >> bitOffset;
            // if any bits spill over to the next byte, place them there
            if (bitOffset != 0)
            {
                fl.chunks[i][byteLoc + 1] |= curByte << (8 - bitOffset);
            }
        }
    }
}

// TODO: fix data race in line 75 and UB (__syncthreads_or in while loop)
__global__ void flCompressionGPU(const byte* data, u64 dataLen, u8* bitDepth, byte* chunks)
{
    u64 tidInBlock = threadIdx.x;
    u64 tidGlobal = blockIdx.x * blockDim.x + tidInBlock;
    if (tidGlobal >= dataLen)
    {
        return;
    }

    byte curByte = data[tidGlobal];
    u8 blockBitDepth = 8;
    while (blockBitDepth > 0 && __syncthreads_or((1 << (blockBitDepth - 1)) & curByte) == 0)
    {
        blockBitDepth--;
    }
    curByte <<= (8 - blockBitDepth);
    if (tidInBlock == 0)
    {
        bitDepth[blockIdx.x] = blockBitDepth;
    }
    u64 bitLoc = 1UL * blockBitDepth * tidInBlock;
    u64 byteLoc = bitLoc >> 3;
    u64 bitOffset = bitLoc & 0b111;
    u64 idx = blockIdx.x * blockDim.x + byteLoc; // the index of the byte we're handling

    // split into 8 batches based on the thread id mod 8
    // to prevent data races
    for (u8 mod = 0; mod < 8; mod++)
    {
        if ((tidInBlock & 0b111) == mod)
        {
            chunks[idx] |= curByte >> bitOffset;
            if (bitOffset != 0)
            {
                chunks[idx + 1] |= curByte << (8 - bitOffset);
            }
        }
        __syncthreads();
    }
}

void flCompression(const std::string& inputPath, const std::string& outputPath, Version version)
{
    FILE* inFile = fopen(inputPath.c_str(), "rb");
    if (inFile == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    FILE* outFile = fopen(outputPath.c_str(), "wb");
    if (outFile == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    u64 rawFileSize = std::filesystem::file_size(inputPath);
    if (rawFileSize == 0)
    {
        fclose(outFile);
        fclose(inFile);
        return;
    }

    FlMetadata flMetadata(rawFileSize);
    flMetadata.writeToFile(outFile);
    u64 batches = ceilDiv(rawFileSize, BATCH_SIZE), lastBatchSize = rawFileSize % BATCH_SIZE;
    for (u64 batchIdx = 1; batchIdx <= batches; batchIdx++)
    {
        printf("Batch %lu out of %lu\n", batchIdx, batches);
        u64 batchSize = batchIdx == batches ? lastBatchSize : BATCH_SIZE;

        TimerCpu timerCpu;
        timerCpu.start();
        std::vector<byte> batch = readInputBatch(inFile, batchSize);
        timerCpu.stop();
        printf("%s\n", timerCpu.formattedResult("[CPU] reading from the input file").c_str());

        Fl fl(flMetadata, batchSize);
        switch (version)
        {
        case Cpu:
            timerCpu.start();
            flCompressionCPU(fl, batch);
            timerCpu.stop();
            printf("%s\n", timerCpu.formattedResult("[CPU] FL compression function").c_str());
            break;
        case Gpu:
            TimerGPU timerGPU;
            timerGPU.start();
            byte* dData;
            CUDA_ERR_CHECK(cudaMalloc(&dData, fl.batchSize * sizeof(byte)));
            CUDA_ERR_CHECK(cudaMemcpy(dData, batch.data(), fl.batchSize * sizeof(byte), cudaMemcpyHostToDevice));
            u8* dBitDepth;
            CUDA_ERR_CHECK(cudaMalloc(&dBitDepth, fl.nChunks * sizeof(u8)));
            byte* dChunks;
            CUDA_ERR_CHECK(cudaMalloc(&dChunks, fl.nChunks * CHUNK_SIZE * sizeof(byte)));
            timerGPU.stop();
            printf("%s\n", timerGPU.formattedResult("[GPU] allocating and copying data to device").c_str());

            timerGPU.start();
            flCompressionGPU<<<fl.nChunks, CHUNK_SIZE>>>(dData, fl.batchSize, dBitDepth, dChunks);
            timerGPU.stop();
            printf("%s\n", timerGPU.formattedResult("[GPU] FL compression kernel").c_str());

            timerGPU.start();
            CUDA_ERR_CHECK(cudaMemcpy(fl.bitDepth.data(), dBitDepth, fl.nChunks * sizeof(u8), cudaMemcpyDeviceToHost));
            for (u64 i = 0; i < fl.nChunks; i++)
            {
                CUDA_ERR_CHECK(cudaMemcpy(fl.chunks[i].data(), dChunks + i * CHUNK_SIZE, CHUNK_SIZE * sizeof(byte),
                    cudaMemcpyDeviceToHost));
            }
            CUDA_ERR_CHECK(cudaFree(dData));
            CUDA_ERR_CHECK(cudaFree(dBitDepth));
            CUDA_ERR_CHECK(cudaFree(dChunks));
            timerGPU.stop();
            printf("%s\n", timerGPU.formattedResult("[GPU] copying data to host and freeing").c_str());
            break;
        }

        timerCpu.start();
        fl.writeToFile(outFile);
        timerCpu.stop();
        printf("%s\n", timerCpu.formattedResult("[CPU] writing to the output file").c_str());
    }

    fclose(outFile);
    fclose(inFile);
}