#include "fl.cuh"
#include "timer.cuh"

void flDecompressionCPU(const Fl& fl, std::vector<byte>& batch)
{
    for (u64 i = 0; i < fl.nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl.batchSize)
        {
            // the last chunk might be shorter than CHUNK_SIZE bytes
            len = fl.batchSize % CHUNK_SIZE;
        }
        u8 bitDepth = fl.bitDepth[i];
        for (u64 j = 0; j < len; j++)
        {
            u64 bitLoc = bitDepth * j;
            u64 byteLoc = bitLoc >> 3;
            u64 bitOffset = bitLoc & 0b111;
            byte mask = 0xff << (8 - bitDepth);
            mask = mask >> bitOffset;
            byte decoded = (fl.chunks[i][byteLoc] & mask) << bitOffset;
            decoded = decoded >> (8 - bitDepth);
            if (bitOffset != 0)
            {
                mask = (0xff << (8 - bitOffset)) << (8 - bitDepth);
                decoded |= ((fl.chunks[i][byteLoc + 1] & mask) >> (8 - bitOffset)) >> (8 - bitDepth);
            }
            batch[i * CHUNK_SIZE + j] = decoded;
        }
    }
}

// takes "flattened" chunks - chunk 0 is [0, CHUNK_SIZE), chunk 1 is [CHUNK_SIZE, 2 * CHUNK_SIZE), ...
__global__ void flDecompressionGPU(byte* data, u64 dataLen, const u8* bitDepth, const byte* chunks)
{
    u64 tidInBlock = threadIdx.x;
    u64 tidGlobal = blockIdx.x * blockDim.x + tidInBlock;
    if (tidGlobal >= dataLen)
    {
        return;
    }

    u8 blockBitDepth = bitDepth[blockIdx.x];
    u64 bitLoc = blockBitDepth * tidInBlock;
    u64 byteLoc = bitLoc >> 3;
    u64 bitOffset = bitLoc & 0b111;
    byte mask = 0xff << (8 - blockBitDepth);
    mask >>= bitOffset;

    u64 idx = blockIdx.x * blockDim.x + byteLoc; // the index of the byte we're handling
    byte decoded = (chunks[idx] & mask) << bitOffset;
    decoded = decoded >> (8 - blockBitDepth);
    if (bitOffset != 0)
    {
        mask = (0xff << (8 - bitOffset)) << (8 - blockBitDepth);
        decoded |= ((chunks[idx + 1] & mask) >> (8 - bitOffset)) >> (8 - blockBitDepth);
    }
    data[tidGlobal] = decoded;
}

void flDecompression(const std::string& inputPath, const std::string& outputPath, Version version)
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
    FlMetadata flMetadata(inFile);
    if (flMetadata.rawFileSizeTotal == 0)
    {
        fclose(outFile);
        fclose(inFile);
        return;
    }

    u64 batches = ceilDiv(flMetadata.rawFileSizeTotal, BATCH_SIZE), lastBatchSize = flMetadata.rawFileSizeTotal % BATCH_SIZE;
    for (u64 batchIdx = 1; batchIdx <= batches; batchIdx++)
    {
        printf("Batch %lu out of %lu\n", batchIdx, batches);
        u64 batchSize = batchIdx == batches ? lastBatchSize : BATCH_SIZE;
        Fl fl(flMetadata, batchSize);
        std::vector<byte> batch(batchSize);

        TimerCpu timerCpu;
        timerCpu.start();
        fl.readFromFile(inFile);
        timerCpu.stop();
        printf("%s\n", timerCpu.formattedResult("[CPU] reading from the input file").c_str());

        switch (version)
        {
        case Cpu:
            timerCpu.start();
            flDecompressionCPU(fl, batch);
            timerCpu.stop();
            printf("%s\n", timerCpu.formattedResult("[CPU] FL decompression function").c_str());
            break;
        case Gpu:
            TimerGPU timerGPU;
            timerGPU.start();
            byte* dData;
            CUDA_ERR_CHECK(cudaMalloc(&dData, fl.batchSize * sizeof(byte)));
            u8* dBitDepth;
            CUDA_ERR_CHECK(cudaMalloc(&dBitDepth, fl.nChunks * sizeof(u8)));
            CUDA_ERR_CHECK(cudaMemcpy(dBitDepth, fl.bitDepth.data(), fl.nChunks * sizeof(u8), cudaMemcpyHostToDevice));
            byte* dChunks;
            CUDA_ERR_CHECK(cudaMalloc(&dChunks, fl.nChunks * CHUNK_SIZE * sizeof(byte)));
            for (u64 i = 0; i < fl.nChunks; i++)
            {
                CUDA_ERR_CHECK(cudaMemcpy(dChunks + i * CHUNK_SIZE, fl.chunks[i].data(), CHUNK_SIZE * sizeof(byte),
                    cudaMemcpyHostToDevice));
            }
            timerGPU.stop();
            printf("%s\n", timerGPU.formattedResult("[GPU] allocating and copying data to device").c_str());

            timerGPU.start();
            flDecompressionGPU<<<fl.nChunks, CHUNK_SIZE>>>(dData, fl.batchSize, dBitDepth, dChunks);
            timerGPU.stop();
            printf("%s\n", timerGPU.formattedResult("[GPU] FL decompression kernel").c_str());

            timerGPU.start();
            CUDA_ERR_CHECK(cudaMemcpy(batch.data(), dData, fl.batchSize * sizeof(byte), cudaMemcpyDeviceToHost));
            CUDA_ERR_CHECK(cudaFree(dData));
            CUDA_ERR_CHECK(cudaFree(dBitDepth));
            CUDA_ERR_CHECK(cudaFree(dChunks));
            timerGPU.stop();
            printf("%s\n", timerGPU.formattedResult("[GPU] copying data to host and freeing").c_str());
            break;
        }

        timerCpu.start();
        writeOutputBatch(outFile, batch);
        timerCpu.stop();
        printf("%s\n", timerCpu.formattedResult("[CPU] writing to the output file").c_str());
    }

    fclose(outFile);
    fclose(inFile);
}