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
                if ((1 << b) & curByte)
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
            u64 bitLoc = j * bitDepth;
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

__global__ void flCompressionGPU(const byte* batch, u64 batchSize, u8* bitDepth, byte* chunks)
{
    u64 tidInBlock = threadIdx.x;
    u64 globalTid = blockIdx.x * blockDim.x + tidInBlock;
    if (globalTid >= batchSize)
    {
        return;
    }

    byte myInByte = batch[globalTid];
    u8 blockBitDepth = 1;
    for (u8 i = 1; i < 8; i++)
    {
        // if some byte in this block has the i-th bit on,
        // then its bit depth >= i + 1
        if (__syncthreads_or((1 << i) & myInByte) != 0)
        {
            blockBitDepth = i + 1;
        }
    }
    if (tidInBlock == 0)
    {
        bitDepth[blockIdx.x] = blockBitDepth;
    }
    // gather bits into my output byte
    byte myOutByte = 0;
    u64 myOffset = blockIdx.x * blockDim.x;
    u64 myEnd = (blockIdx.x + 1) * blockDim.x;
    for (u8 i = 0; i < 8; i++)
    {
        u64 dataByteIdx = myOffset + (tidInBlock * 8 + i) / blockBitDepth;
        u64 dataBitIdx = (tidInBlock * 8 + i) % blockBitDepth;
        if (dataByteIdx >= myEnd)
        {
            break;
        }
        u8 dataByte = batch[dataByteIdx] << (8 - blockBitDepth);
        if ((dataByte & (1 << (7 - dataBitIdx))) != 0)
        {
            myOutByte |= 1 << (7 - i);
        }
    }
    chunks[globalTid] = myOutByte;
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
        printf("Empty file\n");
        fclose(outFile);
        fclose(inFile);
        return;
    }

    FlMetadata flMetadata(rawFileSize);
    flMetadata.writeToFile(outFile);
    u64 batches = ceilDiv(rawFileSize, FL_BATCH_SIZE), lastBatchSize = rawFileSize % FL_BATCH_SIZE;
    std::vector<byte> batch;
    batch.reserve(FL_BATCH_SIZE);
    bool nextBatchReady = false;
    TimerCpu timerCpuInput, timerCpuOutput, timerCpuComputing;
    TimerGpu timerGpuMemHostToDev, timerGpuMemDevToHost, timerGpuComputing;

    byte* dData;
    u8* dBitDepth;
    byte* dChunks;
    if (version == Gpu)
    {
        CUDA_ERR_CHECK(cudaMalloc(&dData, FL_BATCH_SIZE * sizeof(byte)));
        CUDA_ERR_CHECK(cudaMalloc(&dBitDepth, MAX_N_CHUNKS * sizeof(u8)));
        CUDA_ERR_CHECK(cudaMalloc(&dChunks, MAX_N_CHUNKS * CHUNK_SIZE * sizeof(byte)));
    }

    for (u64 batchIdx = 1; batchIdx <= batches; batchIdx++)
    {
        printf("Processing batch %lu out of %lu...\n", batchIdx, batches);
        u64 batchSize = batchIdx == batches ? lastBatchSize : FL_BATCH_SIZE;
        if (!nextBatchReady)
        {
            timerCpuInput.start();
            batch = readInputBatch(inFile, batchSize);
            timerCpuInput.stop();
            nextBatchReady = true;
        }

        Fl fl(flMetadata, batchSize);
        switch (version)
        {
        case Cpu:
            timerCpuComputing.start();
            flCompressionCPU(fl, batch);
            timerCpuComputing.stop();
            nextBatchReady = false;
            break;
        case Gpu:
            timerGpuMemHostToDev.start();
            CUDA_ERR_CHECK(cudaMemcpy(dData, batch.data(), fl.batchSize * sizeof(byte), cudaMemcpyHostToDevice));
            timerGpuMemHostToDev.stop();

            timerGpuComputing.start();
            flCompressionGPU<<<fl.nChunks, CHUNK_SIZE>>>(dData, fl.batchSize, dBitDepth, dChunks);
            // the kernel launch is async so we read the next batch of input in the meantime
            if (batchIdx < batches)
            {
                timerCpuInput.start();
                u64 nextBatchSize = batchIdx + 1 == batches ? lastBatchSize : FL_BATCH_SIZE;
                batch = readInputBatch(inFile, nextBatchSize);
                timerCpuInput.stop();
                nextBatchReady = true;
            }
            timerGpuComputing.stop();

            timerGpuMemDevToHost.start();
            CUDA_ERR_CHECK(cudaMemcpy(fl.bitDepth.data(), dBitDepth, fl.nChunks * sizeof(u8), cudaMemcpyDeviceToHost));
            for (u64 i = 0; i < fl.nChunks; i++)
            {
                CUDA_ERR_CHECK(cudaMemcpy(fl.chunks[i].data(), dChunks + i * CHUNK_SIZE, CHUNK_SIZE * sizeof(byte),
                                          cudaMemcpyDeviceToHost));
            }
            timerGpuMemDevToHost.stop();
            break;
        }

        timerCpuOutput.start();
        fl.writeToFile(outFile);
        timerCpuOutput.stop();
    }

    switch (version)
    {
    case Cpu:
        printCpuTimers(timerCpuInput, timerCpuComputing, timerCpuOutput);
        break;
    case Gpu:
        CUDA_ERR_CHECK(cudaFree(dChunks));
        CUDA_ERR_CHECK(cudaFree(dBitDepth));
        CUDA_ERR_CHECK(cudaFree(dData));
        printGpuTimers(timerCpuInput, timerGpuMemHostToDev, timerGpuComputing, timerGpuMemDevToHost, timerCpuOutput);
        break;
    }
    fclose(outFile);
    fclose(inFile);
}