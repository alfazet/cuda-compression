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

__global__ void flDecompressionGPU(byte* batch, u64 batchLen, const u8* bitDepth, const byte* chunks)
{
    u64 tidInBlock = threadIdx.x;
    u64 tidGlobal = blockIdx.x * blockDim.x + tidInBlock;
    if (tidGlobal >= batchLen)
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
    batch[tidGlobal] = decoded;
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
    Fl fl;
    bool nextBatchReady = false;
    TimerCpu timerCpuInput, timerCpuOutput, timerCpuComputing;
    TimerGpu timerGpuMemHostToDev, timerGpuMemDevToHost, timerGpuComputing;

    for (u64 batchIdx = 1; batchIdx <= batches; batchIdx++)
    {
        printf("Processing batch %lu out of %lu...\n", batchIdx, batches);
        u64 batchSize = batchIdx == batches ? lastBatchSize : BATCH_SIZE;
        std::vector<byte> batch(batchSize);

        if (!nextBatchReady)
        {
            fl = Fl(flMetadata, batchSize);
            timerCpuInput.start();
            fl.readFromFile(inFile);
            timerCpuInput.stop();
            nextBatchReady = true;
        }

        switch (version)
        {
        case Cpu:
            timerCpuComputing.start();
            flDecompressionCPU(fl, batch);
            timerCpuComputing.stop();
            nextBatchReady = false;
            break;
        case Gpu:
            timerGpuMemHostToDev.start();
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
            timerGpuMemHostToDev.stop();

            timerGpuComputing.start();
            flDecompressionGPU<<<fl.nChunks, CHUNK_SIZE>>>(dData, fl.batchSize, dBitDepth, dChunks);
            // the kernel launch is async so we read the next batch of input in the meantime
            u64 tmpBatchSize = fl.batchSize;
            if (batchIdx < batches)
            {
                timerCpuInput.start();
                u64 nextBatchSize = batchIdx + 1 == batches ? lastBatchSize : BATCH_SIZE;
                fl = Fl(flMetadata, nextBatchSize);
                fl.readFromFile(inFile);
                timerCpuInput.stop();
                nextBatchReady = true;
            }
            timerGpuComputing.stop();

            timerGpuMemDevToHost.start();
            CUDA_ERR_CHECK(cudaMemcpy(batch.data(), dData, tmpBatchSize * sizeof(byte), cudaMemcpyDeviceToHost));
            CUDA_ERR_CHECK(cudaFree(dData));
            CUDA_ERR_CHECK(cudaFree(dBitDepth));
            CUDA_ERR_CHECK(cudaFree(dChunks));
            timerGpuMemDevToHost.stop();
            break;
        }

        timerCpuOutput.start();
        writeOutputBatch(outFile, batch);
        timerCpuOutput.stop();
    }

    switch (version)
    {
    case Cpu:
        printCpuTimers(timerCpuInput, timerCpuComputing, timerCpuOutput);
        break;
    case Gpu:
        printGpuTimers(timerCpuInput, timerGpuMemHostToDev, timerGpuComputing, timerGpuMemDevToHost, timerCpuOutput);
        break;
    }
    fclose(outFile);
    fclose(inFile);
}