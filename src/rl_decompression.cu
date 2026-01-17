#include "rl.cuh"
#include "timer.cuh"

void rlDecompressionCPU(const Rl& rl, std::vector<byte>& batch)
{
    u64 p = 0;
    for (u32 i = 0; i < rl.nRuns; i++)
    {
        for (u32 j = 0; j < rl.lengths[i]; j++)
        {
            batch[p] = rl.values[i];
            p++;
        }
    }
}

void rlDecompression(const std::string& inputPath, const std::string& outputPath, Version version)
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
    RlMetadata rlMetadata(inFile);
    if (rlMetadata.rawFileSizeTotal == 0)
    {
        printf("Empty file\n");
        fclose(outFile);
        fclose(inFile);
        return;
    }

    u64 batches = ceilDiv(rlMetadata.rawFileSizeTotal, RL_BATCH_SIZE),
        lastBatchSize = rlMetadata.rawFileSizeTotal % RL_BATCH_SIZE;
    Rl rl;
    bool nextBatchReady = false;
    TimerCpu timerCpuInput, timerCpuOutput, timerCpuComputing;
    TimerGpu timerGpuMemHostToDev, timerGpuMemDevToHost, timerGpuComputing;

    byte* dData;
    byte* dValues;
    thrust::device_vector<u32> dLengths;
    thrust::device_vector<u32> dScannedLengths;
    if (version == Gpu)
    {
        CUDA_ERR_CHECK(cudaMalloc(&dData, RL_BATCH_SIZE * sizeof(byte)));
        CUDA_ERR_CHECK(cudaMalloc(&dValues, RL_BATCH_SIZE * sizeof(byte)));
        dLengths.reserve(RL_BATCH_SIZE);
        dScannedLengths.reserve(RL_BATCH_SIZE);
    }

    for (u64 batchIdx = 1; batchIdx <= batches; batchIdx++)
    {
        printf("Processing batch %lu out of %lu...\n", batchIdx, batches);
        u64 batchSize = batchIdx == batches ? lastBatchSize : RL_BATCH_SIZE;
        std::vector<byte> batch(batchSize);

        if (!nextBatchReady)
        {
            rl = Rl(rlMetadata, batchSize);
            timerCpuInput.start();
            rl.readFromFile(inFile);
            timerCpuInput.stop();
            nextBatchReady = true;
        }

        switch (version)
        {
        case Cpu:
            timerCpuComputing.start();
            rlDecompressionCPU(rl, batch);
            timerCpuComputing.stop();
            nextBatchReady = false;
            break;
        case Gpu:
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
        CUDA_ERR_CHECK(cudaFree(dValues));
        CUDA_ERR_CHECK(cudaFree(dData));
        printGpuTimers(timerCpuInput, timerGpuMemHostToDev, timerGpuComputing, timerGpuMemDevToHost, timerCpuOutput);
        break;
    }
    fclose(outFile);
    fclose(inFile);
}