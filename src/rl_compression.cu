#include "rl.cuh"
#include "timer.cuh"

void rlCompressionCPU(Rl& rl, const std::vector<byte>& batch)
{
    u64 curLen = 1;
    byte curValue = batch[0];
    for (u64 i = 1; i < batch.size(); i++)
    {
        if (batch[i] != curValue)
        {
            rl.lengths.push_back(curLen);
            rl.values.push_back(curValue);
            curLen = 1;
            curValue = batch[i];
        }
        else
        {
            curLen++;
        }
    }
    rl.lengths.push_back(curLen);
    rl.values.push_back(curValue);
    rl.nRuns = rl.lengths.size();
}

void rlCompression(const std::string& inputPath, const std::string& outputPath, Version version)
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

    RlMetadata rlMetadata(rawFileSize);
    rlMetadata.writeToFile(outFile);
    u64 batches = ceilDiv(rawFileSize, RL_BATCH_SIZE), lastBatchSize = rawFileSize % RL_BATCH_SIZE;
    std::vector<byte> batch;
    batch.reserve(RL_BATCH_SIZE);
    bool nextBatchReady = false;
    TimerCpu timerCpuInput, timerCpuOutput, timerCpuComputing;
    TimerGpu timerGpuMemHostToDev, timerGpuMemDevToHost, timerGpuComputing;

    thrust::device_vector<u32> dDiff;
    thrust::device_vector<u32> dScannedDiff;
    thrust::device_vector<byte> dData;
    u32* dCompactedDiff;
    byte* dValues;
    u32* dLengths;
    u32* dNRuns;
    if (version == Gpu)
    {
        dDiff.reserve(RL_BATCH_SIZE);
        dScannedDiff.reserve(RL_BATCH_SIZE);
        dData.reserve(RL_BATCH_SIZE);
        CUDA_ERR_CHECK(cudaMalloc(&dCompactedDiff, (RL_BATCH_SIZE + 1) * sizeof(u32)));
        CUDA_ERR_CHECK(cudaMalloc(&dValues, RL_BATCH_SIZE * sizeof(byte)));
        CUDA_ERR_CHECK(cudaMalloc(&dLengths, RL_BATCH_SIZE * sizeof(u32)));
        CUDA_ERR_CHECK(cudaMalloc(&dNRuns, sizeof(u32)));
    }

    for (u64 batchIdx = 1; batchIdx <= batches; batchIdx++)
    {
        printf("Processing batch %lu out of %lu...\n", batchIdx, batches);
        u64 batchSize = batchIdx == batches ? lastBatchSize : RL_BATCH_SIZE;
        if (!nextBatchReady)
        {
            timerCpuInput.start();
            batch = readInputBatch(inFile, batchSize);
            timerCpuInput.stop();
            nextBatchReady = true;
        }

        Rl rl(rlMetadata, batchSize);
        switch (version)
        {
        case Cpu:
            timerCpuComputing.start();
            rlCompressionCPU(rl, batch);
            timerCpuComputing.stop();
            nextBatchReady = false;
            break;
            break;
        case Gpu:
            break;
        }

        timerCpuOutput.start();
        rl.writeToFile(outFile);
        timerCpuOutput.stop();
    }

    switch (version)
    {
    case Cpu:
        printCpuTimers(timerCpuInput, timerCpuComputing, timerCpuOutput);
        break;
    case Gpu:
        CUDA_ERR_CHECK(cudaFree(dNRuns));
        CUDA_ERR_CHECK(cudaFree(dLengths));
        CUDA_ERR_CHECK(cudaFree(dValues));
        CUDA_ERR_CHECK(cudaFree(dCompactedDiff));
        printGpuTimers(timerCpuInput, timerGpuMemHostToDev, timerGpuComputing, timerGpuMemDevToHost, timerCpuOutput);
        break;
    }
    fclose(outFile);
    fclose(inFile);
}