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

__global__ void computeCompactedDiff(const u32* scannedDiff, u64 batchSize, u32* compactedDiff, u32* nRuns)
{
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batchSize)
    {
        return;
    }
    if (tid == batchSize - 1)
    {
        compactedDiff[scannedDiff[tid]] = tid + 1;
        *nRuns = scannedDiff[tid];
    }

    if (tid == 0)
    {
        compactedDiff[0] = 0;
    }
    else if (scannedDiff[tid] != scannedDiff[tid - 1])
    {
        compactedDiff[scannedDiff[tid] - 1] = tid;
    }
}

__global__ void computeRuns(const u32* compactedDiff, const byte* batch, byte* values, u32* lengths, const u32* nRuns)
{
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *nRuns)
    {
        return;
    }

    u32 l = compactedDiff[tid];
    u32 r = compactedDiff[tid + 1];
    values[tid] = batch[l];
    lengths[tid] = r - l;
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
    u64 rawFileSize = fileSize(inputPath);
    if (rawFileSize == 0)
    {
        printf("Empty file\n");
        fclose(outFile);
        fclose(inFile);
        return;
    }

    RlMetadata rlMetadata(rawFileSize);
    rlMetadata.writeToFile(outFile);
    u64 batches = ceilDiv(rawFileSize, RL_BATCH_SIZE);
    u64 rem = rawFileSize % RL_BATCH_SIZE;
    u64 lastBatchSize = rem == 0 ? RL_BATCH_SIZE : rem;
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
        dDiff = thrust::device_vector<u32>(RL_BATCH_SIZE);
        dDiff[0] = 1;
        dScannedDiff = thrust::device_vector<u32>(RL_BATCH_SIZE);
        dData = thrust::device_vector<byte>(RL_BATCH_SIZE);
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
        case Gpu:
            timerGpuMemHostToDev.start();
            thrust::copy(batch.begin(), batch.end(), dData.begin());
            timerGpuMemHostToDev.stop();

            timerGpuComputing.start();
            thrust::transform(thrust::device, dData.begin() + 1, dData.begin() + static_cast<long>(batchSize),
                              dData.begin(), dDiff.begin() + 1,
                              [] __device__(const byte x, const byte y) { return x == y ? 0 : 1; });
            thrust::inclusive_scan(thrust::device, dDiff.begin(), dDiff.end(), dScannedDiff.begin());
            u64 gridDim = ceilDiv(batchSize, BLOCK_SIZE);
            computeCompactedDiff<<<gridDim, BLOCK_SIZE>>>(thrust::raw_pointer_cast(dScannedDiff.data()), batchSize,
                                                          dCompactedDiff, dNRuns);
            computeRuns<<<gridDim, BLOCK_SIZE>>>(dCompactedDiff, thrust::raw_pointer_cast(dData.data()), dValues,
                                                 dLengths, dNRuns);
            // the kernel launches are async so we read the next batch of input in the meantime
            if (batchIdx < batches)
            {
                timerCpuInput.start();
                u64 nextBatchSize = batchIdx + 1 == batches ? lastBatchSize : RL_BATCH_SIZE;
                batch = readInputBatch(inFile, nextBatchSize);
                timerCpuInput.stop();
                nextBatchReady = true;
            }
            timerGpuComputing.stop();

            timerGpuMemDevToHost.start();
            CUDA_ERR_CHECK(cudaMemcpy(&rl.nRuns, dNRuns, sizeof(u32), cudaMemcpyDeviceToHost));
            rl.lengths.resize(rl.nRuns);
            rl.values.resize(rl.nRuns);
            CUDA_ERR_CHECK(cudaMemcpy(rl.values.data(), dValues, rl.nRuns * sizeof(byte), cudaMemcpyDeviceToHost));
            CUDA_ERR_CHECK(cudaMemcpy(rl.lengths.data(), dLengths, rl.nRuns * sizeof(u32), cudaMemcpyDeviceToHost));
            timerGpuMemDevToHost.stop();
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
