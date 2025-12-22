#include "rl.cuh"
#include "timer.cuh"

void rlCompressionCPU(const std::vector<byte>& data, Rl& rl)
{
    if (data.empty())
    {
        return;
    }
    u64 curRun = 1;
    byte curValue = data[0];
    for (u64 i = 1; i < data.size(); i++)
    {
        if (data[i] != curValue)
        {
            rl.runs.push_back(curRun);
            rl.values.push_back(curValue);
            curRun = 1;
            curValue = data[i];
        }
        else
        {
            curRun++;
        }
    }
    rl.runs.push_back(curRun);
    rl.values.push_back(curValue);
    rl.nRuns = rl.runs.size();
}

__global__ void computeCompactedDiff(const u32* scannedDiff, u64 dataLen, u32* compactedDiff, u64* nRuns)
{
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dataLen)
    {
        return;
    }
    if (tid == dataLen - 1)
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

__global__ void computeRuns(const u32* compactedDiff, const byte* data, byte* values, u32* runs, const u64* nRuns)
{
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *nRuns)
    {
        return;
    }

    u32 l = compactedDiff[tid];
    u32 r = compactedDiff[tid + 1];
    values[tid] = data[l];
    runs[tid] = r - l;
}

void rlCompression(const std::string& inputFile, const std::string& outputFile, Version version)
{
    TimerCPU timer;
    timer.start();
    std::vector<byte> data = readDataFile(inputFile);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] reading the input file").c_str());

    u64 dataLen = data.size();
    if (dataLen == 0)
    {
        printf("Empty input file, no compression required\n");
        writeDataFile(outputFile, {});
        return;
    }

    Rl rl(dataLen);
    switch (version)
    {
    case CPU:
    {
        timer.start();
        rlCompressionCPU(data, rl);
        timer.stop();
        printf("%s\n", timer.formattedResult("[CPU] RL compression function").c_str());
        break;
    }
    case GPU:
    {
        TimerGPU timerGPU;
        timerGPU.start();
        u32* dCompactedDiff;
        CUDA_ERR_CHECK(cudaMalloc(&dCompactedDiff, (dataLen + 1) * sizeof(u32)));
        byte* dValues;
        CUDA_ERR_CHECK(cudaMalloc(&dValues, dataLen * sizeof(byte)));
        u32* dRuns;
        CUDA_ERR_CHECK(cudaMalloc(&dRuns, dataLen * sizeof(u32)));
        u64* dNRuns;
        CUDA_ERR_CHECK(cudaMalloc(&dNRuns, sizeof(u64)));
        auto dData = thrust::device_vector<byte>(data.begin(), data.end());
        auto dDiff = thrust::device_vector<u32>(dataLen);
        auto dScannedDiff = thrust::device_vector<u32>(dataLen);
        timerGPU.stop();
        printf("%s\n", timer.formattedResult("[GPU] allocating and copying data to device").c_str());

        timerGPU.start();
        dDiff[0] = 1;
        thrust::transform(thrust::device, dData.begin() + 1, dData.begin() + static_cast<long>(dataLen), dData.begin(),
                          dDiff.begin() + 1,
                          [] __device__(const int x, const int y)
                          {
                              return x == y ? 0 : 1;
                          });
        thrust::inclusive_scan(thrust::device, dDiff.begin(), dDiff.end(), dScannedDiff.begin());

        u64 gridDim = ceilDiv(dataLen, BLOCK_SIZE);
        computeCompactedDiff<<<gridDim, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(dScannedDiff.data()), dataLen, dCompactedDiff, dNRuns);

        computeRuns<<<gridDim, BLOCK_SIZE>>>(dCompactedDiff, thrust::raw_pointer_cast(dData.data()), dValues, dRuns,
                                             dNRuns);
        timerGPU.stop();
        printf("%s\n", timer.formattedResult("[GPU] RL compression kernels and Thrust functions").c_str());

        timerGPU.start();
        CUDA_ERR_CHECK(cudaMemcpy(&rl.nRuns, dNRuns, sizeof(u64), cudaMemcpyDeviceToHost));
        rl.values.resize(rl.nRuns);
        rl.runs.resize(rl.nRuns);
        CUDA_ERR_CHECK(cudaMemcpy(rl.values.data(), dValues, rl.nRuns * sizeof(byte), cudaMemcpyDeviceToHost));
        CUDA_ERR_CHECK(cudaMemcpy(rl.runs.data(), dRuns, rl.nRuns * sizeof(u32), cudaMemcpyDeviceToHost));
        CUDA_ERR_CHECK(cudaFree(dNRuns));
        CUDA_ERR_CHECK(cudaFree(dRuns));
        CUDA_ERR_CHECK(cudaFree(dValues));
        CUDA_ERR_CHECK(cudaFree(dCompactedDiff));
        timerGPU.stop();
        printf("%s\n", timer.formattedResult("[GPU] copying data to host and freeing").c_str());
        break;
    }
    }

    timer.start();
    rlToFile(outputFile, rl);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] writing the output file").c_str());
}