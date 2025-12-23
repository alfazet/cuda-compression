#include "rl.cuh"
#include "timer.cuh"

void rlDecompressionCPU(const Rl& rl, std::vector<byte>& data)
{
    u64 p = 0;
    for (u64 i = 0; i < rl.nRuns; i++)
    {
        for (u32 j = 0; j < rl.runs[i]; j++)
        {
            data[p] = rl.values[i];
            p++;
        }
    }
}

__global__ void rlFillData(byte* data, u64 dataLen, const u64* scannedRuns, const byte* values, u64 nRuns)
{
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dataLen)
    {
        return;
    }

    // binary search for the smallest i such that scannedRuns[i] - 1 >= tid,
    // (minus one to account for 0-based indexing)
    u64 l = 0, r = nRuns - 1, i = nRuns - 1;
    while (l < r)
    {
        u64 mid = l + (r - l) / 2;
        if (scannedRuns[mid] - 1 < tid)
        {
            l = mid + 1;
        }
        else
        {
            r = mid;
            i = min(i, mid);
        }
    }
    data[tid] = values[i];
}

void rlDecompression(const std::string& inputPath, const std::string& outputPath, Version version)
{
    TimerCPU timer;
    timer.start();
    const Rl rl = rlFromFile(inputPath);
    std::vector<byte> data(rl.dataLen);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] reading the input file").c_str());

    switch (version)
    {
    case CPU:
    {
        timer.start();
        rlDecompressionCPU(rl, data);
        timer.stop();
        printf("%s\n", timer.formattedResult("[CPU] RL decompression function").c_str());
        break;
    }
    case GPU:
    {
        TimerGPU timerGPU;
        timerGPU.start();
        byte* dData;
        CUDA_ERR_CHECK(cudaMalloc(&dData, rl.dataLen * sizeof(byte)));
        byte* dValues;
        CUDA_ERR_CHECK(cudaMalloc(&dValues, rl.nRuns * sizeof(byte)));
        CUDA_ERR_CHECK(cudaMemcpy(dValues, rl.values.data(), rl.nRuns * sizeof(byte), cudaMemcpyHostToDevice));
        auto dRuns = thrust::device_vector<u32>(rl.runs.begin(), rl.runs.end());
        auto dScannedRuns = thrust::device_vector<u64>(rl.nRuns);
        timerGPU.stop();
        printf("%s\n", timerGPU.formattedResult("[GPU] allocating and copying data to device").c_str());

        timerGPU.start();
        thrust::inclusive_scan(thrust::device, dRuns.begin(), dRuns.end(), dScannedRuns.begin());
        rlFillData<<<ceilDiv(rl.dataLen, BLOCK_SIZE), BLOCK_SIZE>>>(dData, rl.dataLen,
                                                                    thrust::raw_pointer_cast(dScannedRuns.data()),
                                                                    dValues, rl.nRuns);
        timerGPU.stop();
        printf("%s\n", timerGPU.formattedResult("[GPU] RL decompression kernels and Thrust functions").c_str());

        timerGPU.start();
        CUDA_ERR_CHECK(cudaMemcpy(data.data(), dData, rl.dataLen * sizeof(byte), cudaMemcpyDeviceToHost));
        CUDA_ERR_CHECK(cudaFree(dValues));
        CUDA_ERR_CHECK(cudaFree(dData));
        timerGPU.stop();
        printf("%s\n", timerGPU.formattedResult("[GPU] copying data to host and freeing").c_str());
        break;
    }
    }

    timer.start();
    writeDataFile(outputPath, data);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] writing the output file").c_str());
}