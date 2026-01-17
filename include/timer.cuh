#ifndef CUDA_COMPRESSION_TIMER_CUH
#define CUDA_COMPRESSION_TIMER_CUH

#include "common.cuh"

class TimerCpu
{
public:
    u64 elapsedMillis = 0;

    void start()
    {
        this->m_start = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        this->m_end = std::chrono::high_resolution_clock::now();
        this->elapsedMillis += std::chrono::duration_cast<std::chrono::milliseconds>(this->m_end - this->m_start).
            count();
    }

private:
    std::chrono::high_resolution_clock::time_point m_start = {};
    std::chrono::high_resolution_clock::time_point m_end = {};
};

class TimerGpu
{
public:
    float elapsedMillis = 0.0f;

    void start()
    {
        CUDA_ERR_CHECK(cudaEventCreate(&this->m_start));
        CUDA_ERR_CHECK(cudaEventCreate(&this->m_end));
        CUDA_ERR_CHECK(cudaEventRecord(this->m_start));
    }

    void stop()
    {
        CUDA_ERR_CHECK(cudaEventRecord(this->m_end));
        CUDA_ERR_CHECK(cudaEventSynchronize(this->m_end));
        float elapsed;
        CUDA_ERR_CHECK(cudaEventElapsedTime(&elapsed, this->m_start, this->m_end));
        this->elapsedMillis += elapsed;
        CUDA_ERR_CHECK(cudaEventDestroy(this->m_start));
        CUDA_ERR_CHECK(cudaEventDestroy(this->m_end));
    }

private:
    cudaEvent_t m_start = {};
    cudaEvent_t m_end = {};
};

inline void printCpuTimers(TimerCpu input, TimerCpu computing, TimerCpu output)
{

    printf("\nDone.\nTotal time required for:\n");
    printf("File input: %lu ms\n", input.elapsedMillis);
    printf("CPU computation: %lu ms\n", computing.elapsedMillis);
    printf("File output: %lu ms\n", output.elapsedMillis);
}

inline void printGpuTimers(TimerCpu input, TimerGpu hostToDev, TimerGpu computing, TimerGpu devToHost, TimerCpu output)
{

    printf("\nDone.\nTotal time required for:\n");
    printf("File input: %lu ms\n", input.elapsedMillis);
    printf("Host->device memory copying: %.3f ms\n", hostToDev.elapsedMillis);
    printf("GPU computation: %.3f ms\n", computing.elapsedMillis);
    printf("Device->host memory copying: %.3f ms\n", devToHost.elapsedMillis);
    printf("File output: %lu ms\n", output.elapsedMillis);
}

#endif //CUDA_COMPRESSION_TIMER_CUH