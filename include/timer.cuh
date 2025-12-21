#ifndef CUDA_COMPRESSION_TIMER_CUH
#define CUDA_COMPRESSION_TIMER_CUH

#include "common.cuh"

class TimerCPU
{
public:
    void start()
    {
        this->m_elapsedMillis = 0;
        this->m_start = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        this->m_end = std::chrono::high_resolution_clock::now();
        this->m_elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>(this->m_end - this->m_start).
            count();
    }

    std::string formattedResult(const char* name) const
    {
        return std::string(name) + ": " + std::to_string(this->m_elapsedMillis) + " ms";
    }

private:
    std::chrono::high_resolution_clock::time_point m_start = {};
    std::chrono::high_resolution_clock::time_point m_end = {};
    u64 m_elapsedMillis = 0;
};

class TimerGPU
{
public:
    void start()
    {
        this->m_elapsedMillis = 0.0f;
        CUDA_ERR_CHECK(cudaEventCreate(&this->m_start));
        CUDA_ERR_CHECK(cudaEventCreate(&this->m_end));
        CUDA_ERR_CHECK(cudaEventRecord(this->m_start));
    }

    void stop()
    {
        CUDA_ERR_CHECK(cudaEventRecord(this->m_end));
        CUDA_ERR_CHECK(cudaEventSynchronize(this->m_end));
        CUDA_ERR_CHECK(cudaEventElapsedTime(&this->m_elapsedMillis, this->m_start, this->m_end));
        CUDA_ERR_CHECK(cudaEventDestroy(this->m_start));
        CUDA_ERR_CHECK(cudaEventDestroy(this->m_end));
    }

    std::string formattedResult(const char* name) const
    {
        return std::string(name) + ": " + std::to_string(this->m_elapsedMillis) + " ms";
    }

private:
    cudaEvent_t m_start = {};
    cudaEvent_t m_end = {};
    float m_elapsedMillis = 0.0f;
};

#endif //CUDA_COMPRESSION_TIMER_CUH