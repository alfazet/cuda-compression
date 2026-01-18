#ifndef CUDA_COMPRESSION_COMMON_CUH
#define CUDA_COMPRESSION_COMMON_CUH

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#include "cuda_runtime.h"

typedef unsigned char byte;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long u64;

enum Version
{
    Cpu,
    Gpu,
};

constexpr char const* USAGE_STR =
    "usage: ./compress <operation> <method> <input_file> <output_file> [cpu (optional, for CPU version)]";

#define ERR_AND_DIE(reason)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        fprintf(stderr, "fatal error in %s, line %d\n", __FILE__, __LINE__);                                           \
        fprintf(stderr, "reason: %s\n", (reason));                                                                     \
        exit(EXIT_FAILURE);                                                                                            \
    } while (0)

#define CUDA_ERR_CHECK(cudaCall)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t res = cudaCall;                                                                                    \
        if (res != cudaSuccess)                                                                                        \
        {                                                                                                              \
            fprintf(stderr, "fatal cuda error in %s, line %d\n", __FILE__, __LINE__);                                  \
            fprintf(stderr, "reason: %s\n", cudaGetErrorString(res));                                                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define FREAD_CHECK(ptr, size, n, stream)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (fread(ptr, size, n, stream) != n)                                                                          \
        {                                                                                                              \
            ERR_AND_DIE("fread");                                                                                      \
        }                                                                                                              \
    } while (0)

#define FWRITE_CHECK(ptr, size, n, stream)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (fwrite(ptr, size, n, stream) != n)                                                                         \
        {                                                                                                              \
            ERR_AND_DIE("fwrite");                                                                                     \
        }                                                                                                              \
    } while (0)

template <typename T> T ceilDiv(T a, T b) { return (a + b - 1) / b; }

inline u64 fileSize(const std::string& path)
{
    struct stat statBuf;
    u32 ret = stat(path.c_str(), &statBuf);

    return ret == 0 ? statBuf.st_size : 0;
}

inline bool fileExists(const std::string& path)
{
    struct stat tmp;
    return stat(path.c_str(), &tmp) == 0;
}

inline std::vector<byte> readInputBatch(FILE* f, u64 batchSize)
{
    std::vector<byte> batch(batchSize);
    FREAD_CHECK(batch.data(), sizeof(byte), batchSize, f);

    return batch;
}

inline void writeOutputBatch(FILE* f, const std::vector<byte>& batch)
{
    FWRITE_CHECK(batch.data(), sizeof(byte), batch.size(), f);
}

#endif // CUDA_COMPRESSION_COMMON_CUH