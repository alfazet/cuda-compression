#ifndef CUDA_COMPRESSION_COMMON_CUH
#define CUDA_COMPRESSION_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <optional>
#include <array>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

#include "cuda_runtime.h"

typedef unsigned char byte;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long u64;

enum Version
{
    CPU,
    GPU,
};

constexpr char const* USAGE_STR =
    "usage: ./compress <operation> <method> <input_file> <output_file> [cpu (optional, for CPU version)]";

#define ERR_AND_DIE(reason) \
    do { \
        fprintf(stderr, "fatal error in %s, line %d\n", __FILE__, __LINE__); \
        fprintf(stderr, "reason: %s\n", (reason)); \
        exit(EXIT_FAILURE); \
    } while (0)

#define CUDA_ERR_CHECK(cudaCall) \
    do { \
        cudaError_t res = cudaCall; \
        if (res != cudaSuccess) { \
            fprintf(stderr, "fatal cuda error in %s, line %d\n", __FILE__, __LINE__); \
            fprintf(stderr, "reason: %s\n", cudaGetErrorString(res)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template <typename T>
T ceilDiv(T a, T b)
{
    return (a + b - 1) / b;
}

inline std::vector<byte> readDataFile(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    fseek(f, 0, SEEK_END);
    u64 fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<byte> content(fileSize);
    if (fread(content.data(), sizeof(byte), fileSize, f) != fileSize)
    {
        ERR_AND_DIE("fread");
    }
    fclose(f);

    return content;
}

inline void writeDataFile(const std::string& path, const std::vector<byte>& data)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    u64 len = data.size();
    if (fwrite(data.data(), sizeof(byte), len, f) != len)
    {
        ERR_AND_DIE("fwrite");
    }
    fclose(f);
}

#endif //CUDA_COMPRESSION_COMMON_CUH