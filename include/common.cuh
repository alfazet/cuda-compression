#ifndef CUDA_COMPRESSION_COMMON_CUH
#define CUDA_COMPRESSION_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "cuda_runtime.h"

typedef unsigned char byte;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long u64;

#define ERR_AND_DIE(reason) \
    do { \
        fprintf(stderr, "fatal error in %s, line %d\n", __FILE__, __LINE__); \
        fprintf(stderr, "reason: %s\n", (reason)); \
        exit(EXIT_FAILURE); \
    } while (0)

inline void cudaErrCheck(const cudaError_t res)
{
    if (res != cudaSuccess)
        ERR_AND_DIE(cudaGetErrorString(res));
}

inline byte* read_all(const char* path, u64* len)
{
    FILE* f = fopen(path, "r");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    fseek(f, 0, SEEK_END);
    u64 file_size = ftell(f);
    *len = file_size;
    fseek(f, 0, SEEK_SET);
    byte* content = new byte[file_size];
    if (content == nullptr)
    {
        ERR_AND_DIE("new");
    }
    if (fread(content, sizeof(byte), file_size, f) != file_size)
    {
        ERR_AND_DIE("fread");
    }
    fclose(f);

    return content;
}

inline void write_all(const char* path, byte* content, u64 len)
{
    FILE* f = fopen(path, "w");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    if (fwrite(content, sizeof(byte), len, f) != len)
    {
        ERR_AND_DIE("fwrite");
    }
    fclose(f);
}

#endif //CUDA_COMPRESSION_COMMON_CUH