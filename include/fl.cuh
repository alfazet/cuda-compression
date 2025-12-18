#ifndef CUDA_COMPRESSION_FL_CUH
#define CUDA_COMPRESSION_FL_CUH

#include "common.cuh"
#include "arena.cuh"

constexpr u64 CHUNK_SIZE = 1024;

struct Fl
{
    u64 dataLen;
    u64 nChunks;
    u8* bitDepth; // # of bits to encode one byte, for each chunk
    byte (*chunks)[CHUNK_SIZE];
};

void flCompression(byte* data, u64 dataLen, const char* outPath, bool cpuVersion);

byte* flDecompression(Fl* fl, const char* outPath, bool cpuVersion);

inline Fl* flInitCPU(Arena* arena, u64 dataLen)
{
    Fl* fl = (Fl*)arenaCPUAlloc(arena, sizeof(Fl));
    fl->dataLen = dataLen;
    fl->nChunks = cdiv(dataLen, CHUNK_SIZE);
    fl->bitDepth = (u8*)arenaCPUAlloc(arena, fl->nChunks * sizeof(u8));
    fl->chunks = (byte(*)[1024])arenaCPUAlloc(arena, fl->nChunks * sizeof(byte*));

    return fl;
}

inline Fl* flInitGPU(Arena* arena, u64 dataLen)
{
    return nullptr;
}

#endif //CUDA_COMPRESSION_FL_CUH