#ifndef CUDA_COMPRESSION_FL_CUH
#define CUDA_COMPRESSION_FL_CUH

#include "common.cuh"
#include "arena.cuh"

#define CHUNK_SIZE 1024

struct Fl
{
    u64 dataLen;
    u64 nChunks;
    u8* bitDepth;
    byte (*chunks)[CHUNK_SIZE];
};

Fl* flCompression(byte* data, u64 dataLen, bool cpuVersion);

byte* flDecompression(Fl* fl, bool cpuVersion);

// TODO: provide an arena
inline Fl* flInitCPU(u64 nChunks, Arena* arena)
{

}

inline Fl* flInitGPU(u64 nChunks, Arena* arena)
{

}

#endif //CUDA_COMPRESSION_FL_CUH