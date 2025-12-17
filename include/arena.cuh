#ifndef CUDA_COMPRESSION_ARENA_CUH
#define CUDA_COMPRESSION_ARENA_CUH

#include "common.cuh"

struct Arena
{
    u64 size;
    u64 offset;
    byte* start;
};

inline Arena* arenaCPUInit(u64 size)
{
    Arena* arena = new Arena;
    arena->size = size;
    arena->offset = 0;
    arena->start = new byte[size];
    if (arena->start == nullptr)
    {
        return nullptr;
    }

    return arena;
}

inline byte* arenaCPUAlloc(Arena* arena, u64 len)
{
    if (arena->offset + len > arena->size)
    {
        return nullptr;
    }
    arena->offset += len;

    return arena->start + arena->offset - len;
}

inline void arenaCPUFree(Arena* arena)
{
    delete[] arena->start;
}

// TODO: GPU versions
inline Arena* arenaGPUInit(u64 size)
{
    return nullptr;
}

inline byte* arenaGPUAlloc(Arena* arena, u64 len)
{
    return nullptr;
}

inline void arenaGPUFree(Arena* arena)
{
    //
}

#endif //CUDA_COMPRESSION_ARENA_CUH