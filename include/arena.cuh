#ifndef CUDA_COMPRESSION_ARENA_CUH
#define CUDA_COMPRESSION_ARENA_CUH

#include "common.cuh"

constexpr u64 ARENA_SIZE = 1 << 20;

struct Arena
{
    u64 size;
    u64 offset;
    byte* start;
};

inline Arena* arenaCPUInit()
{
    Arena* arena = new Arena;
    arena->size = ARENA_SIZE;
    arena->offset = 0;
    // allocate and zero-out the memory
    arena->start = new byte[ARENA_SIZE]();
    if (arena->start == nullptr)
    {
        return nullptr;
    }

    return arena;
}

// allocate `len` bytes in the arena
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

#endif //CUDA_COMPRESSION_ARENA_CUH