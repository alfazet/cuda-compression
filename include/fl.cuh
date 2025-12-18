#ifndef CUDA_COMPRESSION_FL_CUH
#define CUDA_COMPRESSION_FL_CUH

#include "common.cuh"
#include "arena.cuh"

constexpr u64 CHUNK_SIZE = 1024;

struct Fl
{
    u64 dataLen; // length of raw data (before compression)
    u64 nChunks;
    u8* bitDepth; // # of bits to encode one byte, for each chunk
    byte (*chunks)[CHUNK_SIZE];
};

void flCompression(const char* inputFile, const char* outputFile, bool cpuVersion);

void flDecompression(const char* inputFile, const char* outputFile, bool cpuVersion);

inline Fl* flInit(Arena* cpuArena, u64 dataLen)
{
    Fl* fl = (Fl*)arenaCPUAlloc(cpuArena, sizeof(Fl));
    fl->dataLen = dataLen;
    fl->nChunks = cdiv(dataLen, CHUNK_SIZE);
    fl->bitDepth = (u8*)arenaCPUAlloc(cpuArena, fl->nChunks * sizeof(u8));
    fl->chunks = (byte(*)[1024])arenaCPUAlloc(cpuArena, fl->nChunks * sizeof(byte*));

    return fl;
}

inline void flCopyToGPU(Arena* gpuArena, Fl* hFl)
{
    //
}

/*
 *  FL data layout in memory:
 *  - length of uncompressed data (1 x u64)
 *  - (number of chunks is implied so we skip it)
 *  - bit depths of all chunks (nChunks x u8)
 *  - contents of chunks ((bitDepth_i * CHUNK_SIZE) bits for chunk c_i)
 */
inline Fl* flFromFile(const char* path, Arena* cpuArena)
{
    FILE* f = fopen(path, "r");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    u64 dataLen;
    if (fread(&dataLen, sizeof(u64), 1, f) != 1)
    {
        ERR_AND_DIE("fread");
    }
    Fl* fl = flInit(cpuArena, dataLen);
    if (fread(fl->bitDepth, sizeof(u8), fl->nChunks, f) != 1)
    {
        ERR_AND_DIE("fread");
    }
    for (u64 i = 0; i < fl->nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl->dataLen)
        {
            len = fl->dataLen % CHUNK_SIZE;
        }
        u64 bitLen = fl->bitDepth[i] * len;
        u64 nBytes = cdiv(bitLen, 8UL);
        if (fread(fl->chunks[i], sizeof(u8), nBytes, f) != nBytes)
        {
            ERR_AND_DIE("fread");
        }
    }

    return fl;
}

inline void flToFile(const char* path, Fl* fl)
{
    FILE* f = fopen(path, "w");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    if (fwrite(&fl->dataLen, sizeof(u64), 1, f) != 1)
    {
        ERR_AND_DIE("fwrite");
    }
    if (fwrite(fl->bitDepth, sizeof(u8), fl->nChunks, f) != 1)
    {
        ERR_AND_DIE("fwrite");
    }
    for (u64 i = 0; i < fl->nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl->dataLen)
        {
            len = fl->dataLen % CHUNK_SIZE;
        }
        u64 bitLen = fl->bitDepth[i] * len;
        u64 nBytes = cdiv(bitLen, 8UL);
        if (fwrite(fl->chunks[i], sizeof(u8), nBytes, f) != nBytes)
        {
            ERR_AND_DIE("fwrite");
        }
    }
    fclose(f);
}

#endif //CUDA_COMPRESSION_FL_CUH