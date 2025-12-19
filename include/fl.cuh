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
    fl->chunks = (byte(*)[CHUNK_SIZE])arenaCPUAlloc(cpuArena, fl->nChunks * sizeof(byte*));

    return fl;
}

inline void flFreeGPU(Fl* dFl)
{
    // segfaults
    // cudaErrCheck(cudaFree(dFl->bitDepth));
    // cudaErrCheck(cudaFree(dFl->chunks));
    cudaErrCheck(cudaFree(dFl));
}

inline void flCopyToGPU(Fl* dFl, Fl* hFl)
{
    u8* ptr1;
    byte* ptr2;
    cudaErrCheck(cudaMemcpy(&dFl->dataLen, &hFl->dataLen, sizeof(u64), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(&dFl->nChunks, &hFl->nChunks, sizeof(u64), cudaMemcpyHostToDevice));

    u64 nChunks = hFl->nChunks;
    cudaErrCheck(cudaMalloc(&ptr1, nChunks * sizeof(u8)));
    cudaErrCheck(cudaMemcpy(&dFl->bitDepth, &ptr1, sizeof(u8*), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(ptr1, hFl->bitDepth, nChunks * sizeof(u8), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc(&ptr2, nChunks * CHUNK_SIZE * sizeof(byte)));
    cudaErrCheck(cudaMemcpy(&dFl->chunks, &ptr2, sizeof(byte*), cudaMemcpyHostToDevice));
    for (u64 i = 0; i < nChunks; i++)
    {
        cudaErrCheck(cudaMemcpy(ptr2 + i * CHUNK_SIZE * sizeof(byte), hFl->chunks[i], CHUNK_SIZE * sizeof(byte),
                                cudaMemcpyHostToDevice));
    }
}

inline void flCopyToCPU(Fl* hFl, Fl* dFl)
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
    if (fread(fl->bitDepth, sizeof(u8), fl->nChunks, f) != fl->nChunks)
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
    if (fwrite(fl->bitDepth, sizeof(u8), fl->nChunks, f) != fl->nChunks)
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