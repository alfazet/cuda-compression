#ifndef CUDA_COMPRESSION_FL_CUH
#define CUDA_COMPRESSION_FL_CUH

#include "common.cuh"

constexpr u64 CHUNK_SIZE = 1024;

struct Fl
{
    u64 dataLen; // length of raw data (before compression)
    u64 nChunks;
    std::vector<u8> bitDepth;
    std::vector<std::array<byte, CHUNK_SIZE> > chunks;

    explicit Fl(u64 dataLen)
    {
        u64 nChunks = ceilDiv(dataLen, CHUNK_SIZE);
        this->dataLen = dataLen;
        this->nChunks = nChunks;
        this->bitDepth = std::vector<u8>(nChunks);
        this->chunks = std::vector<std::array<byte, CHUNK_SIZE> >(nChunks);
    }
};

void flCompression(const std::string& inputPath, const std::string& outputPath, Version version);

void flDecompression(const std::string& inputPath, const std::string& outputPath, Version version);

/*
 *  FL data layout in memory:
 *  - length of uncompressed data (1 x u64)
 *  - (number of chunks is implied so we skip it)
 *  - bit depths of all chunks (nChunks x u8)
 *  - contents of chunks ((bitDepth_i * CHUNK_SIZE) bits for chunk c_i)
 */
inline Fl flFromFile(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    u64 dataLen;
    FREAD_CHECK(&dataLen, sizeof(u64), 1, f);

    Fl fl(dataLen);
    FREAD_CHECK(fl.bitDepth.data(), sizeof(u8), fl.nChunks, f);
    for (u64 i = 0; i < fl.nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl.dataLen)
        {
            len = fl.dataLen % CHUNK_SIZE;
        }
        u64 nBytes = ceilDiv(fl.bitDepth[i] * len, 8UL);
        FREAD_CHECK(fl.chunks[i].data(), sizeof(u8), nBytes, f);
    }

    return fl;
}

inline void flToFile(const std::string& path, const Fl& fl)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    FWRITE_CHECK(&fl.dataLen, sizeof(u64), 1, f);
    FWRITE_CHECK(fl.bitDepth.data(), sizeof(u8), fl.nChunks, f);
    for (u64 i = 0; i < fl.nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl.dataLen)
        {
            len = fl.dataLen % CHUNK_SIZE;
        }
        u64 nBytes = ceilDiv(fl.bitDepth[i] * len, 8UL);
        FWRITE_CHECK(fl.chunks[i].data(), sizeof(u8), nBytes, f);
    }
    fclose(f);
}

#endif //CUDA_COMPRESSION_FL_CUH
