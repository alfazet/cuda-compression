#ifndef CUDA_COMPRESSION_FL_CUH
#define CUDA_COMPRESSION_FL_CUH

#include "common.cuh"

constexpr u64 CHUNK_SIZE = 1024;
constexpr u64 BATCH_SIZE = 128 * 1024 * 1024; // in bytes

/*
 *  FL data layout in memory:
 *  - length of uncompressed data (1 x u64)
 *  - (number of chunks is implied so we skip it)
 *  - Interleaved:
 *  - bit depths (nChunks x u8)
 *  - contents of chunks ((bitDepth_i * CHUNK_SIZE) bits for the i-th chunk)
 */

struct FlMetadata
{
    u64 rawFileSizeTotal;
    u64 nChunksTotal;

    FlMetadata() = default;

    explicit FlMetadata(u64 _rawFileSizeTotal) : rawFileSizeTotal(_rawFileSizeTotal), nChunksTotal(ceilDiv(_rawFileSizeTotal, CHUNK_SIZE)) {}

    explicit FlMetadata(FILE* f)
    {
        u64 rawFileSizeTotal;
        FREAD_CHECK(&rawFileSizeTotal, sizeof(u64), 1, f);
        this->rawFileSizeTotal = rawFileSizeTotal;
        this->nChunksTotal = ceilDiv(this->rawFileSizeTotal, CHUNK_SIZE);
    }

    void writeToFile(FILE* f) const
    {
        FWRITE_CHECK(&this->rawFileSizeTotal, sizeof(u64), 1, f);
    }
};

struct Fl
{
    FlMetadata metadata;
    u64 batchSize;
    u64 nChunks;
    std::vector<u8> bitDepth;
    std::vector<std::array<byte, CHUNK_SIZE> > chunks;

    Fl() = default;

    Fl(FlMetadata _metadata, u64 _batchSize) : metadata(_metadata), batchSize(_batchSize)
    {
        this->nChunks = ceilDiv(this->batchSize, CHUNK_SIZE);
        this->bitDepth = std::vector<u8>(this->nChunks);
        this->chunks = std::vector<std::array<byte, CHUNK_SIZE> >(this->nChunks);
    }

    // write the result of compressing one batch
    void writeToFile(FILE* f) const
    {
        u64 remLen = this->batchSize;
        for (u64 i = 0; i < this->nChunks; i++)
        {
            FWRITE_CHECK(&this->bitDepth[i], sizeof(u8), 1, f);
            u64 chunkLen = remLen >= CHUNK_SIZE ? CHUNK_SIZE : remLen;
            remLen -= chunkLen;
            u64 compressedLen = ceilDiv(this->bitDepth[i] * chunkLen, 8UL);
            FWRITE_CHECK(this->chunks[i].data(), sizeof(byte), compressedLen, f);
        }
    }

    // read exactly the data needed to decompress one batch
    void readFromFile(FILE* f)
    {
        u64 remLen = this->batchSize;
        for (u64 i = 0; i < this->nChunks; i++)
        {
            FREAD_CHECK(&this->bitDepth[i], sizeof(u8), 1, f);
            u64 chunkLen = remLen >= CHUNK_SIZE ? CHUNK_SIZE : remLen;
            remLen -= chunkLen;
            u64 compressedLen = ceilDiv(this->bitDepth[i] * chunkLen, 8UL);
            FREAD_CHECK(this->chunks[i].data(), sizeof(byte), compressedLen, f);
        }
    }
};

void flCompression(const std::string& inputPath, const std::string& outputPath, Version version);

void flDecompression(const std::string& inputPath, const std::string& outputPath, Version version);

#endif //CUDA_COMPRESSION_FL_CUH
