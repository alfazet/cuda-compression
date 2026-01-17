#ifndef CUDA_COMPRESSION_RL_CUH
#define CUDA_COMPRESSION_RL_CUH

#include "common.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

constexpr u64 BLOCK_SIZE = 1024;
constexpr u64 RL_BATCH_SIZE = 1UL * 128 * 1024 * 1024; // in bytes
constexpr byte LONG_RUN_MARKER = 0x00;

/*
 *  RL data layout in memory:
 *  - length of uncompressed data (1 x u64)
 *  - Interleaved:
 *  - number of `runs` (sequences of consecutive values) (1 x u32)
 *  - run values (1 byte each)
 *  - run lengths (1 byte each (for run lengths that fit in one byte), 5 bytes for longer runs)
 */

struct RlMetadata
{
    u64 rawFileSizeTotal;

    RlMetadata() = default;

    explicit RlMetadata(u64 _rawFileSizeTotal) : rawFileSizeTotal(_rawFileSizeTotal) {}

    explicit RlMetadata(FILE* f)
    {
        u64 rawFileSizeTotal;
        FREAD_CHECK(&rawFileSizeTotal, sizeof(u64), 1, f);
        this->rawFileSizeTotal = rawFileSizeTotal;
    }

    void writeToFile(FILE* f) const { FWRITE_CHECK(&this->rawFileSizeTotal, sizeof(u64), 1, f); }
};

struct Rl
{
    RlMetadata metadata;
    u64 batchSize;
    u32 nRuns = {};
    std::vector<byte> values;
    std::vector<u32> lengths;

    Rl() = default;

    Rl(RlMetadata _metadata, u64 _batchSize) : metadata(_metadata), batchSize(_batchSize) {}

    // write the result of compressing one batch
    void writeToFile(FILE* f) const
    {
        FWRITE_CHECK(&this->nRuns, sizeof(u32), 1, f);
        FWRITE_CHECK(this->values.data(), sizeof(byte), this->nRuns, f);
        for (u64 i = 0; i < this->nRuns; i++)
        {
            u32 len = this->lengths[i];
            if (len <= 255)
            {
                u8 shortLen = static_cast<u8>(len);
                FWRITE_CHECK(&shortLen, sizeof(u8), 1, f);
            }
            else
            {
                fputc(LONG_RUN_MARKER, f);
                FWRITE_CHECK(&len, sizeof(u32), 1, f);
            }
        }
    }

    // read exactly the data needed to decompress one batch
    void readFromFile(FILE* f)
    {
        FREAD_CHECK(&this->nRuns, sizeof(u32), 1, f);
        this->values.resize(this->nRuns);
        FREAD_CHECK(this->values.data(), sizeof(byte), this->nRuns, f);
        this->lengths.resize(this->nRuns);
        for (u32 i = 0; i < this->nRuns; i++)
        {
            u8 len;
            FREAD_CHECK(&len, sizeof(u8), 1, f);
            if (len == LONG_RUN_MARKER)
            {
                FREAD_CHECK(&this->lengths[i], sizeof(u32), 1, f);
            }
            else
            {
                this->lengths[i] = static_cast<u32>(len);
            }
        }
    }
};

void rlCompression(const std::string& inputPath, const std::string& outputPath, Version version);

void rlDecompression(const std::string& inputPath, const std::string& outputPath, Version version);

#endif // CUDA_COMPRESSION_RL_CUH
