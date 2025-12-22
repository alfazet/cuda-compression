#ifndef CUDA_COMPRESSION_RL_CUH
#define CUDA_COMPRESSION_RL_CUH

#include "common.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

constexpr u64 BLOCK_SIZE = 1024;
constexpr byte MARKER = 0x00;

struct Rl
{
    u64 dataLen; // length of raw data (before compression)
    u64 nRuns; // length of `runs` array = length of `values` array
    std::vector<byte> values;
    std::vector<u32> runs;

    Rl(u64 dataLen, u64 nRuns)
    {
        this->dataLen = dataLen;
        this->nRuns = nRuns;
        this->values = std::vector<byte>(nRuns);
        this->runs = std::vector<u32>(nRuns);
    }

    explicit Rl(u64 dataLen)
    {
        this->dataLen = dataLen;
        this->nRuns = 0;
        this->values = {};
        this->runs = {};
    }
};

void rlCompression(const std::string& inputPath, const std::string& outputPath, Version version);

void rlDecompression(const std::string& inputPath, const std::string& outputPath, Version version);

/*
 *  RL data layout in memory:
 *  - length of uncompressed data (1 x u64)
 *  - length of `runs` (= length of `values`) (1 x u64)
 *  - values (1 byte each)
 *  - runs (1 byte each for runs whose length fits in one byte, for longer runs
 *      the first byte is 0x00 and the next 4 bytes after that carry the actual value as a u32)
 */
inline Rl rlFromFile(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    u64 dataLen, nRuns;
    FREAD_CHECK(&dataLen, sizeof(u64), 1, f);
    FREAD_CHECK(&nRuns, sizeof(u64), 1, f);

    Rl rl(dataLen, nRuns);
    FREAD_CHECK(rl.values.data(), sizeof(byte), rl.nRuns, f);
    for (u64 i = 0; i < rl.nRuns; i++)
    {
        u8 len;
        FREAD_CHECK(&len, sizeof(u8), 1, f);
        // TODO: (maybe) reserve more special bytes to handle 8-byte lengths
        if (len == MARKER)
        {
            // read the next 4 bytes because they encode the actual length of this run
            u32 actualLen;
            FREAD_CHECK(&actualLen, sizeof(u32), 1, f);
            rl.runs[i] = actualLen;
        }
        else
        {
            // this single byte is equal to the length
            rl.runs[i] = static_cast<u32>(len);
        }
    }

    return rl;
}

inline void rlToFile(const std::string& path, const Rl& rl)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    FWRITE_CHECK(&rl.dataLen, sizeof(u64), 1, f);
    FWRITE_CHECK(&rl.nRuns, sizeof(u64), 1, f);
    FWRITE_CHECK(rl.values.data(), sizeof(byte), rl.nRuns, f);
    for (u64 i = 0; i < rl.nRuns; i++)
    {
        u32 len = rl.runs[i];
        // we know that len > 0, so
        // the "special" byte 0x00 won't be written here
        if (len <= 255)
        {
            u8 shortLen = static_cast<u8>(len);
            FWRITE_CHECK(&shortLen, sizeof(u8), 1, f);
        }
        else
        {
            fputc(MARKER, f);
            FWRITE_CHECK(&len, sizeof(u32), 1, f);
        }
    }
}

#endif //CUDA_COMPRESSION_RL_CUH