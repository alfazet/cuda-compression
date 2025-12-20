#include "fl.cuh"

void flDecompressionCPU(const Fl& fl, std::vector<byte>& decompressed)
{
    for (u64 i = 0; i < fl.nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl.dataLen)
        {
            // the last chunk might be shorter than CHUNK_SIZE bytes
            len = fl.dataLen % CHUNK_SIZE;
        }
        u8 bitDepth = fl.bitDepth[i];
        for (u64 j = 0; j < len; j++)
        {
            u64 bitLoc = bitDepth * j;
            u64 byteLoc = bitLoc >> 3;
            u64 bitOffset = bitLoc & 0b111;
            byte mask = 0xff << (8 - bitDepth);
            mask = mask >> bitOffset;
            byte decoded = (fl.chunks[i][byteLoc] & mask) << bitOffset;
            decoded = decoded >> (8 - bitDepth);
            if (bitOffset != 0)
            {
                mask = (0xff << (8 - bitOffset)) << (8 - bitDepth);
                decoded |= ((fl.chunks[i][byteLoc + 1] & mask) >> (8 - bitOffset)) >> (8 - bitDepth);
            }
            decompressed[i * CHUNK_SIZE + j] = decoded;
        }
    }
}

// takes "flattened" chunks - chunk 0 is [0, CHUNK_SIZE), chunk 1 is [CHUNK_SIZE, 2 * CHUNK_SIZE), ...
__global__ void flDecompressionGPU(byte* data, u64 dataLen, const u8* bitDepth, const byte* chunks, u64 nChunks)
{
    //
}

void flDecompression(const std::string& inputPath, const std::string& outputPath, Version version)
{
    Fl fl = flFromFile(inputPath);
    std::vector<byte> decompressed(fl.dataLen);
    switch (version)
    {
    case CPU:
    {
        flDecompressionCPU(fl, decompressed);
        break;
    }
    case GPU:
    {
        break;
    }
    }
    writeDataFile(outputPath, decompressed);
}