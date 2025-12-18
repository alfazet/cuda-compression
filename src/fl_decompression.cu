#include "fl.cuh"

__global__ void flDecompressionGPU();

void flDecompressionCPU(const Fl* fl, byte* buf)
{
    for (u64 i = 0; i < fl->nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl->dataLen)
        {
            // the last chunk might be shorter than CHUNK_SIZE bytes
            len = fl->dataLen % CHUNK_SIZE;
        }
        u8 bitDepth = fl->bitDepth[i];
        for (u64 j = 0; j < len; j++)
        {
            u64 bitLoc = bitDepth * j;
            u64 byteLoc = (bitLoc >> 3); // loc = location
            u64 bitOffset = (bitLoc & 0b111);
            byte mask = (0xff << (8 - bitDepth));
            mask = (mask >> bitOffset);
            byte decoded = ((fl->chunks[i][byteLoc] & mask) << bitOffset);
            decoded = (decoded >> (8 - bitDepth));
            if (bitOffset != 0)
            {
                mask = (0xff << (8 - bitOffset)) << (8 - bitDepth);
                decoded |= (((fl->chunks[i][byteLoc + 1] & mask) >> (8 - bitOffset)) >> (8 - bitDepth));
            }
            buf[i * CHUNK_SIZE + j] = decoded;
        }
    }
}

void flDecompression(const char* inputFile, const char* outputFile, bool cpuVersion)
{
    Arena* arena = arenaCPUInit();
    Fl* fl = flFromFile(inputFile, arena);
    if (cpuVersion)
    {
        byte* buf = new byte[fl->dataLen];
        flDecompressionCPU(fl, buf);
        write_all(outputFile, buf, fl->dataLen);
        delete[] buf;
    }
    else
    {
        // create a GPU arena
        // copy Fl from CPU to GPU
        // flDeCompressionGPU();
        // copy byte buffer from GPU to CPU
        // free the GPU arena
    }
    arenaCPUFree(arena);
}
