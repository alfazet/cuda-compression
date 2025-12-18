#include "fl.cuh"

__global__ void flDecompressionGPU();

byte* flDecompressionCPU(Arena* arena, Fl* fl, u64* decompressedLen)
{
    return nullptr;
}

void flDecompression(const char* inputFile, const char* outputFile, bool cpuVersion)
{
    Arena* arena = arenaCPUInit();
    Fl* fl = flFromFile(inputFile, arena);
    if (cpuVersion)
    {
        u64 bufLen;
        byte* buf = flDecompressionCPU(arena, fl, &bufLen);
        write_all(outputFile, buf, bufLen);
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
