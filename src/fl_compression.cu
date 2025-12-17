#include "fl.cuh"

__global__ void flCompressionGPU();

Fl* flCompressionCPU(byte* data, u64 dataLen)
{
    return nullptr;
}

Fl* flCompression(byte* data, u64 dataLen, bool cpuVersion)
{
    if (cpuVersion)
    {
        return flCompressionCPU(data, dataLen);
    }
    else
    {
        // flCompressionGPU();
        return nullptr;
    }
}

__global__ void flDecompressionGPU();

byte* flDecompressionCPU(Fl* fl)
{
    return nullptr;
}

byte* flDecompression(Fl* fl, bool cpuVersion)
{
    if (cpuVersion)
    {
        return flDecompressionCPU(fl);
    }
    else
    {
        // flDecompressionGPU();
        return nullptr;
    }
}