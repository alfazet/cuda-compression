#include "fl.cuh"

__global__ void flCompressionGPU();

void flCompressionCPU(Fl* fl, const byte* data)
{
    for (u64 i = 0; i < fl->nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl->dataLen)
        {
            // the last chunk might be shorter than CHUNK_SIZE bytes
            len = fl->dataLen % CHUNK_SIZE;
        }

        // bit depth of this chunk = the max. MSb of all bytes contained in it
        u8 bitDepth = 0;
        for (u64 j = 0; j < len; j++)
        {
            byte thisByte = data[i * CHUNK_SIZE + j];
            u8 bitDepthHere = 0;
            for (int b = 7; b >= 0; b--)
            {
                if (((byte)1 << b) & thisByte)
                {
                    bitDepthHere = b + 1;
                    break;
                }
            }
            bitDepth = max(bitDepth, bitDepthHere);
        }
        fl->bitDepth[i] = bitDepth;

        for (u64 j = 0; j < len; j++)
        {
            byte thisByte = (data[i * CHUNK_SIZE + j] << (8 - bitDepth));
            u64 bitLoc = bitDepth * j;
            u64 byteLoc = (bitLoc >> 3); // loc = location
            u64 bitOffset = (bitLoc & 0b111);
            // place the relevant bits of this byte
            // starting at the (7 - bitOffset)-th bit of the byteLoc-th byte
            fl->chunks[i][byteLoc] |= (thisByte >> bitOffset);
            // if any bits spill over to the next byte, place them there
            if (bitOffset != 0)
            {
                fl->chunks[i][byteLoc + 1] |= (thisByte << (8 - bitOffset));
            }
        }
    }
}

void flCompression(const char* inputFile, const char* outputFile, bool cpuVersion)
{
    u64 dataLen;
    byte* data = read_all(inputFile, &dataLen);
    Arena* cpuArena = arenaCPUInit();
    Fl* fl = flInit(cpuArena, dataLen);
    if (cpuVersion)
    {
        flCompressionCPU(fl, data);
    }
    else
    {
        // create a GPU arena
        // copy Fl from CPU to GPU
        // flCompressionGPU();
        // copy Fl from GPU to CPU
        // free the GPU arena
    }
    flToFile(outputFile, fl);
    arenaCPUFree(cpuArena);
    delete[] data;
}