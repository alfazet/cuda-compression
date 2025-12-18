#include "fl.cuh"

__global__ void flCompressionGPU();

Fl* flCompressionCPU(Arena* arena, byte* data, u64 dataLen)
{
    Fl* fl = flInitCPU(arena, dataLen);
    // printf("data len: %lu\n", dataLen);
    for (u64 i = 0; i < fl->nChunks; i++)
    {
        // printf("processing chunk %lu\n", i);
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > dataLen)
        {
            len = fl->dataLen % CHUNK_SIZE;
        }
        // printf("bytes: %lu\n", len);

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
                    // printf("msb of %hhu is %hhu\n", thisByte, b + 1);
                    bitDepthHere = b + 1;
                    break;
                }
            }
            bitDepth = max(bitDepth, bitDepthHere);
        }
        fl->bitDepth[i] = bitDepth;
        // printf("bit depth: %hhu\n", bitDepth);

        for (u64 j = 0; j < len; j++)
        {
            byte thisByte = (data[i * CHUNK_SIZE + j] << (8 - bitDepth));
            // printf("current byte: %hhu\n", thisByte);
            u64 bitLoc = bitDepth * j;
            u64 byteLoc = (bitLoc >> 3); // loc = location
            u64 bitOffset = (bitLoc & 0b111);
            // place the relevant bits of this byte
            // starting at the (7 - bitOffset)-th bit of the byteLoc-th byte
            fl->chunks[i][byteLoc] |= (thisByte >> bitOffset);
            // printf("oring byte %lu with %hhu\n", byteLoc, (thisByte >> bitOffset));
            // if any bits spill over to the next byte, place them there
            if (bitOffset != 0)
            {
                fl->chunks[i][byteLoc + 1] |= (thisByte << (8 - bitOffset));
                // printf("spillover: oring byte %lu with %hhu\n", byteLoc + 1, (thisByte << (8 - bitOffset)));
            }
        }
    }

    return fl;
}

void flCompression(byte* data, u64 dataLen, const char* outPath, bool cpuVersion)
{
    if (cpuVersion)
    {
        Arena* arena = arenaCPUInit();
        Fl* fl = flCompressionCPU(arena, data, dataLen);
        write_all(outPath, fl->chunks[0], 3);
        arenaCPUFree(arena);
    }
    else
    {
        // flCompressionGPU();
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