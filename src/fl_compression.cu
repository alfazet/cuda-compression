#include "fl.cuh"

__global__ void flCompressionGPU(u64 dataLen, const byte* data, u8* bitDepth, byte (*chunks)[CHUNK_SIZE])
{
    u64 tidInBlock = threadIdx.x, blockId = blockIdx.x;
    u64 tidGlobal = blockId * blockDim.x + tidInBlock;
    if (tidGlobal >= dataLen)
    {
        return;
    }

    // block <-> chunk, thread <-> byte
    byte curByte = data[tidGlobal];
    u8 blockBitDepth = 8;
    while (blockBitDepth > 0 && __syncthreads_or(((byte)1 << (blockBitDepth - 1)) & curByte) == 0)
    {
        blockBitDepth--;
    }
    curByte <<= (8 - blockBitDepth);
    if (tidInBlock == 0)
    {
        bitDepth[blockId] = blockBitDepth;
    }
    u64 bitLoc = blockBitDepth * tidInBlock;
    u64 byteLoc = (bitLoc >> 3); // loc = location
    u64 bitOffset = (bitLoc & 0b111);
    // in the pessimistic case (bit depth 1), 8 consecutive threads will want to modify the same byte
    // so we have to separate them into 8 "turns" based on their id mod 8
    for (u64 i = 0; i < 8; i++)
    {
        if ((tidInBlock & 0b111) == i)
        {
            chunks[blockId][byteLoc] |= (curByte >> bitOffset);
            if (bitOffset != 0)
            {
                chunks[blockId][byteLoc + 1] |= (curByte << (8 - bitOffset));
            }
        }
        __syncthreads();
    }
}

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
            byte curByte = data[i * CHUNK_SIZE + j];
            u8 bitDepthHere = 0;
            for (int b = 7; b >= 0; b--)
            {
                if (((byte)1 << b) & curByte)
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
            byte curByte = (data[i * CHUNK_SIZE + j] << (8 - bitDepth));
            u64 bitLoc = bitDepth * j;
            u64 byteLoc = (bitLoc >> 3); // loc = location
            u64 bitOffset = (bitLoc & 0b111);
            // place the relevant bits of this byte
            // starting at the (7 - bitOffset)-th bit of the byteLoc-th byte
            fl->chunks[i][byteLoc] |= (curByte >> bitOffset);
            // if any bits spill over to the next byte, place them there
            if (bitOffset != 0)
            {
                fl->chunks[i][byteLoc + 1] |= (curByte << (8 - bitOffset));
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
        byte* dData;
        cudaErrCheck(cudaMalloc(&dData, dataLen * sizeof(byte)));
        cudaErrCheck(cudaMemcpy(dData, data, dataLen * sizeof(byte), cudaMemcpyHostToDevice));

        // CPU -> GPU
        u8* dBitDepth;
        cudaErrCheck(cudaMalloc(&dBitDepth, fl->nChunks * sizeof(u8)));
        cudaErrCheck(cudaMemcpy(dBitDepth, fl->bitDepth, fl->nChunks * sizeof(u8), cudaMemcpyHostToDevice));
        byte (*dChunks)[CHUNK_SIZE];
        cudaErrCheck(cudaMalloc(&dChunks, fl->nChunks * CHUNK_SIZE * sizeof(byte)));
        for (u64 i = 0; i < fl->nChunks; i++)
        {
            cudaErrCheck(cudaMemcpy(dChunks + i, fl->chunks[i], CHUNK_SIZE * sizeof(byte),
                                    cudaMemcpyHostToDevice));
        }

        flCompressionGPU<<<fl->nChunks, CHUNK_SIZE>>>(dataLen, dData, dBitDepth, dChunks);

        // GPU -> CPU
        cudaErrCheck(cudaMemcpy(fl->bitDepth, dBitDepth, fl->nChunks * sizeof(u8), cudaMemcpyDeviceToHost));
        for (u64 i = 0; i < fl->nChunks; i++)
        {
            cudaErrCheck(cudaMemcpy(fl->chunks[i], dChunks + i, CHUNK_SIZE * sizeof(byte),
                                    cudaMemcpyDeviceToHost));
        }

        cudaErrCheck(cudaFree(dData));
        cudaErrCheck(cudaFree(dBitDepth));
        cudaErrCheck(cudaFree(dChunks));
    }
    flToFile(outputFile, fl);
    arenaCPUFree(cpuArena);
    delete[] data;
}