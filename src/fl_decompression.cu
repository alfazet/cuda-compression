#include "fl.cuh"

__global__ void flDecompressionGPU(u64 dataLen, byte* decompressed, const u8* bitDepth,
                                   const byte (*chunks)[CHUNK_SIZE])
{
    u64 tidInBlock = threadIdx.x, blockId = blockIdx.x;
    u64 tidGlobal = blockId * blockDim.x + tidInBlock;
    if (tidGlobal >= dataLen)
    {
        return;
    }

    u8 blockBitDepth = bitDepth[blockId];
    u64 bitLoc = blockBitDepth * tidInBlock;
    u64 byteLoc = (bitLoc >> 3); // loc = location
    u64 bitOffset = (bitLoc & 0b111);
    byte mask = (0xff << (8 - blockBitDepth));
    mask = (mask >> bitOffset);
    byte decoded = ((chunks[blockId][byteLoc] & mask) << bitOffset);
    decoded = (decoded >> (8 - blockBitDepth));
    if (bitOffset != 0)
    {
        mask = (0xff << (8 - bitOffset)) << (8 - blockBitDepth);
        decoded |= (((chunks[blockId][byteLoc + 1] & mask) >> (8 - bitOffset)) >> (8 - blockBitDepth));
    }
    decompressed[tidGlobal] = decoded;
}

void flDecompressionCPU(const Fl* fl, byte* decompressed)
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
            decompressed[i * CHUNK_SIZE + j] = decoded;
        }
    }
}

void flDecompression(const char* inputFile, const char* outputFile, bool cpuVersion)
{
    Arena* arena = arenaCPUInit();
    Fl* fl = flFromFile(inputFile, arena);
    byte* decompressed = new byte[fl->dataLen];
    if (cpuVersion)
    {
        flDecompressionCPU(fl, decompressed);
    }
    else
    {
        byte* dDecompressed;
        cudaErrCheck(cudaMalloc(&dDecompressed, fl->dataLen * sizeof(byte)));

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

        flDecompressionGPU<<<fl->nChunks, CHUNK_SIZE>>>(fl->dataLen, dDecompressed, dBitDepth, dChunks);

        // GPU -> CPU
        cudaErrCheck(cudaMemcpy(decompressed, dDecompressed, fl->dataLen * sizeof(byte), cudaMemcpyDeviceToHost));

        cudaErrCheck(cudaFree(dDecompressed));
        cudaErrCheck(cudaFree(dBitDepth));
        cudaErrCheck(cudaFree(dChunks));
    }
    write_all(outputFile, decompressed, fl->dataLen);
    arenaCPUFree(arena);
    delete[] decompressed;
}