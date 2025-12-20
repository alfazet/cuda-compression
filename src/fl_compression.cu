#include "fl.cuh"

void flCompressionCPU(Fl& fl, const std::vector<byte>& data)
{
    for (u64 i = 0; i < fl.nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl.dataLen)
        {
            // the last chunk might be shorter than CHUNK_SIZE bytes
            len = fl.dataLen % CHUNK_SIZE;
        }

        // bit depth of this chunk = the max. MSb of all bytes contained in it
        u8 bitDepth = 0;
        for (u64 j = 0; j < len; j++)
        {
            byte curByte = data[i * CHUNK_SIZE + j];
            u8 bitDepthHere = 0;
            for (int b = 7; b >= 0; b--)
            {
                if ((1 << b) & curByte)
                {
                    bitDepthHere = b + 1;
                    break;
                }
            }
            bitDepth = max(bitDepth, bitDepthHere);
        }
        fl.bitDepth[i] = bitDepth;
        for (u64 j = 0; j < len; j++)
        {
            byte curByte = data[i * CHUNK_SIZE + j] << (8 - bitDepth);
            u64 bitLoc = bitDepth * j;
            u64 byteLoc = bitLoc >> 3;
            u64 bitOffset = bitLoc & 0b111;
            // place the relevant bits of this byte
            // starting at the (7 - bitOffset)-th bit of the byteLoc-th byte
            fl.chunks[i][byteLoc] |= curByte >> bitOffset;
            // if any bits spill over to the next byte, place them there
            if (bitOffset != 0)
            {
                fl.chunks[i][byteLoc + 1] |= curByte << (8 - bitOffset);
            }
        }
    }
}


// takes "flattened" chunks - chunk 0 is [0, CHUNK_SIZE), chunk 1 is [CHUNK_SIZE, 2 * CHUNK_SIZE), ...
__global__ void flCompressionGPU(const byte* data, u64 dataLen, u8* bitDepth, byte* chunks, u64 nChunks)
{
    u64 tidInBlock = threadIdx.x;
    u64 tidGlobal = blockIdx.x * blockDim.x + tidInBlock;
    if (tidGlobal >= dataLen)
    {
        return;
    }

    // block <-> chunk, thread <-> byte
    byte curByte = data[tidGlobal];
    u8 blockBitDepth = 8;
    while (blockBitDepth > 0 && __syncthreads_or((1 << (blockBitDepth - 1)) & curByte) == 0)
    {
        blockBitDepth--;
    }
    curByte <<= 8 - blockBitDepth;
    if (tidInBlock == 0)
    {
        bitDepth[blockIdx.x] = blockBitDepth;
    }
    u64 bitLoc = 1UL * blockBitDepth * tidInBlock;
    u64 byteLoc = bitLoc >> 3;
    u64 bitOffset = bitLoc & 0b111;
    u64 idx = blockIdx.x * blockDim.x + byteLoc; // the index of the byte we're handling
    // in the pessimistic case (bit depth 1), 8 consecutive threads will want to modify the same byte
    // so we have to separate them into 8 "turns" based on their id mod 8
    for (u64 i = 0; i < 8; i++)
    {
        if ((tidInBlock & 0b111) == i)
        {
            chunks[idx] |= curByte >> bitOffset;
            if (bitOffset != 0)
            {
                chunks[idx + 1] |= curByte << (8 - bitOffset);
            }
        }
        __syncthreads();
    }
}

void flCompression(const std::string& inputFile, const std::string& outputFile, Version version)
{
    std::vector<byte> data = readDataFile(inputFile);
    u64 dataLen = data.size();
    Fl fl(dataLen);
    switch (version)
    {
    case CPU:
    {
        flCompressionCPU(fl, data);
        break;
    }
    case GPU:
    {
        byte* dData;
        CUDA_ERR_CHECK(cudaMalloc(&dData, dataLen * sizeof(byte)));
        CUDA_ERR_CHECK(cudaMemcpy(dData, data.data(), dataLen * sizeof(byte), cudaMemcpyHostToDevice));
        u8* dBitDepth;
        CUDA_ERR_CHECK(cudaMalloc(&dBitDepth, fl.nChunks * sizeof(u8)));
        byte* dChunks;
        CUDA_ERR_CHECK(cudaMalloc(&dChunks, fl.nChunks * CHUNK_SIZE * sizeof(byte)));

        flCompressionGPU<<<fl.nChunks, CHUNK_SIZE>>>(dData, fl.dataLen, dBitDepth, dChunks, fl.nChunks);

        CUDA_ERR_CHECK(cudaMemcpy(fl.bitDepth.data(), dBitDepth, fl.nChunks * sizeof(u8), cudaMemcpyDeviceToHost));
        for (u64 i = 0; i < fl.nChunks; i++)
        {
            CUDA_ERR_CHECK(cudaMemcpy(fl.chunks[i].data(), dChunks + i * CHUNK_SIZE, CHUNK_SIZE * sizeof(byte),
                cudaMemcpyDeviceToHost));
        }

        CUDA_ERR_CHECK(cudaFree(dData));
        CUDA_ERR_CHECK(cudaFree(dBitDepth));
        CUDA_ERR_CHECK(cudaFree(dChunks));
        break;
    }
    }
    flToFile(outputFile, fl);
}
