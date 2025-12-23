#include "fl.cuh"
#include "timer.cuh"

void flCompressionCPU(const std::vector<byte>& data, Fl& fl)
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
                if ((0b1 << b) & curByte)
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
__global__ void flCompressionGPU(const byte* data, u64 dataLen, u8* bitDepth, byte* chunks)
{
    u64 tidInBlock = threadIdx.x;
    u64 tidGlobal = blockIdx.x * blockDim.x + tidInBlock;
    if (tidGlobal >= dataLen)
    {
        return;
    }

    byte curByte = data[tidGlobal];
    u8 blockBitDepth = 8;
    while (blockBitDepth > 0 && __syncthreads_or((0b1 << (blockBitDepth - 1)) & curByte) == 0)
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

    // split into 8 batches based on the thread id mod 8
    // to prevent data races
    for (u8 mod = 0; mod < 8; mod++)
    {
        if ((tidInBlock & 0b111) == mod)
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
    TimerCPU timer;
    timer.start();
    std::vector<byte> data = readDataFile(inputFile);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] reading the input file").c_str());

    u64 dataLen = data.size();
    if (dataLen == 0)
    {
        printf("Empty input file, no compression required\n");
        writeDataFile(outputFile, {});
        return;
    }

    Fl fl(dataLen);
    switch (version)
    {
    case CPU:
    {
        timer.start();
        flCompressionCPU(data, fl);
        timer.stop();
        printf("%s\n", timer.formattedResult("[CPU] FL compression function").c_str());
        break;
    }
    case GPU:
    {
        TimerGPU timerGPU;
        timerGPU.start();
        byte* dData;
        CUDA_ERR_CHECK(cudaMalloc(&dData, dataLen * sizeof(byte)));
        CUDA_ERR_CHECK(cudaMemcpy(dData, data.data(), dataLen * sizeof(byte), cudaMemcpyHostToDevice));
        u8* dBitDepth;
        CUDA_ERR_CHECK(cudaMalloc(&dBitDepth, fl.nChunks * sizeof(u8)));
        byte* dChunks;
        CUDA_ERR_CHECK(cudaMalloc(&dChunks, fl.nChunks * CHUNK_SIZE * sizeof(byte)));
        timerGPU.stop();
        printf("%s\n", timerGPU.formattedResult("[GPU] allocating and copying data to device").c_str());

        timerGPU.start();
        flCompressionGPU<<<fl.nChunks, CHUNK_SIZE>>>(dData, fl.dataLen, dBitDepth, dChunks);
        timerGPU.stop();
        printf("%s\n", timerGPU.formattedResult("[GPU] FL compression kernel").c_str());

        timerGPU.start();
        CUDA_ERR_CHECK(cudaMemcpy(fl.bitDepth.data(), dBitDepth, fl.nChunks * sizeof(u8), cudaMemcpyDeviceToHost));
        for (u64 i = 0; i < fl.nChunks; i++)
        {
            CUDA_ERR_CHECK(cudaMemcpy(fl.chunks[i].data(), dChunks + i * CHUNK_SIZE, CHUNK_SIZE * sizeof(byte),
                cudaMemcpyDeviceToHost));
        }
        CUDA_ERR_CHECK(cudaFree(dData));
        CUDA_ERR_CHECK(cudaFree(dBitDepth));
        CUDA_ERR_CHECK(cudaFree(dChunks));
        timerGPU.stop();
        printf("%s\n", timerGPU.formattedResult("[GPU] copying data to host and freeing").c_str());
        break;
    }
    }

    timer.start();
    flToFile(outputFile, fl);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] writing the output file").c_str());
}