#include "fl.cuh"
#include "timer.cuh"

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
__global__ void flDecompressionGPU(byte* data, u64 dataLen, const u8* bitDepth, const byte* chunks)
{
    u64 tidInBlock = threadIdx.x;
    u64 tidGlobal = blockIdx.x * blockDim.x + tidInBlock;
    if (tidGlobal >= dataLen)
    {
        return;
    }

    u8 blockBitDepth = bitDepth[blockIdx.x];
    u64 bitLoc = blockBitDepth * tidInBlock;
    u64 byteLoc = bitLoc >> 3;
    u64 bitOffset = bitLoc & 0b111;
    byte mask = 0xff << (8 - blockBitDepth);
    mask >>= bitOffset;

    u64 idx = blockIdx.x * blockDim.x + byteLoc; // the index of the byte we're handling
    byte decoded = (chunks[idx] & mask) << bitOffset;
    decoded = decoded >> (8 - blockBitDepth);
    if (bitOffset != 0)
    {
        mask = (0xff << (8 - bitOffset)) << (8 - blockBitDepth);
        decoded |= ((chunks[idx + 1] & mask) >> (8 - bitOffset)) >> (8 - blockBitDepth);
    }
    data[tidGlobal] = decoded;
}

void flDecompression(const std::string& inputPath, const std::string& outputPath, Version version)
{
    TimerCPU timer;
    timer.start();
    Fl fl = flFromFile(inputPath);
    std::vector<byte> data(fl.dataLen);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] reading the input file").c_str());

    switch (version)
    {
    case CPU:
    {
        timer.start();
        flDecompressionCPU(fl, data);
        timer.stop();
        printf("%s\n", timer.formattedResult("[CPU] decompression function").c_str());
        break;
    }
    case GPU:
    {
        TimerGPU timerGPU;
        byte* dData;
        timerGPU.start();
        CUDA_ERR_CHECK(cudaMalloc(&dData, fl.dataLen * sizeof(byte)));
        u8* dBitDepth;
        CUDA_ERR_CHECK(cudaMalloc(&dBitDepth, fl.nChunks * sizeof(u8)));
        CUDA_ERR_CHECK(cudaMemcpy(dBitDepth, fl.bitDepth.data(), fl.nChunks * sizeof(u8), cudaMemcpyHostToDevice));
        byte* dChunks;
        CUDA_ERR_CHECK(cudaMalloc(&dChunks, fl.nChunks * CHUNK_SIZE * sizeof(byte)));
        for (u64 i = 0; i < fl.nChunks; i++)
        {
            CUDA_ERR_CHECK(cudaMemcpy(dChunks + i * CHUNK_SIZE, fl.chunks[i].data(), CHUNK_SIZE * sizeof(byte),
                cudaMemcpyHostToDevice));
        }
        timerGPU.stop();
        printf("%s\n", timer.formattedResult("[GPU] allocating and copying data to device").c_str());

        timerGPU.start();
        flDecompressionGPU<<<fl.nChunks, CHUNK_SIZE>>>(dData, fl.dataLen, dBitDepth, dChunks);
        timerGPU.stop();
        printf("%s\n", timer.formattedResult("[GPU] decompression kernel").c_str());

        timerGPU.start();
        CUDA_ERR_CHECK(cudaMemcpy(data.data(), dData, fl.dataLen * sizeof(byte), cudaMemcpyDeviceToHost));
        CUDA_ERR_CHECK(cudaFree(dData));
        CUDA_ERR_CHECK(cudaFree(dBitDepth));
        CUDA_ERR_CHECK(cudaFree(dChunks));
        timerGPU.stop();
        printf("%s\n", timer.formattedResult("[GPU] copying data to host and freeing").c_str());
        break;
    }
    }

    timer.start();
    writeDataFile(outputPath, data);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] writing the output file").c_str());
}