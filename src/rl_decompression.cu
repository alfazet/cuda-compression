#include "rl.cuh"
#include "timer.cuh"

void rlDecompressionCPU(const Rl& rl, std::vector<byte>& data)
{
    u64 p = 0;
    for (u64 i = 0; i < rl.nRuns; i++)
    {
        for (u32 j = 0; j < rl.runs[i]; j++)
        {
            data[p] = rl.values[i];
            p++;
        }
    }
}

__global__ void rlDecompressionGPU(byte* data, u64 dataLen, const u32* runs, const byte* values, u64 nRuns)
{
    //
}

void rlDecompression(const std::string& inputPath, const std::string& outputPath, Version version)
{
    TimerCPU timer;
    timer.start();
    Rl rl = rlFromFile(inputPath);
    std::vector<byte> data(rl.dataLen);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] reading the input file").c_str());

    switch (version)
    {
    case CPU:
    {
        timer.start();
        rlDecompressionCPU(rl, data);
        timer.stop();
        printf("%s\n", timer.formattedResult("[CPU] RL decompression function").c_str());
        break;
    }
    case GPU:
    {
        break;
    }
    }

    timer.start();
    writeDataFile(outputPath, data);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] writing the output file").c_str());
}