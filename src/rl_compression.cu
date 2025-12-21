#include "rl.cuh"
#include "timer.cuh"

void rlCompressionCPU(const std::vector<byte>& data, Rl& rl)
{
    if (data.empty())
    {
        return;
    }
    u64 curRun = 1;
    byte curValue = data[0];
    for (u64 i = 1; i < data.size(); i++)
    {
        if (data[i] != curValue)
        {
            rl.runs.push_back(curRun);
            rl.values.push_back(curValue);
            curRun = 1;
            curValue = data[i];
        }
        else
        {
            curRun++;
        }
    }
    rl.runs.push_back(curRun);
    rl.values.push_back(curValue);
    rl.nRuns = rl.runs.size();
}

__global__ void rlCompressionGPU(const byte* data, u64 dataLen, u32* runs, byte* values, u64* nRuns)
{
    //
}

void rlCompression(const std::string& inputFile, const std::string& outputFile, Version version)
{
    TimerCPU timer;
    timer.start();
    std::vector<byte> data = readDataFile(inputFile);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] reading the input file").c_str());

    u64 dataLen = data.size();
    Rl rl(dataLen);
    switch (version)
    {
    case CPU:
    {
        timer.start();
        rlCompressionCPU(data, rl);
        timer.stop();
        printf("%s\n", timer.formattedResult("[CPU] RL compression function").c_str());
        break;
    }
    case GPU:
    {
        break;
    }
    }

    timer.start();
    rlToFile(outputFile, rl);
    timer.stop();
    printf("%s\n", timer.formattedResult("[CPU] writing the output file").c_str());
}
