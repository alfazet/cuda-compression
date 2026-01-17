#include "rl.cuh"
#include "timer.cuh"

void rlDecompressionCPU(const Rl& rl, std::vector<byte>& batch)
{
    u64 p = 0;
    for (u32 i = 0; i < rl.nRuns; i++)
    {
        for (u32 j = 0; j < rl.lengths[i]; j++)
        {
            batch[p] = rl.values[i];
            p++;
        }
    }
}