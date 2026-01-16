#include "fl.cuh"
#include "timer.cuh"

void flCompressionCPU(Fl& fl, const std::vector<byte>& batch)
{
    for (u64 i = 0; i < fl.nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl.batchSize)
        {
            // the last chunk might be shorter than CHUNK_SIZE bytes
            len = fl.batchSize % CHUNK_SIZE;
        }

        // the bit depth of a chunk = the max. MSb of all bytes contained in it
        // (with one edge case: when all bytes are 0x00, the bitDepth should be 1, not 0)
        u8 bitDepth = 1;
        for (u64 j = 0; j < len; j++)
        {
            byte curByte = batch[i * CHUNK_SIZE + j];
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
            byte curByte = batch[i * CHUNK_SIZE + j] << (8 - bitDepth);
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

void flCompression(const std::string& inputPath, const std::string& outputPath, Version version)
{
    FILE* inFile = fopen(inputPath.c_str(), "rb");
    if (inFile == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    FILE* outFile = fopen(outputPath.c_str(), "wb");
    if (outFile == nullptr)
    {
        ERR_AND_DIE("fopen");
    }
    u64 rawFileSize = std::filesystem::file_size(inputPath);
    if (rawFileSize == 0)
    {
        fclose(outFile);
        fclose(inFile);
        return;
    }

    FlMetadata flMetadata(rawFileSize);
    flMetadata.writeToFile(outFile);
    u64 batches = ceilDiv(rawFileSize, BATCH_SIZE), lastBatchSize = rawFileSize % BATCH_SIZE;
    for (u64 i = 1; i <= batches; i++)
    {
        printf("Batch %lu out of %lu\n", i, batches);
        u64 batchSize = i == batches ? lastBatchSize : BATCH_SIZE;

        TimerCpu timerCpu;
        timerCpu.start();
        std::vector<byte> batch = readInputBatch(inFile, batchSize);
        timerCpu.stop();
        printf("%s\n", timerCpu.formattedResult("[CPU] reading from the input file").c_str());

        Fl fl(flMetadata, batchSize);
        switch (version)
        {
        case Cpu:
            timerCpu.start();
            flCompressionCPU(fl, batch);
            timerCpu.stop();
            printf("%s\n", timerCpu.formattedResult("[CPU] FL compression function").c_str());
            break;
        case Gpu:
            printf("TODO\n");
            break;
        }

        timerCpu.start();
        fl.writeToFile(outFile);
        timerCpu.stop();
        printf("%s\n", timerCpu.formattedResult("[CPU] writing to the output file").c_str());
    }

    fclose(outFile);
    fclose(inFile);
}