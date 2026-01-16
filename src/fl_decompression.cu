#include "fl.cuh"
#include "timer.cuh"

void flDecompressionCPU(const Fl& fl, std::vector<byte>& batch)
{
    for (u64 i = 0; i < fl.nChunks; i++)
    {
        u64 len = CHUNK_SIZE;
        if ((i + 1) * CHUNK_SIZE > fl.batchSize)
        {
            // the last chunk might be shorter than CHUNK_SIZE bytes
            len = fl.batchSize % CHUNK_SIZE;
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
            batch[i * CHUNK_SIZE + j] = decoded;
        }
    }
}

void flDecompression(const std::string& inputPath, const std::string& outputPath, Version version)
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
    FlMetadata flMetadata(inFile);
    if (flMetadata.rawFileSizeTotal == 0)
    {
        fclose(outFile);
        fclose(inFile);
        return;
    }

    u64 batches = ceilDiv(flMetadata.rawFileSizeTotal, BATCH_SIZE), lastBatchSize = flMetadata.rawFileSizeTotal % BATCH_SIZE;
    for (u64 i = 1; i <= batches; i++)
    {
        printf("Batch %lu out of %lu\n", i, batches);
        u64 batchSize = i == batches ? lastBatchSize : BATCH_SIZE;
        Fl fl(flMetadata, batchSize);
        std::vector<byte> batch(batchSize);

        TimerCpu timerCpu;
        timerCpu.start();
        fl.readFromFile(inFile);
        timerCpu.stop();
        printf("%s\n", timerCpu.formattedResult("[CPU] reading from the input file").c_str());

        switch (version)
        {
        case Cpu:
            timerCpu.start();
            flDecompressionCPU(fl, batch);
            timerCpu.stop();
            printf("%s\n", timerCpu.formattedResult("[CPU] FL decompression function").c_str());
            break;
        case Gpu:
            printf("TODO\n");
            break;
        }

        timerCpu.start();
        writeOutputBatch(outFile, batch);
        timerCpu.stop();
        printf("%s\n", timerCpu.formattedResult("[CPU] writing to the output file").c_str());
    }

    fclose(outFile);
    fclose(inFile);
}