#include "common.cuh"
#include "fl.cuh"

enum OpKind
{
    Compress,
    Decompress,
};

enum AlgoKind
{
    FL,
    RL,
};

struct Args
{
    char* inputFile;
    char* outputFile;
    OpKind opKind;
    AlgoKind algoKind;
    bool cpuVersion;
};

Args* parseArgs(int argc, char** argv)
{
    if (argc < 5)
    {
        return nullptr;
    }
    Args* args = new Args;

    if (strcmp(argv[1], "c") == 0)
    {
        args->opKind = Compress;
    }
    else if (strcmp(argv[1], "d") == 0)
    {
        args->opKind = Decompress;
    }
    else
    {
        fprintf(stderr, "<operation> must be c (compress) or d (decompress)\n");
        return nullptr;
    }
    if (strcmp(argv[2], "fl") == 0)
    {
        args->algoKind = FL;
    }
    else if (strcmp(argv[2], "rl") == 0)
    {
        args->algoKind = RL;
    }
    else
    {
        fprintf(stderr, "<method> must be fl (fixed-legth) or rl (run-length)\n");
        return nullptr;
    }
    args->inputFile = argv[3];
    args->outputFile = argv[4];
    if (argc >= 6 && strcmp(argv[5], "cpu") == 0)
    {
        args->cpuVersion = true;
    }
    else
    {
        args->cpuVersion = false;
    }

    return args;
}

int main(int argc, char** argv)
{
    // TODO: time taken for each stage - reading the input file, computing, ...
    // TODO: testing
    // TODO: FL GPU versions, RL

    Args* args = parseArgs(argc, argv);
    if (args == nullptr)
    {
        ERR_AND_DIE("usage: compress <operation> <method> <input_file> <output_file> [cpu (optional)]");
    }

    if (args->opKind == Compress)
    {
        if (args->algoKind == FL)
        {
            flCompression(args->inputFile, args->outputFile, args->cpuVersion);
        }
        else
        {
            printf("rl compression\n");
        }
    }
    else
    {
        if (args->algoKind == FL)
        {
            flDecompression(args->inputFile, args->outputFile, args->cpuVersion);
        }
        else
        {
            printf("rl decompression\n");
        }
    }
    delete args;

    return EXIT_SUCCESS;
}