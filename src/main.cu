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
    // "Podczas wykonania program powinien wypisywać na standardowe wyjście (z możliwością przekierowania do pliku!) informacje o przebiegu obliczeń."

    Args* args = parseArgs(argc, argv);
    if (args == nullptr)
    {
        ERR_AND_DIE("usage: compress <operation> <method> <input_file> <output_file> [cpu (optional)]");
    }
    u64 dataLen;
    // TODO: measure reading time
    byte* data = read_all(args->inputFile, &dataLen);

    if (args->opKind == Compress)
    {
        if (args->algoKind == FL)
        {
            printf("fl compression\n");
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
            printf("fl decompression\n");
        }
        else
        {
            printf("rl decompression\n");
        }
    }

    delete[] data;
    delete args;

    return EXIT_SUCCESS;
}