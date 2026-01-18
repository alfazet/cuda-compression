#include "common.cuh"
#include "fl.cuh"
#include "rl.cuh"

enum OpKind
{
    Compress,
    Decompress,
};

enum AlgoKind
{
    Fl,
    Rl,
};

struct Args
{
    std::string inputFile;
    std::string outputFile;
    OpKind opKind;
    AlgoKind algoKind;
    Version version;
};

int parseArgs(int argc, char** argv, Args* args)
{
    if (argc < 5)
    {
        return 1;
    }

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
        return 1;
    }
    if (strcmp(argv[2], "fl") == 0)
    {
        args->algoKind = Fl;
    }
    else if (strcmp(argv[2], "rl") == 0)
    {
        args->algoKind = Rl;
    }
    else
    {
        fprintf(stderr, "<method> must be fl (fixed-legth) or rl (run-length)\n");
        return 1;
    }
    args->inputFile = std::string(argv[3]);
    args->outputFile = std::string(argv[4]);
    if (argc >= 6 && strcmp(argv[5], "cpu") == 0)
    {
        args->version = Cpu;
    }
    else
    {
        args->version = Gpu;
    }

    return 0;
}

int main(int argc, char** argv)
{
    Args args;
    if (parseArgs(argc, argv, &args) != 0)
    {
        ERR_AND_DIE(USAGE_STR);
    }
    if (args.inputFile == args.outputFile)
    {
        fprintf(stderr, "The input and output files must be different\n");
        return EXIT_FAILURE;
    }
    if (!fileExists(args.inputFile))
    {
        fprintf(stderr, "File %s not found\n", args.inputFile.c_str());
        return EXIT_FAILURE;
    }

    if (args.opKind == Compress)
    {
        if (args.algoKind == Fl)
        {
            flCompression(args.inputFile, args.outputFile, args.version);
        }
        else
        {
            rlCompression(args.inputFile, args.outputFile, args.version);
        }
    }
    else
    {
        if (args.algoKind == Fl)
        {
            flDecompression(args.inputFile, args.outputFile, args.version);
        }
        else
        {
            rlDecompression(args.inputFile, args.outputFile, args.version);
        }
    }

    return EXIT_SUCCESS;
}
