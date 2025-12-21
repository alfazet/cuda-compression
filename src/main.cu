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
    std::string inputFile;
    std::string outputFile;
    OpKind opKind;
    AlgoKind algoKind;
    Version version;
};

std::optional<Args> parseArgs(int argc, char** argv)
{
    if (argc < 5)
    {
        return {};
    }
    Args args;

    if (strcmp(argv[1], "c") == 0)
    {
        args.opKind = Compress;
    }
    else if (strcmp(argv[1], "d") == 0)
    {
        args.opKind = Decompress;
    }
    else
    {
        fprintf(stderr, "<operation> must be c (compress) or d (decompress)\n");
        return {};
    }
    if (strcmp(argv[2], "fl") == 0)
    {
        args.algoKind = FL;
    }
    else if (strcmp(argv[2], "rl") == 0)
    {
        args.algoKind = RL;
    }
    else
    {
        fprintf(stderr, "<method> must be fl (fixed-legth) or rl (run-length)\n");
        return {};
    }
    args.inputFile = std::string(argv[3]);
    args.outputFile = std::string(argv[4]);
    if (argc >= 6 && strcmp(argv[5], "cpu") == 0)
    {
        args.version = CPU;
    }
    else
    {
        args.version = GPU;
    }

    return args;
}

int main(int argc, char** argv)
{
    // TODO: RL

    std::optional<Args> args = parseArgs(argc, argv);
    if (!args.has_value())
    {
        ERR_AND_DIE(USAGE_STR);
    }

    if (args->opKind == Compress)
    {
        if (args->algoKind == FL)
        {
            flCompression(args->inputFile, args->outputFile, args->version);
        }
        else
        {
            printf("TODO: rl compression\n");
        }
    }
    else
    {
        if (args->algoKind == FL)
        {
            flDecompression(args->inputFile, args->outputFile, args->version);
        }
        else
        {
            printf("TODO: rl decompression\n");
        }
    }

    return EXIT_SUCCESS;
}
