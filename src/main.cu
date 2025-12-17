#include "common.cuh"

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
        args->opKind = OpKind::Compress;
    }
    else if (strcmp(argv[1], "d") == 0)
    {
        args->opKind = OpKind::Decompress;
    }
    else
    {
        fprintf(stderr, "<operation> must be c (compress) or d (decompress)\n");
        return nullptr;
    }
    if (strcmp(argv[2], "fl") == 0)
    {
        args->algoKind = AlgoKind::FL;
    }
    else if (strcmp(argv[2], "rl") == 0)
    {
        args->algoKind = AlgoKind::RL;
    }
    else
    {
        fprintf(stderr, "<method> must be fl (fixed-legth) or rl (run-length)\n");
        return nullptr;
    }
    args->inputFile = argv[3];
    args->outputFile = argv[4];

    return args;
}

int main(int argc, char** argv)
{
    // "Podczas wykonania program powinien wypisywać na standardowe wyjście (z możliwością przekierowania do pliku!) informacje o przebiegu obliczeń."

    Args* args = parseArgs(argc, argv);
    if (args == nullptr)
    {
        ERR_AND_DIE("usage: compress <operation> <method> <input_file> <output_file>");
    }

    // const char* path = "test_data/lorem";
    // long len;
    // byte* content = read_all(path, &len);

    delete args;

    return EXIT_SUCCESS;
}