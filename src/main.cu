#include "common.cuh"

int main(int argc, char** argv)
{
    // usage: compress operation method input_file output_file
    // operation: `c`(ompress) or `d`(ecompress)
    // method: `fl` or `rl`
    // all args are required
    // "Podczas wykonania program powinien wypisywać na standardowe wyjście (z możliwością przekierowania do pliku!) informacje o przebiegu obliczeń."

    const char* path = "../../test_data/lorem";
    long len;
    byte* content = read_all(path, &len);

    printf("the first 3 bytes of content: %c%c%c\n", content[0], content[1], content[2]);
    printf("file length: %ld\n", len);
    free(content);

    return EXIT_SUCCESS;
}