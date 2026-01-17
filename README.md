# CUDA Compression

## Compilation and usage

Requirements:

- GNU Make
- NVCC compiler with CUDA 12.9

```shell
# add DEBUG=1 to compile with debug info
make -j$(nproc)
```

Run the binary with:

```shell
./compress <operation> <method> <input_file> <output_file> [cpu]
```

- `operation`: `c` to compress, `d` to decompress
- `method`: `fl` for fixed-length encoding, `rl` for run length encoding
- `input_file`: the file to compress/decompress
- `output_file`: where to save the output
` `cpu`: **optional** argument to launch the CPU version of the program
