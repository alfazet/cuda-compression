CC=nvcc
FLAGS_COMMON=-Wno-deprecated-gpu-targets
FLAGS_DEBUG=-G
FLAGS_RELEASE=-O3
EXE=build/compress
CUDA_FILES=$(wildcard src/*.cu)
HEADER_FILES=$(wildcard include/*.cuh)
OBJ_FILES=$(patsubst src/%.cu,build/%.o,$(CUDA_FILES))

.PHONY: clean debug release all

debug: $(OBJ_FILES)
	mkdir -p build/debug
	$(CC) $(FLAGS_COMMON) $(FLAGS_DEBUG) $(OBJ_FILES) -o build/debug/compress

release: $(OBJ_FILES)
	mkdir -p build/release
	$(CC) $(FLAGS_COMMON) $(FLAGS_RELEASE) $(OBJ_FILES) -o build/release/compress

$(OBJ_FILES): $(CUDA_FILES) $(HEADER_FILES)
	mkdir -p build
	$(CC) $(FLAGS_COMMON) -Iinclude -c $< -o $@

clean:
	rm -rf build/

all: debug release