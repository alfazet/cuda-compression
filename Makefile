FLAGS_COMMON=-Wno-deprecated-gpu-targets
FLAGS_DEBUG=-G
FLAGS_RELEASE=-O3
EXE=compress
CUDA_FILES=$(wildcard src/*.cu)
HEADER_FILES=$(wildcard include/*.cuh)
OBJ_FILES=$(patsubst src/%.cu,build/%.o,$(CUDA_FILES))

# this variable name makes no sense, but it's needed to make CLion detect the include path properly
C_FLAGS=-Iinclude

.PHONY: release debug all clean

release: $(OBJ_FILES)
	mkdir -p build/release
	nvcc $(FLAGS_COMMON) $(FLAGS_RELEASE) $(OBJ_FILES) -o build/release/$(EXE)

debug: $(OBJ_FILES)
	mkdir -p build/debug
	nvcc $(FLAGS_COMMON) $(FLAGS_DEBUG) $(OBJ_FILES) -o build/debug/$(EXE)

build/%.o: src/%.cu
	mkdir -p build
	nvcc $(FLAGS_COMMON) $(C_FLAGS) -c $< -o $@

all: debug release

clean:
	rm -rf build/
