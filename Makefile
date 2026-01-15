COMMON_FLAGS=-Wno-deprecated-gpu-targets --extended-lambda
FLAGS=
BUILD_DIR=build
DEBUG?=0
ifeq ($(DEBUG),0)
	FLAGS:=$(FLAGS) -O3
	BUILD_DIR:=$(BUILD_DIR)/release
else
	FLAGS:=$(FLAGS) -g -G -O0
	BUILD_DIR:=$(BUILD_DIR)/debug
endif

EXE=compress
CUDA_FILES=$(wildcard src/*.cu)
HEADER_FILES=$(wildcard include/*.cuh)
OBJS=$(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(CUDA_FILES))

C_FLAGS=-Iinclude

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJS)
	mkdir -p $(BUILD_DIR)
	nvcc $(COMMON_FLAGS) $(OBJS) -o $(BUILD_DIR)/$(EXE)

$(BUILD_DIR)/%.o: src/%.cu $(HEADER_FILES)
	mkdir -p $(BUILD_DIR)
	nvcc $(COMMON_FLAGS) $(FLAGS) $(C_FLAGS) -c $< -o $@

clean:
	rm -rf build/
