# build/Makefile

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -arch=sm_86 -O3 -g -std=c++11 -extended-lambda -lineinfo 
INCLUDE_FLAGS = -I../include
LINK_FLAGS = -lcudart

# Directories
SRC_DIR = ../src
BUILD_DIR = .
OBJ_DIR = $(BUILD_DIR)/obj

# Files
SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))

# Target
TARGET = neuralnet

# Default target
all: create_dir $(TARGET)

# Create object directory
create_dir:
	@mkdir -p $(OBJ_DIR)

# Main executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $^ -o $(BUILD_DIR)/$@ $(LINK_FLAGS)

# Compilation rule for CUDA files (TAB-indented)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BUILD_DIR)/$(TARGET)

# Run
run: all
	./$(TARGET)

.PHONY: all clean run create_dir
