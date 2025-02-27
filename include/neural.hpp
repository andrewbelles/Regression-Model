#ifndef __NEURAL_HPP__
#define __NEURAL_HPP__


#include <cuda_runtime.h> 
#include <assert.h>
#include <cstdint>
#include <vector> 
#include <random>
#include <iostream>
#include "../include/cuda_arena.hpp"


#define BLOCKSIZE 16 
#define BLOCKROWS 8


enum ElementOperations {
  Hadamard,
  Add,
  Sub,
};

enum ActivationTypes {
  Leakyrelu,
  Relu,
  Elu,
  Tanh,
  Sigmoid,
};

// Function pointer to activation functions (On device)
typedef __device__ float (*ActivationFunction)(float);

// Column Major Matrix Implementation
// Using C++ Linkage for Matrix so that we can attach cols() and rows() methods
struct Matrix {
  float *data;
  int row;
  int col;

  __host__ __device__ int cols() const { return col; }
  __host__ __device__ int rows() const { return row; }
};

// Structure of function and its derivative 
typedef struct {
  ActivationFunction f, df; 
  ActivationTypes type;
} Activation;

// Single Layer in Network
typedef struct {
  Matrix weights;
  Matrix biases;
  Activation function;
} Layer;

// Full network
struct Network {
  Layer *layers;
  Matrix *activations;
  int layer_count;
  int *sizes;

  __host__ __device__ int* get_sizes() const { return sizes; }
  __host__ __device__ uint get_layer() const { return layer_count; }
};

// Visible Matrix Operations 
__host__ Matrix *new_matrix(int rows, int cols);
__global__ void fill_matrix(Matrix *matrix, float *vector);
__global__ void scale_matrix(Matrix *matrix, float scalar);
__host__ Matrix *transpose_matrix(Matrix *a);
__host__ void matrix_elementwise_operation(Matrix *matrix, Matrix *addend, ElementOperations op);
__global__ void convert_temporary_matrix(Matrix *U, Matrix *temp);

__host__ Matrix *matrix_multiplication(
  uint a_row, uint a_col, uint b_row, uint b_col,
  Matrix *A, Matrix *B, ArenaAllocator &arena
);

__host__ Network *new_network(uint *layer_sizes, uint layer_count, uint input_size, std::vector<Activation> funcs);

__host__ Matrix *input_to_batch_array(
  ArenaAllocator &arena,
  float *input_vector,
  uint64_t input_size,
  uint feature_count,
  uint *batch_count
);


#endif // __NEURAL_HPP__
