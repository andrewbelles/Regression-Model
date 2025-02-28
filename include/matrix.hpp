#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <cuda_runtime.h>
#include <cstdint>
#include <assert.h>
#include <iostream>
#include "../include/cuda_arena.hpp"

enum class Activation {
  Leakyrelu,
  Relu,
  Elu,
  Tanh,
  Sigmoid,
  Identity,
};

enum ElementOperations {
  Hadamard,
  Add,
  Sub,
};

// Column Major Matrix Implementation
// Using C++ Linkage for Matrix so that we can attach cols() and rows() methods
struct alignas(32) Matrix {
  float *data;
  int row;
  int col;

  __host__ __device__ int cols() const { return col; }
  __host__ __device__ int rows() const { return row; }
};

#define BLOCKSIZE 16 

// Visible Matrix Operations 
__host__ Matrix *new_matrix(int rows, int cols);
__global__ void fill_matrix(Matrix *matrix, float *vector);
__global__ void scale_matrix(Matrix *matrix, float scalar);
__host__ Matrix *transpose_matrix(Matrix *a);

__host__ void matrix_elementwise_operation(
  uint m, uint n,
  uint p, uint q,
  Matrix *matrix, 
  Matrix *addend, 
  ElementOperations op 
);

__global__ void convert_temporary_matrix(Matrix *U, Matrix *temp);

__host__ Matrix *matrix_multiplication(
  uint a_row, uint a_col, uint b_row, uint b_col,
  Matrix *A, Matrix *B, ArenaAllocator &arena
);

#endif 
