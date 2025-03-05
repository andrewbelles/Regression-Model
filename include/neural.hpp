#ifndef __NEURAL_HPP__
#define __NEURAL_HPP__


#include <cuda_runtime.h> 
#include <cstdint>
#include <assert.h>
#include <vector> 
#include <random>
#include <iostream>
#include "../include/cuda_arena.hpp"
#include "../include/activation.hpp"
#include "../include/matrix.hpp"

// Matrix Structure Reference
/*
int row
int col 
float *data
*/

// Single Layer in Network
typedef struct {
  ActivationType type;
  Matrix weights;
  Matrix biases;
} Layer;

// Full network
struct Network {
  Layer *layers;
  Matrix *activations;
  int layer_count;
  uint *sizes;
  uint64_t total_size;
  uint *layer_strides;
  uint *activation_strides;

  __host__ __device__ uint* get_sizes() const { return sizes; }
  __host__ __device__ uint get_layer() const { return layer_count; }
};


__host__ Network *new_network(uint *layer_sizes, uint layer_count, uint input_size, std::vector<ActivationType> types);

__host__ Matrix *input_to_batch_array(
  ArenaAllocator &arena,
  float *input_vector,
  uint64_t input_size,
  uint feature_count,
  uint *batch_count
);

__host__ void forward_propagation(
  Network *network,
  Matrix *u_batches,
  ArenaAllocator &arena,
  uint batch_count,
  uint input_size
);


#endif // __NEURAL_HPP__
