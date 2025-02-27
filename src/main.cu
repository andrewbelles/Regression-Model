#include "../include/neural.hpp"
#include "../include/cuda_arena.hpp"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <random>
#include <vector>

float tanh_derivative(float x) {
  return 1.0 - (tanh(x) * tanh(x));
}

__global__ static void set_output_buffer(void *ptr, Matrix *output, uint count, uint row, uint col) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  // If within valid idx 
  if (idx < count) {
    uint64_t offset  = count * sizeof(Matrix) + (idx * row * col * sizeof(float));
    output[idx].row  = row;
    output[idx].col  = col;
    output[idx].data = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset); 
  }
}


int main(int argc, char *argv[]) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-1,1);

  // Create our arena to have 256MB of memory -> Figure a dynamic value we can use
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  ArenaAllocator arena(512 * 512 * 1024, stream);

  std::cout << "Arena Allocated\n";

  uint layer_sizes[] = {1,256,256,1};
  uint layer_count   = 3; 
  uint input_size    = 2048;
  Activation tanac = {
    .f  = [](float x) -> float { return tanh(x); },
    .df = [](float x) -> float { return 1.0 - (tanh(x) * tanh(x)); },
    .type = Tanh,
  };

  std::vector<Activation> funcs;
  for (int i = 0; i < 3; i++) {
    funcs.push_back(tanac);
  }

  float *input = new float[input_size];
  for (int i = 0; i < input_size; i++) {
    input[i] = distribution(gen); 
  }

  Network *network = new_network(layer_sizes, layer_count, input_size, funcs);
  uint batch_count = 0;
  Matrix *batches  = input_to_batch_array(arena, input, input_size, 1, &batch_count);

  // Multiply all batches by first layer and capture result in arena allocated arrays
  // Total memory required for outputs
  uint64_t size = (2048 / 64) * sizeof(Matrix) + (256 * 2048 * sizeof(float));
  Matrix *outputs;
  void *ptr;
  cudaMallocManaged(&ptr, size);
  outputs = static_cast<Matrix*>(ptr);
  cudaMemPrefetchAsync(ptr, size, 0);
  cudaDeviceSynchronize();

  dim3 blocks(256);
  dim3 grid((32 + blocks.x - 1) / blocks.x);

  set_output_buffer<<<grid, blocks>>>(ptr, outputs, 32, 32, 256);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Output buffer invalid: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  
  blocks = dim3(16, 16);
  grid   = dim3((1 + blocks.x - 1) / blocks.x, (256 + blocks.y - 1) / blocks.y);
  for (int i = 0; i < 32; i++) {

    std::cout << "Starting Matmul " << i << '\n';

    Matrix *result = matrix_multiplication(32, 1, 1, 256, &batches[i], &network->layers[0].weights, arena);

    std::cout << "Matmul " << i << " Completed\n";

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "Matmul failure: " << cudaGetErrorString(err) << '\n';
      exit(EXIT_FAILURE);
    }

    convert_temporary_matrix<<<grid, blocks, 0, arena.get_stream()>>>(&outputs[i], result);
    std::cout << "Converted Output\n";
    if (err != cudaSuccess) {
      std::cerr << "Conversion failure: " << cudaGetErrorString(err) << '\n';
      exit(EXIT_FAILURE);
    }

    // Reset arena each iteration 
    std::cout << "Arena Leftover Capacity: " << arena.get_remaining() / 1e6 << " MB\n"; 
    cudaStreamSynchronize(arena.get_stream());
    arena.reset();
  }

  std::cout << "Batch Count: " << batch_count << '\n';
  cudaFree(network);
  cudaFree(batches);
  cudaFree(outputs);
  cudaFree(input);

  return 0;
}


