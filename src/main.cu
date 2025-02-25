#include "../include/neural.hpp"
#include "../include/cuda_arena.hpp"
#include <iostream>
#include <random>
#include <vector>

float tanh_derivative(float x) {
  return 1.0 - (tanh(x) * tanh(x));
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
  uint input_size = 512;
  Activation tanac = {
    .f  = [](float x) -> float { return tanh(x); },
    .df = [](float x) -> float { return 1.0 - (tanh(x) * tanh(x)); },
    .type = Tanh,
  };

  std::vector<Activation> funcs;
  for (int i = 0; i < 3; i++) {
    funcs.push_back(tanac);
  }

  float *input;
  cudaMallocManaged(&input, 512 * sizeof(float));
  for (int i = 0; i < 512; i++) {
    input[i] = distribution(gen); 
  }

  // Fill matrix with values
  Matrix *input_mat = new_matrix(input_size, 1);
  
  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((input_mat->cols() + blocks.x - 1) / blocks.x, (input_mat->rows() + blocks.y - 1) / blocks.y);

  cudaMemPrefetchAsync(input_mat->data, sizeof(float) * input_mat->cols() * input_mat->rows(), 0);
  cudaMemPrefetchAsync(input_mat, sizeof(Matrix), 0);
  cudaMemPrefetchAsync(input, sizeof(float) * input_size, 0);

  fill_matrix<<<grid, blocks>>>(input_mat, input);
  cudaDeviceSynchronize();

  Network *network = new_network(layer_sizes, layer_count, input_size, funcs);

  // Test matmul on layers between 
  std::cout << "First matmul\n";

  const uint a_row = input_mat->rows(), a_col = input_mat->cols();
  uint b_row = network->layers[0].weights.rows(), b_col = network->layers[0].weights.cols();
  Matrix *layer1_out = matrix_multiplication(a_row, a_col, b_row, b_col, input_mat, &network->layers[0].weights, arena);
  std::cout << "Second matmul\n";
  uint n_row = network->layers[1].weights.rows(), n_col = network->layers[1].weights.cols();
  Matrix *layer2_out = matrix_multiplication(a_row, b_col, n_row, n_col, layer1_out, &network->layers[1].weights, arena);
  std::cout << "Third matmul\n";
  n_row = network->layers[2].weights.rows();
  b_col = network->layers[2].weights.cols();
  Matrix *result = matrix_multiplication(a_row, n_col, n_row, b_col, layer2_out, &network->layers[2].weights, arena);
  
  Matrix *output = new_matrix(input_size, 1);
  dim3 grid_out((output->cols() + blocks.x - 1) / blocks.x, (output->rows() + blocks.y - 1) / blocks.y);

  convert_temporary_matrix<<<grid_out, blocks, 0, arena.get_stream()>>>(output, result); 
  cudaDeviceSynchronize();

  // Reset arena
  arena.reset();

  // Print output 
  for (int i = 0; i < output->cols(); i++) {
    std::cout << "[ ";
    for (int j = 0; j < output->rows(); j++) {
      std::cout << output->data[i * output->rows() + j] << " ";
    }
    std::cout << "]\n";
  }

  std::cout << "Network Created\n";

  cudaFree(network);
  cudaFree(input_mat->data);
  cudaFree(input_mat);
  cudaFree(input);

  return 0;
}
