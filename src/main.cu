#include "../include/neural.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <random>

int main(int argc, char *argv[]) {
  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-5.0, 5.0);

  // Compute at compile time 
  constexpr int size = 128;
  constexpr dim3 blocks(16, 16);
  constexpr dim3 grid((size + blocks.x - 1) / blocks.x, (size + blocks.y) / blocks.y);

  // Generate Unified Memory for Matrices 
  Matrix *A = new_matrix(size, 1); 
  Matrix *B = new_matrix(size, 1);  

  // Vectors to fill matrices
  float *a_vec, *b_vec;

  // Allocate as unified
  cudaMallocManaged(&a_vec, sizeof(float) * size);
  cudaMallocManaged(&b_vec, sizeof(float) * size);

  // Fill with random values  
  for (int i = 0; i < size; i++) {
    a_vec[i] = distribution(gen);
    b_vec[i] = distribution(gen);
  }

  std::cout << "Filled array completing Prefetch\n";

  // Send data to gpu 
  cudaMemPrefetchAsync(A, sizeof(Matrix), 0);
  cudaMemPrefetchAsync(B, sizeof(Matrix), 0);
  cudaMemPrefetchAsync(a_vec, sizeof(float) * size * 1, 0);
  cudaMemPrefetchAsync(b_vec, sizeof(float) * size * 1, 0);
  cudaDeviceSynchronize();

  // Fill matrices with data
  fill_matrix<<<grid, blocks>>>(A, a_vec);
  fill_matrix<<<grid, blocks>>>(B, b_vec);
  cudaDeviceSynchronize();

  scale_matrix<<<grid, blocks>>>(B, 3.0); 
  A = transpose_matrix(A);
  std::cout << "Transpose and Scale\n";

  cudaDeviceSynchronize();
  std::cout << "Call to Matrix C\n";
  Matrix *C = matrix_multiplication(A, B);

  // Prefetch back to cpu
  cudaMemPrefetchAsync(C, sizeof(Matrix), cudaCpuDeviceId);
  cudaMemPrefetchAsync(C->data, sizeof(float) * C->rows() * C->cols(), cudaCpuDeviceId);

  std::cout << "Completed mul and prefetch: " << C->rows() << '\n';

  // print result 
  for (int i = 0; i < C->cols(); i++) {
    std::cout << "[";
    for (int j = 0; j < C->rows(); j++) {
      std::cout << " " << C->data[i * size + j]; 
    }
    std::cout << "]\n";
  }

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(a_vec);
  cudaFree(b_vec);

  return 0;
}
