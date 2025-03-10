#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../include/activation.hpp"

template<ActivationType T> 
__global__ void activate_kernel(Matrix *A) {
  const uint row = A->rows(), col = A->cols();
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y; 
  
  // Ensure bounded
  if (x < col && y < row) {
    A->data[x * row + y] = ActivationFunction<T>::activate(A->data[x * row + y]);
  }
}

template<ActivationType T> 
__global__ void derivative_kernel(Matrix *A) {
  const uint row = A->rows(), col = A->cols();
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y; 
  
  // Ensure bounded
  if (x < col && y < row) {
    A->data[x * row + y] = ActivationFunction<T>::derivative(A->data[x * row + y]);
  }
}

__host__ void activate(const uint m, const uint n, Matrix *A, ActivationType type, bool prime) {
  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((m + BLOCKSIZE - 1) / BLOCKSIZE, (n + BLOCKSIZE - 1) / BLOCKSIZE);

  // If not derivative find f(x)
  if (prime == false) {
    switch (type) {
      case ActivationType::Identity:
        activate_kernel<ActivationType::Identity><<<grid, blocks>>>(A); 
        break;
      case ActivationType::Tanh: 
        activate_kernel<ActivationType::Tanh><<<grid, blocks>>>(A);
        break;
      case ActivationType::Relu:
        activate_kernel<ActivationType::Relu><<<grid, blocks>>>(A);
        break;
      default:
        std::cerr << "Invalid Activation Function\n";
        exit(EXIT_FAILURE);
    }
  // Enum switch over f'(x)
  } else {
    switch (type) {
      case ActivationType::Identity:
        derivative_kernel<ActivationType::Identity><<<grid, blocks>>>(A); 
        break;
      case ActivationType::Tanh: 
        derivative_kernel<ActivationType::Tanh><<<grid, blocks>>>(A);
        break;
      case ActivationType::Relu:
        derivative_kernel<ActivationType::Relu><<<grid, blocks>>>(A);
        break;
      default:
        std::cerr << "Invalid Activation Function\n";
        exit(EXIT_FAILURE);
    }
  }
  // Do I have to call?
  cudaDeviceSynchronize();
}

