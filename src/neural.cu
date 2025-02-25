#include "../include/neural.hpp"
#include <random>
// Matrix Kernels/Functions for GPU. 
__host__ Matrix *new_matrix(int rows, int cols) {
  // Uncasted pointer and size of memory  
  void *full_ptr;
  uint64_t size = sizeof(Matrix) + (rows * cols* sizeof(float));

  // Create memory for entire matrix block as contiguous 
  cudaMallocManaged(&full_ptr, size);
  Matrix *result = static_cast<Matrix*>(full_ptr);
  result->data = reinterpret_cast<float*>(result + 1);
  result->row = rows;
  result->col = cols;

  // Prefetch Memory to GPU 
  cudaMemPrefetchAsync(result, size, cudaMemAdviseSetAccessedBy, 0);
  return result;
}

// All proceeding matrix operations will be kernels to utilize GPU acceleration

// Fill matrix with vector, assertion of comparable size happens prior to call
__global__ void fill_matrix(Matrix *matrix, float *vector) {
  const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if ( x < matrix->cols() && y < matrix->rows() ) {
    // Fill in correct place in memory
    matrix->data[x * matrix->rows() + y] = vector[x * matrix->rows() + y];  
  }
} 

// Scales each value in matrix by some scalar float 
__global__ void scale_matrix(Matrix *matrix, float scalar) {
  const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if ( x < matrix->cols() && y < matrix->rows() ) {
    matrix->data[x * matrix->rows() + y] *= scalar;  
  }
}

// Transposition of matrix using shared memory blocks into matrix without copying  
// -- Efficient Transpose Mike Harris
__global__ static void transpose_matrix_kernel(Matrix *a, Matrix *aT) {
  // Shared memory is coalesced. BLOCKSIZE + 1 is to resolve bank conflicts from 32x32
  __shared__ float tile[BLOCKSIZE][BLOCKSIZE+1];

  const uint a_row = a->rows(), a_col  = a->cols();
  const uint t_row = aT->rows(), t_col = aT->cols();

  // Pull x and y indices from block/thread idx
  int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

  // Copy data into shared memory
  for (int i = 0; i < BLOCKSIZE; i += BLOCKROWS) {
    if (x < a_col && (y + i) < a_row) {
      tile[threadIdx.y + i][threadIdx.x] = a->data[x * a_row + (y + i)];
    }
  }

  // Wait for all threads to put data in shared memory
  __syncthreads();
  
  // Fill result matrix with data from shared memory
  for (int i = 0; i <  BLOCKSIZE; i += BLOCKROWS) {
    if ((threadIdx.y + i) < BLOCKSIZE && (y + i) < BLOCKSIZE) {
      aT->data[(y + i) * t_col + x] = tile[threadIdx.x][(threadIdx.y + i)];
    }
  }
}

__host__ Matrix *transpose_matrix(Matrix *a) {
  const uint row = a->rows(), col = a->cols();

  dim3 blocks(16, 16);
  dim3 grid((col + 15) / 16, (row + 15) / 16);

  Matrix *aT = new_matrix(col, row);

  transpose_matrix_kernel<<<grid, blocks>>>(a, aT);
  cudaDeviceSynchronize();

  // Handle free of a
  cudaFree(a);

  return aT;
}


// Passing lambda function through 
template <typename F> 
__global__ static void matrix_element_operation_kernel(Matrix *matrix, const Matrix *addend, F op) {
  // Collect column and row indexes 
  const uint col = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint row = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  const int m_row = matrix->rows(), m_col = matrix->cols();
  const int a_row = addend->rows(), a_col = addend->cols();

  // Handles broadcast intrinstically through y index selection
  if (row < m_row && col < m_col) {

    const int x = col * m_row + row;
    // Index y depends on whether addend is a "Full", column, or vector matrix 
    int y = (a_row == 1) 
      ? col 
      : (a_col == 1) 
        ? row 
        : col * a_row + row;
    
    matrix->data[x] = op(matrix->data[x], addend->data[y]);
  }
}

// Performs an element wise operation between two matrices 
__host__ void matrix_elementwise_operation(Matrix *matrix, Matrix *addend, ElementOperations op) {
  // Collect row and column counts 
  const int m_row = matrix->rows(), m_col = matrix->cols();
  const int a_row = addend->rows(), a_col = addend->cols();

  // Check add and subtract bounding 
  if (op == Hadamard) {
    assert((a_col == 1 || m_col == a_col) && (a_row == 1 || m_row == a_row));  
  }

  // Set block sizes
  dim3 block(32, 8);  
  dim3 grid((m_col + block.x - 1) / block.x, (m_row + block.y - 1) / block.y);

  // Run specified elementwise operation 
  switch (op) {
    case Add:
      matrix_element_operation_kernel<<<grid, block>>>(matrix, addend,
        [] __device__ (float a, float b) {return a + b; });
      break;
    case Sub:
      matrix_element_operation_kernel<<<grid, block>>>(matrix, addend,
        [] __device__ (float a, float b) {return a - b; });
      break;
    case Hadamard:
      // Hadamard Requires more stringent bounding 
      assert(m_row == a_row && m_col == a_col);
      matrix_element_operation_kernel<<<grid, block>>>(matrix, addend,
        [] __device__ (float a, float b) {return a * b; });
      break;
    default: 
      break;
  }
  cudaDeviceSynchronize();
}

// Shared Cache and Memory Coalesced Kernel for efficient Matrix Multiplication
// -- siboehm 
__global__ static void matrix_multiplication_kernel(Matrix *C, const Matrix *A, const Matrix *B) {
  // Shared Cache
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE+1];
  
  // Thread and Block indices 
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Pull Array Sizes 
  const uint M = A->rows();
  const uint N = B->cols();
  const uint K = A->cols();

  // Index for C data 
  const uint C_row = by * BLOCKSIZE + ty;
  const uint C_col = bx * BLOCKSIZE + tx;

  float temp_sum = 0.0;

  // Global Matrix pointers
  // need copies to avoid modifying 
  const float* Aptr = A->data + by * BLOCKSIZE;
  const float* Bptr = B->data + bx * BLOCKSIZE * K;

  for (int block = 0; block < (K + BLOCKSIZE - 1)/(BLOCKSIZE); block++) {
    
    const int load_A_col = block * BLOCKSIZE + tx;
    const int load_A_row = ty; 

    // Check bounds 
    if (load_A_col < K && by * BLOCKSIZE + ty < M) {
      As[ty][tx] = Aptr[load_A_col * M + load_A_row]; 
    } else {
      As[ty][tx] = 0.0;
    }

    const int load_B_col = tx;
    const int load_B_row = block * BLOCKSIZE + ty; 

    if (bx * BLOCKSIZE + tx < N && load_B_row < K) {
      Bs[ty][tx] = Bptr[load_B_col * K + load_B_row];
    } else {
      Bs[ty][tx] = 0.0;
    }

    // Wait for data to load
    __syncthreads();

    Aptr += BLOCKSIZE;
    Bptr += BLOCKSIZE * N;

    for (int dot = 0; dot < BLOCKSIZE; dot++) {
      temp_sum += As[ty][dot] * Bs[dot][tx];
    }

    // Wait for all dot products to compute 
    __syncthreads();

    Aptr += BLOCKSIZE * M;
    Bptr += BLOCKSIZE;
  }
  
  // Check bounds 
  if (C_row < M && C_col < N) {
    C->data[C_col * M + C_row] = temp_sum;
  }
}

// Call to matmul. Returns new sized matrix C
__host__ Matrix *matrix_multiplication(Matrix *A, Matrix *B) {

  const uint A_row = A->rows(), A_col = A->cols();
  const uint B_row = B->rows(), B_col = B->cols();

  // Assert inner sizes match
  assert(A_col == B_row);
  
  dim3 block(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((B_col + BLOCKSIZE - 1) / BLOCKSIZE, (A_row + BLOCKSIZE - 1) / BLOCKSIZE);

  Matrix *C = new_matrix(A_row, B_col);
  cudaMemset(C->data, 0, A_row * B_col * sizeof(float));
  matrix_multiplication_kernel<<<grid, block>>>(C, A, B);
  cudaDeviceSynchronize();

  return C;
}

// Neural Network Operations 

// Accelerated activation
__global__ static void activate_kernel(Matrix *A, ActivationFunction fn) {
  const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  
  A->data[x * A->rows() + y] = fn(A->data[x * A->rows() + y]);
}

// Call to activate kernel using function passed as argument 
__host__ void activate(Matrix *A, ActivationFunction fn) {

  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((A->cols() + BLOCKSIZE - 1) / BLOCKSIZE, (A->rows() + BLOCKSIZE - 1) / BLOCKSIZE);

  // Call to kernel 
  activate_kernel<<<grid, blocks>>>(A, fn);
  cudaDeviceSynchronize();
}

__host__ Layer *new_layer(uint current_size, uint previous_size, Activation function, ActivationTypes type) {
  Layer *L;

  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((current_size + BLOCKSIZE - 1) / BLOCKSIZE, (previous_size + BLOCKSIZE - 1) / BLOCKSIZE);

  std::random_device rd;
  std::mt19937 gen(rd());
  float uniform_range = 0.0;
  float *weight_init, *bias_init;

  void *full;
  uint64_t size = sizeof(Layer) + ((previous_size * current_size * sizeof(float)) + current_size * sizeof(float)); 
  
  cudaMallocManaged(&full, size);
  L = static_cast<Layer*>(full);
  L->weights  = reinterpret_cast<Matrix*>(L + 1);
  L->biases   = reinterpret_cast<Matrix*>(L + 2);
  L->function = function;

  // Determine the type 
  switch (type) {
    case Tanh:
    case Sigmoid:
    default:
      uniform_range = sqrtf(6.0 / static_cast<float>(current_size + previous_size));
      break;
    case Leakyrelu:
    case Relu:
    case Elu:
      uniform_range = sqrtf(2.0 / static_cast<float>(previous_size)); 
      break;
  }

  std::uniform_real_distribution<> distribution(-uniform_range, uniform_range);
  
  cudaMallocManaged(&weight_init, sizeof(float) * previous_size * current_size);
  cudaMallocManaged(&bias_init, sizeof(float) * current_size);

  for (int i = 0; i < current_size; i++) {
    for (int j = 0; j < previous_size; j++) {
      weight_init[i * previous_size + j] = distribution(gen); 
      bias_init[i] = distribution(gen);
    }
  }

  fill_matrix<<<grid, blocks>>>(W, weight_init);
  fill_matrix<<<grid, blocks>>>(B, bias_init);
  cudaDeviceSynchronize();

  return L;
}

// Takes array of sizes and array of Activation functions 
__host__ Network *new_network(uint *sizes, Activation *funcs, ActivationTypes *type) {
  Network *N;

  return N;
}
