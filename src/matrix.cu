#include "../include/matrix.hpp"
#include <memory>

// Matrix Kernels/Functions for GPU. 
__host__ Matrix *new_matrix(int rows, int cols) {
  // Find data size of matrix + data it can hold 
  uint64_t data_size = (rows * cols * sizeof(float));
  uint64_t size = sizeof(Matrix) + data_size;

  Matrix *result;
  cudaError_t err = cudaMallocManaged(&result, size);
  if (err != cudaSuccess) {
    std::cerr << "Matrix Malloc Failure: " << cudaGetErrorString(err) << '\n';
    return NULL;
  }

  // Set pointers
  result->row = rows;
  result->col = cols;
  result->data = reinterpret_cast<float*>(result + 1); 

  // Prefetch Memory to GPU 
  cudaMemPrefetchAsync(result, size, 0);
  cudaMemPrefetchAsync(result->data, data_size, 0);
#ifdef __debug
    std::cout << "Created New Matrix\n";
#endif
  return result;
}

// All proceeding matrix operations will be kernels to utilize GPU acceleration

// Fill matrix with vector, assertion of comparable size happens prior to call
__global__ void fill_matrix(Matrix *matrix, float *vector) {
  // Set contant indexes 
  const uint col = matrix->cols(), row = matrix->rows();
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x < col && y < row ) {
    // Fill in correct place in memory
    const uint index = x * matrix->rows() + y;
    matrix->data[index] = vector[index];  
  }
} 

// Scales each value in matrix by some scalar float 
__global__ void scale_matrix(Matrix *matrix, float scalar) {
  // Matches preceeding kernel 
  const uint col = matrix->cols(), row = matrix->rows();
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x < col && y < row ) {
    matrix->data[x * row + y] *= scalar;  
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
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Copy data into shared memory
  for (int i = 0; i < BLOCKSIZE; i += 8) {
    if (x < a_col && (y + i) < a_row) {
      tile[threadIdx.y + i][threadIdx.x] = a->data[x * a_row + (y + i)];
    }
  }

  // Wait for all threads to put data in shared memory
  __syncthreads();
  
  // Fill result matrix with data from shared memory
  for (int i = 0; i <  BLOCKSIZE; i += 8) {
    if ((threadIdx.y + i) < BLOCKSIZE && (y + i) < BLOCKSIZE) {
      aT->data[x * t_row + (y + i)] = tile[threadIdx.x][(threadIdx.y + i)];
    }
  }
}

// TODO: Rewrite to return a temporary array on the device and take mxn as args
__host__ Matrix *transpose_matrix(Matrix *a) {
  const uint row = a->rows(), col = a->cols();

  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((col + blocks.x - 1) / blocks.x, (row + blocks.y - 1) / blocks.y);

  Matrix *aT = new_matrix(col, row);

  transpose_matrix_kernel<<<grid, blocks>>>(a, aT);
  cudaDeviceSynchronize();

  // Handle free of a
  cudaFree(a);

  return aT;
}


// Passing lambda function through as F op 
template <typename F> 
__global__ static void matrix_element_operation_kernel(Matrix *A, const Matrix *B, F op) {
  // Collect column and row indexes 
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  const int m = A->rows(), n = A->cols();
  const int p = B->rows(), q = B->cols();

  // Handles broadcast intrinstically through y index selection
  if (row < m && col < n) {

    const int x = col * m + row;
    // Index y depends on whether addend is a "Full", column, or vector matrix 
    int y = (p == 1) 
      ? col 
      : (q == 1) 
        ? row 
        : col * p + row;
    
    A->data[x] = op(A->data[x], B->data[y]);
  }
}

// Performs an element wise operation between two matrices 
__host__ void matrix_elementwise_operation(
  uint m,
  uint n,
  uint p,
  uint q,
  Matrix *A,
  Matrix *B,
  ElementOperations op
) {
  // Check add and subtract bounding 
  if (op != Hadamard) {
    assert((p == 1 && n == q) || (q == 1 || m == p) || (m == p && n == q));  
  }

  // Set block sizes
  dim3 block(32, 8);  
  dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

  // Run specified elementwise operation 
  switch (op) {
    case Add:
      matrix_element_operation_kernel<<<grid, block>>>(A, B,
        [] __device__ (float a, float b) {return a + b; });
      break;
    case Sub:
      matrix_element_operation_kernel<<<grid, block>>>(A, B,
        [] __device__ (float a, float b) {return a - b; });
      break;
    case Hadamard:
      // Hadamard Requires more stringent bounding 
      assert(m == p && n == q);
      matrix_element_operation_kernel<<<grid, block>>>(A, B,
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
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // Pull Array Sizes 
  const uint M = A->rows();
  const uint N = B->cols();
  const uint K = A->cols();

  // Index for C data 
  const uint C_row = blockIdx.y * BLOCKSIZE + threadIdx.y;
  const uint C_col = blockIdx.x * BLOCKSIZE + threadIdx.x;

  float temp_sum = 0.0;

  // Global Matrix pointers
  // need copies to avoid modifying 
  const float* Aptr = A->data + C_row * BLOCKSIZE * K;
  const float* Bptr = B->data + C_col * BLOCKSIZE;

  for (int block = 0; block < K; block += BLOCKSIZE) {
    
    // Check bounds 
    if (threadIdx.y * BLOCKSIZE + threadIdx.x < BLOCKSIZE * BLOCKSIZE 
      && threadIdx.y * K + threadIdx.x < M * K 
      && threadIdx.y * N + threadIdx.x < N * K
    ) {
      // Set valid indices equal to pointer
      As[threadIdx.y * BLOCKSIZE + threadIdx.x] = Aptr[threadIdx.y * K + threadIdx.x];
      Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = Bptr[threadIdx.y * N + threadIdx.x]; 
    } else {
      As[threadIdx.y * BLOCKSIZE + threadIdx.x] = 0.0;
      Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = 0.0;
    }

    // Wait for data to load
    __syncthreads();

    Aptr += BLOCKSIZE;
    Bptr += BLOCKSIZE * N;

    for (int dot = 0; dot < BLOCKSIZE; ++dot) {
      if (threadIdx.y * BLOCKSIZE + dot < BLOCKSIZE * BLOCKSIZE 
          && dot * BLOCKSIZE + threadIdx.x < BLOCKSIZE * BLOCKSIZE
      ) {
        temp_sum += As[threadIdx.y * BLOCKSIZE + dot] * Bs[dot * BLOCKSIZE+ threadIdx.x];
      }
    }

    // Wait for all dot products to compute 
    __syncthreads();
  }
  
  // Check bounds 
  if (C_row < M && C_col < N) {
    C->data[C_col * M + C_row] = temp_sum;
  }
}

__global__ static void init_metadata_kernel(Matrix *M, uint row, uint col) {
  M->row  = row;
  M->col  = col;
  M->data = reinterpret_cast<float*>(M + 1);
}

// Has to be 1D block as syncthreads() only coordinates across a single block (?)
__global__ static void set_matrix_kernel(Matrix *M) {

  const uint col = M->cols(), row = M->rows();
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < col && y < row) {
    const uint index = y * col + x;
    M->data[index] = 0.0;
  }
}

// This isn't how shared threads work remember
// They only sync PER BLOCK. This is a 2D block config therefore it will not act how you expect
__global__ void convert_temporary_matrix(Matrix *U, Matrix *temp) {

  const uint col = temp->cols(), row = temp->rows();
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < col && y < row) {
    const uint index = y * col + x;
    U->data[index] = temp->data[index];
  }
}

// Creates a tempory matrix using arena allocator 
__host__ static Matrix *new_temporary_matrix(ArenaAllocator &arena, uint row, uint col) {

  // Call arena for memory
  uint64_t matrix_size = sizeof(Matrix) + (col * row * sizeof(float));
  Matrix *M = static_cast<Matrix*>(arena.allocate(matrix_size));

  // Assert ptr is non-null
  assert(M != nullptr);

  // Initialize metadata

  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((col + BLOCKSIZE - 1) / BLOCKSIZE, (row + BLOCKSIZE - 1) / BLOCKSIZE);


  init_metadata_kernel<<<1, 1>>>(M, row, col);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Init sync error: " << cudaGetErrorString(err) << '\n';
    return NULL;
  }

  set_matrix_kernel<<<grid, blocks>>>(M);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Set matrix sync error: " << cudaGetErrorString(err) << '\n';
    return NULL;
  }

  // Return pointer
  return M;
}

// Call to matmul. Returns new sized matrix C
__host__ Matrix *matrix_multiplication(uint A_row, uint A_col, uint B_row, uint B_col, Matrix *A, Matrix *B, ArenaAllocator &arena) {

  assert(A_col == B_row);

  Matrix *C = new_temporary_matrix(arena, A_row, B_col);

  dim3 block(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((B_col + BLOCKSIZE - 1) / BLOCKSIZE, (A_row + BLOCKSIZE - 1) / BLOCKSIZE);

  matrix_multiplication_kernel<<<grid, block>>>(C, A, B);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Matmul sync error: " << cudaGetErrorString(err) << '\n';
    return NULL;
  }

  return C;
}
