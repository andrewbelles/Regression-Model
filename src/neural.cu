#include "../include/neural.hpp"

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
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ( col < matrix->cols() && row < matrix->rows() ) {
    // Fill in correct place in memory
    matrix->data[col * matrix->rows() + row] = vector[col * matrix->rows() + row];  
  }
} 

// Scales each value in matrix by some scalar float 
__global__ void scale_matrix(Matrix *matrix, float scalar) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ( col < matrix->cols() && row < matrix->rows() ) {
    matrix->data[col * matrix->rows() + row] *= scalar;  
  }
}

// Transposition of matrix using shared memory blocks into matrix without copying  
// -- Efficient Transpose Mike Harris
__global__ static void transpose_matrix_kernel(Matrix *matrix, Matrix *result) {
  // Shared memory is coalesced. BLOCKSIZE + 1 is to resolve bank conflicts from 32x32
  __shared__ float tile[BLOCKSIZE][BLOCKSIZE+1];

  const uint m_row = matrix->rows(), m_col = matrix->cols();
  const uint r_row = result->rows(), r_col = result->cols();

  // Pull x and y indices from block/thread idx
  int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
  int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

  // Copy data into shared memory
  for (int i = 0; i < BLOCKSIZE; i += BLOCKROWS) {
    if (x < m_col && (y + i) < m_row) {
      tile[threadIdx.y + i][threadIdx.x] = matrix->data[x * m_row + (y + i)];
    }
  }

  // Wait for all threads to put data in shared memory
  __syncthreads();

  // Get new x and y indices
  x = blockIdx.y * BLOCKSIZE + threadIdx.x;
  y = blockIdx.x * BLOCKSIZE + threadIdx.y;

  // Fill result matrix with data from shared memory
  for (int i = 0; i <  BLOCKSIZE; i += BLOCKROWS) {
    if ((threadIdx.y + i) < BLOCKSIZE && (y + i)) {
      result->data[(y + i) * r_col + x] = tile[threadIdx.x][(threadIdx.y + i)];
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
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

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
__host__ void matrix_elementwise_operation(Matrix *matrix, Matrix *addend, int op_index) {
  // Collect row and column counts 
  const int m_row = matrix->rows(), m_col = matrix->cols();
  const int a_row = addend->rows(), a_col = addend->cols();

  // Check add and subtract bounding 
  if (op_index != 2) {
    assert((a_col == 1 || m_col == a_col) && (a_row == 1 || m_row == a_row));  
  }

  // Set block sizes
  dim3 block(32, 8);  
  dim3 grid((m_col + block.x - 1) / block.x, (m_row + block.y - 1) / block.y);

  // Run specified elementwise operation 
  switch (op_index) {
    case 0:
      matrix_element_operation_kernel<<<grid, block>>>(matrix, addend,
        [] __device__ (float a, float b) {return a + b; });
      break;
    case 1:
      matrix_element_operation_kernel<<<grid, block>>>(matrix, addend,
        [] __device__ (float a, float b) {return a - b; });
      break;
    case 2:
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

__host__ Matrix *matrix_multiplication(Matrix *A, Matrix *B) {

  const int A_row = A->rows(), A_col = A->cols();
  const int B_row = B->rows(), B_col = B->cols();
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
