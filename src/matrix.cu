#include "../include/matrix.hpp"

// Matrix Kernels/Functions for GPU. 
__host__ Matrix *new_matrix(int rows, int cols) {
  // Uncasted pointer and size of memory  
  void *full_ptr;
  uint64_t size = sizeof(Matrix) + (rows * cols* sizeof(float)) + 31;
  // Create memory for entire matrix block as contiguous 
  // Why the fuck is this failing
  cudaMallocManaged(&full_ptr, size);

  Matrix *result = static_cast<Matrix*>(full_ptr);
  uint64_t matrix_size = sizeof(Matrix);
  uint64_t offset = (matrix_size + 31) & ~31; 
  result->data = reinterpret_cast<float*>(reinterpret_cast<char*>(result) + offset);
  result->row = rows;
  result->col = cols;

  // Prefetch Memory to GPU 
  cudaMemPrefetchAsync(result, size, 0);
#ifdef __debug
    std::cout << "Created New Matrix\n";
#endif
  return result;
}

// All proceeding matrix operations will be kernels to utilize GPU acceleration

// Fill matrix with vector, assertion of comparable size happens prior to call
__global__ void fill_matrix(Matrix *matrix, float *vector) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x < matrix->cols() && y < matrix->rows() ) {
    // Fill in correct place in memory
    const uint index = x * matrix->rows() + y;
#ifdef __debug
    printf("(%u, %u): %u\n", x, y, index);
#endif
    matrix->data[index] = vector[index];  
  }
} 

// Scales each value in matrix by some scalar float 
__global__ void scale_matrix(Matrix *matrix, float scalar) {

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


// Passing lambda function through 
template <typename F> 
__global__ static void matrix_element_operation_kernel(Matrix *matrix, const Matrix *addend, F op) {
  // Collect column and row indexes 
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

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
__host__ void matrix_elementwise_operation(
  uint m,
  uint n,
  uint p,
  uint q,
  Matrix *matrix,
  Matrix *addend,
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
      matrix_element_operation_kernel<<<grid, block>>>(matrix, addend,
        [] __device__ (float a, float b) {return a + b; });
      break;
    case Sub:
      matrix_element_operation_kernel<<<grid, block>>>(matrix, addend,
        [] __device__ (float a, float b) {return a - b; });
      break;
    case Hadamard:
      // Hadamard Requires more stringent bounding 
      assert(m == p && n == q);
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
  uint64_t matrix_size = sizeof(Matrix);
  uint64_t offset = (matrix_size + 31) & ~31;
  M->row  = row;
  M->col  = col;
  M->data = reinterpret_cast<float*>(reinterpret_cast<char*>(M) + offset);
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
  uint col = temp->cols();
  uint row = temp->rows();

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
  uint64_t matrix_size = sizeof(Matrix) + (col * row * sizeof(float) + 31);
  matrix_size = matrix_size & ~31;
  Matrix *M = static_cast<Matrix*>(arena.allocate(matrix_size));

  std::cout << "Arena Allocated Size: " << matrix_size << " to M\n";

  // Assert ptr is non-null
  assert(M != nullptr);

  // Initialize metadata

  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((col + BLOCKSIZE - 1) / BLOCKSIZE, (row + BLOCKSIZE - 1) / BLOCKSIZE);


  init_metadata_kernel<<<1, 1>>>(M, row, col);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Init sync error: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }

  set_matrix_kernel<<<grid, blocks>>>(M);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Set matrix sync error: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }

  // Return pointer
  return M;
}

// Call to matmul. Returns new sized matrix C
__host__ Matrix *matrix_multiplication(uint A_row, uint A_col, uint B_row, uint B_col, Matrix *A, Matrix *B, ArenaAllocator &arena) {

  assert(A_col == B_row);

  Matrix *C = new_temporary_matrix(arena, A_row, B_col);
  std::cout << "Allocated C from arena allocator\n";

  dim3 block(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((B_col + BLOCKSIZE - 1) / BLOCKSIZE, (A_row + BLOCKSIZE - 1) / BLOCKSIZE);

  matrix_multiplication_kernel<<<grid, block>>>(C, A, B);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "matmul sync error: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }

  return C;
}
