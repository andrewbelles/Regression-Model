#include "../include/neural.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>

// #define __debug

// Matrix Kernels/Functions for GPU. 
__host__ Matrix *new_matrix(int rows, int cols) {
  // Uncasted pointer and size of memory  
  void *full_ptr;
  uint64_t size = sizeof(Matrix) + (rows * cols* sizeof(float));
  // Create memory for entire matrix block as contiguous 
  // Why the fuck is this failing
  cudaMallocManaged(&full_ptr, size);

  Matrix *result = static_cast<Matrix*>(full_ptr);
  result->data = reinterpret_cast<float*>(result + 1);
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
    const uint index = y * matrix->cols() + x;
#ifdef __debug
    printf("(%u, %u): %u\n", x, y, index);
#endif
    matrix->data[index] = vector[index];  
  }
} 

// Scales each value in matrix by some scalar float 
__global__ void scale_matrix(Matrix *matrix, float scalar) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ static void set_matrix_kernel(Matrix *M, uint row, uint col, float *data) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    M->row  = row;
    M->col  = col;
    M->data = data;
  }

  // Ensure data ptr is set 
  __syncthreads();

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < col && y < row) {
    const uint index = y * col + x;
    M->data[index] = 0.0;
  }
}

__global__ void convert_temporary_matrix(Matrix *U, Matrix *temp) {
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    U->row = temp->rows();
    U->col = temp->cols();
  }

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  const uint col = temp->cols(), row = temp->rows();

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

  std::cout << "Arena Allocated Size: " << matrix_size << " to M\n";

  // Assert ptr is non-null
  assert(M != nullptr);

  // Initialize metadata

  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((col + BLOCKSIZE - 1) / BLOCKSIZE, (row + BLOCKSIZE - 1) / BLOCKSIZE);

  set_matrix_kernel<<<grid, blocks, 0, arena.get_stream()>>>(M, row, col, reinterpret_cast<float*>(M + 1));

  // Return pointer
  return M;
}

// Call to matmul. Returns new sized matrix C
// Since temporary matrices will be isolated to GPU we want to avoid dereferences. 
// The sizes of temporary matrices will be known and therefore can be passed as args 
__host__ Matrix *matrix_multiplication(uint A_row, uint A_col, uint B_row, uint B_col, Matrix *A, Matrix *B, ArenaAllocator &arena) {

  assert(A_col == B_row);

  Matrix *C = new_temporary_matrix(arena, A_row, B_col);
  std::cout << "Allocated C from arena allocator\n";

  dim3 block(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((B_col + BLOCKSIZE - 1) / BLOCKSIZE, (A_row + BLOCKSIZE - 1) / BLOCKSIZE);

  matrix_multiplication_kernel<<<grid, block, 0, arena.get_stream()>>>(C, A, B);

  cudaError_t err = cudaDeviceSynchronize();
  std::cout << "matmul kernel complete\n";
  if (err != cudaSuccess) {
    std::cerr << "matmul sync error: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }
  cudaStreamSynchronize(arena.get_stream());

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

// Initialize the weights and biases for a layer depending on its activation function
__host__ static void initialize_layer(Layer *layer) {

  const uint col = layer->weights.cols();
  const uint row = layer->weights.rows();

  // Set grid/block sizes for kernel launch
  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((col + BLOCKSIZE - 1) / BLOCKSIZE, (row + BLOCKSIZE - 1) / BLOCKSIZE);
  std::random_device rd;
  std::mt19937 gen(rd());
  float uniform_range = 0.0;
  float *weight_init, *bias_init;

  // Determine the type 
  switch (layer->function.type) {
    case Tanh:
    case Sigmoid:
    default:
      uniform_range = sqrtf(6.0 / static_cast<float>(col + row));
      break;
    case Leakyrelu:
    case Relu:
    case Elu:
      uniform_range = sqrtf(2.0 / static_cast<float>(row)); 
      break;
  }

  std::uniform_real_distribution<> distribution(-uniform_range, uniform_range);
  
  // Create vectors of initial values for weights and biases 
  cudaMallocManaged(&weight_init, sizeof(float) * row * col);
  cudaMallocManaged(&bias_init, sizeof(float) * col);
  for (int i = 0; i < col; i++) {
    for (int j = 0; j < row; j++) {
      weight_init[i * row + j] = distribution(gen); 
      bias_init[i] = 1e-3;
    }
  }

  cudaMemPrefetchAsync(weight_init, sizeof(float) * row * col, 0);
  cudaMemPrefetchAsync(bias_init, sizeof(float) * col, 0);
  cudaDeviceSynchronize();

  // Fill matrices with initializing values
  fill_matrix<<<grid, blocks>>>(&layer->weights, weight_init);
  fill_matrix<<<grid, blocks>>>(&layer->biases, bias_init);
  cudaDeviceSynchronize();
}

// Layer sizes is one longer than layer count 
__host__ static size_t calculate_network_size(uint *layer_sizes, uint layer_count, uint input_size) {
   
  // Find size of metadata
  uint64_t total_size = sizeof(Network);            // Network metadata
  total_size += sizeof(Layer) * layer_count;        // Layer metadata 
  total_size += sizeof(uint) * (layer_count + 1);   // Sizes array 
  total_size += sizeof(Matrix) * (layer_count + 1); // Activation array metadata

  // Iterate over each discrete layer
  for (uint i = 0; i < layer_count; i++) {
    // Find current and previous neuron counts from array
    uint previous_size = layer_sizes[i];
    uint current_size  = layer_sizes[i + 1];

    // Calculate memory for weights and biases ( and each activation )
    uint64_t weights_data    = (previous_size * current_size) * sizeof(float);
    uint64_t bias_data       = (current_size) * sizeof(float);
    uint64_t activation_data = (input_size * current_size) *sizeof(float); 

    // Sum
    total_size += activation_data + weights_data + bias_data; 
  }

  return total_size;
}

// Takes array of sizes and array of Activation functions 
__host__ Network *new_network(
  uint *layer_sizes,
  uint layer_count,
  uint input_size,
  std::vector<Activation> funcs
) {
  Network *network;
  const uint64_t total_size = calculate_network_size(layer_sizes, layer_count, input_size);
  cudaMallocManaged(&network, total_size);

  // Reinterpret initial cast to grab pointers to metadata
  network->layers      = reinterpret_cast<Layer*>(network + 1);
  network->activations = reinterpret_cast<Matrix*>(network->layers + layer_count);
  network->sizes       = reinterpret_cast<int*>(network->activations + layer_count + 1);
  network->layer_count = layer_count; 
  
  for (int i = 0; i <= layer_count; i++) {
    network->sizes[i] = layer_sizes[i];
  }
  
  // Cast address into uint64_t type 
  uint64_t pointer_offset = (uint64_t)(network->sizes + layer_count + 1);

  // Provide ownership to pointers for each Layer
  for (int i = 0; i < layer_count; i++) {
    const uint previous_size = layer_sizes[i];
    const uint current_size  = layer_sizes[i + 1];

    // Set weights metadata and collect location of data pointer from offset 
    network->layers[i].weights.row  = previous_size;
    network->layers[i].weights.col  = current_size; 
    network->layers[i].weights.data = (float*)pointer_offset;
    pointer_offset += static_cast<uint64_t>(sizeof(float) * previous_size * current_size);

    // Set bias metadata and collect location of data pointer from offset 
    network->layers[i].biases.row   = 1;
    network->layers[i].biases.col   = current_size;
    network->layers[i].biases.data  = (float*)pointer_offset;
    pointer_offset += static_cast<uint64_t>(sizeof(float) * current_size);

    // Set layers function 
    network->layers[i].function = funcs[i];
    
    initialize_layer(&network->layers[i]);
  }

  // Pointer offset is onto activations now so the array's metadata can be set 
  for (int i = 0; i <= layer_count; i++) {
    uint current_size = layer_sizes[i];

    network->activations[i].row  = current_size;
    network->activations[i].col  = input_size;
    network->activations[i].data = (float*)pointer_offset;
    pointer_offset += static_cast<uint64_t>(sizeof(float) * current_size * input_size);
  }

  // Send network to GPU. It should not return till free
  cudaError_t err = cudaMemPrefetchAsync(network, total_size, 0);
  if (err != cudaSuccess) {
    std::cerr << "Prefetch Error: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }
  return network;
}

// Efficient copy of memory from one source vector into N batch matrices
// 3D launch configuration to allow for only one kernel call
__global__ static void slice_input_vector(
  Matrix *dests,
  float *src,
  uint batch_count,
  uint batch_size,
  uint feature_count
) {
  const uint block = blockIdx.z;
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (block >= batch_count) return;

  // Pull batch for block thread is operating on 
  Matrix *dest = &dests[block];

  // Check in bounds and copy data 
  if (x < dest->cols() && y < dest->rows()) {
    uint src_index = block * batch_size * feature_count + y * feature_count + x;
    dest->data[y * dest->cols() + x] = src[src_index];
  }
}

// Takes some input vector and converts to N batch array of data split into sections. Forward pass will be a kernel operating on each batch(?)
// Allocate on arena? I don't think so(?) I kind of want activations to be the compilation of batches in a nice shared memory type layout 
__host__ Matrix *input_to_batch_array(
  ArenaAllocator &arena,
  float *input_vector,
  uint64_t input_size,
  uint feature_count,
  uint *batch_count
) {
  const uint BATCHSIZE = 64; 
  Matrix *batch_matrix;
  arena.reset();  // Force a reset to arena to ensure enough space 

  // Allocate memory on arena
  float *d_inputs = static_cast<float*>(arena.allocate(input_size * sizeof(float)));
  // Async call to copy data to input vector 
  cudaMemcpyAsync(d_inputs, input_vector, input_size * sizeof(float), cudaMemcpyHostToDevice, arena.get_stream());

  // Memory of input_matrix == batch_matrix 
  // Compute memory cost for array of matrices

  // Quotient + 1 will be allocated.
  uint quotient = input_size / BATCHSIZE;
  uint rem      = input_size % BATCHSIZE; 
  quotient += (rem != 0) ? 1 : 0;

  // Matrices will be BATCHSIZE x feature_count 
  // Allocate data arrays contiguous to matrix array 
  // Rem is just the row count of the batch, it'll be allocated to 32 byte aligned for simplicity
  uint64_t matrix_size = sizeof(Matrix);
  uint64_t matrix_data = (BATCHSIZE * feature_count) * sizeof(float);
  uint64_t total_size  = quotient * (matrix_data + matrix_size);

  // Allocate entire block
  void *full_ptr;
  cudaMallocManaged(&full_ptr, total_size);
  // Fetch start 
  batch_matrix = static_cast<Matrix*>(full_ptr);

  for (int i = 0; i < quotient; i++) {
    // Get offsets and pointer arithemetic/cast pointer to owner
    uint64_t matrix_offset = i * (matrix_size + matrix_data);
    Matrix *current_matrix = reinterpret_cast<Matrix*>(static_cast<char*>(full_ptr) + matrix_offset);
    uint64_t data_offset   = matrix_offset + matrix_size;
    float *current_data    = reinterpret_cast<float*>(static_cast<char*>(full_ptr) + data_offset);

    // Copy metadata into current matrix 
    current_matrix->row  = (rem != 0 && i == quotient - 1) ? rem : BATCHSIZE;
    current_matrix->col  = feature_count;
    current_matrix->data = current_data;

    // Copy current matrix into batch array
    batch_matrix[i] = *current_matrix;
  }

  // Fill calls
  dim3 blocks(BLOCKSIZE, BLOCKSIZE);
  dim3 grid;

  // Ensure memory has been copied 
  cudaDeviceSynchronize();

  // Array slice loop 
  grid = dim3((feature_count + BLOCKSIZE - 1) / BLOCKSIZE, (BATCHSIZE + BLOCKSIZE - 1) / BLOCKSIZE, quotient);
  slice_input_vector<<<grid, blocks, 0, arena.get_stream()>>>(batch_matrix, d_inputs, quotient, BATCHSIZE, feature_count);

  // Reset arena and prefetch the batch matrices to the gpu.
  arena.reset();
  cudaMemPrefetchAsync(batch_matrix, total_size, 0);

  // Return batch array and update count
  *batch_count = quotient;
  return batch_matrix;
}
/*
__host__ void forward_propagation(
  Matrix *input,
  Matrix *output
) {

} */
