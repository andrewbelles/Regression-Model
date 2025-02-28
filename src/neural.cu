#include "../include/neural.hpp"
#include <cuda_runtime_api.h>

// #define __debug
// Neural Network Operations 

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
  switch (layer->type) {
    case ActivationType::Tanh:
    case ActivationType::Sigmoid:
    default:
      uniform_range = sqrtf(6.0 / static_cast<float>(col + row));
      break;
    case ActivationType::Leakyrelu:
    case ActivationType::Relu:
    case ActivationType::Elu:
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
  total_size += sizeof(Matrix) * (layer_count); // Activation array metadata

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
  std::vector<ActivationType> types
) {
  Network *network;
  const uint64_t total_size = calculate_network_size(layer_sizes, layer_count, input_size);
  cudaMallocManaged(&network, total_size);

  // Reinterpret initial cast to grab pointers to metadata
  network->layers      = reinterpret_cast<Layer*>(network + 1);
  network->activations = reinterpret_cast<Matrix*>(network->layers + layer_count);
  network->sizes       = reinterpret_cast<int*>(network->activations + layer_count + 1);
  network->layer_count = layer_count; 
  network->total_size  = total_size;
  
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
    network->layers[i].type = types[i];
    
    initialize_layer(&network->layers[i]);
  }

  // Pointer offset is onto activations now so the array's metadata can be set 
  for (int i = 0; i < layer_count; i++) {
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
  matrix_data = (matrix_data + 31) & ~31; // Ensure 32 byte aligned 
  uint64_t total_size  = quotient * (matrix_data + matrix_size);

  // Allocate entire block
  void *full_ptr;
  cudaMallocManaged(&full_ptr, total_size);

  assert(reinterpret_cast<uint64_t>(full_ptr) % 32 == 0);

  // Fetch start 
  batch_matrix = static_cast<Matrix*>(full_ptr);

  for (int i = 0; i < quotient; i++) {
    // Get offsets and pointer arithemetic/cast pointer to owner
    uint64_t matrix_offset = i * (matrix_size + matrix_data);
    Matrix *current_matrix = reinterpret_cast<Matrix*>(static_cast<char*>(full_ptr) + matrix_offset);
    uint64_t data_offset   = matrix_offset + matrix_size;
    float *current_data    = reinterpret_cast<float*>(static_cast<char*>(full_ptr) + data_offset);

    assert(reinterpret_cast<uint64_t>(current_matrix) % 32 == 0);
    assert(reinterpret_cast<uint64_t>(current_data) % 32 == 0);

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

__global__ void insert_output(Matrix *activation, Matrix *d_output, uint batch, uint batch_size) {
  // Place Array into activation at specified location 
  const uint row = activation->rows(), col = activation->cols();
  const uint out_row = d_output->rows(), out_col = d_output->cols();
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < out_row && y < out_col) {
    activation->data[batch * (row * col) + x * row + y] = d_output->data[x * row + y];
  }
}

// We don't want d_outputs until an output is to be read.
// d_outputs is structured identical to u_batches but is allocated on arena 
__host__ void forward_propagation(
  Network *network,
  Matrix *u_batches,
  ArenaAllocator &arena,
  uint batch_count,
  uint input_size
) {
  const int *sizes = network->get_sizes();
  const uint layer_count = network->get_layer();
  dim3 blocks(BLOCKSIZE, BLOCKSIZE), grid;

  for (uint batch = 0; batch < batch_count; batch++) {
    uint current_row = u_batches[batch].rows();
    Matrix *d_output = nullptr;
    
    cudaMemPrefetchAsync(network, network->total_size, 0);
    for (uint i = 0; i < layer_count; i++) {

      std::cout << "Batch: " << batch << " Layer: " << i << '\n';
      std::cout << "Current Row: " << current_row << '\n';
      std::cout << "Size: " << sizes[i+1] << '\n';
      
      // Multiply by weights 
      Matrix *current_input = (i == 0) ? &u_batches[batch] : d_output;
      if (i != 0) assert(d_output != nullptr);

      std::cout << arena.get_remaining() / 1e6 << " MB remaining\n";
      d_output = matrix_multiplication(
        current_row,
        sizes[i],
        sizes[i],
        sizes[i+1],
        current_input,
        &network->layers[i].weights,
        arena
      );

      // Add biases
      matrix_elementwise_operation(
        current_row,
        sizes[i+1],
        network->layers[i].biases.rows(),
        network->layers[i].biases.cols(),
        d_output,
        &network->layers[i].biases,
        Add
      );

      grid = dim3((current_row + blocks.x - 1) / blocks.x, (sizes[i+1] + blocks.y - 1) / blocks.y);

      // Apply activation function to z
      if (i != layer_count - 1) {
        activate(current_row, sizes[i+1], d_output, network->layers[i].type, false);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          std::cerr << "Activate Error: " << cudaGetErrorString(err) << '\n';
          exit(EXIT_FAILURE);
        }
        // Must be fully activated before insertion 
      }      

      // Append d_output to u_outputs
      insert_output<<<grid, blocks>>>(&network->activations[i + 1], d_output, batch, batch_count);
      cudaDeviceSynchronize();
      // Can't be done async since we need d_output to not be overwritten by batch 
    }
    arena.reset();
  }
}
