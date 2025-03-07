#include "../include/neural.hpp"
#include <cmath>

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
    bias_init[i] = 0.0;
    for (int j = 0; j < row; j++) {
      weight_init[j * row + i] = distribution(gen); 
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

  total_size += (2 * sizeof(Matrix) * layer_count);

  // Iterate over each discrete layer
  for (uint i = 0; i < layer_count; i++) {
    // Find current and previous neuron counts from array
    uint previous_size = layer_sizes[i];
    uint current_size  = layer_sizes[i + 1];

    // Calculate memory for weights and biases ( and each activation )
    uint64_t weights_data    = (previous_size * current_size) * sizeof(float);
    uint64_t bias_data       = (current_size) * sizeof(float);
    uint64_t activation_data = (input_size * current_size) * sizeof(float); 

    // Sum
    total_size += (activation_data + weights_data + bias_data); 
  }

  total_size += (input_size * layer_sizes[0]) * sizeof(float);

  return total_size;
}

__global__ void print_matrix(Matrix *A) {
  const uint col = A->cols(), row = A->rows();
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y; 

  if (x < col && y < row) {
    const uint index = x * A->rows() + y;
    printf("(%u, %u): %f\n", x, y, A->data[index]);
  }
}

__host__ Network *new_network(
  uint *layer_sizes,
  uint layer_count,
  uint input_size,
  std::vector<ActivationType> types
) {
  Network *network; 
  const uint64_t total_size = calculate_network_size(layer_sizes, layer_count, input_size);
#ifdef __debug
  std::cout << "Layer_Count: " << layer_count << '\n';
#endif
  cudaError_t err = cudaMallocManaged(&network, total_size);
  if (err != cudaSuccess) {
    std::cerr << "Network Malloc Failure: " << cudaGetErrorString(err) << '\n';
    return nullptr;
  }
  
  network->total_size  = total_size;
  network->layer_count = layer_count;

  // Set pointers of struct manually 
  char* ptr = reinterpret_cast<char*>(network) + sizeof(Network);
  network->layers = reinterpret_cast<Layer*>(ptr);
  ptr += sizeof(Layer) * layer_count;
#ifdef __debug
  std::cout << "Layer Address: 0x" << std::hex << reinterpret_cast<uint64_t>(network->layers) << '\n';
#endif
  network->sizes = reinterpret_cast<uint*>(ptr);
  ptr += sizeof(uint) * (layer_count + 1);
#ifdef __debug  
  std::cout << "Sizes Address: 0x" << std::hex << reinterpret_cast<uint64_t>(network->sizes) << '\n';
#endif
  network->activations = reinterpret_cast<Matrix*>(ptr);
  ptr += sizeof(Matrix) * (layer_count + 1);
#ifdef __debug  
  std::cout << "Activations Address: 0x" << std::hex << reinterpret_cast<uint64_t>(network->activations) << '\n';
#endif
  for (int i = 0; i < layer_count; i++) {
    // Collect layer
    Layer& layer = network->layers[i];
    layer.weights.data = reinterpret_cast<float*>(ptr); 
    // Advance ptr 
    ptr += layer_sizes[i] * layer_sizes[i+1] * sizeof(float);

    layer.biases.data  = reinterpret_cast<float*>(ptr);
    ptr += layer_sizes[i+1] * sizeof(float);
  }

  for (int i = 0; i < layer_count; i++) {
    Layer& layer = network->layers[i];

    layer.type = types[i];
    layer.weights.row = layer_sizes[i];
    layer.weights.col = layer_sizes[i+1];
    layer.biases.row  = 1;
    layer.biases.col  = layer_sizes[i+1];
    initialize_layer(&layer);
  }

  for (int i = 0; i < layer_count + 1; i++) {
    Matrix& activation = network->activations[i];
    activation.row  = input_size;
    activation.col  = layer_sizes[i];
    activation.data = reinterpret_cast<float*>(ptr);
    ptr += (input_size * layer_sizes[i] * sizeof(float));
  }

  for (int i = 0; i < layer_count + 1; i++) {
    network->sizes[i] = layer_sizes[i];
  }

  return network;
}

__host__ Matrix *input_to_batch_array(
  ArenaAllocator& arena,
  float *input_vector,
  uint64_t input_size,
  uint feature_count,
  uint *batch_count
) {
  const uint batchsize = 64;
  Matrix *batches;
  arena.reset();  // Force arena reset 
  
  float *d_inputs = reinterpret_cast<float*>(arena.allocate(input_size * feature_count * sizeof(float)));
  cudaError_t err = cudaMemcpyAsync(d_inputs, input_vector, input_size * feature_count * sizeof(float), cudaMemcpyHostToDevice, arena.get_stream());
  if (err != cudaSuccess) {
    std::cerr << "Memcpy Failure: " << cudaGetErrorString(err) << '\n';
    return nullptr;
  }

  // Calculate size of batch matrix 
  uint quotient  = input_size / batchsize; 
  uint remainder = input_size % batchsize;
  *batch_count = quotient + (remainder != 0 ? 1 : 0);

  uint64_t matrix_size = *batch_count * sizeof(Matrix);
  uint64_t data_size   = (quotient * batchsize + remainder) * feature_count * sizeof(float);
  uint64_t total_size  = matrix_size + data_size;

  err = cudaMallocManaged(&batches, total_size);
  if (err != cudaSuccess) {
    std::cerr << "Batches Malloc Error: " << cudaGetErrorString(err) << '\n';
    return NULL;
  }

  char* ptr = reinterpret_cast<char*>(batches) + matrix_size;
  for (int i = 0; i < *batch_count; i++) {
    const uint rows = (i == *batch_count - 1 && remainder != 0) ? remainder : batchsize;

    batches[i].row  = rows;
    batches[i].col  = feature_count;
    batches[i].data = reinterpret_cast<float*>(ptr);
    ptr += (rows * feature_count * sizeof(float));
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Sync Error: " << cudaGetErrorString(err) << '\n';
    return nullptr;
  }
  
  // Shift d_inputs into each batch array
  cudaStream_t stream = arena.get_stream();
  for (int i = 0; i < *batch_count; i++) {
    uint rows = batches[i].row;
    uint cols = feature_count * sizeof(float);
    
    float *src = d_inputs + i * batchsize * feature_count;
    float *dst = batches[i].data;

    // Copy 2d flattened array to batch
    err = cudaMemcpy2DAsync(dst, cols, src, cols, cols, rows, cudaMemcpyDeviceToDevice, stream);
  
    if (err != cudaSuccess) {
      std::cerr <<"d_inputs to Batches Failure: " << cudaGetErrorString(err) << '\n';
      return nullptr;
    }
  }

  cudaStreamSynchronize(stream);


  arena.reset();

  return batches;
}


__global__ void insert_output(Matrix *activation, Matrix *d_output, uint batch, uint batch_size) {
  // Place Array into activation at specified location 
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Fill activation at correct index 
  if (x < d_output->rows() && y < d_output->cols()) {
    uint activation_start = batch * batch_size * activation->cols();
    uint activation_idx   = activation_start + x * activation->cols() + y;
    activation->data[activation_idx] = d_output->data[x * d_output->cols() + y];
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
  const uint *sizes = network->get_sizes();
  const uint layer_count = network->get_layer();
  dim3 blocks(BLOCKSIZE, BLOCKSIZE), grid;
  const uint start_row = u_batches->rows(), start_col = u_batches->cols();

  cudaError_t err = cudaMemPrefetchAsync(network, network->total_size, 0);
  if (err != cudaSuccess) {
    std::cerr << "Prefetch Error: " << cudaGetErrorString(err) << '\n';
    return;
  }

  Matrix *d_output = nullptr;

  for (uint b = 0; b < batch_count; b++) {

    Matrix& batch = u_batches[b];
    
    uint current_row = batch.rows();
    assert(current_row == 64); 

    for (uint i = 0; i < layer_count; i++) {
      assert(network->layers[i].weights.data != nullptr);
      
      // Multiply by weights 
      Matrix *current_input = (i == 0) ? &batch : d_output;
      if (i != 0) assert(d_output != nullptr);

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
      // std::cout << "{" << current_row << "}x{" << sizes[i+1] << "}\n";

      // Apply activation function to z
      if (i != layer_count - 1) {
        activate(current_row, sizes[i+1], d_output, network->layers[i].type, false);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          std::cerr << "Activate Error: " << cudaGetErrorString(err) << '\n';
          exit(EXIT_FAILURE);
        }
        // Must be fully activated before insertion 
      }      

      // Append d_output to u_outputs
      insert_output<<<grid, blocks>>>(&network->activations[i + 1], d_output, b, batch_count);
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::cerr << "Insert Output Failure: " << cudaGetErrorString(err) << '\n';
        return;
      }
    }
    arena.reset();
  }
}
