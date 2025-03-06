#include "../include/matrix.hpp"
#include "../include/activation.hpp"
#include "../include/neural.hpp"

#include <vector> 
#include <random>
#include <iostream>

int main(void) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distr(-2.5, 2.5);

  uint sizes[] = {1,128,128,1};
  const uint layer_count = 3;
  std::vector<ActivationType> types;
  const uint input_size = 1024;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  ArenaAllocator arena(512 * 512 * 1024, stream);

  std::cout << "Arena Allocated to: " << 512 * 512 * 1024 / 1e6 << " MB\n";

  for (int i = 0; i < layer_count; i++) {
    types.push_back(ActivationType::Relu);
  }

  Network *network = new_network(sizes, layer_count, input_size, types);
  
  std::cout << "Network Created\n";

  uint batch_count = 0;
  uint feature_count = 1;
  float *inputs = new float[input_size];

  for (int i = 0; i < input_size; i++) {
    inputs[i] = distr(gen);
  }

  Matrix *batches = input_to_batch_array(arena, inputs, input_size, feature_count, &batch_count);

  std::cout << "Inputs batched to " << batch_count << " batches\n";

  forward_propagation(network, batches, arena, batch_count, input_size);

  std::cout << "Forward Propagated\n";

  // Read last activation layer 
  const uint count = network->get_layer();
  for (int i = 0 ; i < network->activations[layer_count].cols(); i++) {
    std::cout << "[ ";
    for (int j = 0; j < network->activations[layer_count].rows(); j++) {
      std::cout << network->activations[layer_count].data[i * network->activations[layer_count].rows() + j] << " ";
    }
    std::cout << "]\n";
  }

  cudaFree(network);
  cudaFree(batches);
  delete[] inputs;

  return 0;
}
