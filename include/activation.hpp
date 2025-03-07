#ifndef __ACTIVATION_HPP__
#define __ACTIVATION_HPP__

#include <cuda_runtime.h>
#include <cmath>
#include "../include/matrix.hpp"

enum class ActivationType {
  Leakyrelu,
  Relu,
  Elu,
  Tanh,
  Sigmoid,
  Identity,
};

template<ActivationType T> struct ActivationFunction;

template<> struct ActivationFunction<ActivationType::Relu> {
  __device__ static float activate(float x) { 
    if (x > 0.0) {
      return x;
    } else {
      return 0.0;
    }
  }
  __device__ static float derivative(float x) {
    if (x > 0.0) {
      return 1.0; 
    } else {
      return 0.0;
    }
  }
};

template<> struct ActivationFunction<ActivationType::Tanh> {
  __device__ static float activate(float x) { return tanh(x); }
  __device__ static float derivative(float x) {
    float t = tanh(x);
    return 1.0 - t * t;
  }
};

template<> struct ActivationFunction<ActivationType::Identity> {
  __device__ static float activate(float x) { return x; }
  __device__ static float derivative(float x) { return 1.0; }
};

__host__ void activate(const uint m, const uint n, Matrix *A, ActivationType type, bool prime);

#endif // __ACTIVATION_HPP__
