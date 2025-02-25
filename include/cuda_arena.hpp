#ifndef __CUDA_ARENA__
#define __CUDA_ARENA__

#include <cuda_runtime.h>
#include <cstdint>

// Arena allocator using cuda async memory operations 
  // Allocates the entire arena at the start of the program 
  // Arena could be for forward or backwards propagation, etc, to avoid overhead 
class ArenaAllocator {
private: 
  void *pool_ptr = nullptr;
  uint64_t pool_size_;
  uint64_t offset; 
  cudaStream_t stream_;
public:
  ArenaAllocator(uint64_t pool_size, cudaStream_t stream=0)
    : pool_size_(pool_size), stream_(stream) {
    cudaMallocAsync(&pool_ptr, pool_size, stream_);
  }
  
  ~ArenaAllocator() {
    cudaFreeAsync(pool_ptr, stream_);
  }

  // Allocate memory to a temp matrix, batch, vector, etc.
  void *allocate(uint64_t size, uint64_t alignment=256) {
    uint64_t aligned_size = (size + alignment - 1) & ~(alignment - 1);  // Bitwise & and ~ to get value aligned to 256
    if (offset + aligned_size > pool_size_) return nullptr;  // No memory 
    
    // Sets ptr to next unallocated block in memory using ptr arithmetic 
    void *ptr = static_cast<char*>(pool_ptr) + offset; 
    offset += aligned_size;
    // Shifts offset and returns ptr to caller
    return ptr;
  }

  void reset() {
    offset = 0;
  }

  cudaStream_t get_stream() const { return stream_; }

  // Cannot copy arena (unique)
  ArenaAllocator(const ArenaAllocator&) = delete;
  ArenaAllocator& operator=(const ArenaAllocator) = delete;
};

#endif // __CUDA_ARENA__
