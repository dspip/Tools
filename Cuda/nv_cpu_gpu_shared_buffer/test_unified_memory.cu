#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel to perform computation on GPU
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel to modify data on GPU
__global__ void squareElements(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

struct unified_buffer
{
    void * data;
    uint64_t size;
};

unified_buffer unified_buffer_allocate(const int N)
{
    size_t size = N;
    unified_buffer r_buffer= {};
    cudaMallocManaged(&r_buffer.data, size);
    r_buffer.size = N ;

    return r_buffer;
    // Allocate Unified Memory – accessible from both CPU and GPU
    
}
void unified_buffer_free(unified_buffer & buff)
{
    cudaFree(buff.data);
    buff.size = 0;
}


int main() {

    // Initialize data on CPU
    std::cout << "Initializing data on CPU..." << std::endl;
    uint64_t N = 100000;
    unified_buffer af =  unified_buffer_allocate(N * sizeof(float));
    unified_buffer bf =  unified_buffer_allocate(N * sizeof(float));
    unified_buffer cf =  unified_buffer_allocate(N * sizeof(float));
    unified_buffer df =  unified_buffer_allocate(N * sizeof(float));

    float * a, *b, *c, *d;
    a = (float*)af.data;
    b = (float*)bf.data;
    c = (float*)cf.data;
    d = (float*)df.data;

    for (int i = 0; i < N; i++) {
        a[i] = sin(i);
        b[i] = cos(i);
        c[i] = 0;
        d[i] = sin(i) + cos(i);
    }
    
    // Print first 5 elements from CPU before GPU computation
    std::cout << "\nFirst 5 elements before GPU computation (CPU read):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << std::endl;
    }
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // Launch kernel on GPU
    std::cout << "\nExecuting vector addition on GPU..." << std::endl;
    vectorAdd<<<numBlocks, blockSize>>>(a, b, c, N);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Read results on CPU (no explicit memory transfer needed!)
    std::cout << "\nFirst 5 results after GPU computation (CPU read):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "c[" << i << "] = " << c[i] << " expected:  " << d[i] << std::endl;
    }
    
    // Modify data on CPU
    std::cout << "\nModifying first element on CPU..." << std::endl;
    c[0] = 999.0f;
    std::cout << "c[0] set to " << c[0] << " on CPU" << std::endl;
    
    // Use GPU to square all elements
    std::cout << "\nSquaring all elements on GPU..." << std::endl;
    squareElements<<<numBlocks, blockSize>>>(c, N);
    cudaDeviceSynchronize();
    
    // Read modified data on CPU
    std::cout << "\nFirst 5 elements after squaring (CPU read):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }
    
    // Verify correctness for a few elements
    std::cout << "\nVerification:" << std::endl;
    bool correct = true;
    for (int i = 1; i < 5; i++) {  // Skip first element as we modified it
        float expected = d[i] * d[i];

        if (fabs(c[i] - expected) > 1e-5) {
            correct = false;
            std::cout << "Error at index " << i << ": expected " << expected 
                     << ", got " << c[i] << std::endl;
        }
    }
    if (correct) {
        std::cout << "Results verified successfully!" << std::endl;
    }
    
    
    // Free Unified Memory
    unified_buffer_free(af);
    unified_buffer_free(bf);
    unified_buffer_free(cf);
    unified_buffer_free(df);
    
    std::cout << "\nUnified Memory demonstration complete!" << std::endl;
    
    return 0;
}

/* 
Compilation instructions:
nvcc -o unified_memory unified_memory.cu -arch=sm_60

Key features demonstrated:
1. cudaMallocManaged() allocates memory accessible from both CPU and GPU
2. CPU can initialize data directly without explicit transfers
3. GPU kernels can operate on the same memory
4. CPU can read results without explicit cudaMemcpy
5. Both CPU and GPU can modify the same memory space

Requirements:
- CUDA 6.0+ (for basic Unified Memory support)
- GPU with compute capability 3.0+ (Kepler or newer)
- For best performance: Pascal architecture (6.0+) or newer

Notes:
- The CUDA runtime automatically handles page migration between CPU and GPU
- On Pascal and newer GPUs, page faults enable on-demand migration
- Prefetching can improve performance by moving data before it's needed
- Unified Memory simplifies code but may have performance implications
  compared to explicit memory management in some cases
*/
