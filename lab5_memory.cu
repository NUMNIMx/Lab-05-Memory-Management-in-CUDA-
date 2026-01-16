#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

// CUDA Error checking helper
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// 1. Global Memory Kernel
__global__ void vectorAddGlobal(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 2. Shared Memory Kernel
__global__ void vectorAddShared(float* A, float* B, float* C, int N) {
    extern __shared__ float temp[]; // Dynamic shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < N) {
        // Load data into shared memory could be optimized (coalesced), 
        // but following the PDF pattern where we just compute and store temporarily?
        // Wait, the PDF example implies calculating *in* shared memory or loading to it?
        // PDF: temp[tid] = A[idx] + B[idx]; __syncthreads(); C[idx] = temp[tid];
        // This is strictly essentially just using shared mem as a buffer.
        
        temp[tid] = A[idx] + B[idx];
        __syncthreads(); 
        C[idx] = temp[tid];
    }
}

// 3. Register Memory Kernel (Explicit usage)
__global__ void vectorAddRegister(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Load to registers first
        float a = A[idx];
        float b = B[idx];
        // Compute in register
        float res = a + b;
        // Store back
        C[idx] = res;
    }
}

void randomInit(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

void runBenchmark(const char* name, void (*kernel)(float*, float*, float*, int), 
                  float* d_A, float* d_B, float* d_C, int N, int threadsPerBlock, int blocks, 
                  int sharedMemSize = 0) {
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warmup
    kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("%-20s %-15d %-15d %-15.5f\n", name, threadsPerBlock, blocks, milliseconds);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

int main() {
    int N = 1000000; // 10^6 elements
    size_t size = N * sizeof(float);

    printf("Vector Addition (N=%d)\n", N);
    printf("========================================================================\n");
    printf("%-20s %-15s %-15s %-15s\n", "Memory Type", "Threads/Block", "Blocks", "Time (ms)");
    printf("========================================================================\n");

    // Allocate Host Memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize inputs
    randomInit(h_A, N);
    randomInit(h_B, N);

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, size));
    CHECK(cudaMalloc((void**)&d_B, size));
    CHECK(cudaMalloc((void**)&d_C, size));

    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksList[] = {1000, 2000, 3907, 8000}; 
    // PDF said 3097 (calculated), but ceil(10^6/256) = 3907. 
    // We stick to the plan: use 3907 as the correct calculated value.

    // 1. Global Memory
    for (int b : blocksList) {
        runBenchmark("Global Memory", vectorAddGlobal, d_A, d_B, d_C, N, threadsPerBlock, b, 0);
    }

    printf("------------------------------------------------------------------------\n");

    // 2. Shared Memory
    for (int b : blocksList) {
        // Shared mem size: threadsPerBlock * sizeof(float)
        runBenchmark("Shared Memory", vectorAddShared, d_A, d_B, d_C, N, threadsPerBlock, b, threadsPerBlock * sizeof(float));
    }

    printf("------------------------------------------------------------------------\n");

    // 3. Register Memory
    for (int b : blocksList) {
        runBenchmark("Register Memory", vectorAddRegister, d_A, d_B, d_C, N, threadsPerBlock, b, 0);
    }
    
    printf("========================================================================\n");

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
