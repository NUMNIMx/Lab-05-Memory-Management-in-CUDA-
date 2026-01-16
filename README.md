# Lab 5: Memory Management in CUDA

This project demonstrates the performance differences between **Global**, **Shared**, and **Register** memory in CUDA using a Vector Addition example.

## 📂 File Structure
- `lab5_memory.cu`: The complete source code containing:
  - **Global Memory Kernel**: Standard implementation.
  - **Shared Memory Kernel**: Uses shared memory as a buffer.
  - **Register Memory Kernel**: Explicitly uses registers for computation.
  - **Benchmark**: Runs tests with block sizes [1000, 2000, 3907, 8000].

## 🚀 How to Run

Since there is only one file, you can compile it directly with `nvcc`.

### 1. Compile
```bash
nvcc -o lab5_memory lab5_memory.cu
```

### 2. Run
```bash
./lab5_memory
```

## 📊 Expected Output
You will see a table comparing the execution time (ms) for each memory type across different grid sizes.

Example:
```text
Vector Addition (N=1000000)
========================================================================
Memory Type          Threads/Block   Blocks          Time (ms)      
========================================================================
Global Memory        256             3907            0.01206        
Shared Memory        256             3907            0.01229        
Register Memory      256             3907            0.01203        
...
```

## 📝 Analysis
- **Register Memory**: Fastest, as it uses on-chip registers with the lowest latency.
- **Shared Memory**: Comparable to Global in this specific case (no data reuse), but essential for more complex algorithms.
- **Global Memory**: High latency, but effectively hidden by the GPU's coalescing and caching mechanisms in sequential access patterns.
