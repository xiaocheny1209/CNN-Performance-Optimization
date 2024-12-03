# Overview
This repository contains optimized implementations of core components in deep learning, including a GPT-2 model, an attention mechanism, and a Convolutional Neural Network (CNN) kernel. The project explores and integrates multiple performance enhancements, such as SIMD, multi-threading, tiling/blocking, and sparsity techniques for matrix multiplication. It also includes profiling through the Performance Monitoring Unit (PMU) for top-down analysis of resource usage and computational efficiency.

# Features
- GPT-2 Model Optimization:
  - Implemented a simplified GPT-2 model.
  - Applied SIMD, multi-threading, and blocking to optimize attention computation and improve runtime performance.
  - Achieved significant speedups across various input sizes, as detailed in the performance report.
- CNN Kernel: Low-level implementation of a Convolutional Neural Network kernel.
- Attention Mechanism: Scaled dot-product attention for sequence data.
- Optimization Techniques:
  - Tiling and Blocking: Enhanced cache locality and memory utilization by breaking large matrices into manageable tiles.
  - Sparse matrix multiplication: Implemented sparse matrix multiplication using the Compressed Sparse Row (CSR) format to optimize both computation and memory usage. Integrated this sparse matrix multiplication into the convolution layer, replacing the pairwise multiplication to further enhance performance.
  - Multi-threading: Increased computational speed by leveraging concurrent processing.
  - SIMD (Single Instruction Multiple Data): Utilized SIMD intrinsics to accelerate linear algebra operations, including matrix addition, GeLU activation, and scaled dot-product attention.
- Top-down Profiling with PMU: Used the Performance Monitoring Unit (PMU) for top-down analysis, enabling identification of bottlenecks and fine-tuning of resource utilization.

# Performance
The `report.pdf` and `gpt2/report.md` file contains reports for performance data across various input sizes. Performance profiling with the PMU tool includes a top-down analysis of bottlenecks and efficiency, enabling continuous optimization of computational and memory performance.
