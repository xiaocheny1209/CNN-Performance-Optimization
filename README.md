# Overview
This repository contains optimized implementations of core components in deep learning, including a Convolutional Neural Network (CNN) kernel, an attention mechanism, and multiple performance enhancements, such as tiling, blocking, multi-threading, and sparsity techniques. The project integrates these components into an end-to-end neural network (NN) inference framework and includes profiling through the Performance Monitoring Unit (PMU) for in-depth, top-down analysis of resource usage and computational efficiency.

# Features
- CNN Kernel: Low-level implementation of a Convolutional Neural Network kernel.
- Attention Mechanism: Scaled dot-product attention for sequence data.
- Optimization Techniques:
  - Tiling and Blocking: Enhanced cache locality and memory utilization by breaking large matrices into manageable tiles.
  - Sparse matrix multiplication: Implemented sparse matrix multiplication using the Compressed Sparse Row (CSR) format to optimize both computation and memory usage. Integrated this sparse matrix multiplication into the convolution layer, replacing the pairwise multiplication to further enhance performance.
  - Multi-threading: Increased computational speed by leveraging concurrent processing.
- Integrated Inference Framework: Combined optimized components into a full NN inference pipeline, achieving efficient end-to-end processing.
- Top-down Profiling with PMU: Used the Performance Monitoring Unit (PMU) for top-down analysis, enabling identification of bottlenecks and fine-tuning of resource utilization.

# Performance
The `report.pdf` file contains reports for performance data across various input sizes. Performance profiling with the PMU tool includes a top-down analysis of bottlenecks and efficiency, enabling continuous optimization of computational and memory performance.
