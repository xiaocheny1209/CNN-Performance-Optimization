# Overview
This repository contains optimized implementations of core components in deep learning, including a Convolutional Neural Network (CNN) kernel, an attention mechanism, and multiple performance enhancements, such as tiling, blocking, multi-threading, and sparsity techniques. The project integrates these components into an end-to-end neural network (NN) inference framework and includes profiling through the Performance Monitoring Unit (PMU) for in-depth, top-down analysis of resource usage and computational efficiency.

# Features
- CNN Kernel: Low-level implementation of a Convolutional Neural Network kernel.
- Attention Mechanism: Scaled dot-product attention for sequence data.
- Optimization Techniques:
  - Tiling and Blocking: Enhances cache locality and memory utilization by breaking large matrices into manageable tiles.
  - Multi-threading: Increases computational speed by leveraging concurrent processing.
  - Sparsity: Reduces computation and memory usage by optimizing operations for sparse data.
- Integrated Inference Framework: Combines optimized components into a full NN inference pipeline, achieving efficient end-to-end processing.
- Top-down Profiling with PMU: Uses the Performance Monitoring Unit (PMU) for top-down analysis, enabling identification of bottlenecks and fine-tuning of resource utilization.

# Performance
The `reports/` directory contains reports for performance data across various input sizes. Performance profiling with the PMU tool includes a top-down analysis of bottlenecks and efficiency, enabling continuous optimization of computational and memory performance.
