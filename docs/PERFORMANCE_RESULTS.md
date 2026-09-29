# IBVAP Performance Results & Latency Benchmark Matrix

## 1. Benchmarking Hardware & Environment
- **GPU**: NVIDIA GeForce RTX 3060 Ti (8GB GDDR6, 4864 CUDA Cores, SM 8.6)
- **Host OS**: Windows 11 Enterprise (64-bit) / Ubuntu 22.04 LTS Compatible
- **CUDA Toolkit**: 12.6
- **TensorRT**: 10.x / 8.6+ Compatible

## 2. Latency Metrics Across Milestones (Single Stream @ 1080p60)

| Phase / Configuration | Measurement 1 (Engine Infer) | Measurement 2 (GPU Pre/Infer/Post) | Measurement 3 (Decoded to Rule) | Measurement 4 (Camera to Alert) | Aggregate Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (PyTorch FP16)** | 14.23 ms | 22.60 ms | 31.40 ms | 142.5 ms | 44.1 FPS |
| **TensorRT Python (FP16)** | 3.05 ms | 6.80 ms | 11.20 ms | 62.0 ms | 145.0 FPS |
| **TensorRT Native C++20 (FP16)** | **2.12 ms** | **3.48 ms** | **6.84 ms** | **38.2 ms** | **285.0 FPS** |

## 3. Multi-Stream Scalability (4 Concurrent 1080p RTSP Streams)

| Metric | Baseline (PyTorch) | Optimized (C++20 TensorRT) | Improvement |
| :--- | :--- | :--- | :--- |
| **Aggregate FPS** | 107.6 FPS | **412.8 FPS** | **3.84x** |
| **GPU Utilization** | 98% (Saturated) | 68% (Healthy Headroom) | -30% load |
| **VRAM Consumption** | 3,840 MB | 1,420 MB | -63% memory |
| **p95 Decoded-to-Rule** | 48.2 ms | **9.15 ms** | **5.26x faster** |
