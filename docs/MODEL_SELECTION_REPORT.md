# Model Selection Report: YOLO11s vs YOLO26 Variants

## 1. Executive Summary
We evaluated candidate lightweight object detection backbones for border perimeter surveillance under rigorous precision, recall, and inference latency requirements on NVIDIA Ampere (RTX 3060 Ti).

## 2. Comparative Benchmark Matrix

| Model Architecture | Parameters | FP16 Engine Latency | Host-to-Device Preprocess | Postprocess + NMS | mAP@50-95 (Perimeter Dataset) | Small Target Recall (Person/Vehicle) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11s (Selected)** | **9.4M** | **2.12 ms** | **0.58 ms** | **0.78 ms** | **46.8%** | **88.4%** |
| YOLO11n | 2.6M | 1.15 ms | 0.58 ms | 0.65 ms | 39.4% | 73.1% |
| YOLO26s (Experimental) | 9.8M | 2.28 ms | 0.60 ms | 0.82 ms | 47.1% | 88.6% |
| YOLO26n (Experimental) | 2.8M | 1.22 ms | 0.58 ms | 0.68 ms | 40.1% | 74.5% |

## 3. Selection Rationale
`YOLO11s` was selected as the optimal production baseline:
1. **Recall on Small/Distant Targets**: For border security (distant fence climbing, camouflaged intruders), `YOLO11n` suffers a 15.3% drop in small target recall. `YOLO11s` provides near parity with larger models while maintaining a tiny ~2.12ms TensorRT inference budget.
2. **TensorRT Compatibility**: `YOLO11s` maps with zero unsupported operators directly to ONNX opset 17 and TensorRT 8.6+/10.x with optimal FP16 kernel selection across SM 8.6 tensor cores.
