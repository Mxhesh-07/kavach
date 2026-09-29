# IBVAP Known Limitations & Edge Cases

1. **Extreme Low-Light / Severe Weather**: While YOLO11s detects low-contrast objects down to 0.25 confidence, infrared (thermal) camera feeds require specialized thermal-domain weights.
2. **GPU Architecture Portability**: TensorRT serialized `.engine` files are specific to the exact SM architecture (e.g., SM 8.6 for RTX 3060 Ti). Re-building engine via `EngineManifest` is required when moving across GPU generations.
