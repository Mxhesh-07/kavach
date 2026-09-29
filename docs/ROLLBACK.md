# KAVACH Rollback Procedure

If the native C++20 / TensorRT engine encounters an unexpected driver exception or hardware mismatch:
1. Set `DETECTOR_BACKEND=pytorch` in `.env`.
2. The `BackendDetector` in `core/backend.py` will automatically fall back to the pure PyTorch/Ultralytics backend with zero service downtime.
3. Verify operation via `python verify_system.py`.
