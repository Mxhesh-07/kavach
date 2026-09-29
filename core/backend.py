"""
KAVACH backend selector and fallback coordinator.
Routes inference between Native C++20/CUDA TensorRT engine and PyTorch/Ultralytics fallback.
"""
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class BackendDetector:
    """Unified interface for detector backend routing."""
    def __init__(self, model_path: Optional[str] = "yolo11s.pt", engine_path: Optional[str] = None):
        self.model_path = model_path
        self.engine_path = engine_path or (model_path.replace(".pt", ".engine") if model_path else "yolo11s.engine")
        self.backend_type = "pytorch"
        self._native_engine = None
        self._torch_model = None
        self._init_backend()

    def _init_backend(self):
        # Try loading C++ native engine if available
        if os.path.exists(self.engine_path):
            try:
                # The extension was renamed ibvap_native -> kavach_native with
                # the project rename. A deployment that built the engine before
                # the rename still has the old .so on disk, so fall back to it
                # rather than reporting a missing engine that is present.
                try:
                    import kavach_native
                except ImportError:
                    import ibvap_native as kavach_native
                # The pybind module historically exposed only TensorRTEngine /
                # NativeScheduler, with no synchronous ``detect()`` binding, so
                # an ``kavach_native.Engine(...).detect(...)`` call could never
                # work and this branch silently claimed a native backend it
                # could not drive (then fell through on AttributeError). Require
                # a detect()-capable entry point up front instead of assuming it.
                engine_api = getattr(kavach_native, "Engine", None)
                detect_fn = getattr(engine_api, "detect", None) if engine_api else None
                if not engine_api or not detect_fn:
                    raise ImportError(
                        "kavach_native has no synchronous detect() binding"
                    )
                logger.info(f"Loading native TensorRT engine from {self.engine_path}")
                self._native_engine = engine_api(self.engine_path)
                self.backend_type = "tensorrt_native"
                return
            except ImportError as exc:
                logger.info("Native TensorRT engine not usable (%s); checking TensorRT Python...", exc)
                try:
                    import tensorrt as trt
                    self.backend_type = "tensorrt_python"
                    return
                except ImportError:
                    logger.info("TensorRT Python not available; falling back to PyTorch.")
            except Exception as e:
                logger.warning(f"Failed to initialize native TensorRT engine: {e}; falling back.")
        
        # Fallback to PyTorch
        self.backend_type = "pytorch"
        from cv.detector import Detector
        self._torch_model = Detector(model_path=self.model_path)
        logger.info("PyTorch/Ultralytics backend initialized successfully.")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25):
        if self.backend_type == "tensorrt_native" and self._native_engine is not None:
            return self._native_engine.detect(frame, conf_threshold, 0.45)
        elif self._torch_model is not None:
            return self._torch_model.detect(frame)
        else:
            raise RuntimeError("No valid backend detector initialized.")

    def raw_detect(self, frame: np.ndarray):
        if self.backend_type == "tensorrt_native" and self._native_engine is not None:
            boxes, scores, class_ids = self._native_engine.detect(frame, 0.25, 0.45)
            return boxes, scores, class_ids, 2.0
        elif self._torch_model is not None:
            return self._torch_model.raw_detect(frame)
        else:
            raise RuntimeError("No valid backend detector initialized.")
