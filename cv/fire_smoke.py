"""
Fire & Smoke Detection using YOLOv8n fine-tuned on D-Fire dataset.

The D-Fire dataset has 2 classes:
- Class 0: smoke
- Class 1: fire

This module provides a lightweight detector that runs on CPU and can
operate alongside the main object detection pipeline.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("ibvap.fire_smoke")
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)-18s %(message)s"))
    log.addHandler(handler)


@dataclass
class FireSmokeDetection:
    """Result of fire/smoke analysis on a single frame."""
    fire_detected: bool = False
    smoke_detected: bool = False
    fire_confidence: float = 0.0
    smoke_confidence: float = 0.0
    fire_boxes: list = field(default_factory=list)
    smoke_boxes: list = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None


class FireSmokeDetector:
    """
    YOLOv8n-based fire and smoke detector.

    Uses a model fine-tuned on the D-Fire dataset for detecting
    fire and smoke in video frames. Runs on CPU for edge deployment.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.30,
        imgsz: int = 320,
    ):
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._imgsz = imgsz
        self._model = None
        self._lock = threading.Lock()
        self._loaded = False
        self._load_attempted = False

    def _load_model(self):
        """Lazy-load the YOLO model."""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            from ultralytics import YOLO

            if self._model_path and Path(self._model_path).exists():
                self._model = YOLO(self._model_path)
            else:
                # Try to find the model in common locations
                possible_paths = [
                    "models/fire_smoke_yolov8n.pt",
                    "fire_smoke_yolov8n.pt",
                    "yolov8n-fire-smoke.pt",
                ]
                for path in possible_paths:
                    if Path(path).exists():
                        self._model = YOLO(path)
                        break

                if self._model is None:
                    log.warning(
                        "No fire/smoke model found. Fire/smoke detection will be disabled. "
                        "Download a D-Fire fine-tuned YOLOv8n model to enable this feature."
                    )
                    return

            self._loaded = True
            log.info("Fire & smoke model loaded successfully")

        except Exception as exc:
            log.error("Failed to load fire/smoke model: %s", exc)
            self._model = None

    def analyze(self, frame: np.ndarray) -> Optional[FireSmokeDetection]:
        """
        Analyze a frame for fire and smoke.

        Returns None if the model is not available.
        """
        with self._lock:
            self._load_model()

            if self._model is None:
                return None

            try:
                results = self._model.predict(
                    source=frame,
                    conf=self._confidence_threshold,
                    imgsz=self._imgsz,
                    verbose=False,
                    device="cpu",
                )

                detection = FireSmokeDetection()

                for result in results:
                    if result.boxes is None:
                        continue

                    boxes = result.boxes
                    for i in range(len(boxes)):
                        cls = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        xyxy = boxes.xyxy[i].cpu().numpy()

                        if cls == 1:  # fire
                            detection.fire_detected = True
                            detection.fire_confidence = max(detection.fire_confidence, conf)
                            detection.fire_boxes.append(xyxy)
                        elif cls == 0:  # smoke
                            detection.smoke_detected = True
                            detection.smoke_confidence = max(detection.smoke_confidence, conf)
                            detection.smoke_boxes.append(xyxy)

                    # Generate annotated frame
                    detection.annotated_frame = result.plot()

                return detection

            except Exception as exc:
                log.warning("Fire/smoke analysis failed: %s", exc)
                return None

    def get_metrics(self) -> dict:
        return {
            "loaded": self._loaded,
            "confidence_threshold": self._confidence_threshold,
            "imgsz": self._imgsz,
        }
