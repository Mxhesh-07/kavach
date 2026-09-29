"""
Smoke Detection Module.

Uses a YOLO model fine-tuned on the D-Fire dataset to detect smoke
in video frames. Provides confidence scoring and temporal filtering
to reduce false positives.
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

from core.config import settings

log = logging.getLogger("kavach.smoke")


@dataclass
class SmokeDetection:
    """Result of smoke analysis on a single frame."""
    smoke_detected: bool
    confidence: float = 0.0
    bbox: Optional[tuple] = None  # (x, y, w, h)
    frame_idx: int = 0


class SmokeDetector:
    """
    Real-time smoke detection using a trained YOLO model.

    The model is loaded lazily on first use to avoid blocking startup.
    Temporal filtering requires multiple consecutive detections before
    triggering an alert, reducing false positives from transient artifacts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_confidence: float = 0.50,
        min_smoke_frames: int = 3,
        cooldown_seconds: float = 5.0,
    ):
        self._model_path = model_path or str(
            Path(settings.BASE_DIR) / "models" / "smoke_yolov8n.pt"
        )
        self._min_confidence = min_confidence
        self._min_smoke_frames = min_smoke_frames
        self._cooldown_seconds = cooldown_seconds

        self._model = None
        self._lock = threading.Lock()
        self._consecutive_detections: int = 0
        self._last_alert_time: float = 0.0
        self._frame_idx: int = 0

    def _load_model(self):
        """Lazy-load the YOLO smoke detection model."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            log.info("Smoke detection model loaded: %s", self._model_path)
        except Exception as exc:
            log.error("Failed to load smoke detection model: %s", exc)
            self._model = None

    def analyze(self, frame: np.ndarray) -> SmokeDetection:
        """
        Analyze a frame for smoke.

        Returns a SmokeDetection with smoke_detected=True only when
        the temporal threshold is met (multiple consecutive detections).
        """
        self._frame_idx += 1

        # Lazy load model
        if self._model is None:
            self._load_model()
        if self._model is None:
            return SmokeDetection(smoke_detected=False, frame_idx=self._frame_idx)

        try:
            results = self._model.predict(
                source=frame,
                conf=self._min_confidence,
                verbose=False,
                device="cpu",
            )

            max_confidence = 0.0
            best_bbox = None

            for result in results:
                if result.boxes is None:
                    continue
                for i in range(len(result.boxes)):
                    conf = float(result.boxes.conf[i].item())
                    cls = int(result.boxes.cls[i].item())
                    # D-Fire dataset: class 0 = smoke
                    if cls == 0 and conf > max_confidence:
                        max_confidence = conf
                        xyxy = result.boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        best_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # Temporal filtering: require consecutive detections
            if max_confidence > 0:
                self._consecutive_detections += 1
            else:
                self._consecutive_detections = 0

            # Check if we have enough consecutive detections
            if self._consecutive_detections >= self._min_smoke_frames:
                # Check cooldown
                now = time.time()
                if now - self._last_alert_time >= self._cooldown_seconds:
                    self._last_alert_time = now
                    return SmokeDetection(
                        smoke_detected=True,
                        confidence=round(max_confidence, 4),
                        bbox=best_bbox,
                        frame_idx=self._frame_idx,
                    )

            return SmokeDetection(
                smoke_detected=False,
                confidence=round(max_confidence, 4),
                bbox=best_bbox,
                frame_idx=self._frame_idx,
            )

        except Exception as exc:
            log.warning("Smoke analysis failed: %s", exc)
            return SmokeDetection(smoke_detected=False, frame_idx=self._frame_idx)

    def reset(self):
        """Reset temporal state (called when camera restarts)."""
        self._consecutive_detections = 0
        self._last_alert_time = 0.0
        self._frame_idx = 0

    def get_metrics(self) -> dict:
        return {
            "model_loaded": self._model is not None,
            "consecutive_detections": self._consecutive_detections,
            "min_confidence": self._min_confidence,
            "min_smoke_frames": self._min_smoke_frames,
            "cooldown_seconds": self._cooldown_seconds,
        }


class SmokeAlarm:
    """
    Continuous beeping alarm for smoke detection.

    Uses winsound.Beep on Windows or a simple tone on other platforms.
    Runs in a daemon thread so it never blocks the pipeline.
    """

    def __init__(
        self,
        beep_interval: float = 0.8,
        beep_frequency: int = 880,
        beep_duration: int = 300,
    ):
        self._beep_interval = beep_interval
        self._beep_frequency = beep_frequency
        self._beep_duration = beep_duration

        self._active: bool = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the continuous beep alarm."""
        if self._active:
            return
        self._active = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._beep_loop, daemon=True, name="smoke-alarm")
        self._thread.start()
        log.warning("SMOKE ALARM ACTIVE — continuous beep started")

    def stop(self) -> None:
        """Stop the continuous beep alarm."""
        if not self._active:
            return
        self._active = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("Smoke alarm stopped")

    @property
    def is_active(self) -> bool:
        return self._active

    def _beep_loop(self) -> None:
        """Background thread that beeps continuously until stopped."""
        while not self._stop_event.is_set():
            try:
                import winsound
                winsound.Beep(self._beep_frequency, self._beep_duration)
            except ImportError:
                # Fallback for non-Windows platforms
                try:
                    import os
                    os.system(f'play -nq -t alsa synth {self._beep_interval / 1000} sine {self._beep_frequency} 2>/dev/null &')
                except Exception:
                    pass
            # Wait for interval, but can be interrupted by stop
            self._stop_event.wait(self._beep_interval)
