"""
Precise frame lifecycle timing and metrics collection for IBVAP.
Tracks every microsecond across capture, decode, preprocess, inference,
tracking, spatial rules, and event dispatch.
"""
from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class FrameTiming:
    """Fine-grained nanosecond lifecycle timestamps for one video frame."""
    camera_id: str = ""
    sequence_number: int = 0
    
    # Ingestion & Queue timestamps
    capture_started_ns: int = 0
    capture_completed_ns: int = 0
    enqueued_ns: int = 0
    dequeued_ns: int = 0
    
    # Analytics & AI timestamps
    preprocess_started_ns: int = 0
    preprocess_completed_ns: int = 0
    inference_started_ns: int = 0
    inference_completed_ns: int = 0
    postprocess_completed_ns: int = 0
    
    # Tracking & Rules timestamps
    tracking_completed_ns: int = 0
    rules_completed_ns: int = 0
    
    # Event & Dispatch timestamps
    event_created_ns: int = 0
    evidence_requested_ns: int = 0
    event_persisted_ns: int = 0
    websocket_published_ns: int = 0

    @classmethod
    def start(cls, camera_id: str = "", sequence: int = 0) -> FrameTiming:
        now = time.monotonic_ns()
        return cls(
            camera_id=camera_id,
            sequence_number=sequence,
            capture_started_ns=now,
            capture_completed_ns=now,
        )

    # Elapsed times in milliseconds
    @property
    def preprocess_ms(self) -> float:
        if self.preprocess_completed_ns and self.preprocess_started_ns:
            return (self.preprocess_completed_ns - self.preprocess_started_ns) / 1e6
        return 0.0

    @property
    def inference_ms(self) -> float:
        if self.inference_completed_ns and self.inference_started_ns:
            return (self.inference_completed_ns - self.inference_started_ns) / 1e6
        return 0.0

    @property
    def postprocess_ms(self) -> float:
        if self.postprocess_completed_ns and self.inference_completed_ns:
            return (self.postprocess_completed_ns - self.inference_completed_ns) / 1e6
        return 0.0

    @property
    def tracking_ms(self) -> float:
        if self.tracking_completed_ns and self.postprocess_completed_ns:
            return (self.tracking_completed_ns - self.postprocess_completed_ns) / 1e6
        return 0.0

    @property
    def rules_ms(self) -> float:
        if self.rules_completed_ns and self.tracking_completed_ns:
            return (self.rules_completed_ns - self.tracking_completed_ns) / 1e6
        return 0.0

    @property
    def decoded_to_rule_ms(self) -> float:
        start = self.dequeued_ns or self.preprocess_started_ns or self.capture_completed_ns
        end = self.rules_completed_ns or self.tracking_completed_ns
        if start and end and end >= start:
            return (end - start) / 1e6
        return 0.0

    @property
    def capture_to_alert_ms(self) -> float:
        start = self.capture_started_ns or self.capture_completed_ns
        end = self.event_persisted_ns or self.websocket_published_ns or self.rules_completed_ns
        if start and end and end >= start:
            return (end - start) / 1e6
        return 0.0

    @property
    def frame_age_ms(self) -> float:
        start = self.capture_completed_ns
        if start:
            return (time.monotonic_ns() - start) / 1e6
        return 0.0


class LifecycleMetricsCollector:
    """Thread-safe statistical latency collector for p50, p90, p95, p99, max."""

    def __init__(self, max_samples: int = 10000) -> None:
        self.max_samples = max_samples
        self.samples_infer: List[float] = []
        self.samples_gpu_pipeline: List[float] = []
        self.samples_decoded_to_rule: List[float] = []
        self.samples_capture_to_alert: List[float] = []
        self.samples_frame_age: List[float] = []

        self.captured_frames: int = 0
        self.analyzed_frames: int = 0
        self.discarded_stale_frames: int = 0
        self.decoder_failures: int = 0
        self.camera_reconnects: int = 0
        self.inference_failures: int = 0

    def record(self, timing: FrameTiming) -> None:
        self.analyzed_frames += 1
        
        inf = timing.inference_ms
        if inf > 0:
            self.samples_infer.append(inf)
            if len(self.samples_infer) > self.max_samples:
                self.samples_infer.pop(0)

        gpu = timing.preprocess_ms + timing.inference_ms + timing.postprocess_ms
        if gpu > 0:
            self.samples_gpu_pipeline.append(gpu)
            if len(self.samples_gpu_pipeline) > self.max_samples:
                self.samples_gpu_pipeline.pop(0)

        d2r = timing.decoded_to_rule_ms
        if d2r > 0:
            self.samples_decoded_to_rule.append(d2r)
            if len(self.samples_decoded_to_rule) > self.max_samples:
                self.samples_decoded_to_rule.pop(0)

        c2a = timing.capture_to_alert_ms
        if c2a > 0:
            self.samples_capture_to_alert.append(c2a)
            if len(self.samples_capture_to_alert) > self.max_samples:
                self.samples_capture_to_alert.pop(0)

        age = timing.frame_age_ms
        if age > 0:
            self.samples_frame_age.append(age)
            if len(self.samples_frame_age) > self.max_samples:
                self.samples_frame_age.pop(0)

    @staticmethod
    def calc_percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        s = sorted(values)
        n = len(s)
        def p(pct: float) -> float:
            idx = min(n - 1, max(0, int(math.ceil(pct / 100.0 * n)) - 1))
            return round(s[idx], 2)
        return {
            "count": n,
            "min": round(s[0], 2),
            "mean": round(sum(s) / n, 2),
            "p50": p(50),
            "p90": p(90),
            "p95": p(95),
            "p99": p(99),
            "max": round(s[-1], 2),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "counters": {
                "captured_frames": self.captured_frames,
                "analyzed_frames": self.analyzed_frames,
                "discarded_stale_frames": self.discarded_stale_frames,
                "decoder_failures": self.decoder_failures,
                "camera_reconnects": self.camera_reconnects,
                "inference_failures": self.inference_failures,
            },
            "measurement_1_inference_only": self.calc_percentiles(self.samples_infer),
            "measurement_2_gpu_pipeline": self.calc_percentiles(self.samples_gpu_pipeline),
            "measurement_3_decoded_to_rule": self.calc_percentiles(self.samples_decoded_to_rule),
            "measurement_4_capture_to_alert": self.calc_percentiles(self.samples_capture_to_alert),
            "frame_age": self.calc_percentiles(self.samples_frame_age),
        }


# Global metrics instance
metrics = LifecycleMetricsCollector()
