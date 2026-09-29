"""
Phase 13: End-to-End Native Pipeline Integration & Multi-Stream Orchestration Tests.
Validates:
1. Native backend routing and fallback mechanisms.
2. Multi-camera scheduler micro-batching and thread isolation.
3. Decoupled operator preview stream behavior.
4. Frame timing telemetry across all 4 measurement boundaries.
5. Thermal governor state machine integration.
"""
import time
import numpy as np
import pytest
import threading

from core.timing import FrameTiming, LifecycleMetricsCollector
from core.scheduler import InferenceScheduler, FrameJob
from core.backend import BackendDetector
from core.preview import PreviewStream
from core.thermal import ThermalGovernor
from core.trt_manifest import EngineManifest


def test_backend_detector_fallback():
    """Verify BackendDetector initializes and handles frame detection properly."""
    detector = BackendDetector(model_path="yolo11s.pt")
    assert detector.backend_type in ("tensorrt_native", "tensorrt_python", "pytorch")
    
    # Create synthetic test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(frame, conf_threshold=0.25)
    assert isinstance(detections, list)
    
    boxes, scores, class_ids, elapsed = detector.raw_detect(frame)
    assert isinstance(boxes, (list, np.ndarray))
    assert isinstance(scores, (list, np.ndarray))
    assert isinstance(class_ids, (list, np.ndarray))
    assert elapsed >= 0.0


def test_multi_stream_scheduler_orchestration():
    """Verify multi-camera scheduling with micro-batching and round-robin dispatch."""
    def mock_backend(frames):
        return [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "class_ids": np.zeros(0)} for _ in frames]
        
    scheduler = InferenceScheduler(backend_fn=mock_backend, max_wait_ms=5.0, max_batch_size=4)
    scheduler.start()
    
    num_cameras = 4
    results = {}
    
    def worker(camera_id: int):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        job = scheduler.submit(f"cam_{camera_id}", camera_id, dummy_frame)
        job.event.wait(timeout=1.0)
        results[camera_id] = job.result

    threads = [threading.Thread(target=worker, args=(cam_id,), daemon=True) for cam_id in range(num_cameras)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
        
    scheduler.stop()
    
    # Check that all cameras received inference results
    assert len(results) == num_cameras
    for cam_id, res in results.items():
        assert res is not None
        assert "boxes" in res
        assert "scores" in res
        assert "class_ids" in res


def test_preview_stream_decoupling():
    """Verify PreviewStream maintains a non-blocking decoupled buffer without lagging."""
    preview = PreviewStream(max_fps=15, target_resolution=(640, 360))
    
    frame1 = np.ones((1080, 1920, 3), dtype=np.uint8) * 50
    preview.push_frame(frame1)
    
    disp1 = preview.get_display_frame()
    assert disp1 is not None
    assert disp1.shape == (1080, 1920, 3)
    
    # Immediately querying again should return None due to max_fps throttling
    disp2 = preview.get_display_frame()
    assert disp2 is None


def test_lifecycle_metrics_collection():
    """Verify LifecycleMetricsCollector accurately calculates 4-boundary percentiles."""
    collector = LifecycleMetricsCollector()
    
    for i in range(100):
        t = FrameTiming(
            camera_id=str(i % 4),
            sequence_number=i,
            capture_started_ns=1_000_000_000,
            capture_completed_ns=1_010_000_000,
            enqueued_ns=1_010_000_000,
            dequeued_ns=1_015_000_000,
            preprocess_started_ns=1_015_000_000,
            preprocess_completed_ns=1_016_000_000,
            inference_started_ns=1_016_000_000,
            inference_completed_ns=1_018_500_000,
            postprocess_completed_ns=1_019_500_000,
            tracking_completed_ns=1_020_000_000,
            rules_completed_ns=1_021_000_000,
            event_created_ns=1_022_000_000,
            event_persisted_ns=1_030_000_000,
            websocket_published_ns=1_040_000_000,
        )
        collector.record(t)
        
    summary = collector.summary()
    assert "measurement_1_inference_only" in summary
    assert "measurement_2_gpu_pipeline" in summary
    assert "measurement_3_decoded_to_rule" in summary
    assert "measurement_4_capture_to_alert" in summary
    
    # Check p50 values match expected
    assert summary["measurement_1_inference_only"]["p50"] == pytest.approx(2.5, rel=0.1)
    assert summary["measurement_2_gpu_pipeline"]["p50"] == pytest.approx(4.5, rel=0.1)


def test_thermal_governor_adaptation():
    """Verify ThermalGovernor reports valid telemetry and calculates throttling factors."""
    governor = ThermalGovernor()
    telemetry = governor.check_telemetry()
    
    assert "temperature_c" in telemetry
    assert "power_watts" in telemetry
    assert "throttle_state" in telemetry
    assert "fps_scale" in telemetry
    assert 0.0 < telemetry["fps_scale"] <= 1.0


def test_engine_manifest_validation(tmp_path):
    """Verify EngineManifest generation and validation check."""
    engine_file = tmp_path / "model.engine"
    engine_file.write_bytes(b"dummy_engine_data_12345")
    
    manifest = EngineManifest(str(engine_file))
    manifest.write_manifest(
        model_name="yolo11s",
        precision="FP16",
        input_shape=(1, 3, 640, 640),
        sm_arch="SM_86",
        trt_version="10.2.0"
    )
    
    assert manifest.validate() is True
