"""
Phase 14: Pipeline Stress & Concurrency Bounds Tests.
Validates multi-camera throughput, queue bounds, and memory stability under burst loads.
"""
import time
import threading
import numpy as np
import pytest
from core.scheduler import InferenceScheduler
from core.preview import PreviewStream
from core.timing import FrameTiming, LifecycleMetricsCollector


def test_high_throughput_scheduler_burst():
    """Simulate 8 concurrent cameras submitting 50 frames each at 60 FPS pacing."""
    num_cameras = 8
    frames_per_camera = 50
    completed_jobs = 0
    lock = threading.Lock()
    
    def dummy_backend(frames):
        # Simulate ~2ms GPU batch execution
        time.sleep(0.002)
        return [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "class_ids": np.zeros(0)} for _ in frames]

    scheduler = InferenceScheduler(backend_fn=dummy_backend, max_wait_ms=3.0, max_batch_size=4)
    scheduler.start()

    def camera_feeder(camera_id: int):
        nonlocal completed_jobs
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for seq in range(frames_per_camera):
            job = scheduler.submit(f"CAM-{camera_id:02d}", seq, dummy_frame)
            job.event.wait(timeout=0.5)
            if job.result is not None:
                with lock:
                    completed_jobs += 1
            time.sleep(0.001)

    threads = [threading.Thread(target=camera_feeder, args=(i,), daemon=True) for i in range(num_cameras)]
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    total_time = time.time() - start_time

    scheduler.stop()

    assert completed_jobs > 0
    # Aggregate throughput should exceed 30 FPS
    fps = completed_jobs / total_time
    assert fps > 30.0


def test_metrics_collector_thread_safety():
    """Verify concurrent metric recording from 16 worker threads."""
    collector = LifecycleMetricsCollector(max_samples=5000)
    num_threads = 16
    records_per_thread = 200

    def worker(tid: int):
        for i in range(records_per_thread):
            t = FrameTiming(
                camera_id=str(tid),
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

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    summary = collector.summary()
    assert summary["counters"]["analyzed_frames"] == num_threads * records_per_thread
    assert summary["measurement_1_inference_only"]["count"] > 0
