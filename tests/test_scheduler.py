"""
Tests for Phase 4: Centralized InferenceScheduler, bounded micro-batching,
fair scheduling, and stale job supersession.
"""
import time
import numpy as np
import pytest

from core.scheduler import InferenceScheduler, FrameJob, JobStatus


def test_scheduler_lifecycle():
    """Scheduler starts, accepts jobs, and stops cleanly."""
    executed = []
    def mock_backend(batch):
        executed.append(len(batch))
        return [f"result_{i}" for i in range(len(batch))]

    sched = InferenceScheduler(backend_fn=mock_backend, max_batch_size=4, max_wait_ms=5.0)
    sched.start()

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    job = sched.execute_sync("cam1", 1, frame, timeout_s=1.0)
    
    assert job.status == JobStatus.COMPLETED
    assert job.result == "result_0"
    assert len(executed) >= 1

    sched.stop()
    assert sched._running is False


def test_scheduler_micro_batch_formation():
    """Multiple concurrent jobs are batched up to max_batch_size."""
    batch_sizes = []
    def mock_backend(batch):
        batch_sizes.append(len(batch))
        time.sleep(0.01)
        return [f"res_{i}" for i in range(len(batch))]

    sched = InferenceScheduler(backend_fn=mock_backend, max_batch_size=4, max_wait_ms=20.0)
    sched.start()

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # Submit 3 jobs rapidly from 3 distinct cameras
    j1 = sched.submit("cam1", 1, frame)
    j2 = sched.submit("cam2", 1, frame)
    j3 = sched.submit("cam3", 1, frame)

    j1.event.wait(timeout=1.0)
    j2.event.wait(timeout=1.0)
    j3.event.wait(timeout=1.0)

    assert j1.status == JobStatus.COMPLETED
    assert j2.status == JobStatus.COMPLETED
    assert j3.status == JobStatus.COMPLETED
    assert any(size >= 2 for size in batch_sizes)

    sched.stop()


def test_stale_job_supersession():
    """A new job for the same camera supersedes the previous pending job."""
    sched = InferenceScheduler(max_batch_size=4, max_wait_ms=100.0)
    # Do not start worker immediately so jobs queue up
    sched._running = True

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    j1 = sched.submit("cam1", 1, frame)
    j2 = sched.submit("cam1", 2, frame)

    assert j1.status == JobStatus.SUPERSEDED
    assert j2.status == JobStatus.PENDING
    assert sched.total_jobs_superseded == 1


def test_malformed_frame_rejection():
    """Empty or None frames are immediately rejected with MALFORMED_FRAME."""
    sched = InferenceScheduler()
    sched.start()

    j1 = sched.submit("cam1", 1, None)
    assert j1.status == JobStatus.MALFORMED_FRAME

    j2 = sched.submit("cam1", 2, np.zeros((0,), dtype=np.uint8))
    assert j2.status == JobStatus.MALFORMED_FRAME

    sched.stop()
