"""
Centralized InferenceScheduler for KAVACH.
Owns the active AI model backend, forms micro-batches with bounded wait times,
enforces per-camera latest-job fairness, and dispatches results asynchronously.
"""
from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.config import settings

log = logging.getLogger("kavach.scheduler")


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    SUPERSEDED = "SUPERSEDED"
    TIMED_OUT = "TIMED_OUT"
    MALFORMED_FRAME = "MALFORMED_FRAME"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class FrameJob:
    """An inference request submitted by a camera stream."""
    camera_id: str
    sequence_number: int
    frame: np.ndarray
    enqueued_ns: int = field(default_factory=time.monotonic_ns)
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    event: threading.Event = field(default_factory=threading.Event)

    def finish(self, result: Any = None, status: JobStatus = JobStatus.COMPLETED, error: Optional[str] = None) -> None:
        self.result = result
        self.status = status
        self.error = error
        self.event.set()


class InferenceScheduler:
    """
    Dedicated GPU inference scheduler.
    
    Guarantees:
    - Exactly at most one pending job per camera (supersedes stale jobs).
    - Micro-batching up to max_batch_size with timeout window max_wait_ms.
    - Round-robin camera fairness.
    - Zero starvation of slower cameras.
    """

    def __init__(
        self,
        backend_fn: Optional[Callable[[List[np.ndarray]], List[Any]]] = None,
        max_batch_size: Optional[int] = None,
        max_wait_ms: Optional[float] = None,
    ) -> None:
        self.backend_fn = backend_fn
        self.max_batch = max_batch_size or getattr(settings, "INFERENCE_BATCH_MAX", 4)
        self.max_wait_ms = max_wait_ms if max_wait_ms is not None else float(getattr(settings, "INFERENCE_BATCH_TIMEOUT_MS", 3.0))


        self._pending_jobs: Dict[str, FrameJob] = {}
        self._camera_order: List[str] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Scheduler metrics
        self.total_batches = 0
        self.total_jobs_completed = 0
        self.total_jobs_superseded = 0
        self.total_jobs_failed = 0
        self.batch_sizes: List[int] = []
        self.wait_times_ms: List[float] = []

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._scheduler_loop, name="inference-scheduler", daemon=True
            )
            self._worker_thread.start()
            log.info("InferenceScheduler started (max_batch=%d, wait_ms=%.1f)", self.max_batch, self.max_wait_ms)

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            # Cancel all pending jobs
            for job in self._pending_jobs.values():
                job.finish(status=JobStatus.SHUTDOWN, error="Scheduler stopped")
            self._pending_jobs.clear()
            self._condition.notify_all()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        log.info("InferenceScheduler stopped")

    def submit(self, camera_id: str, sequence: int, frame: np.ndarray) -> FrameJob:
        """Submit a frame for inference. If a job for this camera is already pending, supersede it."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            job = FrameJob(camera_id=camera_id, sequence_number=sequence, frame=frame)
            job.finish(status=JobStatus.MALFORMED_FRAME, error="Invalid or empty frame")
            return job

        job = FrameJob(camera_id=camera_id, sequence_number=sequence, frame=frame)

        with self._lock:
            if not self._running:
                job.finish(status=JobStatus.SHUTDOWN, error="Scheduler not running")
                return job

            # Check if camera already has a pending job
            old_job = self._pending_jobs.get(camera_id)
            if old_job is not None:
                old_job.finish(status=JobStatus.SUPERSEDED, error="Superseded by fresher frame")
                self.total_jobs_superseded += 1

            self._pending_jobs[camera_id] = job
            if camera_id not in self._camera_order:
                self._camera_order.append(camera_id)
            self._condition.notify()

        return job

    def execute_sync(self, camera_id: str, sequence: int, frame: np.ndarray, timeout_s: float = 2.0) -> FrameJob:
        """Submit and block until completion or timeout."""
        job = self.submit(camera_id, sequence, frame)
        if not job.event.wait(timeout=timeout_s):
            job.finish(status=JobStatus.TIMED_OUT, error=f"Inference timed out after {timeout_s}s")
        return job

    def _scheduler_loop(self) -> None:
        """Main worker loop: forms bounded micro-batches and runs inference."""
        while True:
            batch_jobs: List[FrameJob] = []

            with self._condition:
                while self._running and not self._pending_jobs:
                    self._condition.wait()

                if not self._running:
                    break

                # Wait window for micro-batch formation if batch size < max_batch
                start_wait = time.monotonic()
                deadline = start_wait + (self.max_wait_ms / 1000.0)

                while len(self._pending_jobs) < self.max_batch and time.monotonic() < deadline:
                    remaining = max(0.0001, deadline - time.monotonic())
                    self._condition.wait(timeout=remaining)
                    if not self._running:
                        break

                if not self._running:
                    break

                # Fair extraction in round-robin order
                cams_to_process = []
                for cam_id in list(self._camera_order):
                    if cam_id in self._pending_jobs:
                        cams_to_process.append(cam_id)
                        if len(cams_to_process) >= self.max_batch:
                            break

                for cam_id in cams_to_process:
                    job = self._pending_jobs.pop(cam_id)
                    batch_jobs.append(job)
                    # Rotate processed camera to back for fairness
                    self._camera_order.remove(cam_id)
                    self._camera_order.append(cam_id)

            if not batch_jobs:
                continue

            # Metrics are mutated here on the worker thread but read by ``submit()``
            # (superseded) and ``stats()`` from other threads; guard them so a
            # concurrent ``stats()`` snapshot cannot read a half-appended list.
            now_ns = time.monotonic_ns()
            with self._lock:
                self.total_batches += 1
                self.batch_sizes.append(len(batch_jobs))
                if len(self.batch_sizes) > 1000:
                    self.batch_sizes.pop(0)
                # Record queue wait time
                for j in batch_jobs:
                    wait_ms = (now_ns - j.enqueued_ns) / 1e6
                    self.wait_times_ms.append(wait_ms)
                    if len(self.wait_times_ms) > 1000:
                        self.wait_times_ms.pop(0)

            # Execute forward pass on batch
            try:
                if self.backend_fn is None:
                    # Fallback default using Detector.get().raw_detect
                    from cv.detector import Detector
                    det = Detector.get()
                    for job in batch_jobs:
                        res = det.raw_detect(job.frame)
                        job.finish(result=res, status=JobStatus.COMPLETED)
                        with self._lock:
                            self.total_jobs_completed += 1
                else:
                    frames = [j.frame for j in batch_jobs]
                    results = self.backend_fn(frames)
                    for job, res in zip(batch_jobs, results):
                        job.finish(result=res, status=JobStatus.COMPLETED)
                        with self._lock:
                            self.total_jobs_completed += 1
            except Exception as exc:
                log.exception("Batch inference execution failed: %s", exc)
                with self._lock:
                    self.total_jobs_failed += len(batch_jobs)
                for job in batch_jobs:
                    job.finish(status=JobStatus.INFERENCE_FAILED, error=str(exc))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_batches = self.total_batches
            total_jobs_completed = self.total_jobs_completed
            total_jobs_superseded = self.total_jobs_superseded
            total_jobs_failed = self.total_jobs_failed
            sizes = list(self.batch_sizes)
            waits = list(self.wait_times_ms)
        return {
            "total_batches": total_batches,
            "total_jobs_completed": total_jobs_completed,
            "total_jobs_superseded": total_jobs_superseded,
            "total_jobs_failed": total_jobs_failed,
            "avg_batch_size": round(sum(sizes) / max(1, len(sizes)), 2),
            "avg_queue_wait_ms": round(sum(waits) / max(1, len(waits)), 2),
        }
