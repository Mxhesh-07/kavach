"""
Per-camera background render/encode/publish worker.

Moved off the analytics thread: JPEG encode (``cv2.imencode``),
``FrameBuffer.publish`` and the clip pre-roll ``clips.push``. Together those
were the 100ms tail that inflated ``latency_ms`` and blocked new frames, so
they now run on a daemon thread with **drop-oldest** backpressure — exactly the
latest-frame philosophy the rest of the pipeline already uses (``maxlen=1``
deque in the decoder, shed-frames in the inference queue).

Contract
--------
* ``submit(result, captured_at)`` overwrites any not-yet-rendered frame, so a
  camera that is falling behind publishes its *last* analysed frame, never a
  backlog of stale ones.
* ``stop()`` signals the worker non-blockingly; ``join()`` drains it under a
  deadline shared with the rest of teardown.
* ``_sync_publish`` is the inline synchronous fallback used when the worker
  thread is not alive (e.g. a processor that was constructed but never
  ``start()``-ed, or during hard teardown) — this keeps
  ``tests/test_regressions.py:97`` and the offline-card path working without a
  running thread.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from core.config import settings

if TYPE_CHECKING:  # avoid import cycles at module load
    from core.analytics import AnalysisResult
    from core.camera import CameraProcessor
    from core.backend import FrameBuffer

log = logging.getLogger("kavach.renderworker")


class RenderWorker:
    """One daemon thread per CameraProcessor; owns encode + publish + clips."""

    def __init__(self, owner: "CameraProcessor") -> None:
        self._owner = owner
        self._cond = threading.Condition()
        self._stop = threading.Event()
        # A single latest-wins slot. None = idle, non-None = pending work.
        self._slot: Optional[tuple["AnalysisResult", float, "FrameBuffer"]] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"render-cam{self._owner.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop. Non-blocking; returns in microseconds."""
        self._stop.set()
        with self._cond:
            # Discard the pending slot so the worker wakes immediately instead
            # of rendering one last stale frame during teardown.
            self._slot = None
            self._cond.notify_all()

    def join(self, timeout: float) -> bool:
        """Wait for the worker thread to exit. Returns False on timeout."""
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    @property
    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # -- the hot path -------------------------------------------------------- #

    def submit(self, result: "AnalysisResult", captured_at: float,
               buffer: "FrameBuffer") -> None:
        """Hand a frame to the worker. Overwrites any older pending frame."""
        if not self._owner._running and not self._owner._paused:
            # Processor is tearing down; publish synchronously so the last
            # frame is not lost during hard-stop / cleanup paths.
            self._sync_publish(result, captured_at, buffer)
            return
        with self._cond:
            self._slot = (result, captured_at, buffer)
            self._cond.notify_all()

    # -- internals ----------------------------------------------------------- #

    def _loop(self) -> None:
        owner = self._owner
        analyzer = owner.analyzer
        clips = owner.clips

        while not self._stop.is_set():
            with self._cond:
                if self._slot is None:
                    # Wait to be signalled, but wake at least every 50 ms so a
                    # missed notify cannot strand us during teardown.
                    self._cond.wait(timeout=0.05)
                slot = self._slot
                if slot is None:
                    continue
                result, captured_at, buffer = slot
                self._slot = None  # consume; next submit overwrites if stale

            if not owner._running and not owner._paused:
                # Tearing down — publish inline, skip clip writes.
                self._sync_publish(result, captured_at, buffer, clip=False)
                continue

            try:
                self._render(result, captured_at, buffer, analyzer, clips)
            except Exception:  # pragma: no cover - guard the daemon thread
                log.exception("render worker: frame dropped during render")

        # Final drain of one last frame on the way out so the dashboard tile
        # holds the final analysed frame instead of a black offline card.
        with self._cond:
            slot = self._slot
            self._slot = None
        if slot is not None:
            result, captured_at, buffer = slot
            try:
                self._render(result, captured_at, buffer, analyzer, clips, clip=False)
            except Exception:  # pragma: no cover
                log.exception("render worker: final frame dropped")

    def _render(self, result: "AnalysisResult", captured_at: float,
                buffer: "FrameBuffer", analyzer, clips, clip: bool = True) -> None:
        annotated = analyzer.render_frame(result)
        self._publish(annotated, result, captured_at, buffer, clips, clip=clip)

    def _sync_publish(self, result: "AnalysisResult", captured_at: float,
                      buffer: "FrameBuffer", clip: bool = True) -> None:
        """Synchronous inline publish — fallback when no thread is running."""
        if not self._owner._running:
            return
        annotated = self._owner.analyzer.render_frame(result)
        self._publish(annotated, result, captured_at, buffer, self._owner.clips,
                      clip=clip)

    def _publish(self, annotated, result, captured_at, buffer, clips,
                 clip: bool = True) -> None:
        """Encode + FrameBuffer.publish + optional clip push."""
        ok, encoded = cv2.imencode(
            ".jpg", annotated,
            [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY],
        )
        jpeg = encoded.tobytes() if ok else None
        buffer.publish(
            self._owner.camera_id, annotated, jpeg,
            clean=result.raw_frame,
        )
        if settings.EVIDENCE_ENABLED and clips is not None and clip:
            clips.push(annotated)
