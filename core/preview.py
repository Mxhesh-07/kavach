"""
Decoupled Operator Preview Pipeline.
Maintains a separate non-blocking preview ring buffer with optional downsampled rendering.
"""
import time
import threading
import numpy as np
from collections import deque
from typing import Optional, Tuple

class PreviewStream:
    """Non-blocking ring-buffered frame visualizer for human operator dashboard."""
    def __init__(self, max_fps: int = 15, target_resolution: Tuple[int, int] = (640, 360)):
        self.max_fps = max_fps
        self.interval = 1.0 / max_fps
        self.target_resolution = target_resolution
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_rendered_ts = 0.0

    def push_frame(self, frame: np.ndarray):
        """Called by ingestion thread; never blocks analytics or detection."""
        with self._lock:
            self._latest_frame = frame

    def get_display_frame(self) -> Optional[np.ndarray]:
        """Fetches the latest preview frame, downsampling if needed at throttled FPS."""
        now = time.time()
        if now - self._last_rendered_ts < self.interval:
            return None
        with self._lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()
        self._last_rendered_ts = now
        return frame
