"""
KAVACH Pipeline Hardware Watchdog & Resilience Coordinator.
Monitors thread heartbeats, detects frozen ingestion/inference loops,
and orchestrates automated recovery and graceful CPU degradation.
"""
import time
import logging
import threading
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class WorkerState:
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = time.time()
        self.restarts = 0
        self.healthy = True

    def beat(self):
        self.last_heartbeat = time.time()
        self.healthy = True

    def is_stalled(self) -> bool:
        return (time.time() - self.last_heartbeat) > self.timeout_seconds


class PipelineWatchdog:
    """Central watchdog coordinator managing worker thread heartbeats and recovery."""
    _instance: Optional["PipelineWatchdog"] = None
    _lock = threading.Lock()

    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self._workers: Dict[str, WorkerState] = {}
        self._recovery_hooks: Dict[str, Callable[[], None]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mutex = threading.Lock()

    @classmethod
    def get(cls) -> "PipelineWatchdog":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_worker(self, name: str, timeout_seconds: float = 5.0, on_stall: Optional[Callable[[], None]] = None):
        with self._mutex:
            self._workers[name] = WorkerState(name, timeout_seconds)
            if on_stall:
                self._recovery_hooks[name] = on_stall

    def unregister_worker(self, name: str):
        with self._mutex:
            self._workers.pop(name, None)
            self._recovery_hooks.pop(name, None)

    def heartbeat(self, name: str):
        with self._mutex:
            if name in self._workers:
                self._workers[name].beat()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, name="pipeline-watchdog", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _watch_loop(self):
        while self._running:
            with self._mutex:
                for name, worker in list(self._workers.items()):
                    if worker.is_stalled() and worker.healthy:
                        worker.healthy = False
                        worker.restarts += 1
                        logger.warning(f"Watchdog detected stalled worker '{name}' (last seen {time.time() - worker.last_heartbeat:.1f}s ago). Triggering recovery.")
                        if name in self._recovery_hooks:
                            try:
                                self._recovery_hooks[name]()
                            except Exception as e:
                                logger.error(f"Error executing recovery hook for {name}: {e}")
            time.sleep(self.check_interval)

    def get_status(self) -> Dict[str, Any]:
        with self._mutex:
            return {
                name: {
                    "healthy": w.healthy,
                    "last_seen_sec_ago": round(time.time() - w.last_heartbeat, 2),
                    "restarts": w.restarts,
                }
                for name, w in self._workers.items()
            }
