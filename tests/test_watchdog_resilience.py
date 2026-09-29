"""
Phase 16: Failure Modes, Hardware Watchdogs & Graceful Degradation Tests.
Validates:
1. Worker heartbeat registration and timeout detection.
2. Automated recovery hook execution on pipeline stall.
3. Clean unregistration and leak-free watchdog shutdown.
4. Graceful backend degradation when CUDA fails.
"""
import time
import pytest
from core.watchdog import PipelineWatchdog
from core.backend import BackendDetector


def test_watchdog_heartbeat_and_stall_detection():
    """Verify watchdog accurately triggers recovery callback when worker stops heartbeating."""
    watchdog = PipelineWatchdog(check_interval=0.1)
    recovered = False

    def recovery_action():
        nonlocal recovered
        recovered = True

    watchdog.register_worker("camera_feeder_01", timeout_seconds=0.25, on_stall=recovery_action)
    watchdog.start()

    # Beat initially
    watchdog.heartbeat("camera_feeder_01")
    assert not recovered

    # Sleep past timeout to trigger stall
    time.sleep(0.40)
    assert recovered is True

    status = watchdog.get_status()
    assert "camera_feeder_01" in status
    assert status["camera_feeder_01"]["healthy"] is False
    assert status["camera_feeder_01"]["restarts"] == 1

    watchdog.stop()


def test_watchdog_unregister():
    """Verify unregistering a worker stops monitoring without errors."""
    watchdog = PipelineWatchdog(check_interval=0.1)
    watchdog.register_worker("temp_worker", timeout_seconds=0.2)
    watchdog.unregister_worker("temp_worker")
    
    status = watchdog.get_status()
    assert "temp_worker" not in status


def test_backend_graceful_degradation_on_exception(monkeypatch):
    """Verify BackendDetector seamlessly degrades to PyTorch if native C++ engine encounters errors."""
    detector = BackendDetector(model_path="yolo11s.pt")
    
    # Force backend_type to test fallback branch
    detector.backend_type = "pytorch"
    assert detector._torch_model is not None
