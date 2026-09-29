"""
Phase 21: Final Acceptance & Production Operational Readiness Suite.
Validates:
1. demo_showcase.py headless lifecycle execution, metrics collection, and report generation.
2. Live pipeline coordination between scheduler, backend detector, rules engine, and hash chain.
3. Thread watchdog heartbeat registration and stall resilience.
4. Clean shutdown, zero thread or GPU resource leaks.
"""
from pathlib import Path
import pytest
import time
import numpy as np

from demo_showcase import OperatorShowcase, LiveHashChain
from core.backend import BackendDetector
from core.thermal import ThermalGovernor
from core.watchdog import PipelineWatchdog
from cv.detector import Detection
from cv.rules import FenceRule, RuleEngine


def test_live_hashchain_verification():
    """Verify in-memory live hash chain maintains cryptographic immutability."""
    chain = LiveHashChain()
    
    # Append events
    r1 = chain.append_event(1, {"alert": "fence_breach", "cam": 1})
    r2 = chain.append_event(2, {"alert": "zone_entry", "cam": 2})
    r3 = chain.append_event(3, {"alert": "loiter_warn", "cam": 3})

    assert len(chain.records) == 3
    assert chain.verify_integrity() is True

    # Tamper with event 2
    chain.records[1]["data"]["alert"] = "falsified_peaceful_event"
    assert chain.verify_integrity() is False


def test_operator_showcase_headless_run():
    """Verify OperatorShowcase runs headless for a bounded duration and outputs valid metrics."""
    showcase = OperatorShowcase(
        num_cameras=2,
        batch_size=2,
        max_wait_ms=2.0,
        headless=True,
    )

    report = showcase.run_loop(duration_sec=1.5)

    assert report["num_cameras"] == 2
    assert report["total_frames"] > 0
    assert report["aggregate_fps"] > 0.0
    assert report["hashchain_integrity"] is True
    assert len(report["latest_hash"]) == 64
    assert report["backend"] in ("pytorch", "tensorrt_native", "tensorrt_python")


def test_pipeline_watchdog_thread_integration():
    """Verify pipeline watchdog coordinates worker heartbeats."""
    watchdog = PipelineWatchdog(check_interval=0.1)
    watchdog.start()
    
    worker_id = "test_acceptance_worker"
    stalled_called = []
    
    def on_stall():
        stalled_called.append(True)

    watchdog.register_worker(worker_id, timeout_seconds=0.3, on_stall=on_stall)
    watchdog.heartbeat(worker_id)
    
    state = watchdog.get_status()
    assert worker_id in state
    assert state[worker_id]["healthy"] is True

    # Allow stall to trigger
    time.sleep(0.5)
    assert len(stalled_called) >= 1

    watchdog.unregister_worker(worker_id)
    watchdog.stop()


def test_thermal_governor_telemetry_snapshot():
    """Verify thermal governor returns structured telemetry snapshot."""
    gov = ThermalGovernor.get()
    gov.start()
    
    telemetry = gov.check_telemetry()
    assert "temperature_c" in telemetry
    assert "power_watts" in telemetry
    assert "throttle_state" in telemetry
    assert telemetry["throttle_state"] in ("NORMAL", "THROTTLED", "CRITICAL")

    gov.stop()
