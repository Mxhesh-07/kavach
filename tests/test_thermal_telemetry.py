"""
Phase 15: Edge Hardware Telemetry, Thermal Management & Throttling Tests.
Validates:
1. Dynamic temperature monitoring and NVML fallback.
2. Multi-stage throttling state transitions (NORMAL -> THROTTLED -> CRITICAL).
3. Cooling hysteresis and de-escalation recovery.
4. Background monitor thread lifecycle and snapshot retrieval.
"""
import time
import pytest
from core.thermal import ThermalGovernor


def test_thermal_governor_lifecycle():
    """Verify background monitor thread starts, records telemetry, and shuts down cleanly."""
    gov = ThermalGovernor(poll_interval=0.1)
    gov.start()
    assert gov._running is True
    assert gov._thread is not None and gov._thread.is_alive()
    
    # Wait for monitor loop tick
    time.sleep(0.25)
    
    snapshot = gov.get_telemetry_snapshot()
    assert "temperature_c" in snapshot
    assert "power_watts" in snapshot
    assert "fps_scale" in snapshot
    assert snapshot["fps_scale"] in (0.5, 0.75, 1.0)
    
    gov.stop()
    assert gov._running is False


def test_thermal_state_transitions():
    """Simulate temperature spikes and test state machine escalation & hysteresis."""
    gov = ThermalGovernor(
        target_temp_c=70.0,
        throttle_temp_c=80.0,
        critical_temp_c=85.0
    )
    # Mock NVML active
    gov.nvml_initialized = True
    
    class MockMem:
        used = 2 * 1024 * 1024 * 1024
    class MockUtil:
        gpu = 85
        
    class MockPynvml:
        NVML_TEMPERATURE_GPU = 0
        current_temp = 65
        @classmethod
        def nvmlDeviceGetTemperature(cls, handle, sensor):
            return cls.current_temp
        @classmethod
        def nvmlDeviceGetPowerUsage(cls, handle):
            return 115000 # 115W
        @classmethod
        def nvmlDeviceGetUtilizationRates(cls, handle):
            return MockUtil()
        @classmethod
        def nvmlDeviceGetMemoryInfo(cls, handle):
            return MockMem()

    import sys
    sys.modules["pynvml"] = MockPynvml

    # 1. Nominal
    MockPynvml.current_temp = 65
    t1 = gov.check_telemetry()
    assert t1["throttle_state"] == "NORMAL"
    assert t1["fps_scale"] == 1.0

    # 2. Throttled (>= 80C)
    MockPynvml.current_temp = 82
    t2 = gov.check_telemetry()
    assert t2["throttle_state"] == "THROTTLED"
    assert t2["fps_scale"] == 0.75

    # 3. Critical (>= 85C)
    MockPynvml.current_temp = 89
    t3 = gov.check_telemetry()
    assert t3["throttle_state"] == "CRITICAL"
    assert t3["fps_scale"] == 0.50

    # 4. Hysteresis (75C stays in current state until <= 70C)
    MockPynvml.current_temp = 75
    t4 = gov.check_telemetry()
    assert t4["throttle_state"] == "CRITICAL"
    assert t4["fps_scale"] == 0.50

    # 5. Cooled back down to <= 70C
    MockPynvml.current_temp = 68
    t5 = gov.check_telemetry()
    assert t5["throttle_state"] == "NORMAL"
    assert t5["fps_scale"] == 1.0
