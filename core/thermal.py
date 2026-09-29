"""
IBVAP Dynamic Thermal & Power Throttling Monitor.
Monitors GPU temperature, power, and clock states to adaptively adjust stream FPS.
"""
import time
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ThermalGovernor:
    """Monitors GPU/CPU temperature and adapts processing pipeline parameters."""
    _instance: Optional["ThermalGovernor"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        target_temp_c: float = 75.0,
        throttle_temp_c: float = 83.0,
        critical_temp_c: float = 88.0,
        poll_interval: float = 1.0,
    ):
        self.target_temp_c = target_temp_c
        self.throttle_temp_c = throttle_temp_c
        self.critical_temp_c = critical_temp_c
        self.poll_interval = poll_interval
        
        self.current_fps_multiplier = 1.0
        self.current_state = "NORMAL"
        self.nvml_initialized = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshot_lock = threading.Lock()
        
        self._last_telemetry: Dict[str, Any] = {
            "temperature_c": 50.0,
            "power_watts": 70.0,
            "gpu_utilization_pct": 45,
            "memory_used_mb": 1400,
            "throttle_state": "NORMAL",
            "fps_scale": 1.0,
            "timestamp": time.time(),
        }

        self._init_nvml()

    @classmethod
    def get(cls) -> "ThermalGovernor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml_initialized = True
            logger.info("NVML thermal monitoring initialized successfully.")
        except Exception as e:
            logger.info(f"NVML unavailable ({e}); running in simulated passive mode.")
            self.nvml_initialized = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, name="thermal-governor", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _monitor_loop(self):
        while self._running:
            self.check_telemetry()
            time.sleep(self.poll_interval)

    def check_telemetry(self) -> Dict[str, Any]:
        """Query GPU temperature, power draw, memory, and clock rate."""
        telemetry = {}
        if not self.nvml_initialized:
            telemetry = {
                "temperature_c": 55.0,
                "power_watts": 80.0,
                "gpu_utilization_pct": 50,
                "memory_used_mb": 1500,
                "throttle_state": "NORMAL",
                "fps_scale": 1.0,
                "timestamp": time.time(),
            }
        else:
            try:
                import pynvml
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                # Hysteresis state machine
                if temp >= self.critical_temp_c:
                    self.current_fps_multiplier = 0.50
                    self.current_state = "CRITICAL"
                elif temp >= self.throttle_temp_c:
                    self.current_fps_multiplier = 0.75
                    self.current_state = "THROTTLED"
                elif temp <= self.target_temp_c:
                    self.current_fps_multiplier = 1.0
                    self.current_state = "NORMAL"

                telemetry = {
                    "temperature_c": temp,
                    "power_watts": power,
                    "gpu_utilization_pct": util.gpu,
                    "memory_used_mb": mem.used / (1024 * 1024),
                    "throttle_state": self.current_state,
                    "fps_scale": self.current_fps_multiplier,
                    "timestamp": time.time(),
                }
            except Exception as e:
                logger.error(f"Failed to query NVML telemetry: {e}")
                telemetry = {
                    "temperature_c": 60.0,
                    "power_watts": 80.0,
                    "gpu_utilization_pct": 50,
                    "memory_used_mb": 1500,
                    "throttle_state": "NORMAL",
                    "fps_scale": 1.0,
                    "timestamp": time.time(),
                }

        with self._snapshot_lock:
            self._last_telemetry = telemetry
        return telemetry

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        with self._snapshot_lock:
            return dict(self._last_telemetry)
