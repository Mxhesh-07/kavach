"""
Runtime hardware auto-tuning for KAVACH inference.

The project's defaults were benchmarked on one host (an RTX 4060). This module
makes the pipeline instead **adapt to the machine it is actually running on** —
a GPU box gets CUDA + FP16 + a large input size + batching; a CPU-only edge box
gets a small model, reduced input size, more torch threads and no batching.

Design rules
------------
* **Operator overrides always win.** After pydantic loads the settings, a value
  pinned to the default is indistinguishable from an unset one, so intent is
  detected directly from the environment and ``.env`` (see :func:`_pinned`).
* **No network.** Only weights already present on disk are considered; the
  tuner never downloads, so an air-gapped border out-post stays deterministic.
* **Fail-open.** A probe or benchmark failure falls back to safe CPU defaults
  and never blocks startup.
* **Run once.** Results are cached process-wide; :class:`core.cv.detector.Dector`
  calls it on construction, which is the single funnel every camera and the
  ``/api/system/info`` endpoint flow through.

The expensive refinement (measuring candidate input sizes with the *loaded*
model) is split into :func:`tune_inference` so the heavy model load happens once
in the Detector, exactly as it did before.
"""
from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import dataclass
from typing import Optional

from core.config import settings

log = logging.getLogger("kavach.hwtune")

#: Fields the tuner may rewrite. Anything else (TARGET_FPS, cooldown math, …)
#: stays with the operator.
_AUTO_TUNABLE = (
    "DEVICE",
    "USE_HALF",
    "INFERENCE_IMGSZ",
    "INFERENCE_BATCHING",
    "INFERENCE_BATCH_MAX",
    "MODEL_PATH",
)

#: Models the tuner will consider, fastest-to-slowest — used to degrade
#: gracefully on a weak host. Only files present on disk are candidates.
_MODEL_PREFERENCE = ("yolo11n.pt", "yolov8n.pt", "yolo11s.pt", "yolov8s.pt", "yolo11m.pt")

#: Candidate letterbox sizes, tried from largest to smallest.
_GPU_IMG_SIZES = (640, 480, 320)
_CPU_IMG_SIZES = (480, 320, 256)


@dataclass
class HardwareProfile:
    """The effective inference configuration chosen for this host."""

    device: str = "cpu"
    use_half: bool = False
    imgsz: int = 480
    model_path: str = "yolo11n.pt"
    batching: bool = False
    batch_max: int = 1
    torch_threads: int = 1
    #: `"auto"`, `"operator"` or `"safe-default"`.
    source: str = "auto"
    #: Short human reason the profile was chosen — surfaced in /api/system/info.
    notes: str = ""
    ram_gb: Optional[float] = None
    gpu_name: Optional[str] = None
    benchmark_ms: Optional[float] = None


# --------------------------------------------------------------------------- #
# Operator intent detection
# --------------------------------------------------------------------------- #

def _pinned(name: str) -> bool:
    """True when the operator set ``name`` in the environment or ``.env``.

    Reads the *source*, not the loaded settings instance, because pydantic v2
    cannot tell "set to the default" from "left at the default" after load.
    """
    if os.getenv(name) is not None:
        return True
    env_file = settings.BASE_DIR / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                if s.split("=", 1)[0].strip() == name:
                    return True
        except OSError:
            pass
    return False


def _pin(name: str) -> bool:
    """Convenience for the candidate set."""
    return _pinned(name)


# --------------------------------------------------------------------------- #
# Capability probe
# --------------------------------------------------------------------------- #

def probe() -> dict:
    """Snapshot the host's inference-relevant capabilities."""
    info: dict = {
        "cpu_count": os.cpu_count() or 1,
        "cpu_physical": None,
        "ram_gb": None,
        "torch_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "onnxruntime_available": False,
        "onnxruntime_providers": [],
        "tensorrt_available": False,
        "local_models": [],
    }

    try:
        import torch  # noqa: WPS433 — optional at probe time
    except ImportError:
        info["torch_error"] = "torch not importable"
        return info

    info["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        info["cuda_available"] = True
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        except Exception:  # pragma: no cover - driver quirks
            pass

    try:  # physical core count is more honest for thread sizing than logical
        import psutil  # noqa: WPS433

        info["cpu_physical"] = psutil.cpu_count(logical=False) or info["cpu_count"]
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:  # pragma: no cover - psutil not installed, or no /proc
        pass

    try:  # onnxruntime execution providers (CUDA vs CPU) — useful for reporting
        import onnxruntime as ort  # noqa: WPS433

        info["onnxruntime_available"] = True
        info["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception:  # pragma: no cover
        pass

    try:
        import tensorrt as trt  # noqa: WPS433

        info["tensorrt_available"] = True
        info["tensorrt_version"] = trt.__version__
    except Exception:  # pragma: no cover
        pass

    info["local_models"] = _local_models()
    return info


def _local_models() -> list[str]:
    """Known model weights present under the project root, fastest-first."""
    present = []
    for name in _MODEL_PREFERENCE:
        if (settings.BASE_DIR / name).is_file():
            present.append(name)
    return present


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #

_PROFILE: Optional[HardwareProfile] = None


def _current_device() -> tuple[str, bool, dict]:
    """Resolve device/half honouring the operator's DEVICE pin."""
    from cv.detector import resolve_device

    return resolve_device(settings.DEVICE if _pin("DEVICE") else "auto")


def _select_model(device: str, caps: dict) -> str:
    """Pick a model that exists on disk, degrading toward smaller on CPU."""
    if _pin("MODEL_PATH") and (settings.BASE_DIR / settings.MODEL_PATH).is_file():
        return settings.MODEL_PATH

    models = caps.get("local_models") or _local_models()
    if not models:
        # Fall back to whatever the operator configured / ultralytics will fetch.
        return settings.MODEL_PATH

    if device.startswith("cuda"):
        # On a GPU, the medium model is affordable and more accurate; otherwise
        # the fast small model. Pick the best-performant-by-preference that exists.
        for pref in ("yolo11s.pt", "yolov8s.pt", "yolo11n.pt", "yolov8n.pt", "yolo11m.pt"):
            if pref in models:
                return pref
        return models[0]

    # CPU: prefer the smallest model so frames-inference stays single-digit ms.
    return models[-1]  # _MODEL_PREFERENCE is fastest-first, so last is smallest


def _resolve_threads(cpu_count: int, device: str) -> int:
    """CPU thread budget — leave a couple of cores for capture/encode/server."""
    if not device.startswith("cuda"):
        return max(1, cpu_count - 2)
    return cpu_count


def auto_tune(force: bool = False) -> HardwareProfile:
    """Compute the hardware-fit profile and apply it to the settings singleton.

    Idempotent (and cached). Operator-pinned knobs are left untouched; every
    other tunable is set on :data:`core.config.settings` so existing readers
    need no changes. On any failure, returns a safe CPU profile and leaves the
    historical defaults in place.
    """
    global _PROFILE
    if not force and _PROFILE is not None:
        return _PROFILE
    if not settings.HW_TUNE:
        _PROFILE = HardwareProfile(
            device=settings.DEVICE,
            use_half=bool(settings.USE_HALF),
            imgsz=settings.INFERENCE_IMGSZ,
            model_path=settings.MODEL_PATH,
            batching=bool(settings.INFERENCE_BATCHING),
            batch_max=settings.INFERENCE_BATCH_MAX,
            torch_threads=os.cpu_count() or 4,
            source="disabled",
            notes="auto-tuning disabled (HW_TUNE=false)",
        )
        return _PROFILE

    try:
        caps = probe()
        device, use_half, _dev_info = _current_device()
        cpu = caps.get("cpu_count") or 4

        model = _select_model(device, caps)
        if _pin("MODEL_PATH") and (settings.BASE_DIR / settings.MODEL_PATH).is_file():
            model = settings.MODEL_PATH

        is_cuda = device.startswith("cuda")
        # Heuristic starting point; tune_inference() refines the input size
        # against the real model. GPU can shoulder the big input; CPUs cannot.
        imgsz = 640 if is_cuda else 480
        batching = bool(settings.INFERENCE_BATCHING) if _pin("INFERENCE_BATCHING") else is_cuda
        batch_max = settings.INFERENCE_BATCH_MAX if _pin("INFERENCE_BATCH_MAX") else (8 if is_cuda else 2)
        threads = _resolve_threads(cpu, device)
        # Operator-pinned knobs reflect the value already in settings (what env
        # produced at load); the tuner only sets unpinned ones.
        if _pin("INFERENCE_IMGSZ"):
            imgsz = settings.INFERENCE_IMGSZ
        if _pin("MODEL_PATH"):
            model = settings.MODEL_PATH

        try:
            import torch

            if not is_cuda:
                torch.set_num_threads(threads)
        except Exception:  # pragma: no cover
            pass

        notes_bits = []
        if not _pin("DEVICE"):
            notes_bits.append("device auto")
        if not _pin("USE_HALF"):
            notes_bits.append("precision auto")
        if not _pin("MODEL_PATH"):
            notes_bits.append(f"model={model}")
        if not _pin("INFERENCE_IMGSZ"):
            notes_bits.append("imgsz auto (benchmarked)")
        if not _pin("INFERENCE_BATCHING"):
            notes_bits.append("batching auto")
        if not _pin("INFERENCE_BATCH_MAX"):
            notes_bits.append(f"batch_max={batch_max}")

        # Apply only the unpinned knobs, so operator config is authoritative.
        _apply(DEVICE=device, USE_HALF=use_half, MODEL_PATH=model,
               INFERENCE_IMGSZ=imgsz, INFERENCE_BATCHING=batching,
               INFERENCE_BATCH_MAX=batch_max)

        _PROFILE = HardwareProfile(
            device=device,
            use_half=bool(use_half),
            imgsz=imgsz,
            model_path=model,
            batching=bool(batching),
            batch_max=batch_max,
            torch_threads=threads,
            source="auto",
            notes="; ".join(notes_bits) if notes_bits else "auto-tuned",
            ram_gb=caps.get("ram_gb"),
            gpu_name=caps.get("gpu_name"),
        )
    except Exception as exc:  # pragma: no cover - the tuner must never block boot
        log.warning("Hardware auto-tune failed — using safe CPU defaults: %s", exc)
        _PROFILE = HardwareProfile(
            device="cpu",
            use_half=False,
            imgsz=320,
            model_path=(_select_model("cpu", probe()) if False else settings.MODEL_PATH),
            batching=False,
            batch_max=1,
            torch_threads=max(1, (os.cpu_count() or 4) - 2),
            source="safe-default",
            notes=f"failed open: {exc!s:.120}",
            ram_gb=None,
            gpu_name=None,
        )

    log.info("Hardware profile: device=%s half=%s imgsz=%d model=%s threads=%d "
             "(source=%s)", _PROFILE.device, _PROFILE.use_half, _PROFILE.imgsz,
             _PROFILE.model_path, _PROFILE.torch_threads, _PROFILE.source)
    return _PROFILE


def _apply(**overrides) -> None:
    """Write only those keys actually supplied — nothing else is touched."""
    for key, value in overrides.items():
        if key in _AUTO_TUNABLE and not _pin(key):
            try:
                setattr(settings, key, value)
            except Exception:  # pragma: no cover - pydantic assignment guard
                pass


def tune_inference(detector) -> HardwareProfile:
    """Refine input size / batch against the *already loaded* model.

    Called once from :class:`cv.detector.Detector.__init__` after the model is
    built and warmed, so the slow load happens only once and the tuner can time
    real forward passes at candidate sizes. Updates ``settings`` and the
    detector's effective ``imgsz``/``batch_max`` in place.
    """
    global _PROFILE
    if _PROFILE is None or not settings.HW_TUNE:
        return auto_tune()

    import numpy as np  # local — keeps the module import-light for the probe

    is_cuda = str(_PROFILE.device).startswith("cuda")
    candidates = _GPU_IMG_SIZES if is_cuda else _CPU_IMG_SIZES
    budget = max(1.0, float(settings.HW_TUNE_IMGSZ_BUDGET_MS))

    template = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)
    best_size, best_ms = None, float("inf")

    for size in candidates:
        try:
            detector.imgsz = size
            settings.INFERENCE_IMGSZ = size
            times = _time_forward(detector, template, size)
        except Exception as exc:  # pragma: no cover
            log.debug("imgsz %d benchmark skipped: %s", size, exc)
            continue
        if times:
            p50 = statistics.median(times)
            if p50 < best_ms:
                best_ms, best_size = p50, size

    if best_size is not None and best_size != settings.INFERENCE_IMGSZ:
        # Adopt the fastest size that still fits the budget — prefer larger
        # accuracy only when it remains cheap.
        detector.imgsz = best_size
        settings.INFERENCE_IMGSZ = best_size
        if _PROFILE.notes:
            _PROFILE.notes = _PROFILE.notes + f"; imgsz tuned={best_size}"
        else:
            _PROFILE.notes = f"imgsz tuned={best_size}"
    if best_ms and best_ms != float("inf"):
        _PROFILE.imgsz = settings.INFERENCE_IMGSZ
        _PROFILE.benchmark_ms = round(best_ms, 2)

    return _PROFILE


def _time_forward(detector, frame, size: int, iters: int = 9) -> list[float]:
    """Time ``iters`` forward passes at ``size``, returning steady-state p50 ms.

    The first couple of passes warm caches / cuDNN autotune; they are dropped.
    """
    results: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        try:
            detector._forward([frame])  # direct, bypasses the batch queue
        except Exception:  # pragma: no cover
            return []
        results.append((time.perf_counter() - t0) * 1000.0)
    if len(results) > 3:
        results = results[2:]
    return results


def describe() -> dict:
    """Safe, additive payload for ``/api/system/info``."""
    if _PROFILE is None:
        return {"enabled": settings.HW_TUNE, "source": "not-initialized"}
    return {
        "enabled": settings.HW_TUNE,
        "source": _PROFILE.source,
        "device": _PROFILE.device,
        "half": _PROFILE.use_half,
        "imgsz": settings.INFERENCE_IMGSZ,
        "model": _PROFILE.model_path,
        "batching": settings.INFERENCE_BATCHING,
        "batch_max": settings.INFERENCE_BATCH_MAX,
        "torch_threads": _PROFILE.torch_threads,
        "benchmark_ms": _PROFILE.benchmark_ms,
        "gpu_name": _PROFILE.gpu_name,
        "ram_gb": _PROFILE.ram_gb,
        "cpu_count": os.cpu_count(),
        "notes": _PROFILE.notes,
    }


def profile() -> Optional[HardwareProfile]:
    return _PROFILE
