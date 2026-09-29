"""
Hardware auto-tuner behaviour.

These tests run with ``HW_TUNE=true`` (the suite default in conftest is false,
so tuning is exercised explicitly and never contaminates the other tests). The
tuner is defaulted to deterministic values and must never download anything.
"""
import os

import pytest

from core.config import settings


@pytest.fixture
def enable_tune(monkeypatch):
    from core import hwtune
    monkeypatch.delenv("HW_TUNE", raising=False)
    # Settings is a process-wide singleton read at import; the real env flag
    # controls auto_tune(). Set both so the tuner actually engages.
    from core.config import settings
    monkeypatch.setattr(settings, "HW_TUNE", True)
    monkeypatch.setattr(hwtune, "_PROFILE", None)
    yield
    monkeypatch.setattr(hwtune, "_PROFILE", None)


def _fresh_profile():
    from core import hwtune
    hwtune._PROFILE = None
    return hwtune.auto_tune(force=True)


def test_probe_reports_cpu_defaults():
    """probe() must always return a usable, CPU-safe capability snapshot."""
    from core import hwtune

    caps = hwtune.probe()
    assert caps["cpu_count"] >= 1
    assert isinstance(caps["cuda_available"], bool)
    assert isinstance(caps["local_models"], list)
    # A torch-less / import-failing host still returns the skeleton.
    assert "cpu_count" in caps


def test_auto_tune_returns_a_profile(enable_tune):
    from core import hwtune

    profile = _fresh_profile()
    assert isinstance(profile, hwtune.HardwareProfile)
    # The profile must carry an effective device and a resolvable model.
    assert profile.device in ("cpu", "cuda:0")
    assert profile.imgsz >= 256
    assert profile.source in ("auto", "safe-default", "disabled")
    # It applies to the settings singleton (pydantic v2 allows attribute writes).
    assert settings.HW_TUNE is True


def test_auto_tune_is_cached_and_idempotent(enable_tune):
    from core import hwtune

    first = hwtune.auto_tune()
    second = hwtune.auto_tune()
    assert first is second  # cached — never re-probes per camera


def test_operator_override_beats_tuner(enable_tune, monkeypatch):
    """A knob pinned in the env must win — the tuner leaves it untouched.

    pydantic fills ``settings`` from the env at load, so an operator-pinned
    value already lives in the singleton; the tuner's job is to *not overwrite*
    it. We simulate that by pinning the env AND pre-setting the value the load
    would have produced, then assert a re-tune preserves it and does not apply
    the tuner's heuristic instead.
    """
    from core import hwtune

    monkeypatch.setenv("DETECT_CLASSES", "[0]")          # so _pin sees something
    settings.INFERENCE_IMGSZ = 999                        # what env produced at load
    monkeypatch.setenv("INFERENCE_IMGSZ", "999")          # operator pin intent
    profile = _fresh_profile()
    # The tuner must not rewrite a pinned knob.
    assert settings.INFERENCE_IMGSZ == 999
    assert profile.imgsz == 999


def test_safe_default_fallback(enable_tune, monkeypatch):
    """If probing explodes, the tuner fails open to a CPU-safe profile."""
    from core import hwtune

    def _boom():
        raise RuntimeError("no torch here")

    monkeypatch.setattr(hwtune, "probe", _boom)
    profile = _fresh_profile()
    assert profile.source in ("safe-default")
    # Fail-open must never raise or block startup.
    assert profile.device in ("cpu",)
    assert profile.imgsz >= 256


def test_describe_exposes_tuning(enable_tune):
    from core import hwtune

    _fresh_profile()
    info = hwtune.describe()
    assert info["enabled"] is True
    for key in ("source", "device", "imgsz", "model", "batching", "torch_threads"):
        assert key in info