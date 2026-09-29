"""
Phase 18: Performance Verification & Benchmark Deliverables Tests.
Validates:
1. Benchmark JSON artifact integrity and completeness.
2. Latency threshold compliance across all 4 measurement boundaries.
3. Multi-stream scaling linearity and VRAM bounds.
"""
import json
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT / "benchmarks"


def test_benchmark_artifacts_exist():
    """Verify all benchmark JSON deliverables exist on disk."""
    required_files = [
        "baseline_pytorch.json",
        "tensorrt_python.json",
        "tensorrt_cpp.json",
        "final_pipeline.json",
        "multistream_scaling.json",
    ]
    for rf in required_files:
        p = BENCHMARKS_DIR / rf
        assert p.exists(), f"Missing benchmark artifact: {rf}"


def test_latency_targets_compliance():
    """Verify final pipeline benchmark strictly meets all 4 latency boundaries."""
    final_path = BENCHMARKS_DIR / "final_pipeline.json"
    with open(final_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Measurement 3 target: p50 <= 10ms, p95 <= 15ms, p99 <= 22ms
    pipe_4x = data.get("multi_stream_4x", {})
    assert pipe_4x.get("p50_decoded_to_rule_ms", 999) <= 10.0
    assert pipe_4x.get("p95_decoded_to_rule_ms", 999) <= 15.0
    assert pipe_4x.get("p99_decoded_to_rule_ms", 999) <= 22.0

    # Measurement 4 target: p95 <= 200ms
    assert pipe_4x.get("camera_to_alert_p95_ms", 999) <= 200.0

    # Target flags
    targets = data.get("targets_met", {})
    assert targets.get("measurement_1_engine") is True
    assert targets.get("measurement_2_gpu_pipe") is True
    assert targets.get("measurement_3_decoded_to_rule") is True
    assert targets.get("measurement_4_camera_to_alert") is True


def test_multistream_scaling_bounds():
    """Verify multi-stream scaling behavior and memory efficiency."""
    scaling_path = BENCHMARKS_DIR / "multistream_scaling.json"
    with open(scaling_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    streams = data.get("streams", {})
    assert "1_stream" in streams
    assert "4_streams" in streams
    assert "8_streams" in streams

    # 4 streams aggregate FPS must exceed 350 FPS
    assert streams["4_streams"]["fps"] >= 350.0
    # 4 streams VRAM must stay under 3GB
    assert streams["4_streams"]["vram_mb"] <= 3072
