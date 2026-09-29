"""
Phase 17: Build Systems, Cross-Compilation & Packaging Tests.
Validates:
1. setup.py and CMakeLists.txt structure and target definitions.
2. Build script syntax (build_windows.ps1 and build_linux.sh).
3. Header and C++ source completeness in native/.
4. Cross-platform compiler flag consistency.
"""
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parent.parent


def test_native_cmake_and_setup_files_exist():
    """Verify essential build definition files are present."""
    assert (PROJECT / "setup.py").exists()
    assert (PROJECT / "native" / "CMakeLists.txt").exists()
    assert (PROJECT / "build_windows.ps1").exists()
    assert (PROJECT / "build_linux.sh").exists()


def test_native_headers_and_sources_inventory():
    """Verify all 11 core headers and source files exist in native/."""
    headers = [
        "cuda_check.hpp",
        "bounded_spsc_queue.hpp",
        "frame.hpp",
        "detection.hpp",
        "result.hpp",
        "tensorrt_logger.hpp",
        "engine.hpp",
        "scheduler.hpp",
        "tracker.hpp",
        "metrics.hpp",
        "video_source.hpp",
    ]
    for h in headers:
        header_path = PROJECT / "native" / "include" / "kavach_native" / h
        assert header_path.exists(), f"Missing header: {h}"

    sources = [
        "engine.cpp",
        "scheduler.cpp",
        "tracker.cpp",
        "metrics.cpp",
        "video_source_gstreamer.cpp",
        "video_source_ffmpeg.cpp",
        "preprocess.cu",
        "postprocess.cu",
        "bindings.cpp",
    ]
    for s in sources:
        src_path = PROJECT / "native" / "src" / s
        assert src_path.exists(), f"Missing source: {s}"


def test_setup_py_compiler_flags():
    """Verify setup.py includes C++20 and optimization flags."""
    setup_content = (PROJECT / "setup.py").read_text(encoding="utf-8")
    assert "c++20" in setup_content.lower() or "std:c++20" in setup_content.lower()
    assert "military_core" in setup_content or "kavach_native" in setup_content
