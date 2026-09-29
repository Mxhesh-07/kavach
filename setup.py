"""
Build script for military_core C++ extension module.

Compiles the TensorRT + GStreamer + CUDA inference engine as a Python extension.

Requirements:
- TensorRT 8.6+ (or 10.0+)
- CUDA Toolkit 11.8+ or 12.x
- GStreamer 1.18+ with nvv4l2decoder, nvvidconv plugins
- pybind11 2.10+
- Python 3.8+

Ubuntu 22.04/24.04:
  sudo apt install python3-dev pybind11-dev gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev

TensorRT (from NVIDIA):
  Download TensorRT tar.gz from developer.nvidia.com, extract to /opt/tensorrt

CUDA:
  Download from developer.nvidia.com/cuda-toolkit

Build:
  python3 setup.py build_ext --inplace

Install:
  pip install -e .
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import platform

# ============================================================================
# Configuration
# ============================================================================

# Base paths - override via environment variables
TENSORRT_ROOT = os.environ.get("TENSORRT_ROOT", "/opt/tensorrt")
CUDA_ROOT = os.environ.get("CUDA_ROOT", "/usr/local/cuda")
GSTREAMER_ROOT = os.environ.get("GSTREAMER_ROOT", "/usr")

# Auto-detect TensorRT version
def find_tensorrt():
    candidates = [
        "/opt/tensorrt",
        "/usr/local/tensorrt",
        "/usr/lib/x86_64-linux-gnu",
    ]
    for base in candidates:
        if os.path.exists(os.path.join(base, "lib", "libnvinfer.so")):
            return base
    return None

TENSORRT_ROOT = find_tensorrt() or TENSORRT_ROOT

# Auto-detect CUDA
def find_cuda():
    candidates = [
        "/usr/local/cuda",
        "/opt/cuda",
    ]
    for base in candidates:
        if os.path.exists(os.path.join(base, "include", "cuda_runtime.h")):
            return base
    return None

CUDA_ROOT = find_cuda() or CUDA_ROOT

# ============================================================================
# Include / Library Paths
# ============================================================================

include_dirs = [
    # pybind11
    *subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"]).decode().strip().split(),
    # Python
    *subprocess.check_output([sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip().split(),
    # TensorRT
    os.path.join(TENSORRT_ROOT, "include"),
    # CUDA
    os.path.join(CUDA_ROOT, "include"),
    # GStreamer
    os.path.join(GSTREAMER_ROOT, "include", "gstreamer-1.0"),
    os.path.join(GSTREAMER_ROOT, "include", "glib-2.0"),
    os.path.join(GSTREAMER_ROOT, "lib", "x86_64-linux-gnu", "glib-2.0", "include"),
    # Local src
    "src",
]

library_dirs = [
    os.path.join(TENSORRT_ROOT, "lib"),
    os.path.join(CUDA_ROOT, "lib64"),
    os.path.join(GSTREAMER_ROOT, "lib", "x86_64-linux-gnu"),
]

libraries = [
    "nvinfer",      # TensorRT runtime
    "nvparsers",    # TensorRT ONNX parser (if needed)
    "cudart",       # CUDA runtime
    "cuda",         # CUDA driver
    "gstreamer-1.0",
    "gobject-2.0",
    "glib-2.0",
    "gstapp-1.0",
    "gstbase-1.0",
    "gstvideo-1.0",
]

# Windows adjustments
if platform.system() == "Windows":
    library_dirs = [
        os.path.join(TENSORRT_ROOT, "lib"),
        os.path.join(CUDA_ROOT, "lib", "x64"),
    ]
    libraries = [
        "nvinfer",
        "nvparsers",
        "cudart",
        "cublas",
        "gstreamer-1.0",
        "gobject-2.0",
        "glib-2.0",
        "gstapp-1.0",
        "gstbase-1.0",
        "gstvideo-1.0",
    ]
    include_dirs = [
        d for d in include_dirs if "glib-2.0/include" not in d and "lib" not in d
    ]
    # Add MSVC-compatible GStreamer paths
    gst_win = os.environ.get("GSTREAMER_1_0_ROOT_X86_64", "C:/gstreamer/1.0/x86_64")
    include_dirs.extend([
        os.path.join(gst_win, "include", "gstreamer-1.0"),
        os.path.join(gst_win, "include", "glib-2.0"),
        os.path.join(gst_win, "lib", "glib-2.0", "include"),
    ])
    library_dirs.append(os.path.join(gst_win, "lib"))

# ============================================================================
# Extension Definition
# ============================================================================

_is_win = platform.system().lower() == "windows"
extra_compile_args = [
    "/std:c++20", "/O2", "/DNDEBUG", "/EHsc", "/W3",
] if _is_win else [
    "-std=c++20", "-O3", "-fPIC", "-DNDEBUG", "-Wall", "-Wextra",
    "-Wno-unused-parameter", "-Wno-sign-compare",
]

extra_link_args = [] if _is_win else [
    "-Wl,-rpath,$ORIGIN/../lib",
    "-Wl,-rpath," + os.path.join(TENSORRT_ROOT, "lib"),
    "-Wl,-rpath," + os.path.join(CUDA_ROOT, "lib64"),
]


# Source files
sources = [
    "src/military_core.cpp",
]

military_core_ext = Extension(
    "military_core",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
    define_macros=[
        ("VERSION_INFO", '"dev"'),
    ],
)

# ============================================================================
# Custom Build Extension
# ============================================================================

class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Verify dependencies exist
        self.verify_dependencies()
        super().build_extensions()

    def verify_dependencies(self):
        missing = []

        # Check TensorRT
        trt_lib = "nvinfer.lib" if platform.system() == "Windows" else "libnvinfer.so"
        trt_path = os.path.join(TENSORRT_ROOT, "lib", trt_lib)
        if not os.path.exists(trt_path):
            missing.append(f"TensorRT library not found at {trt_path}")

        # Check CUDA
        cuda_lib = "cudart.lib" if platform.system() == "Windows" else "libcudart.so"
        cuda_path = os.path.join(CUDA_ROOT, "lib64" if platform.system() != "Windows" else "lib/x64", cuda_lib)
        if not os.path.exists(cuda_path):
            missing.append(f"CUDA library not found at {cuda_path}")

        # Check GStreamer
        gst_lib = "gstreamer-1.0.lib" if platform.system() == "Windows" else "libgstreamer-1.0.so"
        gst_path = os.path.join(GSTREAMER_ROOT, "lib", "x86_64-linux-gnu" if platform.system() != "Windows" else "", gst_lib)
        if not os.path.exists(gst_path):
            # Try alternate
            gst_alt = os.path.join(GSTREAMER_ROOT, "lib", gst_lib)
            if not os.path.exists(gst_alt):
                missing.append(f"GStreamer library not found (tried {gst_path} and {gst_alt})")

        # Check pybind11
        try:
            import pybind11
        except ImportError:
            missing.append("pybind11 not installed (pip install pybind11)")

        if missing:
            print("WARNING: Missing dependencies:")
            for m in missing:
                print(f"  - {m}")
            print("\nSet environment variables to override paths:")
            print("  TENSORRT_ROOT=/path/to/tensorrt")
            print("  CUDA_ROOT=/path/to/cuda")
            print("  GSTREAMER_ROOT=/path/to/gstreamer")
            print("\nContinuing anyway - build may fail...")


# ============================================================================
# Setup
# ============================================================================

setup(
    name="ibvap-military-core",
    version="0.1.0",
    description="IBVAP Military-Grade C++ TensorRT Inference Engine",
    author="IBVAP Team",
    ext_modules=[military_core_ext],
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.10",
        "numpy>=1.21",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)