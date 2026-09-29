#!/bin/bash
#
# Build military_core C++ extension on Linux (Ubuntu 22.04/24.04)
#
# Prerequisites:
#   sudo apt update && sudo apt install -y \
#       python3-dev python3-venv \
#       build-essential cmake pkg-config \
#       libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
#       libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-bad \
#       gstreamer1.0-libav \
#       pybind11-dev
#
# TensorRT: Download from NVIDIA, extract to /opt/tensorrt
# CUDA: Download from NVIDIA, install to /usr/local/cuda
#
# Usage:
#   ./build_linux.sh
#   ./build_linux.sh --clean
#   TENSORRT_ROOT=/custom/tensorrt ./build_linux.sh

set -euo pipefail

# Configuration (override via environment)
TENSORRT_ROOT="${TENSORRT_ROOT:-/opt/tensorrt}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
PYTHON_EXE="${PYTHON_EXE:-python3}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== Build Configuration ===${NC}"
echo "Python:        $($PYTHON_EXE --version)"
echo "TensorRT:      $TENSORRT_ROOT"
echo "CUDA:          $CUDA_ROOT"
echo ""

# Verify dependencies
check_path() {
    local path="$1"
    local desc="$2"
    if [[ -e "$path" ]]; then
        echo -e "${GREEN}✓${NC} $desc: $path"
        return 0
    else
        echo -e "${RED}✗${NC} $desc NOT FOUND: $path"
        return 1
    fi
}

MISSING=0

# Check TensorRT
check_path "$TENSORRT_ROOT/include/NvInfer.h" "TensorRT headers" || MISSING=1
check_path "$TENSORRT_ROOT/lib/libnvinfer.so" "TensorRT library" || MISSING=1

# Check CUDA
check_path "$CUDA_ROOT/include/cuda_runtime.h" "CUDA headers" || MISSING=1
check_path "$CUDA_ROOT/lib64/libcudart.so" "CUDA library" || MISSING=1

# Check GStreamer
check_path "/usr/include/gstreamer-1.0/gst/gst.h" "GStreamer headers" || MISSING=1
check_path "/usr/lib/x86_64-linux-gnu/libgstreamer-1.0.so" "GStreamer library" || MISSING=1

# Check pybind11
if $PYTHON_EXE -c "import pybind11" 2>/dev/null; then
    PYBIND_INCLUDE=$($PYTHON_EXE -m pybind11 --include)
    echo -e "${GREEN}✓${NC} pybind11: $PYBIND_INCLUDE"
else
    echo -e "${RED}✗${NC} pybind11 not installed (pip install pybind11)"
    MISSING=1
fi

# Check Python dev headers
PYTHON_INCLUDE=$($PYTHON_EXE -c "import sysconfig; print(sysconfig.get_path('include'))")
check_path "$PYTHON_INCLUDE/Python.h" "Python headers" || MISSING=1

if [[ $MISSING -eq 1 ]]; then
    echo -e "\n${RED}Missing dependencies. Install them or set environment variables:${NC}"
    echo "  TENSORRT_ROOT=/path/to/tensorrt"
    echo "  CUDA_ROOT=/path/to/cuda"
    echo ""
    echo "TensorRT: https://developer.nvidia.com/tensorrt"
    echo "CUDA:     https://developer.nvidia.com/cuda-toolkit"
    echo "GStreamer: sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \\"
    echo "                  libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-libav"
    exit 1
fi

# Clean if requested
if [[ "${1:-}" == "--clean" ]]; then
    echo -e "\n${YELLOW}Cleaning build artifacts...${NC}"
    rm -rf build *.so *.pyd
fi

# Create build directory
mkdir -p build
cd build

# Compiler flags
CXXFLAGS=(
    -std=c++20
    -O3
    -fPIC
    -DNDEBUG
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-sign-compare
    -DVERSION_INFO=\"dev\"
)

# Include directories
INCLUDES=(
    -I"$PYTHON_INCLUDE"
    $($PYTHON_EXE -m pybind11 --include | tr ' ' '\n' | sed 's/^/-I/')
    -I"$TENSORRT_ROOT/include"
    -I"$CUDA_ROOT/include"
    -I/usr/include/gstreamer-1.0
    -I/usr/include/glib-2.0
    -I/usr/lib/x86_64-linux-gnu/glib-2.0/include
    -I../src
)

# Library directories
LIBDIRS=(
    -L"$TENSORRT_ROOT/lib"
    -L"$CUDA_ROOT/lib64"
    -L/usr/lib/x86_64-linux-gnu
)

# Libraries
LIBS=(
    -lnvinfer
    -lnvparsers
    -lcudart
    -lcuda
    -lgstreamer-1.0
    -lgobject-2.0
    -lglib-2.0
    -lgstapp-1.0
    -lgstbase-1.0
    -lgstvideo-1.0
)

# Source
SOURCE="../src/military_core.cpp"

# Determine output name (platform-specific extension)
EXT_SUFFIX=$($PYTHON_EXE -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="military_core$EXT_SUFFIX"

echo -e "\n${CYAN}=== Compiling ===${NC}"
CMD=(g++ "${CXXFLAGS[@]}" "${INCLUDES[@]}" "$SOURCE" -c -o military_core.o)
echo "${CMD[*]}"
"${CMD[@]}"

echo -e "\n${CYAN}=== Linking ===${NC}"
CMD=(g++ -shared -o "$OUTPUT" military_core.o "${LIBDIRS[@]}" "${LIBS[@]}"
    -Wl,-rpath,"$TENSORRT_ROOT/lib"
    -Wl,-rpath,"$CUDA_ROOT/lib64"
    -Wl,-rpath,'$ORIGIN/../lib')
echo "${CMD[*]}"
"${CMD[@]}"

# Copy to project root
cp "$OUTPUT" ../
echo -e "\n${GREEN}=== Build Successful ===${NC}"
echo "Output: $OUTPUT"
echo "Copied to project root for import"
echo ""
echo -e "${CYAN}Test import:${NC}"
echo "  python3 -c \"import military_core; print('OK')\""