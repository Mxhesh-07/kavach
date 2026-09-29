<#
.SYNOPSIS
    Build military_core C++ extension on Windows with MSVC.

.DESCRIPTION
    Compiles the TensorRT + GStreamer + CUDA inference engine as a Python extension
    using Visual Studio's MSVC compiler.

.PREREQUISITES
    - Visual Studio 2022 with "Desktop development with C++" workload
    - CUDA Toolkit 11.8+ or 12.x (installed to default C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y)
    - TensorRT 8.6+ or 10.0+ (extracted to C:\tensorrt or similar)
    - GStreamer 1.22+ for Windows (installed via MSI to C:\gstreamer\1.0\x86_64)
    - Python 3.8+ with pybind11 (pip install pybind11)
    - CMake 3.20+ (optional, for alternative build)

.USAGE
    # From Developer PowerShell for VS 2022 (x64 Native Tools Command Prompt)
    .\build_windows.ps1

    # Or with custom paths
    .\build_windows.ps1 -TensorRTPath "C:\tensorrt" -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" -GStreamerPath "C:\gstreamer\1.0\x86_64"

.OUTPUT
    military_core*.pyd in the current directory (ready for python import)
#>

param(
    [string]$TensorRTPath = "C:\tensorrt",
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    [string]$GStreamerPath = "C:\gstreamer\1.0\x86_64",
    [string]$PythonPath = (Get-Command python).Source,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Find Python include/lib
$pythonExe = $PythonPath
$pythonInclude = & $pythonExe -c "import sysconfig; print(sysconfig.get_path('include'))"
$pythonLib = & $pythonExe -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
$pybindInclude = & $pythonExe -m pybind11 --include

if (-not $pythonInclude -or -not $pythonLib) {
    Write-Error "Could not determine Python include/lib paths"
    exit 1
}

Write-Host "=== Build Configuration ===" -ForegroundColor Cyan
Write-Host "Python:        $pythonExe"
Write-Host "Python Include: $pythonInclude"
Write-Host "Python Lib:     $pythonLib"
Write-Host "pybind11:       $pybindInclude"
Write-Host "TensorRT:       $TensorRTPath"
Write-Host "CUDA:           $CudaPath"
Write-Host "GStreamer:      $GStreamerPath"
Write-Host ""

# Verify paths
$pathsToCheck = @(
    @($TensorRTPath, "include", "NvInfer.h"),
    @($TensorRTPath, "lib", "nvinfer.lib"),
    @($CudaPath, "include", "cuda_runtime.h"),
    @($CudaPath, "lib", "x64", "cudart.lib"),
    @($GStreamerPath, "include", "gstreamer-1.0", "gst", "gst.h"),
    @($GStreamerPath, "lib", "gstreamer-1.0.lib"),
)

foreach ($p in $pathsToCheck) {
    $full = Join-Path @p
    if (-not (Test-Path $full)) {
        Write-Warning "Missing: $full"
    } else {
        Write-Host "Found: $full" -ForegroundColor Green
    }
}

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force *.pyd -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force *.obj -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force *.lib -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force *.exp -ErrorAction SilentlyContinue
}

# Create build directory
New-Item -ItemType Directory -Force -Path build | Out-Null
Set-Location build

# Compiler flags
$cxxFlags = @(
    "/std:c++20",
    "/O2",
    "/DNDEBUG",
    "/EHsc",
    "/W3",
    "/MD",
    "/DVERSION_INFO=\"dev\"",
)

# Include directories
$includes = @(
    $pythonInclude,
    $pybindInclude.Trim().Split(" ") | ForEach-Object { $_.Trim('"') },
    Join-Path $TensorRTPath "include",
    Join-Path $CudaPath "include",
    Join-Path $GStreamerPath "include" "gstreamer-1.0",
    Join-Path $GStreamerPath "include" "glib-2.0",
    Join-Path $GStreamerPath "lib" "glib-2.0" "include",
    "..\src",
)

# Library directories
$libDirs = @(
    Join-Path $TensorRTPath "lib",
    Join-Path $CudaPath "lib" "x64",
    Join-Path $GStreamerPath "lib",
    $pythonLib,
)

# Libraries
$libs = @(
    "nvinfer.lib",
    "nvparsers.lib",
    "cudart.lib",
    "cublas.lib",
    "gstreamer-1.0.lib",
    "gobject-2.0.lib",
    "glib-2.0.lib",
    "gstapp-1.0.lib",
    "gstbase-1.0.lib",
    "gstvideo-1.0.lib",
    "python3.lib",
)

# Source file
$source = "..\src\military_core.cpp"
$output = "military_core.pyd"

# Build command
$clArgs = @(
    $cxxFlags,
    ($includes | ForEach-Object { "/I`"$($_)`"" }),
    $source,
    "/Fo:military_core.obj",
    "/c",
)

Write-Host "=== Compiling ===" -ForegroundColor Cyan
$compileCmd = "cl.exe " + ($clArgs -join " ")
Write-Host "Command: $compileCmd"
& cl.exe @clArgs
if ($LASTEXITCODE -ne 0) { Write-Error "Compilation failed"; exit 1 }

# Link command
$linkArgs = @(
    "military_core.obj",
    "/DLL",
    "/OUT:$output",
    "/MACHINE:X64",
    ($libDirs | ForEach-Object { "/LIBPATH:`"$($_)`"" }),
    $libs,
)

Write-Host "=== Linking ===" -ForegroundColor Cyan
$linkCmd = "link.exe " + ($linkArgs -join " ")
Write-Host "Command: $linkCmd"
& link.exe @linkArgs
if ($LASTEXITCODE -ne 0) { Write-Error "Linking failed"; exit 1 }

# Copy to root
Copy-Item $output ..\$output -Force
Write-Host ""
Write-Host "=== Build Successful ===" -ForegroundColor Green
Write-Host "Output: $output"
Write-Host "Copied to project root for import"
Write-Host ""
Write-Host "Test import:" -ForegroundColor Cyan
Write-Host "  python -c \"import military_core; print('OK')\""