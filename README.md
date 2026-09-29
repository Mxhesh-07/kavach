# KAVACH — Intelligent Border Video Analytics Platform

**Smart India Hackathon 2026 · Problem Statement 26187 · Ministry of Home Affairs / Sashastra Seema Bal**

KAVACH is an ultra-low latency, mission-critical video analytics platform designed to transform standard perimeter CCTV, thermal, and aerial camera feeds into an AI-powered border surveillance system. Engineered for high-throughput edge deployment, the platform combines a native **C++20 / CUDA / TensorRT** critical data plane with a robust Python micro-service orchestration architecture.

The platform provides deterministic sub-3ms inference dispatch, sub-10ms decoded-to-rule evaluation, multi-camera micro-batching, spatial vector geometry rules, facial recognition, automatic number plate recognition (ANPR), and cryptographically sealed SHA-256 tamper-evident event logging — operating entirely on commodity edge compute with zero dependency on proprietary smart cameras or closed appliances.

Any frame source is supported: RTSP / RTSPS (TLS 1.3) feeds, USB webcams, mobile IP cameras, YouTube live streams, and pre-recorded surveillance video files.

---

## Contents

- [Key Capabilities](#key-capabilities)
- [Technology Stack](#technology-stack)
- [Architecture & Concurrency Model](#architecture--concurrency-model)
- [Performance & Benchmark Verification](#performance--benchmark-verification)
- [Installation & Setup for First-Timers](#installation--setup-for-first-timers)
- [Quick Start & Everyday CLI Commands](#quick-start--everyday-cli-commands)
- [Verification & Automated Test Suites](#verification--automated-test-suites)
- [(Optional) Native C++20 / CUDA Engine Compilation](#optional-native-c20--cuda-engine-compilation)
- [Analytics & Vectorized Rule Engine](#analytics--vectorized-rule-engine)
- [Evidence Security & Cryptographic Integrity](#evidence-security--cryptographic-integrity)
- [Edge Telemetry & Hardware Watchdogs](#edge-telemetry--hardware-watchdogs)
- [Configuration Reference](#configuration-reference)
- [REST API & WebSocket Surface](#rest-api--websocket-surface)
- [Comprehensive Test Suite](#comprehensive-test-suite)
- [Deployment & Air-Gapped Operation](#deployment--air-gapped-operation)
- [Project Structure](#project-structure)
- [License](#license)

---

## Key Capabilities

### 1. Detection & Tracking Data Plane
- **Dual-Engine Architecture**: Hardware-accelerated TensorRT 8.6+/10.x native C++20 engine with automatic, zero-downtime fallback to PyTorch / Ultralytics.
- **Micro-Batching Inference Scheduler**: Centralized micro-batching (dynamic batch sizes 1 to 4 with 3.0 ms timeout) coordinating multi-camera feeds with round-robin fairness.
- **Multi-Object Tracking**: Vectorized ByteTracker with independent 8-state Kalman filters and two-stage IoU Hungarian association per camera feed.
- **Track Identity Continuity**: Eight tracked surveillance classes (person, bicycle, car, motorcycle, bus, train, truck, boat) maintaining stable trajectories across occlusions.
- **Low-Light & Night Enhancement**: Automatic CLAHE histogram equalization on low-luma and infrared frames.

### 2. Vectorized Spatial Rule Engine
| Rule Type | Geometric Primitive | Operational Trigger Condition |
|---|---|---|
| **Tripwire (Virtual Fence)** | Line Segment | Directional crossing (entry / exit) based on trajectory-segment intersection |
| **Restricted Zone** | Arbitrary Polygon | Vectorized point-in-polygon entry, sustained presence, and exit |
| **Loitering Zone** | Arbitrary Polygon | Cumulative dwell time exceeding threshold with boundary hysteresis |
| **Direction of Travel** | Directed Vector | Velocity vector angular divergence against permitted direction |
| **Night-Time Movement** | Global Scene | Sustained trajectory displacement under low-illumination scene conditions |

### 3. Biometrics & Optical Character Recognition
- **Facial Recognition**: SCRFD face localisation with ArcFace deep embeddings; cosine distance matching against an encrypted watchlist.
- **ANPR**: Plate bounding box localization, EasyOCR text extraction, Indian vehicle registration syntax validation, and temporal multi-frame consensus voting.

### 4. Operator Interface & WebUI Stream
- **Decoupled Operator Ring Buffer**: 15 FPS preview pipeline completely isolated from the 60 FPS critical analytics path, eliminating UI rendering lag.
- **Interactive Spatial Tooling**: Real-time browser-based polygon and tripwire drawing directly onto live feeds.
- **Forensic Playback Controls**: Frame-accurate pause, slow-motion review (0.5x to 2.0x), digital pan/zoom, and continuous zone occupancy displays.

### 5. Evidence Protection & Tamper Evidence
- **Cryptographic Hash Chaining**: Every alert row is bound to its predecessor via SHA-256 (`chain_hash`), preventing retrospective row modification or deletion.
- **Merkle Checkpointing**: Periodic Merkle root checkpoints sealed in SQLite WAL mode for third-party integrity verification.
- **HMAC Digital Signatures**: Pre-roll contextual MP4 evidence clips cryptographically signed via HMAC-SHA256.

---

## Technology Stack

| Domain | Layer / Tool | Technology & Version | Role |
|---|---|---|---|
| **Accelerated Native Path** | Native Engine | C++20 / CUDA 12.x / TensorRT 8.6+ / 10.x | Ultra-low latency GPU preprocessing, inference, and postprocessing |
| | Parallel Kernels | Custom CUDA Kernels (`preprocess.cu`, `postprocess.cu`) | Fused bilinear resize, BGR->RGB, normalization, and parallel NMS |
| | Concurrency | Lock-Free `BoundedSPSCQueue` (Cacheline padded 64B) | Zero-copy pinned memory frame ingestion without lock contention |
| | Interop | Pybind11 (with `py::gil_scoped_release`) | Python bindings with zero Global Interpreter Lock (GIL) stalls |
| **Python Orchestration** | Language Runtime | Python 3.11+ / 3.14 compatible | Camera lifecycle, rule coordinator, API, and database management |
| | Fallback Detection | PyTorch 2.x / Ultralytics YOLO11 | Transparent zero-configuration CPU/CUDA fallback detector |
| | Tracking & Numerics | NumPy 2.x / SciPy | Kalman filter state estimation and vectorized spatial geometry |
| | Web Framework | FastAPI / Uvicorn ASGI | High-concurrency REST endpoints and live WebSocket telemetry |
| | Database & Storage | SQLite (WAL Mode) / SQLAlchemy 2.0 | Append-only hash-chained alert logging and evidence indexing |
| | Hardware Telemetry | NVIDIA NVML (`pynvml`) | Real-time GPU temperature, power draw (W), and clock monitoring |
| | Build & Packaging | CMake 3.20+ / MSVC 2022 / GCC 11+ | Multi-platform build tooling for Windows and Linux |

---

## Architecture & Concurrency Model

```
+----------------------------------------------------------------------------------------------------+
|                                    KAVACH Unified Data Plane                                        |
|                                                                                                    |
|   [Camera Stream 1..8] (RTSP / USB / File)                                                         |
|             |                                                                                      |
|             v  (Hardware Ingestion & FFmpeg / GStreamer Decoders)                                  |
|   [Bounded SPSC Queue (Capacity=2)]  -->  [Pinned Host Memory Zero-Copy Buffer]                    |
|             |                                                                                      |
|             v  (Non-blocking CUDA Stream 0)                                                        |
|   [Fused Preprocess CUDA Kernel (preprocess.cu)]                                                   |
|   - Bilinear Resize + BGR->RGB + 1/255.0f Normalization + HWC->NCHW Transposition                 |
|             |                                                                                      |
|             v                                                                                      |
|   [Central Inference Scheduler (core/scheduler.py & scheduler.cpp)]                                |
|   - Dynamic micro-batching (1..4 batch, 3ms timeout window)                                       |
|   - Per-camera job supersession (stale frame elimination) & round-robin fairness                  |
|             |                                                                                      |
|             v                                                                                      |
|   [TensorRT 8.6+/10.x Engine (engine.cpp)]                                                         |
|   - FP16 Tensor Cores, IExecutionContext::enqueueV3, zero steady-state heap allocations           |
|             |                                                                                      |
|             v                                                                                      |
|   [Fused Postprocess & Parallel NMS CUDA Kernel (postprocess.cu)]                                  |
|   - Fast candidate box decode + ArgMax class score + Bitmask Parallel NMS                         |
|             |                                                                                      |
|             v                                                                                      |
|   [C++20 Vectorized Multi-Camera ByteTracker (tracker.cpp)]                                        |
|   - 8-State Kalman Filter motion prediction + 2-Stage IoU Hungarian Association                    |
|             |                                                                                      |
|             v                                                                                      |
|   [Spatial Rule Engine (cv/rules.py)]                                                              |
|   - Vectorized Ray-Casting Polygon Containment & Segment Intersection Tripwires                   |
|             |                                                                                      |
|             +----------------------------+----------------------------+                            |
|             |                            |                            |                            |
|             v                            v                            v                            |
|   [Immediate Event Alert]      [SQLite WAL Async DB]        [Decoupled Preview (15 FPS)]           |
|   - WebSocket / Push notify     - Append-only Hashchain      - Ring-buffer WebUI stream            |
|   - Evidence Clip HMAC sign     - SHA-256 Merkle Checkpoint  - Zero analytics blocking             |
+----------------------------------------------------------------------------------------------------+
```

### Concurrency Guarantees
1. **Latest-Frame Ingestion (`deque(maxlen=1)`)**: The capture thread never waits for inference, and analytics never stalls the decoder. If the GPU falls behind, frames are dropped at ingestion rather than buffered, keeping end-to-end latency flat at 0.0 ms backlog.
2. **Daemon Thread Invariants**: Every thread spawned in the system is explicitly configured with `daemon=True`, preventing orphaned processes from retaining GPU contexts or camera device handles upon process shutdown.
3. **Connection Pooling Safety**: All database interactions use strict `try ... finally: session.close()` blocks, eliminating connection leaks under burst alert loads.

---

## Performance & Benchmark Verification

Empirically validated on an **NVIDIA GeForce RTX 3060 Ti (8GB VRAM, SM 8.6)** running at 1080p60:

### 1. Latency Across Four Rigorous Boundaries
| Measurement Boundary | Target Budget | Measured Latency (TensorRT FP16) | Operational Outcome |
| :--- | :--- | :--- | :--- |
| **Measurement 1: Engine Inference Only** | p50 $\le$ 3.0 ms, p95 $\le$ 4.0 ms | **p50 = 2.12 ms, p95 = 2.85 ms** | Sub-3ms edge compute |
| **Measurement 2: GPU Pipeline (Pre + Infer + Post)** | p50 $\le$ 5.0 ms, p95 $\le$ 7.0 ms | **p50 = 3.48 ms, p95 = 4.62 ms** | End-to-end GPU acceleration |
| **Measurement 3: Decoded Frame to Rule Complete** | p50 $\le$ 10.0 ms, p95 $\le$ 15.0 ms | **p50 = 6.84 ms, p95 = 9.15 ms** | Sub-10ms alert determination |
| **Measurement 4: Camera Capture to Alert Dispatched** | p50 $\le$ 100 ms, p95 $\le$ 200 ms | **p50 = 38.2 ms, p95 = 64.7 ms** | Real-time C2 escalation |

### 2. Multi-Stream Scalability (1080p60 RTSP Feeds)
| Stream Count | Aggregate Throughput | Decoded-to-Rule p95 Latency | GPU VRAM Allocated | GPU Core Utilization |
| :--- | :--- | :--- | :--- | :--- |
| **1 Stream** | 285.0 FPS | 9.15 ms | 1,120 MB | 28% |
| **2 Streams** | 348.0 FPS | 9.32 ms | 1,240 MB | 46% |
| **4 Streams** | **412.8 FPS** | **9.15 ms** | **1,420 MB** | **68%** |
| **8 Streams** | 480.0 FPS | 12.10 ms | 1,850 MB | 89% |

---

## Installation & Setup for First-Timers

### Prerequisites
- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04/24.04 LTS, or macOS.
- **Python**: Version 3.11 to 3.14 (Python 3.11 or 3.12 recommended for broadest binary wheel compatibility).
- **Git**: Installed and available in PATH.
- **Hardware Acceleration**:
  - *With NVIDIA GPU*: RTX / GTX / Quadro / Jetson Orin with CUDA 11.8+ or 12.x for hardware acceleration.
  - *Without GPU (CPU Fallback)*: Standard x86_64 or ARM64 processor. The platform automatically runs in CPU fallback mode with zero extra configuration.

---

### Step-by-Step Setup Guide

#### 1. Clone the Repository
```bash
git clone https://github.com/the-srirup/SIH26187.git
cd SIH26187
```

#### 2. Create and Activate a Virtual Environment
*On Windows (PowerShell):*
```powershell
python -m venv .venv
# If PowerShell blocks script execution, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

*On Linux / macOS (Bash):*
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> **Model Weights**: You do NOT need to manually download weights. On your first run, Ultralytics will automatically download `yolo11s.pt` (19.3 MB) into the project root if it is not already present.

#### 4. Initialize Database & Seed Demo Data
Initialize the SQLite database schema and seed the bundled demo border camera with a virtual fence tripwire rule:
```bash
python manage.py init
python manage.py seed
```

#### 5. Launch the Platform
```bash
python manage.py run
```
The server will boot on `http://127.0.0.1:8000`. Open your browser to access:
- **Operator Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive REST API Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **System Health Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## Quick Start & Everyday CLI Commands

All management actions can be performed via the WebUI dashboard or through `manage.py`:

### Managing Cameras
```bash
# List all registered cameras
python manage.py cameras

# Register a local USB webcam (camera index 0)
python manage.py camera-add --name "Webcam-Gate" --url 0 --location "Main Checkpoint"

# Register an IP / RTSP security camera
python manage.py camera-add --name "North-Fence" --url "rtsp://admin:pass@192.168.1.100:554/stream"

# Register a pre-recorded surveillance video file
python manage.py camera-add --name "Sector-4" --url "samples/sample_border_scenario.mp4"

# Remove a camera by ID
python manage.py camera-rm --id 2
```

### Managing Rules & Virtual Fences
```bash
# List all active rules for camera 1
python manage.py rules --camera 1

# Add a virtual fence tripwire crossing line
python manage.py rule-add --camera 1 --type line --geometry "[[20,180],[620,180]]"

# Add a polygon restricted zone
python manage.py rule-add --camera 1 --type zone --geometry "[[100,100],[300,100],[300,300],[100,300]]"

# Remove a rule by ID
python manage.py rule-rm --id 3
```

### Forensic Auditing & Security Integrity
```bash
# Inspect recent alerts and incidents
python manage.py alerts --limit 20

# Cryptographically verify the tamper-evident SHA-256 hash chain
python manage.py integrity

# Seal a Merkle root checkpoint for external audit
python manage.py checkpoint

# View database and storage disk statistics
python manage.py stats
```

---

## Verification & Automated Test Suites

Verify that your local environment is 100% operational:

```bash
# 1. Run the entire automated test suite (446+ unit and integration tests)
pytest

# 2. Run the capability audit against sample video clips
python verify_capabilities.py --seconds 15

# 3. Run full HTTP/WebSocket end-to-end system acceptance
python verify_system.py
```

---

## (Optional) Native C++20 / CUDA Engine Compilation

By default, KAVACH operates out of the box with the PyTorch/CUDA detector (`cv/detector.py`). For edge servers processing 4–8 concurrent 1080p60 RTSP streams at maximum throughput (400+ FPS), you can optionally compile the native C++20 / CUDA TensorRT engine:

### Prerequisites for Native Compilation
- NVIDIA CUDA Toolkit 12.x or 11.8 (`nvcc` on PATH)
- NVIDIA TensorRT 8.6+ or 10.x SDK
- C++20 Compiler: Visual Studio 2022 (Windows) or GCC 11+ (Linux)

### Build Commands
```powershell
# On Windows (PowerShell with VS 2022 Build Tools):
.\build_windows.ps1
```

```bash
# On Linux (Ubuntu 22.04 / NVIDIA Jetson):
chmod +x build_linux.sh
./build_linux.sh
```

```bash
# Direct Pybind11 extension compilation:
python setup.py build_ext --inplace
```

---

## Analytics & Vectorized Rule Engine

Every rule in KAVACH is deterministic, geometric, and decoupled from AI inference:

```
[Detection BBoxes] -> [Ground Foot Point Projection] -> [Spatial Rule Engine] -> [Debounce & Hysteresis] -> [Sealed Alert]
```

- **Ground Foot Point Projection**: Spatial evaluations project bounding boxes to bottom-center ground coordinates:
  $$P_{\text{foot}} = \left(\frac{x_1 + x_2}{2}, y_2\right)$$
  ensuring fence crossings and restricted zones reflect actual ground contact rather than bounding-box perspective overlap.
- **Segment Trajectory Intersection**: Crossings detect genuine geometric line segment intersection between frame $t-1$ and $t$:
  $$(P_t - P_{t-1}) \times (S_2 - S_1) \neq 0$$
  completely eliminating false alerts caused by infinite line extensions.
- **Multi-Frame Debouncing**: Event generation requires configurable persistence (`ANCHOR_CONFIRMATION_FRAMES`), rejecting transient detection flicker.

---

## Evidence Security & Cryptographic Integrity

KAVACH implements an immutable audit log for all security events:

1. **SHA-256 Hash Chain**:
   $$H_i = 	ext{SHA-256}\left(	ext{canonical\_json}(	ext{Alert}_i) \parallel H_{i-1}ight)$$
   Starting from genesis hash $H_0 = 0^{64}$, every alert row is mathematically bound to its predecessor.
2. **Audit Verification**:
   Running `GET /api/integrity/verify` checks every row in `alerts.db`. If any record was modified, deleted, or inserted out of sequence, the exact tampering index is reported.
3. **Exportable Integrity Certificate**:
   Generates a cryptographically verifiable JSON certificate covering Merkle root checkpoints and historical alert integrity for forensic presentation.

---

## Edge Telemetry & Hardware Watchdogs

Designed for harsh, unattended edge deployments:

- **Thermal Governor ([`core/thermal.py`](core/thermal.py))**:
  Monitors GPU temperature, power (Watts), and utilization via NVML:
  - $\le 75^\circ	ext{C}$: `NORMAL` (1.0x FPS multiplier)
  - $\ge 83^\circ	ext{C}$: `THROTTLED` (0.75x adaptive frame rate pacing)
  - $\ge 88^\circ	ext{C}$: `CRITICAL` (0.50x emergency thermal pacing)
- **Pipeline Watchdog ([`core/watchdog.py`](core/watchdog.py))**:
  Continuous 10s heartbeat monitor tracking camera capture threads, inference queues, and database writers. Automatically restarts stalled decoder threads and gracefully degrades to CPU fallback if GPU context loss occurs.

---

## Comprehensive Test Suite

The repository contains **446 unit, integration, regression, and stress tests** covering 100% of pipeline invariants:

```bash
pytest tests/
```

### Test Suite Inventory:
| Test Suite | Focus Area & Invariants Verified |
|---|---|
| `test_native_pipeline.py` | Native backend routing, scheduler micro-batching, preview decoupling |
| `test_numerical_equivalence.py` | IoU $\ge 0.99$ equivalence between PyTorch and native engine |
| `test_pipeline_stress.py` | 8 concurrent cameras under 60 FPS burst load, thread safety |
| `test_thermal_telemetry.py` | NVML telemetry, multi-stage throttling, hysteresis recovery |
| `test_watchdog_resilience.py` | Worker heartbeat timeouts, auto-restart hooks, CPU degradation |
| `test_build_packaging.py` | C++20 header/source presence, setup.py flags, cross-platform build |
| `test_benchmarks.py` | All 4 latency boundaries compliance, multi-stream scaling bounds |
| `test_evidence_security.py` | SHA-256 hash chain tampering detection, HMAC signatures, URL sanitization |
| `test_scheduler.py` | Micro-batching queues, 3ms timeouts, latest-job supersession |
| `test_stale_frames.py` | Ingestion `deque(maxlen=1)` queue lag elimination |
| `test_resources.py` | Daemon thread compliance, database session `finally` closure |
| `test_rules.py` | Polygon containment, tripwire line crossing, loitering dwell time |
| `test_alert_policy.py` | Severity grading (CRITICAL, HIGH, MEDIUM, LOW, INFO) |
| `test_api.py` | REST endpoints, WebSocket telemetry, validation errors |

---

## Project Structure

```
SIH26187/
├── native/                           # C++20 / CUDA Native Inference Engine
│   ├── CMakeLists.txt                # Standalone native build configuration
│   ├── include/kavach_native/         # C++20 headers
│   │   ├── bounded_spsc_queue.hpp    # Cacheline-padded lock-free SPSC ring buffer
│   │   ├── cuda_check.hpp            # CUDA runtime error validation macros
│   │   ├── engine.hpp                # TensorRT 8.6+/10.x enqueueV3 execution wrapper
│   │   ├── frame.hpp                 # Zero-copy pinned memory frame buffers
│   │   ├── detection.hpp             # High-density bounding box representations
│   │   ├── scheduler.hpp             # Micro-batching inference scheduler
│   │   ├── tracker.hpp               # Vectorized ByteTracker with Kalman Filter
│   │   ├── metrics.hpp               # Nanosecond telemetry collector
│   │   ├── video_source.hpp          # Native video source abstraction
│   │   └── tensorrt_logger.hpp       # Severity-filtered TRT logger
│   ├── src/                          # C++ and CUDA implementations
│   │   ├── preprocess.cu             # Fused bilinear resize + BGR->RGB + norm kernel
│   │   ├── postprocess.cu            # Fused bbox decode + parallel NMS kernel
│   │   ├── engine.cpp                # Engine deserialization & tensor binding
│   │   ├── scheduler.cpp             # Multi-camera micro-batch coordinator
│   │   ├── tracker.cpp               # Multi-camera Kalman filter & Hungarian matching
│   │   ├── bindings.cpp              # Pybind11 bridge with GIL release
│   │   ├── video_source_ffmpeg.cpp   # Dedicated FFmpeg ingestion worker
│   │   └── video_source_gstreamer.cpp# Hardware GStreamer ingestion worker
│   └── tests/                        # C++ unit & numerical equivalence tests
├── core/                             # Core Python Infrastructure
│   ├── backend.py                    # Dynamic backend router (TensorRT vs PyTorch)
│   ├── scheduler.py                  # Python micro-batching inference scheduler
│   ├── timing.py                     # Nanosecond FrameTiming telemetry collector
│   ├── thermal.py                    # NVML GPU thermal & power adaptive governor
│   ├── watchdog.py                   # Pipeline heartbeat & resilience watchdog
│   ├── preview.py                    # Decoupled 15 FPS operator preview ring buffer
│   ├── trt_manifest.py               # Engine hardware signature & manifest validator
│   ├── camera.py                     # CameraProcessor, CameraManager, FrameBuffer
│   ├── video_source.py               # Credential sanitization & maxlen=1 ingest
│   ├── analytics.py                  # FrameAnalyzer with nanosecond telemetry
│   ├── hashchain.py                  # SHA-256 tamper-evident audit log & Merkle roots
│   ├── evidence.py                   # Contextual MP4 clip & snapshot retention
│   ├── database.py                   # SQLite engine, connection pooling, WAL mode
│   └── config.py                     # Application configuration & thresholds
├── cv/                               # Computer Vision & Spatial Algorithms
│   ├── detector.py                   # YOLO11 detector & fallback coordinator
│   ├── rules.py                      # Vectorized polygon & tripwire spatial rules
│   ├── geometry.py                   # Ray-casting, winding number, line intersection
│   ├── face.py                       # SCRFD face detection & ArcFace embedding
│   ├── anpr.py                       # Number plate localization & OCR voting
│   └── overlay.py                    # Visual overlay & heads-up display rendering
├── api/                              # Web Interface & APIs
│   ├── main.py                       # FastAPI application & WebSocket handlers
│   └── schemas.py                    # Pydantic request & response models
├── benchmarks/                       # Benchmark Deliverables & Telemetry
│   ├── baseline_pytorch.json         # PyTorch baseline measurements
│   ├── tensorrt_cpp.json             # Native C++20 / TensorRT measurements
│   ├── multistream_scaling.json      # 1, 2, 4, 8 stream scalability benchmarks
│   ├── final_pipeline.json           # Validated 4-boundary latency matrix
│   └── baseline_gpu_telemetry.csv    # Hardware GPU telemetry trace
├── docs/                             # Engineering Documentation
│   ├── ARCHITECTURE_TARGET.md        # Complete target production architecture
│   ├── PERFORMANCE_RESULTS.md        # Detailed benchmark & latency matrix
│   ├── MODEL_SELECTION_REPORT.md     # YOLO11s vs YOLO26 evaluation report
│   ├── SECURITY_MODEL.md             # Security, TLS 1.3, & audit safeguards
│   ├── DEFECT_REGISTER.md            # DEF-001 through DEF-008 remediations
│   ├── KNOWN_LIMITATIONS.md          # Hardware portability & environmental limits
│   ├── MIGRATION.md                  # Compilation, serialization, & deployment guide
│   └── ROLLBACK.md                   # Zero-downtime fallback procedures
├── tests/                            # Comprehensive Automated Test Suite (446 tests)
├── dashboard/                        # WebUI Operator Interface
├── setup.py                          # Cross-platform C++20 / CUDA build script
├── build_windows.ps1                 # Windows automated build script
├── build_linux.sh                    # Linux automated build script
├── Dockerfile                        # Container specification
├── docker-compose.yml                # Multi-container orchestration
└── manage.py                         # Unified command-line interface
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
