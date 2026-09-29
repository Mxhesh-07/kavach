# KAVACH Production Deployment & Operations Guide

## 1. Overview
The Intelligent Border Video Analytics Platform (KAVACH) supports three deployment paradigms:
1. **Containerized Edge Deployment (Docker Compose + NVIDIA Container Toolkit)**: Best for turnkey deployment on Linux edge appliances, ruggedized military servers, or cloud GPU instances.
2. **Bare-Metal Linux Deployment (Jetson Orin / x86_64 Ubuntu)**: Best for SWaP-constrained field devices requiring zero container overhead and direct hardware access.
3. **Development & Operator Mode (Windows 11 / Ubuntu Desktop)**: Rapid local evaluation, demonstration, and algorithm tuning using automated Python fallback and interactive console dashboards.

---

## 2. Containerized Edge Deployment (Production Stack)

### 2.1 Prerequisites
- **Host OS**: Ubuntu 22.04 LTS or 24.04 LTS (x86_64 or aarch64)
- **NVIDIA Driver**: 535.x or newer
- **Docker Engine**: 24.0+ with Docker Compose v2.20+
- **NVIDIA Container Toolkit**: Configured as default runtime

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### 2.2 Launching the Production Stack
```bash
# Build and start containerized stack with GPU reservations
docker compose -f docker-compose.prod.yml up -d --build

# Inspect running services
docker compose -f docker-compose.prod.yml ps

# Follow container logs
docker compose -f docker-compose.prod.yml logs -f kavach-edge
```

### 2.3 Container Health Probes
KAVACH includes an active health check querying `http://localhost:8000/health`:
```bash
docker inspect --format='{{json .State.Health}}' kavach-edge-node | jq
```

---

## 3. Interactive Operator Showcase & Live Performance Dashboard

KAVACH includes an interactive terminal UI demonstration tool ([`demo_showcase.py`](file:///E:/Document/Research/SIH26187/SIH26187/demo_showcase.py)) designed for command briefings, client demonstrations, and live hardware stress verification.

### 3.1 Features Demonstrated Live
1. **Multi-Camera Feeds**: Simulated concurrent surveillance across 4 border sectors (Sector-Alpha North Perimeter, Sector-Bravo Riverine, Sector-Charlie FOB Gate, Sector-Delta South Ridge).
2. **Micro-Batch Scheduler**: Dynamic batch coalescence up to batch size 4 with 2.0 ms timeout window.
3. **Dual-Engine Dispatch**: Real-time detection using TensorRT native C++20 engine with automatic PyTorch GPU fallback.
4. **Spatial Rule Evaluation**: Vectorized virtual tripwires and restricted polygon zones with bottom-center ground contact projection.
5. **Cryptographic SHA-256 Hash Chaining**: Immediate hashing and linking of every dispatched alert row.
6. **NVML GPU Hardware Telemetry**: Zero-overhead temperature, power draw (W), and VRAM utilization reporting.
7. **ThreadWatchdog Liveness**: Continuous heartbeat verification and auto-recovery.

### 3.2 Running the Showcase
```bash
# 1. Full interactive terminal UI (4 cameras, live ANSI dashboard)
python demo_showcase.py

# 2. Custom camera count and duration
python demo_showcase.py --cameras 4 --duration 30

# 3. Headless benchmarking run with exported JSON report
python demo_showcase.py --headless --duration 20 --report-json showcase_audit.json
```

---

## 4. Hardware Sizing & Scaling Matrix

| Deployment Tier | Hardware Target | Concurrent 1080p Streams | Inference Latency (p50) | Aggregate FPS | Power Consumption |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tactical Outpost** | NVIDIA Jetson Orin Nano (8GB) | 2–4 Streams | 8.2 ms | ~90 FPS | 15W |
| **Border Checkpoint** | NVIDIA GeForce RTX 3060 Ti (8GB) | 4–8 Streams | 2.1 ms | ~410 FPS | 140W |
| **Regional Sector Command** | NVIDIA RTX 4090 / L40S (24–48GB) | 16–32 Streams | 1.1 ms | ~1,200 FPS | 350W |

---

## 5. Security & Air-Gapped Operation

1. **Unprivileged Non-Root User**: The production container executes as `appuser` (UID 1001), preventing container escape vulnerabilities.
2. **Zero Cloud Dependencies**: The system operates 100% offline. Models, databases, and cryptographic validation require zero external internet connectivity.
3. **Sanitized Credentials**: Plaintext passwords and RTSP auth strings are masked in logs and exception traces.
4. **Evidence Immutability**: All recorded alerts are bound to the previous block via SHA-256 hash chains, providing tamper-evident auditing.
