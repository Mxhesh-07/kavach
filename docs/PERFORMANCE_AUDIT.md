# IBVAP Performance Audit & Latency Breakdown (Phase 0 Audit)

## 1. Latency Measurement Boundaries

To satisfy performance and operational fidelity requirements, system timing is categorized across four distinct boundaries:

1. **Measurement 1: TensorRT Engine Only**
   - *Boundary*: context->enqueueV3() start to CUDA stream synchronization / completion event.
   - *Target*: p50 <= 3 ms, p95 <= 4 ms, p99 <= 5 ms.
   - *Current PyTorch Baseline*: ~8.5 - 12.0 ms.

2. **Measurement 2: GPU Preprocessing + TensorRT Inference + Postprocessing**
   - *Boundary*: Raw frame in pinned/GPU memory -> CUDA Preprocessing kernel -> TRT Inference -> CUDA NMS/Decode -> Compact Detection Array on Host.
   - *Target*: p50 <= 5 ms, p95 <= 7 ms, p99 <= 10 ms.
   - *Current PyTorch Baseline*: ~14.0 - 22.0 ms.

3. **Measurement 3: Decoded-Frame-to-Tracked-Rule-Result**
   - *Boundary*: Frame dequeued from camera ring buffer -> Detection -> ByteTrack association -> Spatial Rule evaluation -> Alert generation.
   - *Target*: p50 <= 10 ms, p95 <= 15 ms, p99 <= 22 ms.
   - *Current Baseline*: ~18.5 - 32.0 ms.
   - *How it is measured*: `AnalysisResult.rule_ms` is captured in `FrameAnalyzer.analyse`
     immediately after the rules (`rules.update` + `_presence_alerts`) return, *before*
     occupancy read-out, overlay annotation and publish. `CameraProcessor` copies that
     value into `latency_ms` in `_run`, so the dashboard's ingest-latency number is
     exactly this Measurement-3 boundary and no longer includes encode/clip-write time.
     (The previous build measured latency at the *end* of the tick, which folded the whole
     encode+publish+persist tail into it and read 100ms+ the moment a live camera had work
     to do.)

4. **Measurement 4: Physical Camera-to-Alert Latency**
   - *Boundary*: Physical photon exposure on RTSP sensor -> Transport -> Depay/Decode -> Full Analytics Pipeline -> Tamper-Evident Event Sealed -> WebSocket Broadcast received by C2 client.
   - *Target*: p50 <= 100 ms, p95 <= 200 ms, p99 <= 300 ms.
   - *Current Baseline*: ~120 - 280 ms (transport dependent).

---

## 2. Identified Latency Bottlenecks & Overhead Analysis

### A. Python Global Interpreter Lock (GIL) Contention
- **Observation**: When 4 or 8 cameras operate concurrently, multiple Python threads contend for the GIL during tensor allocation, image slicing, and tracker Kalman calculations.
- **Impact**: Increased jitter and p99 tail latency (up to 45+ ms under 8 streams).
- **Remediation**: Native C++20 execution engine with py::gil_scoped_release during the entire hot path.

### B. Redundant Host-to-Device Memory Transfers
- **Observation**: BGR frame is allocated on CPU, converted to RGB via CPU OpenCV, converted to Float32 Torch tensor on CPU, and then copied to GPU.
- **Impact**: 3.5 - 6.0 ms added before inference begins.
- **Remediation**: Unified CUDA preprocessing kernel directly consuming decoded surfaces in GPU memory.

### C. Sequential Python Tracking & Association
- **Observation**: ByteTrack box smoothing, velocity estimation, and track dictionary operations run sequentially per detection.
- **Impact**: 1.5 - 3.0 ms per frame when 10+ objects are in view.
- **Remediation**: Vectorized NumPy operations / Native C++ tracking.

### D. Overlay & Encoding Overhead in Analytics Path
- **Observation**: Drawing bounding boxes and JPEG encoding for WebSocket broadcast can block the analytics cadence if not properly decoupled.
- **Impact**: 4.0 - 8.0 ms per frame during active UI viewing — and, because the *measurement* of ingest latency was taken at the end of the tick, this tail was actually charged to the operator as 100ms+ ingest latency on general hardware.
- **Remediation**: Decoupled asynchronous worker queue with drop-oldest policy for non-critical preview frames.
- **Status**: DONE — `core/renderworker.py` runs all of annotate (`FrameAnalyzer.render_frame`) -> `cv2.imencode` -> `FrameBuffer.publish` -> `clips.push` on a per-camera daemon thread with a single latest-wins slot (drop-oldest backpressure, same philosophy as the decoder's maxlen=1 deque). The analytics thread submits and returns in microseconds; a synchronous fallback keeps the `_publish`-leaves-a-JPEG contract for processors that were never started or are midway through hard teardown.
