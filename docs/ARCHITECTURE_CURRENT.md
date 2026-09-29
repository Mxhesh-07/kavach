# IBVAP Current Architecture Document (Phase 0 Audit)

## 1. System Overview
The Intelligent Border Video Analytics Platform (IBVAP) is an edge-native tactical video analytics system designed for border defense, perimeter security, and checkpoint monitoring. It runs on tactical edge devices (e.g., NVIDIA RTX laptops, edge workstations) to process multi-stream high-definition video feeds in real-time.

---

## 2. End-to-End Frame Processing Pipeline

The current execution flow traverses the following sequential and concurrent stages:

`
[Physical Camera / RTSP / Video File]
                 │
                 ▼
      [1. Capture & Demux / Decode]
         (LiveSource / FileSource via OpenCV / FFmpeg)
                 │
                 ▼
      [2. Frame Buffer / Latest-Frame Handoff]
         (deque maxlen=1 per camera)
                 │
                 ▼
      [3. Resize & Normalization / Enhancement]
         (cv2.resize / CLAHE / colorspace conversion)
                 │
                 ▼
      [4. Detector Submission & Model Lock]
         (Adaptive inference queue / batcher)
                 │
                 ▼
      [5. Preprocessing & Tensor Creation]
         (PyTorch tensor conversion / letterboxing)
                 │
                 ▼
      [6. Core AI Inference]
         (Ultralytics YOLO / TensorRT Engine)
                 │
                 ▼
      [7. Postprocessing & NMS]
         (Box decoding, confidence filtering, IoU suppression)
                 │
                 ▼
      [8. Multi-Object Tracking]
         (ByteTrack wrapper: Kalman filter + box smoothing)
                 │
                 ▼
      [9. Spatial Rules Analytics]
         (Fence / Tripwire, Zone Intrusion, Loiter, Direction, Night Movement)
                 │
                 ▼
      [10. Event Creation & Hash Chain Audit]
         (SHA-256 tamper-evident merkle chain)
                 │
                 ▼
      [11. Secondary Enrichment Stages (Async)]
         (Face Recognition + ANPR via _AsyncStage)
                 │
                 ▼
      [12. Evidence Packaging & Persistence]
         (Snapshot crops, video clips, SQLite database)
                 │
                 ▼
      [13. Dispatch & Live Broadcast]
         (WebSocket streaming, Push notifications, Web HUD)
`

---

## 3. Pipeline Component Specifications

| Stage | Thread / Process Owner | Locks / Synchronization | Queues / Capacity | Blocking Calls | Copies & Allocations |
|---|---|---|---|---|---|
| **Capture & Ingestion** | Background worker thread (LiveSource._capture_loop) | None on hot path | deque(maxlen=1) | cv2.VideoCapture.read() | Decoded BGR numpy array allocation |
| **Handoff** | Capture thread -> Analytics thread | Single-element exchange | Capacity: 1 frame | Non-blocking pop() / ppend() | None (pointer exchange) |
| **Preprocessing** | Analytics thread / Inference batcher | GIL / Batch lock | Batch buffer | None | cv2.resize, color conversion, normalization |
| **Inference** | Shared Detector Singleton | _lock (threading.Lock) | _queue (maxsize=4) | PyTorch CUDA kernel sync / TRT execute | Host-to-Device tensor DMA copy |
| **Postprocessing** | Detector / Analytics thread | None | None | None | Tensor-to-CPU tensor/numpy conversion |
| **Tracking** | Camera Analytics Processor | None (per-camera tracker) | None | None | Kalman state allocation, box smoothing dicts |
| **Rules Engine** | Camera Analytics Processor | None | None | None | Geometry vectorization & list comprehensions |
| **Audit Hash Chain** | Analytics thread / Event Logger | Database / Chain Lock | None | SHA-256 digest computation | String encoding & formatting |
| **Enrichment (Face/ANPR)**| Shared Background Worker (_AsyncStage) | _budget_lock | None (drop if busy) | Model inference & OCR | Frame copy for worker safety |
| **Evidence & Storage** | Background thread / Storage pool | DB connection pool | Queue capacity: 100 | Disk write / DB commit | Snapshot JPEG encoding (cv2.imencode) |
| **WebSockets & UI** | FastAPI / Starlette async event loop | Client set lock | Queue per client (maxlen=10) | Async socket send | JPEG frame encoding |

---

## 4. Concurrency & Resource Isolation
- **Per-Camera Isolation**: Each camera runs an independent analytics processing loop (CameraProcessor).
- **Shared Predictor**: All camera threads funnel into a centralized detector singleton (Detector.get()).
- **Resource Protection**: Shared enrichment stages (Face, ANPR) use a shared duty-cycle gate (_AsyncStage) to prevent CPU/GPU starvation.
- **Daemon Threads**: All background worker threads are explicitly set as daemon threads to ensure clean process termination.
