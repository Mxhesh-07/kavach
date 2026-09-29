# KAVACH Information Architecture & Control Room Layout Specification
**Document ID**: `KAVACH-SPEC-IA-003`  
**System**: Intelligent Border Video Analytics Platform (Controlled Evaluation Build)  
**Standard Compliance**: ISO 11064-3 (Control Centre Design), MIL-STD-1472H §5.10 (Displays)

---

## 1. Top-Level Structural Layout

The KAVACH interface is structured to ensure that **video is never obscured by navigation**, critical alarms are always visible, and hardware health is monitored continuously:

```
+-------------------------------------------------------------------------------------------------------------------------+
| GLOBAL HEADER: Deployment Name | UTC/IST Clocks | Mode | Critical Count | Cams Online/Total | Storage | Operator | Shift|
+----+----------------------------------------------------------------------------------------------+---------------------+
|    | MAIN OPERATIONAL WORKSPACE (Active View)                                                     | ALERTS DRAWER       |
|    |                                                                                              | (Collapsible / F12) |
| N  |  1. Live Operations (1x1, 2x2, 3x3, 4x4, Focus+Strip)                                        |                     |
| A  |  2. Alert Queue (Multi-Filter Triage, Batch Acknowledge)                                     | Critical: 2         |
| V  |  3. Incident Review (Synchronized Multi-Cam Playback, Timeline)                              | Warning: 5          |
|    |  4. Site Map (Offline Vector Plan, Camera Cones, Live Targets)                               | Info: 12            |
| B  |  5. Cameras (Stream Onboarding, RTSP Health, Video File Loops)                               |                     |
| A  |  6. Rules (Virtual Tripwires, Zones, Loitering, Direction, Night)                            | [Ack All] [Triage]  |
| R  |  7. Evidence (Cryptographic Clips, Clean/Annotated Snaps, HMAC)                              |                     |
|    |  8. System Health (NVML Telemetry, Thermal State, Watchdogs)                                 | J/K to navigate     |
|    |  9. Audit Log (SHA-256 Tamper-Evident Hash Chain, Merkle Roots)                              | Enter to open       |
|    | 10. Administration (Air-Gap Secrets, User Roles, Sweeper Retention)                          |                     |
+----+----------------------------------------------------------------------------------------------+---------------------+
| PERSISTENT SYSTEM STATUS STRIP: Backend | Model | GPU Temp | VRAM | Infer p95 | Frame Age p95 | Storage | Time Sync OK   |
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Top-Level Functional Areas

### 2.1 Live Operations (`/operations`)
* **Primary Mission**: Real-time visual monitoring of high-risk perimeter sectors.
* **Layout Grid Options**:
  * `1x1 Focus View`: High-resolution primary feed with zoom/pan inspector.
  * `2x2 Standard Quad`: Recommended for standard 1080p control room stations (4 channels).
  * `3x3 Multi-Sector`: High-density monitoring (9 channels) with snapshot overflow throttling.
  * `Focus + Strip`: Primary selected camera displayed at 70% viewport width with a synchronized vertical thumbnail strip of remaining cameras.
* **Per-Tile Interactive Tools**:
  * Live frame age badge (`< 150 ms`).
  * Overlay toggles: Bounding boxes, ground foot contact points, trajectory vectors, rules geometry, and privacy zones.
  * Instant forensic freeze (`Pause` button) with 1x–8x digital pan-zoom.
  * Manual snapshot & 10-second retrospective evidence bookmark.

### 2.2 Alert Queue (`/alerts`)
* **Primary Mission**: High-velocity alarm triage, escalation, and verification.
* **Filter Taxonomy**:
  * Severity: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`.
  * Status: `NEW`, `ACKNOWLEDGED`, `UNDER_REVIEW`, `ESCALATED`, `RESOLVED`, `FALSE_ALARM`.
  * Spatial Origin: By Camera ID, Location, or Border Sector.
  * Rule Primitive: `Tripwire Crossing`, `Restricted Zone`, `Loitering`, `Direction of Travel`, `Night Movement`, `ANPR Match`, `Watchlist Face Match`.
* **Action Drawer**: Displays annotated snapshot, clean unannotated ground-truth frame, cropped target preview, trajectory history, and disposition selector.

### 2.3 Incident Review (`/incidents`)
* **Primary Mission**: Multi-alert post-event investigation and case collation.
* **Features**:
  * Unique Incident Identifier (`INC-YYYYMMDD-SECTOR-XXXX`).
  * Synchronized multi-camera video scrubbing across up to 4 concurrent feeds.
  * Speed scaling: `0.25x`, `0.5x`, `1.0x`, `2.0x` and single-frame step forwards/backwards.
  * Dual-stream comparison: Annotated AI render vs Clean optical evidence.
  * Immutable timeline: Chronological record of detections, operator notes, escalation triggers, and dispatch confirmations.

### 2.4 Site Map (`/map`)
* **Primary Mission**: Spatial situational awareness across wide border sectors without requiring external internet map services.
* **Features**:
  * Offline vector / raster site plan loader (supports uploaded SVG, high-res PNG, or local GeoJSON).
  * Interactive camera placement markers with dynamic FOV (field of view) coverage cones.
  * Real-time ground target foot-point projection mapped directly onto tactical grid coordinates.
  * Color-coded zone polygons matching server-side spatial rule definitions.

### 2.5 Cameras Management (`/cameras`)
* **Primary Mission**: Video stream lifecycle management, RTSP onboarding, and hardware diagnostics.
* **Features**:
  * Live camera inventory table with stream URLs (masked credentials), resolution, current FPS, analytics latency, and connection health.
  * Add Camera modal supporting live RTSP/HTTP streams, local webcams (`/dev/video0` or index `0`), and looping surveillance MP4s.
  * Maintenance mode toggle with scheduled duration and operator rationale.

### 2.6 Spatial Rules Engine (`/rules`)
* **Primary Mission**: Interactive definition of mathematical boundaries for border surveillance.
* **Features**:
  * Interactive canvas calibrated to native analytics pixel dimensions (e.g., 640x384).
  * Virtual Tripwires: Bidirectional or unidirectional line segments with custom traversal arrows.
  * Restricted Zones: Arbitrary convex and non-convex polygons with hysteresis buffer margins.
  * Loiter Zones: Polygon boundaries with configurable dwell-time thresholds (e.g., 30 seconds) and exit excursion grace windows.

### 2.7 Evidence Vault (`/evidence`)
* **Primary Mission**: Chain-of-custody archive for forensic investigation and review boards.
* **Features**:
  * Browsable repository of pre-roll/post-roll MP4 clips and paired high-res clean/annotated JPEGs.
  * Direct HMAC-SHA256 cryptographic signature display.
  * Exportable evidence bundle builder: Compiles selected clips, snapshots, and tamper-evident audit logs into a signed, password-protected archive.

### 2.8 System Health & Edge Telemetry (`/health`)
* **Primary Mission**: Hardware survivability, thermal protection, and watchdog supervision.
* **Telemetry Panels**:
  * NVML GPU vitals: Core temperature (°C), fan speed (%), power wattage (W), VRAM allocation (MB), and GPU core load (%).
  * Dynamic Thermal Governor state: `NORMAL` (1.0x FPS), `WARM` (0.75x), `HOT` (0.50x), `CRITICAL` (0.25x).
  * Pipeline Hardware Watchdog: Thread heartbeats, stall counters, and auto-restart activity.
  * Video Decoder Metrics: Dropped frames, ingestion latency, and jitter buffers.

### 2.9 Audit Log & Tamper Verification (`/audit`)
* **Primary Mission**: Evidentiary verification of log immutability.
* **Features**:
  * Full inspection of the SHA-256 hash-chained event table.
  * Interactive "Verify Log Integrity" tool that recomputes all link hashes from genesis (`0`*64).
  * Periodic Merkle Checkpoints table with root hashes, start/end alert IDs, and timestamp seals.
  * Cryptographic certificate export generating verifiable JSON and printable PDF audit sheets.

### 2.10 System Administration (`/admin`)
* **Primary Mission**: Air-gapped operational configuration, storage limits, and retention policies.
* **Features**:
  * Evidence retention sweep controls (max disk usage GB, max retention days).
  * Outbound notification endpoints (Relay dry-contacts, Webhook endpoints, local SMS gateways).
  * Model selection and confidence calibration overrides.

---

## 3. Global Headers & Persistent Status Strip Contracts

### 3.1 Global Header Metadata Elements
1. **Deployment Identity**: Sector / Post name (e.g., `BOP-04 SOUTH PERIMETER`).
2. **Synchronized Clocks**: Real-time IST (Indian Standard Time, UTC+05:30) alongside standard UTC clock.
3. **Operational Mode**: `LIVE OPERATIONAL` (Green) or `EVALUATION TEST MODE` (Amber).
4. **Critical Alert Pill**: Real-time counter of unacknowledged high/critical alerts with audio mute toggle.
5. **Camera Fleet Summary**: `X/Y Online` (e.g., `4/4 Active`).
6. **Storage Capacity**: Visual meter of evidence disk space remaining (e.g., `142.4 GB / 82% Free`).
7. **Operator Session**: Active watchstander ID and shift duration timer.

### 3.2 Persistent Status Strip Elements (Fixed Bottom Bar)
1. **Analytics Engine**: `TENSORRT C++20 (FP16)` or `PYTORCH CUDA (FALLBACK)`.
2. **Active Neural Network**: `YOLO11s (640x384)`.
3. **GPU Core Temperature**: e.g., `57.0°C` (Color-coded: Green < 70°C, Amber 70–82°C, Red > 83°C).
4. **GPU Power Consumption**: e.g., `27.5 W` (Live NVML readout).
5. **VRAM Footprint**: e.g., `1,420 MB / 8,192 MB` (17.3%).
6. **Inference Latency (p95)**: e.g., `2.85 ms`.
7. **Frame Freshness Age (p95)**: e.g., `14.2 ms`.
8. **Watchdog Liveness**: `ALL WORKERS HEALTHY (100% ACK)`.
9. **Hash Chain State**: `SEALED (CHAIN TIP: 1cb8a8fd...)`.
