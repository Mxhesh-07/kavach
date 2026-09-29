# IBVAP Operator Workflows & Standard Operating Procedures
**Document ID**: `IBVAP-SOP-OPS-002`  
**System**: Intelligent Border Video Analytics Platform (Controlled Evaluation Build)  
**Audience**: Duty Watch Officers, Perimeter Surveillance Operators, Systems Administrators  
**Compliance Standard**: ISO 11064 (Control Centre Ergonomics), MIL-STD-1472H (Human Engineering)

---

## 1. Operational Overview & Roles

The IBVAP platform supports three standardized operational roles in field evaluation:
1. **Surveillance Operator (Watchstander)**: Real-time monitoring of live camera grids, keyboard-first alert triage, immediate perimeter intrusion acknowledgement, and alarm escalation.
2. **Duty Incident Commander (Watch Officer)**: Incident lifecycle review, multi-camera correlation, forensic video playback, disposition verification, and evidence package sealing.
3. **Systems & Security Administrator**: Camera onboarding, virtual tripwire/polygon geometry definition, NVML hardware health auditing, tamper-evident hash chain verification, and audit trail export.

---

## 2. Alert Lifecycle & State Transition Engine

Every detection or spatial rule trigger enters a deterministic, auditable state machine. Alerts **cannot be silently deleted**; every transition requires role attribution, timestamp, and justification.

```
       [ NEW ] 
          │
          ├──────────────────────────┐
          ▼                          ▼
   [ ACKNOWLEDGED ]        [ SYSTEM GENERATED ]
          │                          │
          ├──────────────────────────┤
          ▼                          ▼
   [ UNDER REVIEW ] ──────────> [ DUPLICATE ]
          │                          │
          ├───► [ ESCALATED ]        │
          │         │                │
          │         ▼                │
          │    (INCIDENT             │
          │    CREATED)              │
          │         │                │
          ▼         ▼                ▼
   [ RESOLVED / DISPOSITION ]   [ DISMISSED AS FALSE ALARM ]
          │                              │
          └──────────────┬───────────────┘
                         ▼
             [ EVIDENCE ARCHIVED ]
```

### 2.1 State Definitions

| State | Trigger / Condition | Visual Indicator | Required Operator Action |
| :--- | :--- | :--- | :--- |
| **`NEW`** | Real-time event dispatched by analytics engine | Flashing critical/caution badge with high-contrast indicator | Immediate triage within 15 seconds; acknowledge via `A` or click. |
| **`ACKNOWLEDGED`** | Operator claims initial situational awareness | Solid amber/blue badge; tone silenced | Operator reviews video snippet or camera tile. |
| **`UNDER REVIEW`** | Operator opens forensic detail or associates alert with an incident | Blue badge with active operator name tag | Operator inspects clean frame, trajectory, and OCR/face match. |
| **`ESCALATED`** | Intrusion confirmed; forwarded to Quick Reaction Team (QRT) / Duty Officer | Pulsing red double-border badge | Operator creates an Incident or links to existing operational event. |
| **`RESOLVED`** | Incident closed; sector secured by field team | Green badge with disposition text (e.g., `Apprehended`, `Secured`) | Record disposition code, notes, and closing officer ID. |
| **`DISMISSED AS FALSE ALARM`** | Legitimate non-threat (animal, debris, light reflection, authorized personnel) | Muted grey badge with strikethrough | **Mandatory reason code** (e.g., `FA-WILDLIFE`, `FA-WEATHER`, `FA-MAINTENANCE`). |
| **`DUPLICATE`** | Concurrent rule fires from single object trajectory | Grey badge linked to parent alert ID | Link to primary anchor alert ID. |
| **`EVIDENCE PENDING`** | Post-roll MP4 writer or forensic JPEG export in progress | Yellow clock badge | Automated transition upon disk flush. |
| **`EVIDENCE READY`** | HMAC-SHA256 signed evidence clip and clean frame committed to disk | Blue shield badge | Ready for export or court bundle. |
| **`EVIDENCE FAILED`** | Disk full, write error, or decoder buffer timeout | Red hazard badge with diagnostic log | Immediate notification to Systems Administrator. |

---

## 3. Keyboard-First Alert Triage Workflow

During a multi-sector intrusion, operators must never be forced to reach for a mouse. The dashboard implements a deterministic, non-interfering keyboard navigation system:

### 3.1 Global & Alert Queue Keybindings

| Keybinding | Scope | Operational Action | Safety Guard / Invariant |
| :--- | :--- | :--- | :--- |
| **`J`** / **`↓`** | Alert Queue | Select next alert down in list | Does not fire when input field has focus. |
| **`K`** / **`↑`** | Alert Queue | Select previous alert up in list | Does not fire when input field has focus. |
| **`Enter`** | Selected Alert | Open alert detail drawer / forensic inspector | Focus trapped inside drawer until dismissed. |
| **`A`** | Selected Alert | **Acknowledge Alert** (transitions to `ACKNOWLEDGED`) | Records operator session ID and timestamp. |
| **`E`** | Selected Alert | **Escalate to QRT / Incident** | Prompts for incident assignment modal. |
| **`R`** | Selected Alert | **Resolve Alert** | Opens disposition selector (default: `Normal Routine`). |
| **`F`** | Selected Alert | **Mark as False Alarm** | Opens mandatory justification dropdown. |
| **`C`** | Selected Alert | **Jump to Camera** (switches to camera focus view) | Automatically selects corresponding camera tile. |
| **`1` – `4`** | Operations | Switch grid layout (`1`=Single, `2`=2x2, `3`=3x3, `4`=Focus) | Instant layout re-arrangement. |
| **`M`** | Audio Subsystem | Master Mute / Unmute alarm siren | Visual indicator reflects armed state. |
| **`Esc`** | Any Modal / Drawer | Dismiss active dialog / drawer | Focus restored to triggering element. |

---

## 4. Multi-Camera Incident Aggregation Workflow

A single border event (e.g., vehicle approaching perimeter, cutting fence, dismounting personnel) triggers multiple alerts across adjacent sectors. The **Incident Review Workspace** aggregates these:

### 4.1 Incident Aggregation Lifecycle
1. **Incident Creation**: An operator selects one or more alerts and clicks **`Create Incident`** (or presses `E`).
2. **Unique Identifier**: System generates a collision-free ID: `INC-YYYYMMDD-SECTOR-XXXX` (e.g., `INC-20260918-NORTH-0042`).
3. **Multi-Camera Association**: Operator links adjacent camera feeds (e.g., Camera 01 North Fence + Camera 02 Riverine Approach).
4. **Synchronized Multi-View Playback**:
   * Synchronized time-scrubbing across all involved cameras.
   * Playback speeds: `0.25x`, `0.5x`, `1.0x`, `2.0x` and frame-by-frame stepping (`←` / `→`).
   * Dual-view rendering: Original clean video vs AI-annotated trajectory view.
5. **Timeline & Action Log**:
   * Every note, assignment, and status change is appended to the incident's immutable action log.
6. **Evidentiary Sealing & Export**:
   * Cryptographic compilation: Concatenates all relevant alert hashes, snapshots, clips, and audit records.
   * Exports an encrypted, HMAC-signed ZIP bundle with a human-readable PDF incident certificate.

---

## 5. Camera Health & Degradation Management

Surveillance cameras in border environments experience dirty lenses, power fluctuations, severed fiber, and radio interference. IBVAP mandates explicit operational states:

```
+---------------+--------------------------------------------------------------------------+
| State         | Visual Presentation & Operational Treatment                              |
+---------------+--------------------------------------------------------------------------+
| LIVE          | Solid neutral border. Millisecond frame age counter active (< 250 ms).   |
+---------------+--------------------------------------------------------------------------+
| STALE         | Amber dashed border. "STALE FEED" watermark. Retains last frame.         |
|               | Frame age displays in red (e.g., "+3,420 ms"). Alarm logged if > 2.0s.   |
+---------------+--------------------------------------------------------------------------+
| OFFLINE       | Slate background. Red status banner. Shows last known picture with       |
|               | disconnection duration, retry counter, and manual "Reconnect" button.    |
+---------------+--------------------------------------------------------------------------+
| DEGRADED      | Yellow warning badge: "DEGRADED - CPU FALLBACK (15 FPS)".                |
|               | Explains active fallback path (e.g., GPU thermal throttling or failover).|
+---------------+--------------------------------------------------------------------------+
| MAINTENANCE   | Purple diagonal border. Operator ID, reason, and scheduled expiry shown. |
|               | Alerts suppressed from primary siren; recorded to audit trail.           |
+---------------+--------------------------------------------------------------------------+
| UNKNOWN       | Never masked as normal. High-contrast flashing badge: "SIGNAL UNVERIFIED"|
+---------------+--------------------------------------------------------------------------+
```

---

## 6. Destructive Action Safeguards & Audit Logging

To prevent operator error or malicious repudiation:
1. **Camera Removal**:
   * Requires confirmation with reason code (`REPAIR`, `RELOCATION`, `DECOMMISSION`).
   * Never deletes associated alert logs (soft-archives camera row to preserve SHA-256 hash chain).
2. **Rule Modification**:
   * Editing tripwire coordinates records a versioned snapshot of the prior rule geometry.
3. **Database Maintenance**:
   * Hard resets and evidence sweeps require administrative role elevation and generate an unalterable audit log entry.
