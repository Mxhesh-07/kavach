# IBVAP User Interface & Human Factors Forensic Audit
**Document ID**: `IBVAP-AUD-UI-001`  
**System**: Intelligent Border Video Analytics Platform (Controlled Evaluation Build)  
**Evaluator**: Principal Product Designer, Human Factors & Frontend Architecture Team  
**Evaluation Scope**: `dashboard/index.html`, `static/css/style.css`, `static/js/app.js`, API & WebSocket Interfaces  
**Compliance Baseline**: Human Factors for Command & Control (MIL-STD-1472H / ISO 11064 / ISO 9241-210), WCAG 2.2 AA

---

## 1. Executive Summary

This forensic audit evaluates the operational readiness and human-factors ergonomics of the IBVAP WebUI. While the underlying backend pipeline (micro-batching scheduler, dynamic fallback, cryptographic hash chaining, and NVML telemetry) exhibits strong determinism and resilience, the existing operator dashboard displays severe characteristics of a **hackathon-style demonstration interface** rather than a mission-critical, control-room-ready operational tool.

Critical deficiencies identified include:
1. **Visual Noise & Arbitrary Styling**: Decorative CSS gradients, glowing radial neon box-shadows, emoji icons in system headers and feeds, and inconsistent border-radius definitions.
2. **Ambiguous Status & Telemetry Signaling**: Color-only status indicators (e.g., lone red/green/amber dots without redundant text labels or geometric symbols), failing WCAG 2.2 AA Success Criterion 1.4.1.
3. **Unmitigated Stale Video Ambiguity**: Video tiles lack explicit data-freshness timestamps, heartbeat watermarks, or clear visual degradation states when upstream decoders stall.
4. **Mouse-Bound Alert Triage**: Alert feeds lack keyboard navigation (J/K selection, Acknowledge, Escalate, Resolve shortcuts), forcing operators to rely on point-and-click actions during high-stress alerts.
5. **Lack of Incident Aggregation**: High-velocity alert streams flood the sidebar with discrete events rather than grouping correlated alerts across camera sectors into actionable incidents.
6. **Destructive Action Hazards**: Deleting camera sources or purging logs lacks explicit role safeguards, reason tracking, or confirmation invariants.

---

## 2. Granular Defect Register

### 2.1 Visual Hierarchy & Styling Deficiencies

| Defect ID | Component | Current Implementation | Human-Factors / Ergonomic Impact | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-UI-01** | Topbar Brand | `linear-gradient(135deg, var(--accent), #1c7bb5)` on `.brand-mark` with `IB` text | Consumer SaaS / hackathon styling; distracts from operational status indicators. | Medium |
| **AUD-UI-02** | Status Indicators | Glowing CSS box-shadows (`box-shadow: 0 0 8px var(--ok)`) on `.dot-ok`, `.dot-pending`, `.dot-down` | Creates blooming/glare on 24/7 dark-room displays; visually exhausts operators during 8-hour shifts. | High |
| **AUD-UI-03** | Iconography | Raw emojis (`🛡️`, `🔇`, `🔕`, `🔗`, `🎞️`, `🚨`, `⚠️`, `🚶`, `🚗`) in header, buttons, and alert cards | Emojis render inconsistently across OS platforms (Windows Segoe Color vs Linux Noto Emoji) and convey casual consumer software. | Critical |
| **AUD-UI-04** | Heading Scales | `h1` (18px), `h2` (14px), `h3` (13px) with arbitrary letter-spacing and conflicting line-heights | Inconsistent scale across view panels and modal titles; lacks typographic discipline. | Medium |
| **AUD-UI-05** | Spacing System | Arbitrary padding values (`0 18px`, `5px 11px`, `9px 14px`, `11px 14px`, `0 11px 9px`) | Violates standard 4px/8px spatial grid; produces visual misalignment across side-by-side viewports. | Medium |

### 2.2 Telemetry, Freshness & Stale-Video Ambiguity

| Defect ID | Component | Current Implementation | Human-Factors / Ergonomic Impact | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-UI-06** | Camera Tile Freshness | Tile displays static text string `fps` and `inference_ms` but does **not display frame age in milliseconds** or absolute timestamp | If an RTSP stream hangs or a decoder freezes on a frame, the operator cannot distinguish a frozen live picture from an active feed. | Critical |
| **AUD-UI-07** | Stale / Disconnected States | Tile only distinguishes `.offline` (red border). No intermediate `STALE`, `DEGRADED`, `MAINTENANCE`, or `RECONNECTING` state | Binary offline/online distinction is inadequate for intermittent wireless/tactical links. | Critical |
| **AUD-UI-08** | Statbar Information Density | Horizontal bar with 9 wide cards (`Active Cameras`, `Persons`, `Vehicles`, etc.) taking 64px vertical height | Consumes valuable vertical display space without providing historical trend lines or health context. | High |
| **AUD-UI-09** | Color-Only Communication | Critical alerts rely exclusively on red background/border without redundant iconography or textured outlines | Operators with protanopia or deuteranopia (red-green color blindness) cannot quickly identify severity. | Critical |

### 2.3 Alert Handling, Triage & Incident Aggregation

| Defect ID | Component | Current Implementation | Human-Factors / Ergonomic Impact | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-UI-10** | Alert Feed Flooding | Unbounded vertical card list (`.alert-feed`) capped at 60 DOM nodes | During a mass crossing or convoy event, dozens of individual vehicle/person alerts scroll past uncontrollably. | High |
| **AUD-UI-11** | Missing Alert Workflow | Alerts only have a modal inspection; no states for `ACKNOWLEDGED`, `UNDER_REVIEW`, `ESCALATED`, or `FALSE_ALARM` | The operator cannot triage or take operational ownership of alerts; no audit trail of who acted on what. | Critical |
| **AUD-UI-12** | Absence of Incidents | Every detection/rule fire is an isolated alert row; no grouping of multi-camera tracks into an `Incident` | Forces command staff to mentally correlate 15 alerts across 3 cameras into a single perimeter breach event. | Critical |
| **AUD-UI-13** | Zero Keyboard Shortcuts | Triage is 100% mouse driven; no hotkeys for next alert, acknowledge, escalate, or switch camera | Slows operator response latency during critical intrusion events. | Critical |

### 2.4 Ergonomics, Modals & Safety Safeguards

| Defect ID | Component | Current Implementation | Human-Factors / Ergonomic Impact | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-UI-14** | Modal Overload | Camera creation, alert detail, and virtual fence configuration all use popup modals (`.modal`) | Modals obscure live video tiles; an operator configuring a fence is blinded to perimeter alerts occurring behind the modal. | High |
| **AUD-UI-15** | Destructive Action Hazards | Camera removal uses immediate confirm dialog; no role authorization, reason input, or verification gate | High risk of accidental camera deletion or configuration loss during shift operations. | High |
| **AUD-UI-16** | Focus Management | When modals open or close, keyboard focus is not trapped or restored to the triggering element | Fails WCAG 2.2 AA (Criterion 2.4.3 Focus Order, Criterion 2.4.7 Focus Visible). | High |
| **AUD-UI-17** | Zoom & Scalability | Layout breaks or causes horizontal overflow at 200% browser zoom and 1366x768 display resolution | Fails tactical ruggedized laptop field evaluations and WCAG 1.4.4 Resize Text requirements. | High |

### 2.5 Network, DOM & Client Performance

| Defect ID | Component | Current Implementation | Operational & Technical Consequence | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-UI-18** | Polling Redundancy | Client polls `/api/system/stats` at 1 Hz and `/api/system/zones` at 5s while WebSocket `/ws/alerts` is also active | Generates redundant HTTP request bursts that compete with video snapshot transfers on low-bandwidth links. | High |
| **AUD-UI-19** | WebSocket Reconnection | Simple exponential backoff without connection status indicator or offline queue | If the link drops, alerts missed during disconnection are not automatically re-synchronized from the database. | High |
| **AUD-UI-20** | Snapshot Canvas Redraw | Canvas redraws fence coordinates on every mouse movement without dirty-rect throttling | Unnecessary CPU cycles on lower-powered operator workstations. | Medium |

---

## 3. Human Factors & Accessibility Compliance Matrix

| Requirement / Standard | Current State | Required Target State |
| :--- | :--- | :--- |
| **WCAG 2.2 1.4.1 (Use of Color)** | Non-compliant (lone color dots used for camera online/offline and alert severity) | Fully compliant (all statuses combine shape, text label, and color token) |
| **WCAG 2.2 1.4.3 (Contrast Minimum)** | Partial (some muted text tokens at `#66718a` on `#141924` fail 4.5:1 ratio) | Fully compliant (minimum 4.5:1 for body text; 3:1 for large graphical components) |
| **WCAG 2.2 2.1.1 (Keyboard Navigation)** | Non-compliant (no shortcut keys for alert triage, drawer toggle, or camera switching) | Fully compliant (complete keyboard workflow: `J`/`K` navigation, `A` ack, `E` escalate) |
| **WCAG 2.2 2.4.7 (Focus Visible)** | Partial (`:focus` styles inconsistent or overridden by `outline: none`) | Fully compliant (distinct 2px high-contrast focus ring on all interactive elements) |
| **MIL-STD-1472H (Alarms & Signals)** | Non-compliant (repetitive unacknowledged audio; no visual acknowledgment state) | Fully compliant (clear alarm states: NEW, ACKNOWLEDGED, ESCALATED, RESOLVED) |
| **1366x768 Laptop Usability** | Degraded (vertical scrollbars on statbar and camera grid) | Fully compliant (dense responsive workspace optimized for 1366x768 and 1080p) |

---

## 4. Remediation Architecture Roadmap

1. **Phase 1**: Establish unified Design System tokens (`docs/UI_COMPONENT_INVENTORY.md`) and Information Architecture (`docs/INFORMATION_ARCHITECTURE.md`).
2. **Phase 2**: Implement non-destructive Operator Workflow Engine in frontend and API (`docs/OPERATOR_WORKFLOWS.md`).
3. **Phase 3**: Refactor `dashboard/index.html` and `static/css/style.css` into a restrained, military-spec dark UI with dedicated views (Live Operations, Alert Queue, Incident Review, Site Map, Cameras, Rules, Evidence, System Health, Audit Log, Administration).
4. **Phase 4**: Upgrade `static/js/app.js` with keyboard navigation, state management, heartbeat tracking, and zero-flicker rendering.
5. **Phase 5**: Implement comprehensive automated UI and API verification test suites.
