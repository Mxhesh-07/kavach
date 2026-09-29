# KAVACH UI Component Inventory & Design System Tokens
**Document ID**: `KAVACH-SPEC-DYS-004`  
**System**: Intelligent Border Video Analytics Platform (Controlled Evaluation Build)  
**Standard Compliance**: WCAG 2.2 Level AA, ISO 9241-303 (Electronic Visual Displays), WAI-ARIA 1.2

---

## 1. Design Token Architecture

All interface elements are bound to strict CSS variables. Hard-coded HEX colors, arbitrary pixel paddings, and floating gradients are strictly prohibited.

### 1.1 Color Tokens (Dark Command Palette & High-Contrast Light Theme)

```css
:root {
  /* Surface & Canvas Hierarchy */
  --color-bg-canvas:       #0a0d12;  /* Deepest base layer */
  --color-bg-surface:      #12161f;  /* Primary panel surface */
  --color-bg-surface-raised:#181d29; /* Card & modal surface */
  --color-bg-surface-sunken:#07090d; /* Video viewport container */
  --color-bg-overlay:      rgba(10, 13, 18, 0.88);

  /* Borders & Dividers */
  --color-border-subtle:   #1f2633;  /* 1px structural grid lines */
  --color-border-default:  #2b3547;  /* Component outlines */
  --color-border-strong:   #414e68;  /* Active / focused borders */
  --color-border-focus:    #3b82f6;  /* High-visibility keyboard focus */

  /* Text & Typography */
  --color-text-primary:    #f1f5f9;  /* Titles, numbers, active labels */
  --color-text-secondary:  #94a3b8;  /* Column headers, descriptions */
  --color-text-muted:      #64748b;  /* Inactive states, metadata */
  --color-text-disabled:   #475569;  /* Deactivated buttons */
  --color-text-inverse:    #0f172a;  /* Text on bright badges */

  /* Operational Status Tokens (Color + Shape + Text Label) */
  --color-status-ok:       #10b981;  /* Normal / Healthy / Live */
  --color-status-ok-bg:    #064e3b;
  --color-status-info:     #38bdf8;  /* Routine notice / System */
  --color-status-info-bg:  #0c4a6e;
  --color-status-caution:  #fbbf24;  /* Medium warning / Acknowledged */
  --color-status-caution-bg:#78350f;
  --color-status-warning:  #f97316;  /* High threat / Zone intrusion */
  --color-status-warning-bg:#7c2d12;
  --color-status-critical: #ef4444;  /* Tripwire breach / Red alert */
  --color-status-critical-bg:#7f1d1d;
  
  /* Telemetry & Degraded Pipeline States */
  --color-state-live:      #10b981;
  --color-state-stale:     #eab308;
  --color-state-offline:   #64748b;
  --color-state-degraded:  #d97706;
  --color-state-maint:     #a855f7;
  --color-state-unknown:   #ec4899;

  /* Typography Fonts */
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: "Cascadia Mono", "Segoe UI Mono", "Liberation Mono", Menlo, Consolas, monospace;

  /* Spatial Grid (Strict 4px / 8px increments) */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;

  /* Restrained Border Radii */
  --radius-xs: 2px;
  --radius-sm: 4px;
  --radius-md: 6px;
}

/* Optional High-Contrast Light Theme */
[data-theme="high-contrast-light"] {
  --color-bg-canvas:       #ffffff;
  --color-bg-surface:      #f8fafc;
  --color-bg-surface-raised:#f1f5f9;
  --color-bg-surface-sunken:#e2e8f0;
  --color-border-subtle:   #cbd5e1;
  --color-border-default:  #94a3b8;
  --color-border-strong:   #475569;
  --color-border-focus:    #1d4ed8;
  --color-text-primary:    #0f172a;
  --color-text-secondary:  #334155;
  --color-text-muted:      #64748b;
  --color-text-disabled:   #94a3b8;
  --color-text-inverse:    #ffffff;
  --color-status-ok:       #047857;
  --color-status-ok-bg:    #d1fae5;
  --color-status-warning:  #c2410c;
  --color-status-warning-bg:#ffedd5;
  --color-status-critical: #b91c1c;
  --color-status-critical-bg:#fee2e2;
}
```

---

## 2. Typography Hierarchy

| Typographic Level | Font Family | Size | Weight | Line Height | Usage Context |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Page Title** | Sans-serif | 20px | 600 | 28px | Main Workspace View Titles |
| **Section Title** | Sans-serif | 15px | 600 | 20px | Panel Headers, Drawer Titles |
| **Sub-Header** | Sans-serif | 13px | 600 | 18px | Group Headings, Form Labels |
| **Body Primary** | Sans-serif | 13px | 400 | 18px | Descriptions, Narrative Logs |
| **Table Data / Compact**| Sans-serif (Tabular Nums) | 12px | 400 | 16px | Data Grid Rows, Attribute Lists |
| **Status Badge** | Sans-serif | 11px | 700 | 14px | Allcaps Status Labels (`CRITICAL`) |
| **Telemetry Numbers** | Monospace (Tabular Nums) | 13px | 600 | 18px | FPS, Millisecond Latency, Clock |
| **Cryptographic Hash** | Monospace | 12px | 400 | 16px | SHA-256 Hashes, Device UUIDs |

*Minimum Font Size Guard*: No essential operational text or metric may render below **12px** (9pt equivalent).

---

## 3. UI Component Specifications

### 3.1 Camera Video Tile Component (`<div class="camera-tile">`)
Every camera tile must communicate data freshness without relying on memory-leaking DOM reconstitutions:
* **Frame Container**: Fixed 16:9 aspect-ratio box on `--color-bg-surface-sunken`.
* **Tile Header Bar**:
  * Left: Camera ID (e.g., `CAM-01`) + Sector Name + Stream Source Badge (`LIVE RTSP` / `VIDEO FILE`).
  * Right: Freshness Timestamp (`19:42:15.820 IST`) + Frame Age Meter (`42 ms`).
* **Tile Overlay State Watermarks**:
  * *LIVE*: Subtle green dot + 1px default border.
  * *STALE*: High-contrast amber dashed border + persistent watermark: `[STALE FEED - 4.2s LAG]`.
  * *OFFLINE*: Muted dark slate panel + red hazard symbol + `[OFFLINE - RECONNECTING (3s)]`.
  * *DEGRADED*: Amber warning chip: `[DEGRADED: CPU FALLBACK]`.
  * *MAINTENANCE*: Violet border + `[MAINTENANCE MODE - OPERATOR #4]`.
* **Tile Footer Action Bar**:
  * Snapshot quick-button (`Ctrl+S`).
  * Pause/Play freeze control (`Space`).
  * Zoom controls (`-`, `1x`, `+`) with pan drag indicator.
  * Rules geometry toggle.

### 3.2 Status Badge Matrix (Multi-Channel Encoding)
WCAG 2.2 AA compliant — every badge combines a **distinct geometric icon**, a **bold uppercase text label**, and a **prescribed semantic background**:

```
[●] LIVE          --> Shape: Filled Circle    | Label: LIVE         | Palette: OK (Green)
[▲] WARNING       --> Shape: Triangle Warning | Label: WARNING      | Palette: Warning (Orange)
[■] CRITICAL      --> Shape: Filled Square    | Label: CRITICAL     | Palette: Critical (Red)
[◆] STALE FEED    --> Shape: Diamond Rhombus  | Label: STALE FEED   | Palette: Caution (Amber)
[✕] OFFLINE       --> Shape: Cross Symbol     | Label: OFFLINE      | Palette: Disabled (Slate)
[⚙] MAINTENANCE   --> Shape: Gear Symbol      | Label: MAINTENANCE  | Palette: Purple
```

### 3.3 Alert Queue Card Component (`<div class="alert-item">`)
* Compact table-row or card layout with fixed height (44px compact, 72px expanded).
* Keyboard Selection indicator: 2px solid `--color-border-focus` on active item.
* Layout:
  1. Severity Flag (Shape + Color + Text).
  2. Timestamp (IST with tabular numerals).
  3. Camera Sector & Track ID (e.g., `CAM-02 · TRACK #104`).
  4. Rule Description (e.g., `Perimeter Tripwire Inbound Breach`).
  5. Evidence Availability Icon (`[JPG]` clean/annotated, `[MP4]` clip).
  6. Action Buttons: `[A] Acknowledge`, `[E] Escalate`, `[R] Resolve`, `[F] False Alarm`.

### 3.4 Interactive Modal & Drawer Architecture
* Standard popups are replaced with **Right-Side Sliding Operational Drawers**:
  * Allows live camera feeds to remain 100% visible and un-obscured while an operator inspects alert evidence or camera properties.
  * Focus is trapped inside the drawer; `Esc` closes drawer and restores focus to the invoking row.
* Critical Destructive Action Dialogs:
  * Modal only used when an action is irreversible (Camera Removal, Database Purge).
  * Requires explicit reason selection from dropdown and confirmation button with 3-second hold or typed confirmation.

### 3.5 Global Header & System Status Strip
* **Header Height**: Strictly 48px fixed height.
* **Footer Status Strip Height**: Strictly 32px fixed height.
* Maximizes vertical viewport for 1080p and 1366x768 operator monitors.
