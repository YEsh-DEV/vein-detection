# 📘 Technical Architecture & Engineering Deep-Dive Report
## Edge-Native Sub-Dermal Palm Vein Biometrics & Contactless Payment System
**System Architecture Specification & Technical Reference Manual**  
**Version:** 3.0.0 (Production Master)  
**Target Audience:** Planning Agents, Principal Systems Architects, Core CV/ML Engineers, Senior Full-Stack Developers  

---

## 📑 Table of Contents
1. [Executive Summary & System Genesis](#1-executive-summary--system-genesis)
2. [End-to-End System Topology](#2-end-to-end-system-topology)
3. [Computer Vision & Biometric Pipeline (Deep Dive)](#3-computer-vision--biometric-pipeline-deep-dive)
   - 3.1 Anatomical Principle & Sensor Physics
   - 3.2 Camera Abstraction & Thread-Safe Hardware Locking
   - 3.3 Landmark Detection & Knuckle Valley Geometry
   - 3.4 Canonical ROI Normalization (Ma et al. 2017)
   - 3.5 Vessel Enhancement via CLAHE & Bilateral Filtering
   - 3.6 2D Gabor Wavelet Feature Extraction (VeinCode Phase Binarization)
   - 3.7 Shift-Tolerant Modified Normalized Hamming Distance (MNHD)
4. [Search Engine & In-Memory Matching Architecture](#4-search-engine--in-memory-matching-architecture)
   - 4.1 Layer 1: 16-Float Euclidean Spatial Filter (RAM)
   - 4.2 Layer 2: 4-Core Parallel Multiprocessing Matcher
   - 4.3 Multi-Sample Aggregation & Decision Boundary
5. [Storage Layer & SQLite Data Vault](#5-storage-layer--sqlite-data-vault)
   - 5.1 Schema Specification & BLOB Compression
   - 5.2 Thread-Safe Database Access Manager
6. [Backend API Architecture (FastAPI)](#6-backend-api-architecture-fastapi)
   - 6.1 Concurrency & Async Threadpool Dispatch
   - 6.2 Dual-Stream MJPEG vs Still Capture Switching
   - 6.3 Timeout Guards & Session Bleed Protection
   - 6.4 Comprehensive Endpoint Catalog & Data Contracts
7. [Frontend Architecture (Granular Deep Dive)](#7-frontend-architecture-granular-deep-dive)
   - 7.1 Modern Neobrutalism Design System & Styling Tokens
   - 7.2 Global State Tree & Hook Lifecycle Management
   - 7.3 Component Tree & Layout Shell
   - 7.4 Page-by-Page Architectural Breakdown
   - 7.5 Modal Portals & Notification Systems
   - 7.6 Edge Case Resilience & Network Fault Handling
8. [Edge Diagnostic & Dataset Collection Tools](#8-edge-diagnostic--dataset-collection-tools)
   - 8.1 `cam_test.py`: Interactive Terminal Diagnostic
   - 8.2 `collect_samples.py`: ML Dataset Harvesting Engine
9. [Raspberry Pi 5 Deployment & Verification Checklist](#9-raspberry-pi-5-deployment--verification-checklist)

---

## 1. Executive Summary & Core Concept

### 1.1 Project Mission
The **Palm Vein Biometrics & Contactless Payment System** is an edge-native biometric identification platform engineered specifically for embedded single-board computers (Raspberry Pi 5) equipped with Near-Infrared (NIR 850nm) NoIR camera modules. 

Unlike conventional optical fingerprints, 2D facial recognition, or iris scans, palm vein biometrics operates in the sub-dermal vascular domain. Deoxygenated hemoglobin flowing through the venous network of the human palm strongly absorbs near-infrared radiation between 760nm and 850nm. When illuminated by an NIR cluster, subcutaneous veins appear as dark, high-contrast branching silhouettes beneath the skin surface.

### 1.2 Core Architectural Advantages
1. **Liveness-Inherent Anti-Spoofing:** Vein patterns can only be imaged in living tissue containing active deoxygenated hemoglobin. Dead hands, silicone molds, 3D resin prints, or high-resolution photographic prints cannot reflect or absorb NIR radiation identically to human blood flow.
2. **Contactless Hygiene & POS Usability:** Users place their hand 10–15cm above the camera aperture without physical contact. This eliminates latent print residue, optical smudge degradation, and viral cross-contamination in public POS environments.
3. **High Discriminative Entropy:** The vascular structure of the human palm is completely formed during embryonic gestation and remains anatomically stable throughout adulthood. Even identical twins possess totally distinct, uncorrelated palm vascular trees.
4. **Sub-Second Edge Execution:** The platform executes complete landmarking, affine normalization, Gabor wavelets, and candidate search entirely on the Raspberry Pi 5 without external cloud dependencies or GPU acceleration.

---

## 2. End-to-End System Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PHYSICAL HARDWARE LAYER                                │
│   Raspberry Pi 5 (Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4GHz + 8GB LPDDR4X RAM)    │
│   ├── Pi NoIR Camera v2 / OV5647 / IMX219 / IMX708 via CSI Cable (CFE / ISP device)   │
│   └── 850nm / 940nm High-Power Infrared LED Ring Illuminator Array                     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              CAMERA DRIVER SUBSYSTEM                                  │
│   Thread-Safe Lock: _camera_lock = threading.Lock()                                    │
│   ├── Mode A: Preview Stream (640x480 RGB888, 30 FPS, Fixed Shutter 5000µs, Gain 1.0)  │
│   └── Mode B: Still Capture  (1640x1232 RGB888 High-Res Capture Mode)                  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             BIOMETRIC CV PIPELINE                                      │
│  [1] MediaPipe HandLandmarker (Float16) ──► 21 3D Joint Landmarks                      │
│  [2] Knuckle Valley Derivation         ──► Pv1 (Index-Middle) & Pv2 (Ring-Pinky)       │
│  [3] Canonical Affine Normalization    ──► 256x256 Scaled Palm ROI (Ma et al. 2017)    │
│  [4] Sub-dermal Vessel Enhancement     ──► Bilateral Smoothing + CLAHE (16x16 Grid)   │
│  [5] 2D Gabor Wavelet Filter Bank      ──► Structure Tensor -> VR (Real) & VI (Imag)   │
│  [6] Spatial Fingerprint Extraction    ──► 16-Float Euclidean Signature Vector         │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                            2-LAYER BIOMETRIC SEARCH ENGINE                             │
│  Layer 1: Fast RAM Euclidean Pre-Filter ──► Filters thousands down to Top 80 in <1ms  │
│  Layer 2: 4-Core Parallel MNHD Matcher ──► Shift-Tolerant [-8..+8] Hamming Distance   │
│  Decision Engine: Score = 0.7 * Min + 0.3 * Mean <= Threshold (0.3800)                 │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                            FASTAPI ASYNC HTTP SERVER (8000)                            │
│  ├── /api/video_feed (MJPEG Streaming) ──► Low-latency multipart stream with HUD box   │
│  ├── /api/scan       (Scan Endpoint)   ──► 15s CV timeout + 10s matcher timeout        │
│  ├── /api/enroll/*   (Enrollment)      ──► Multi-sample cache + cancellation hook      │
│  └── SQLite Database (palm_vein.db)    ──► Indexed, zlib-compressed VeinCode storage   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       NEOBRUTALISM REACT 18 WEB APPLICATION                            │
│  ├── Framework: React 18 + TypeScript + Vite + Tailwind CSS                            │
│  ├── Architecture: Mobile-first simulated smartphone terminal (480px width)           │
│  ├── Features: 5s Auto-Countdown, Real-Time Feedback, Confetti, Diagnostics Matrix     │
│  └── Pages: Landing (Hero), Scan (POS), Enroll (Studio), Friends (Users), Stats (Admin)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Computer Vision & Biometric Pipeline (Deep Dive)

### 3.1 Anatomical Principle & Sensor Physics
Human skin is composed of three primary layers: the epidermis, dermis, and subcutaneous hypodermis. Deoxygenated blood flowing back to the cardiopulmonary system through the superficial palmar venous arches contains concentrated deoxygenated hemoglobin ($Hb$). 

While visible light (400–700nm) is largely scattered by the melanin and epidermis, near-infrared radiation in the optical window of **760nm to 850nm** penetrates epidermal layers up to 3–5mm deep. Deoxygenated hemoglobin exhibits a sharp optical extinction peak at 850nm. When NIR light illuminates the palm, light penetrating tissue around the veins is scattered back toward the sensor, while light striking the veins is absorbed by hemoglobin. Consequently, the camera sensor registers veins as low-intensity (dark) silhouettes against bright back-scattered palmar tissue.

### 3.2 Camera Abstraction & Thread-Safe Hardware Locking
On the Raspberry Pi 5, the CSI camera interface is driven by `libcamera` through the Broadcom PiSP (Raspberry Pi Image Signal Processor) pipeline. Direct concurrent access to the camera by both a live video streaming thread and a still capture routine results in hardware lockup, missing V4L2 buffers, or pipeline crashes.

To solve this, `server.py` and `cam_test.py` implement an explicit threading mutual exclusion lock:

```python
_camera_lock = threading.Lock()
```

#### Dual-Mode Picamera2 Operation
```python
# Hardware Preview Mode (Continuous Display):
preview_cfg = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

# Hardware Still Capture Mode (High Resolution):
still_cfg = picam2.create_still_configuration(
    main={"size": (1640, 1232), "format": "RGB888"}
)

# Manual Exposure and Gain Enforcement:
picam2.set_controls({
    "AeEnable": False,       # Disable auto-exposure fluctuation
    "ExposureTime": 5000,    # 5000 microseconds (5ms fixed shutter)
    "AnalogueGain": 1.0,     # Unity analog gain (eliminates sensor noise)
})
```

When a still frame is requested by `/api/scan` or `/api/enroll/sample`, `capture_frame_gray()` acquires `_camera_lock`, dynamically switches the ISP pipeline from `preview_cfg` to `still_cfg`, captures a crisp 1640×1232 array, restores `preview_cfg`, and releases the lock.

---

### 3.3 Landmark Detection & Knuckle Valley Geometry
Traditional palmprint algorithms used boundary contour tracing and convexity defect analysis. Under variable lighting or slight finger angling, convexity defects jump chaotically between skin folds.

This system replaces contour heuristics with Google's **MediaPipe HandLandmarker** (`hand_landmarker.task`, Float16 runtime model). MediaPipe outputs 21 anatomical 3D coordinates.

```
                    Finger Tip Landmarks
                       (4, 8, 12, 16, 20)
                             │
                             ▼
                    ┌───┬───┬───┬───┐
                    │   │   │   │   │
                    │   │   │   │   │
         Index MCP  [5] │   [9] │  [13] │  [17]  Pinky MCP
                    └───┴───┴───┴───┘
                       ▲       ▲
                      Pv1     Pv2
                       │       │
      Pv1 = Mid(5, 9) ─┘       └─ Pv2 = Mid(13, 17)
```

The system computes two immutable physiological reference points:
1. **$Pv_1$ (Index–Middle Valley Anchor):**
   $$\mathbf{Pv_1} = \frac{\mathbf{L}_5 + \mathbf{L}_9}{2}$$
2. **$Pv_2$ (Ring–Pinky Valley Anchor):**
   $$\mathbf{Pv_2} = \frac{\mathbf{L}_{13} + \mathbf{L}_{17}}{2}$$

These metacarpophalangeal (MCP) joints are anchored to skeletal bone structures and remain immune to skin elasticity or fingernail artifacts.

---

### 3.4 Canonical ROI Normalization (Ma et al. 2017)
Using anchors $Pv_1$ and $Pv_2$, the system establishes a canonical, scale-invariant, rotation-invariant coordinate frame based on the landmark paper by *Ma et al. (2017)*:

```
                  Pv1 ─────────────── Pv2
                   \        │        /
                    \   midpoint    /
                     \      │      /
                      \     ▼ d0  /
                       ┌─────────┐
                       │         │  L (Side = 1.5 * dist_pv)
                       │   ROI   │
                       │ (256px) │
                       └─────────┘
```

1. **Inter-Valley Vector:**
   $$\Delta x = Pv_2.x - Pv_1.x, \quad \Delta y = Pv_2.y - Pv_1.y$$
2. **Euclidean Inter-Valley Distance ($D_{pv}$):**
   $$D_{pv} = \sqrt{\Delta x^2 + \Delta y^2}$$
3. **Hand Incline Angle ($\theta$):**
   $$\theta = \arctan2(\Delta y, \Delta x)$$
4. **Midpoint Vector:**
   $$\mathbf{M} = \left( \frac{Pv_1.x + Pv_2.x}{2}, \frac{Pv_1.y + Pv_2.y}{2} \right)$$
5. **Affine Rotation Matrix ($M_{rot}$):**
   The entire grayscale frame is rotated around $\mathbf{M}$ by $-\theta$ using bilinear interpolation:
   $$M_{rot} = \text{cv2.getRotationMatrix2D}(\mathbf{M}, \theta, 1.0)$$
6. **Distance Transform Orientation Check:**
   A binary silhouette of the hand is generated via Otsu thresholding and morphological closing. A Euclidean distance transform (`cv2.DIST_L2`) identifies the centroid of maximal tissue mass, establishing the positive direction toward the palm center ($direction \in \{+1, -1\}$).
7. **Bounding Box Formulation:**
   $$L = \text{round}(1.5 \times D_{pv})$$
   $$d_0 = \text{round}(0.35 \times D_{pv})$$
   $$x_1 = M_x - \frac{L}{2}, \quad x_2 = M_x + \frac{L}{2}$$
   $$y_1 = M_y + d_0, \quad y_2 = y_1 + L \quad (\text{if } direction > 0)$$
8. **Canonical Resampling:**
   The bounded patch is cropped and resized using cubic interpolation to exactly **$256 \times 256$ pixels**.

---

### 3.5 Vessel Enhancement via CLAHE & Bilateral Filtering
Raw subcutaneous infrared imagery suffers from low contrast and skin scattering attenuation. To bring out vascular ridges:
1. **MinMax Dynamic Range Stretching:** Expands pixel intensities to span the full dynamic range $[0, 255]$.
2. **Edge-Preserving Bilateral Smoothing:**
   $$\text{Filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i) - I(x)\|) g_s(\|x_i - x\|)$$
   Configured with diameter $d=7$, $\sigma_{color}=35$, $\sigma_{space}=35$. This eliminates camera sensor shot noise while keeping sharp vein boundary gradients intact.
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE):**
   CLAHE divides the $256 \times 256$ patch into a $16 \times 16$ grid of contextual tiles (each $16 \times 16$ pixels). A clip limit of $2.5$ prevents noise over-amplification in uniform areas. The resulting enhanced image highlights micro-vascular branching channels.

---

### 3.6 2D Gabor Wavelet Feature Extraction (VeinCode Phase Binarization)
The enhanced $256 \times 256$ ROI is tessellated into non-overlapping **$32 \times 32$ pixel blocks** (64 blocks total). For each block, an adaptive 2D Gabor wavelet is synthesized based on local tissue properties.

#### A. Dominant Orientation Estimation via Structure Tensor
For each $32 \times 32$ block, spatial gradient vectors are derived using $3 \times 3$ Sobel operators:
$$J_{xx} = G_{\sigma=1} * (I_x^2), \quad J_{yy} = G_{\sigma=1} * (I_y^2), \quad J_{xy} = G_{\sigma=1} * (I_x \cdot I_y)$$
The dominant orientation angle $\theta$ is:
$$\theta = \frac{1}{2} \arctan2(2 \overline{J_{xy}}, \overline{J_{xx}} - \overline{J_{yy}})$$
The computed $\theta$ is discretized to the closest orientation in $[0^\circ, 30^\circ, 60^\circ, 90^\circ, 120^\circ, 150^\circ]$.

#### B. Scale & Radial Frequency Mapping
The standard deviation of pixel intensities $D = \text{std}(block)$ determines the scale parameter $\sigma$:
$$
\sigma = \begin{cases} 
1.0 & D \le 0.05 \\
\sqrt{2} & 0.05 < D \le 0.10 \\
2\sqrt{2} & 0.10 < D \le 0.18 \\
4\sqrt{2} & D > 0.18 
\end{cases}
$$
The carrier center frequency $\mu$ is selected proportionally:
$$\mu = \frac{\text{raw\_table}[\sigma]}{32}$$

#### C. DC-Free 2D Gabor Kernel Generation
The spatial filter is defined as:
$$G(x, y; \sigma, \mu, \theta) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{x^2+y^2}{2\sigma^2} \right) \exp\left( j 2\pi\mu(x\cos\theta + y\sin\theta) \right)$$
The real ($G_R$) and imaginary ($G_I$) kernels are made DC-free by subtracting their respective spatial means:
$$G_R(x, y) = \text{Re}(G) - \overline{\text{Re}(G)}, \quad G_I(x, y) = \text{Im}(G) - \overline{\text{Im}(G)}$$

#### D. Fast FFT Convolution & Phase Binarization
Each block is extracted with symmetric boundary padding ($pad = 7$). Fast Fourier Transform convolution (`scipy.signal.fftconvolve`) computes the response:
$$C_R = I_{patch} * G_R, \quad C_I = I_{patch} * G_I$$
Phase responses are binarized into two boolean bit-planes ($256 \times 256$ bits each):
$$V_R(x, y) = \begin{cases} 1 & C_R(x, y) \ge 0 \\ 0 & C_R(x, y) < 0 \end{cases}$$
$$V_I(x, y) = \begin{cases} 1 & C_I(x, y) \ge 0 \\ 0 & C_I(x, y) < 0 \end{cases} \quad (\text{if } \mu > 0, \text{ else } 0)$$

---

### 3.7 Shift-Tolerant Modified Normalized Hamming Distance (MNHD)
Due to minor hand placement jitter between sessions, probe VeinCodes exhibit small linear translations relative to enrolled templates. The system compensates using spatial shift-tolerant Modified Normalized Hamming Distance.

For a horizontal displacement $s \in [-8, +8]$ and vertical displacement $t \in [-8, +8]$ (289 displacement evaluations):
$$\text{NHD}(P, Q, s, t) = \frac{\sum_{(x,y)} \left( P_R(x-s, y-t) \oplus Q_R(x, y) \right) + \sum_{(x,y)} \left( P_I(x-s, y-t) \oplus Q_I(x, y) \right)}{2 \times H_{\text{overlap}} \times W_{\text{overlap}}}$$
The minimal distance across all shifts defines the match score:
$$\text{MNHD}(P, Q) = \min_{s \in [-8, 8], t \in [-8, 8]} \text{NHD}(P, Q, s, t)$$

* A score of **$0.00$** indicates an identical match.
* Scores below **$0.3800$** indicate verified authentic match.
* Scores around **$0.50$** indicate completely uncorrelated impostor hands or white noise.

---

## 4. Search Engine & In-Memory Matching Architecture

Exhaustive bit-level MNHD calculation takes approximately $2\text{ms}$ per template on a Raspberry Pi 5 core. To scale efficiently across thousands of biometric templates, `search_engine.py` implements a **Hierarchical Two-Layer Search Engine**.

```
[ Incoming Probe VeinCode ]
           │
           ▼
[ Extract 16-Float Signature ]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  LAYER 1: RAM Vector Pre-Filter                        │
│  L2 Euclidean Distance against all DB Signatures       │
│  - Executes in ~0.5ms across 5,000 templates in RAM   │
│  - Filters database down to Top-K (K <= 80 candidates) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ Top Candidates
┌────────────────────────────────────────────────────────┐
│  LAYER 2: Parallel Multi-Core MNHD Engine              │
│  - Dispatches templates across 4 Raspberry Pi 5 Cores │
│  - Computes exact Shift-Tolerant Hamming Distance      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Multi-Sample User Aggregation                         │
│  Score = 0.7 * min(scores) + 0.3 * mean(scores)        │
│  Threshold: Best Score <= 0.3800                       │
└────────────────────────────────────────────────────────┘
```

### 4.1 Layer 1: 16-Float Euclidean Spatial Filter (RAM)
During enrollment, each $256 \times 256$ binary VeinCode matrix $V_R$ is downsampled into a compact spatial feature vector:
1. Reshaped into an $8 \times 8$ grid of $32 \times 32$ blocks.
2. Grouped into $4 \times 4$ quadrants (each $2 \times 2$ blocks).
3. Means computed across blocks produce a **16-element float32 signature** $\mathbf{S} \in \mathbb{R}^{16}$.

During boot, `SearchEngine.refresh_cache()` preloads all signatures from SQLite into a contiguous in-memory NumPy matrix $\mathbf{M}_{sig} \in \mathbb{R}^{N \times 16}$.
When an unknown hand is scanned, the Euclidean distance vector is calculated in one vectorized BLAS call:
$$\mathbf{D} = \|\mathbf{M}_{sig} - \mathbf{S}_{probe}\|_2$$
Only candidates with $\|\mathbf{D}\|_2 < 0.25$ (capped at `TOP_K = 80`) proceed to Layer 2.

### 4.2 Layer 2: 4-Core Parallel Multiprocessing Matcher
The filtered candidate templates are loaded from disk/cache and distributed across a persistent worker pool (`multiprocessing.Pool(processes=4)`). Each core executes `match_templates()`, searching the $[-8, +8]$ displacement space in parallel.

### 4.3 Multi-Sample Aggregation & Decision Boundary
Users enroll multiple samples (between 3 and 6) to capture natural hand posture variation. For a candidate user with $k$ enrolled templates, Layer 2 produces scores $[s_1, s_2, \dots, s_k]$. 

The aggregate identity score is weighted to reward the closest matching posture while penalizing noisy outliers:
$$S_{\text{user}} = 0.7 \times \min(s_1, \dots, s_k) + 0.3 \times \frac{1}{k}\sum_{i=1}^k s_i$$
If $S_{\text{user}} \le \text{MATCH\_THRESHOLD}$ ($0.3800$), identity is confirmed.

---

## 5. Storage Layer & SQLite Data Vault

The storage layer is encapsulated entirely in `db_manager.py`. No other module connects directly to SQLite.

### 5.1 Schema Specification & BLOB Compression
Raw $256 \times 256$ VeinCode bit-matrices contain $65,536$ bytes each. Storing uncompressed raw bitmaps consumes significant disk space and I/O bandwidth. The system compresses $V_R$ and $V_I$ using standard `zlib.compress()`:

* **Uncompressed Bit-Plane:** 65,536 bytes
* **Zlib Compressed BLOB:** ~1,100 to 1,400 bytes (over **98% compression ratio**)
* **16-Float Signature:** 64 bytes (`np.float32.tobytes()`)

```sql
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT UNIQUE NOT NULL COLLATE NOCASE,
    enrolled_at  TEXT DEFAULT (datetime('now')),
    active       INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS templates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    sample_idx   INTEGER NOT NULL DEFAULT 0,
    vr_blob      BLOB NOT NULL,
    vi_blob      BLOB NOT NULL,
    signature    BLOB NOT NULL,
    vr_mean      REAL,
    vi_mean      REAL,
    enrolled_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, sample_idx)
);

CREATE TABLE IF NOT EXISTS access_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER,
    score        REAL NOT NULL,
    accepted     INTEGER NOT NULL,
    scan_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_templates_user ON templates(user_id);
CREATE INDEX IF NOT EXISTS idx_users_active   ON users(active, username);
```

---

## 6. Backend API Architecture (FastAPI)

### 6.1 Concurrency & Async Threadpool Dispatch
FastAPI routes run on an asynchronous event loop (`async def`). However, OpenCV transformations, Gabor filtering, and SQLite queries are CPU-bound blocking operations. Running them directly inside async route handlers would block the event loop, freezing HTTP responses and MJPEG streams.

Every blocking call is offloaded to FastAPI's background threadpool via `run_in_threadpool`:

```python
clahe_roi, probe_code = await run_in_threadpool(process_image, gray)
username, score = await run_in_threadpool(engine.identify, probe_code)
```

### 6.2 Dual-Stream MJPEG vs Still Capture Switching
* **Live Display (`/api/video_feed`):** An infinite generator yields boundary frames at ~25 FPS with targeting reticles overlaid outside the camera lock.
* **Biometric Snap (`/api/scan`, `/api/enroll/sample`):** Temporarily acquires `_camera_lock`, flips the camera hardware into 1640×1232 still mode, grabs a raw uncompressed frame, returns to preview mode, and releases the lock.

### 6.3 Timeout Guards & Session Bleed Protection
On an edge device, difficult hand placement or low infrared illumination can cause computer vision detectors to stall. `server.py` guards all critical pathways with `asyncio.wait_for`:

1. **Pipeline Timeout:** 15.0 seconds maximum for MediaPipe landmarking + Gabor extraction.
2. **Search Engine Timeout:** 10.0 seconds maximum for parallel MNHD matching.
3. **Session Bleed Prevention:** An in-memory dictionary `enrollment_cache = {}` stores uncommitted samples. If an error occurs, or if the user navigates away from the Enroll page, `/api/enroll/cancel` purges the cache using `enrollment_cache.pop(uname, None)`.

### 6.4 Comprehensive Endpoint Catalog

#### Core System & Telemetry Endpoints
* `GET /health`: Health probe returning `{"ok": True}`.
* `GET /api/status`: System status, camera driver name, enrolled user count, total template count, and matching threshold.
* `GET /api/video_feed`: Multipart MJPEG continuous video stream.

#### Biometric Authentication & Verification
* `POST /api/scan`:
  * Captures still frame under hardware lock.
  * Extracts 256×256 CLAHE ROI and VeinCode.
  * Dispatches search across enrolled database.
  * Logs transaction to `access_log` table.
  * Saves audit capture PNGs to `captures/` and `roi_clahe/`.
  * Returns JSON matching `ScanResponse`:
    ```json
    {
      "accepted": true,
      "username": "alice",
      "score": 0.2841,
      "threshold": 0.3800,
      "time_ms": 342,
      "clahe_base64": "iVBORw0KGgoAAA..."
    }
    ```

#### Enrollment Lifecycle
* `POST /api/enroll/sample`:
  * Validates username regex (`^[a-z0-9][a-z0-9_-]{1,29}$`).
  * Rejects duplicate registered usernames with HTTP 409.
  * Limits active enrollment sessions to a maximum of 6 samples.
  * Extracts Gabor VeinCode and appends it to `enrollment_cache`.
  * Returns JSON matching `SampleResponse` with base64 thumbnail.
* `POST /api/enroll/save`:
  * Enforces minimum requirement of 3 valid samples.
  * Commits samples to SQLite table `templates`.
  * Triggers `SearchEngine.refresh_cache()` to update RAM index.
  * Cleans up `enrollment_cache` entry.
* `POST /api/enroll/cancel`:
  * Takes `{ "username": "alice" }` and discards uncommitted samples from RAM.

#### Diagnostics & Administration
* `GET /api/users`: Returns list of all active enrolled users and sample counts.
* `DELETE /api/users/{username}`: Soft-deletes user and updates search index.
* `DELETE /api/database/reset`: Wipes all enrolled biometric records.
* `GET /api/report`: Computes intra-user self-matches and inter-user cross-matches for separation matrix analytics.

---

## 7. Frontend Architecture (Granular Deep Dive)

### 7.1 Modern Neobrutalism Design System & Styling Tokens
The frontend is built with a **Neobrutalism** aesthetic. It avoids generic muted flat designs in favor of high-contrast, tactile, retro-futuristic styling:

```
┌────────────────────────────────────────────────────────┐
│                   NEOBRUTALIST TOKENS                  │
├───────────────────┬────────────────────────────────────┤
│ Backdrop Color    │ Dribbble Yellow (#FFC800)          │
│ Polka Dot Pattern │ Radial #121212 (22px grid)         │
│ Container Canvas  │ Cream (#FFFDF0, micro-dots 16px)   │
│ Ink / Borders     │ Pitch Black (#121212, 3px - 4px)   │
│ Drop Shadows      │ Hard 4px - 10px #121212 (No blur)  │
│ Accent Pink       │ Hot Pink (#FF4081)                 │
│ Accent Cyan       │ Electric Cyan (#38BDF8)            │
│ Accent Lime       │ Neon Lime (#CCFF00)                │
│ Primary Amber     │ Electric Amber (#FFDE59)           │
│ Typography        │ 'Plus Jakarta Sans', sans-serif    │
└───────────────────┴────────────────────────────────────┘
```

#### Mechanical Button Physics (`index.css`)
```css
.neo-btn {
  transition: transform 0.12s ease-out, box-shadow 0.12s ease-out;
  will-change: transform, box-shadow;
}
.neo-btn:hover:not(:disabled) {
  transform: translate(2px, 2px);
  box-shadow: 2px 2px 0px #121212 !important;
}
.neo-btn:active:not(:disabled) {
  transform: translate(4px, 4px);
  box-shadow: 0px 0px 0px #121212 !important;
}
```

### 7.2 Global State Tree & Hook Lifecycle Management

```typescript
// Core Navigation State
const [activeTab, setActiveTab] = useState<'landing'|'scan'|'enroll'|'users'|'admin'>('landing');

// Hardware Health
const [cameraReady, setCameraReady] = useState<boolean>(false);
const [cameraType, setCameraType] = useState<string>('Checking...');

// Biometric Verification
const [isScanning, setIsScanning] = useState<boolean>(false);
const [scanCountdown, setScanCountdown] = useState<number | null>(null);
const [lastScan, setLastScan] = useState<ScanResult | null>(null);
const [resultOverlay, setResultOverlay] = useState<ScanResult | null>(null);
const [selectedAuthAction, setSelectedAuthAction] = useState({
  id: 'pay', name: 'Palm Pay Auth', desc: 'Payment Token', icon: '💳'
});

// Biometric Enrollment
const [enrollUsername, setEnrollUsername] = useState<string>('');
const [enrollSamples, setEnrollSamples] = useState<Array<{ vr_mean: number; thumb: string }>>([]);
const [isCapturingSample, setIsCapturingSample] = useState<boolean>(false);
const [enrollCountdown, setEnrollCountdown] = useState<number | null>(null);
const [enrollStatusMsg, setEnrollStatusMsg] = useState<string>('');

// Modals & Feedback
const [users, setUsers] = useState<User[]>([]);
const [searchQuery, setSearchQuery] = useState<string>('');
const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
const [reportModalOpen, setReportModalOpen] = useState<boolean>(false);
const [reportData, setReportData] = useState<ReportData | null>(null);
const [toast, setToast] = useState<{ msg: string; type: 'success'|'warn'|'error' } | null>(null);
```

#### Lifecycle Hooks
1. **Heartbeat Polling:** `useEffect` runs on mount, invoking `loadUsers()` and setting up a 5000ms interval for `loadStatus()`.
2. **Session Cleanup Hook:**
   ```typescript
   useEffect(() => {
     if (activeTab !== 'enroll' && enrollSamples.length > 0 && enrollUsername) {
       fetch('/api/enroll/cancel', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ username: enrollUsername }),
       }).catch(() => {});
       setEnrollSamples([]);
       setEnrollUsername('');
       setEnrollStatusMsg('');
     }
   }, [activeTab]);
   ```

---

### 7.3 Component Tree & Layout Shell
The application is structured as a single mobile phone terminal canvas (`max-w-[480px]`, `min-h-[854px]`) centered on desktop viewports and taking full screen on mobile or kiosk touch displays:

```
<App>
  ├── Top Service Bar (Clock, Driver Tag, Dual-Color Live Sensor Bead)
  ├── Header Profile Bar (Custom PalmIcon SVG, User Counter, Home Nav)
  ├── Toast Notification Portal (Animated bounce banner)
  ├── Main Viewport (<main> with custom scrollbar)
  │     ├── [Tab: landing] Animated Hero Showcase
  │     ├── [Tab: scan]    Authentication Terminal & Countdown
  │     ├── [Tab: enroll]  6-Sample Guided Studio & Timer
  │     ├── [Tab: users]   Directory, Search, & Delete Management
  │     └── [Tab: admin]   Accuracy Separation Matrix & Hardware Specs
  ├── Sliding Bottom Navbar (Spring-physics indicator pill)
  ├── Floating Action Button (+) on Users Tab
  ├── Fullscreen Result Overlay (Confidence gauge, confetti trigger)
  ├── Delete Confirmation Modal
  └── Biometric Matrix Report Modal
```

---

### 7.4 Page-by-Page Architectural Breakdown

#### Page 1: Landing (`landing`)
* **Hero Visual:** 3D styled smartphone card showing sub-dermal vein tracks with pulsing sensor beads.
* **Real-Time Telemetry:** Warns immediately if camera hardware is detached.
* **Primary Action:** Direct deep-link CTA button into `/scan` with animated arrow icon.
* **Feature Badges:** Highlights 5-second countdown timers and sub-dermal security advantages.

#### Page 2: Palm Scan Terminal (`scan`)
* **Action Mode Selector:** 3-button segmented selector (`Palm Pay Auth`, `Door Access`, `Identity Verify`).
* **Viewport Window:** Streams `/api/video_feed` within a 3px black-bordered frame.
* **3-Second Countdown HUD:** Clicking **START 3s PALM SCAN** counts down: `3 -> 2 -> 1 -> CAPTURE`.
* **Recent Verification Card:** Displays the user's name, match distance score, millisecond latency, and base64 CLAHE ROI crop.
* **Celebration:** Matches trigger a 5-color confetti explosion using `canvas-confetti`.

#### Page 3: Enrollment Studio (`enroll`)
* **Username Input:** Real-time sanitization to lowercase alphanumeric characters.
* **Progress Grid:** 6 dynamic cells showing sample status, green checks, and live CLAHE thumbnails.
* **Camera Feed & 5s Countdown:** Large yellow numbers overlay the feed during the 5-second countdown.
* **Dynamic Posture Guidance:**
  * Sample 1: *"Hold palm flat, centered ~10-15cm above sensor"*
  * Sample 2: *"Tilt palm slightly to the LEFT (~5 degrees)"*
  * Sample 3: *"Tilt palm slightly to the RIGHT (~5 degrees)"*
  * Sample 4: *"Raise palm slightly HIGHER (~15-18cm)"*
  * Sample 5: *"Spread fingers slightly wider"*
  * Sample 6: *"Hold palm flat for final confirmation"*
* **Database Commit Button:** Unlocks once $\ge 3$ samples are acquired. Commits templates and redirects to the user directory.

#### Page 4: User Directory (`users`)
* **Live Search:** Instant client-side filtering by username.
* **Profile Cards:** Avatar icons with deterministic color rotation (`#FFDE59`, `#38BDF8`, `#CCFF00`, `#FF4081`, `#A855F7`), sample counts, and enrollment timestamps.
* **Soft Delete:** Clicking the trash button triggers the confirmation modal and sends `DELETE /api/users/{username}`.

#### Page 5: System Diagnostics (`admin`)
* **Hardware Specs Grid:** Displays camera driver, 4-core worker pool configuration, RAM Euclidean filter specifications, and decision thresholds.
* **SQLite Storage Card:** Displays user count and zlib template metrics.
* **Accuracy Matrix Modal:** Displays intra-user self-match stability ($min, avg, max$) and inter-user pairwise cross-match separation scores.
* **Database Wipe Button:** Clears all biometric templates and logs back to factory state.

---

### 7.5 Modal Portals & Notification Systems
* **Fullscreen Result Overlay:** Covers the viewport with an animated icon, status title, confidence gauge percentage bar, and biometric MNHD score.
* **Delete Modal:** Modal backdrop preventing accidental deletions.
* **Biometric Matrix Report Modal:** Scrollable list showing verification quality metrics.

---

## 8. Edge Diagnostic & Dataset Collection Tools

In addition to the web stack, the codebase provides two standalone tools for terminal environments and dataset collection.

### 8.1 `cam_test.py`: Interactive Terminal Diagnostic
A lightweight OpenCV script that runs directly on the Raspberry Pi 5 terminal (or via SSH with `DISPLAY=:0`):

```bash
python3 cam_test.py
```

* **Live Alignment HUD:** Shows green target box, enrolled user count, and operating threshold.
* **Key Controls:**
  * `N`: Runs full 6-sample enrollment with 5-second countdowns and consistency validation.
  * `S`: Runs 3-second countdown scan and prints score, latency, and match outcome.
  * `L`: Lists registered identities in terminal.
  * `Q`: Releases hardware and cleanly exits.
* **Automatic Logging:** Saves raw grayscale captures to `captures/` and CLAHE ROIs to `roi_clahe/`.

### 8.2 `collect_samples.py`: ML Dataset Harvesting Engine
Designed for machine learning engineers collecting raw palm datasets for fine-tuning preprocessing pipelines:

```bash
python3 collect_samples.py
```

* **No Overhead:** Bypasses MediaPipe, Gabor filtering, and SQLite storage to capture pure, uncompressed frames at full camera resolution (1640×1232).
* **Automated Structure:** Organizes images into `dataset/<subject_name>/<subject>_<hand>_<idx>_<ts>.png`.
* **Metadata Logging:** Records resolution, timestamp, mean brightness, and exposure time to `dataset/dataset_log.csv`.
* **Hotkeys:**
  * `SPACE` / `C`: Instant raw capture.
  * `B`: Automated 6-sample guided batch capture with 4-second countdowns.
  * `H`: Toggle active hand (`RIGHT` ⟷ `LEFT`).
  * `U`: Switch subject ID.
  * `[` / `]`: Adjust exposure in 1000µs increments for NIR lighting tuning.
  * `L`: Print dataset collection statistics.
  * `Q`: Clean exit.

---

## 9. Raspberry Pi 5 Deployment & Verification Checklist

### 9.1 Hardware Setup
1. Mount the **Raspberry Pi NoIR Camera** to CSI port 1 or 2 using a 15-to-22 pin ribbon cable.
2. Connect the **850nm Infrared Illuminator Ring** to a 5V/GND header or external supply.
3. Power the Pi using an official 27W USB-C Power Supply.

### 9.2 Software Installation
```bash
# Extract deployment package
unzip palm_vein_pi5.zip -d palm_vein
cd palm_vein

# Install Python requirements
pip install -r requirements.txt --break-system-packages

# Install Raspberry Pi camera support (if not pre-installed)
sudo apt update && sudo apt install python3-picamera2 -y

# Verify camera detection
libcamera-hello --list-cameras
```

### 9.3 Launching the Application
```bash
# Terminal Test Mode
python3 cam_test.py

# Full Web Application
python3 server.py
```
Open **`http://localhost:8000`** in Chromium to use the terminal.

---

## 10. Summary File Map

```
vein-detection1/
├── server.py              # FastAPI server, camera lock, async route handlers
├── gabor.py               # 2D Gabor wavelets, structure tensor, shift-tolerant MNHD
├── mediapipe_img.py       # MediaPipe HandLandmarker, valley derivation, ROI crop
├── search_engine.py       # 2-layer search engine (RAM Euclidean + parallel MNHD)
├── db_manager.py          # SQLite interface, zlib BLOB compression, access logging
├── cam_test.py            # Standalone terminal CV test tool
├── collect_samples.py     # Dataset collection script for ML training
├── hand_landmarker.task   # MediaPipe hand landmark model
├── requirements.txt       # Production dependencies
├── README.md              # Deployment guide and hardware instructions
├── static/                # Built React frontend assets
└── web/                   # Frontend source code (React 18, TypeScript, Vite)
```
