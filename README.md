# 🖐️ Palm Vein Biometrics & Contactless Payment System
> **Edge-Native Sub-Dermal Vascular Biometric Authentication for Raspberry Pi 5**  
> *Powered by Near-Infrared (NIR 850nm) Imaging, MediaPipe Hand Landmarking, 2D Gabor Wavelets, and a Neobrutalism React Terminal.*

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red.svg?logo=raspberry-pi)](https://www.raspberrypi.com/)
[![Camera](https://img.shields.io/badge/Sensor-Pi%20NoIR%20Camera%20%28850nm%29-blue.svg)](https://www.raspberrypi.com/documentation/accessories/camera.html)
[![Backend](https://img.shields.io/badge/Backend-FastAPI%20%2B%20Picamera2-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-React%2018%20%2B%20TypeScript%20%2B%20Tailwind-61dafb.svg?logo=react)](https://react.dev/)
[![Biometrics](https://img.shields.io/badge/Matching-Modified%20Normalized%20Hamming%20Distance-orange.svg)]()
[![Documentation](https://img.shields.io/badge/Architecture-Deep--Dive%20Report-purple.svg)](SYSTEM_ARCHITECTURE_OVERVIEW.md)

---

## 🌟 Overview & Scientific Principles

The **Palm Vein Biometrics & Contactless Payment Terminal** is an edge-native biometric verification system built to provide non-contact, spoof-proof identification.

Unlike surface fingerprints or 2D facial recognition, palm veins operate in the **sub-dermal vascular domain**:
1. **NIR Absorption:** Deoxygenated hemoglobin ($Hb$) flowing through palmar venous arches absorbs near-infrared light at **850nm**.
2. **Sub-Surface Imaging:** Back-scattered infrared reflection reveals veins as dark branching vascular silhouettes beneath the skin.
3. **Liveness Guarantee:** Biometric patterns cannot be replicated from surface residue, photograph prints, or silicone prosthetics.
4. **Sub-Second Edge Execution:** 100% on-device pipeline execution on the Raspberry Pi 5 without cloud connectivity or GPU acceleration.

> 📖 **Looking for full mathematical equations, Gabor wavelet formulation, and deep-dive architecture specs?**  
> See the [**Complete Technical Architecture Report (`SYSTEM_ARCHITECTURE_OVERVIEW.md`)**](SYSTEM_ARCHITECTURE_OVERVIEW.md).

---

## ⚡ Key Capabilities

* **📷 Dual Camera Hardware Engine:** Direct hardware-level support for Raspberry Pi NoIR Camera (`Picamera2` with locked `5000µs` exposure and `1.0` analog gain) with automatic fallback to USB webcams (`OpenCV`).
* **🔒 Thread-Safe Streaming & Stills:** Threading mutual exclusion lock (`_camera_lock`) eliminates camera bus crashes between the continuous 640×480 MJPEG display and 1640×1232 high-resolution biometric still captures.
* **🖐️ MediaPipe Landmark Knuckle Valleys:** Replaces noisy contour heuristics with 21 anatomical landmarks to calculate invariant Cartesian anchors ($Pv_1$ and $Pv_2$) from MCP knuckle joints.
* **📐 Canonical ROI Normalization (Ma et al. 2017):** Invariant scale normalization ($L = 1.5 \times D_{pv}$) and vertical offset calculation ($d_0 = 0.35 \times D_{pv}$) resampled to standard $256 \times 256$ patches.
* **🌊 Adaptive 2D Gabor Phase Binarization:** Multi-orientation wavelet filter bank extracting real ($V_R$) and imaginary ($V_I$) phase VeinCodes.
* **🚀 Hierarchical 2-Layer Search Engine:**
  * **Layer 1 (RAM Vector Filter):** Compares 16-float spatial signatures in RAM to filter candidate pools in $<0.5\text{ms}$.
  * **Layer 2 (Parallel Multi-Core MNHD):** Parallel shift-tolerant Modified Normalized Hamming Distance matching ($\pm 8$ pixels) across 4 Pi 5 CPU cores.
* **🎨 Premium Neobrutalism React UI:** Tactile retro-modern interface with 5-second guided positioning countdowns, live alignment reticles, confetti celebration bursts, and biometric separation diagnostics.
* **🛠️ CLI Diagnostics & Dataset Tools:** Dedicated offline terminal utility (`cam_test.py`) and dataset harvesting engine (`collect_samples.py`) for machine learning research.

---

## 🚀 Raspberry Pi 5 Deployment Guide

### 1. Hardware Assembly
1. Connect the **Raspberry Pi NoIR Camera Module** to CSI Port 1 or 2 using the appropriate 15-to-22 pin ribbon cable.
2. Mount the **850nm Infrared Illuminator Array** to illuminate the palm area directly above the camera lens.
3. Power the Raspberry Pi 5 using an official **27W USB-C Power Supply**.

### 2. Software Installation

```bash
# Transfer and unzip deployment bundle
scp palm_vein_pi5.zip pi@<PI_IP>:~/
ssh pi@<PI_IP>
unzip palm_vein_pi5.zip -d palm_vein
cd palm_vein

# Install Python production dependencies
pip install -r requirements.txt --break-system-packages

# Ensure picamera2 package is installed on Pi OS
sudo apt update && sudo apt install python3-picamera2 -y

# Verify camera sensor detection
libcamera-hello --list-cameras
```

---

## 🖥️ Operating Modes

### Mode 1: Full Web Application (`server.py`)

Launches the asynchronous FastAPI backend and serves the compiled Neobrutalism Single-Page Application:

```bash
python3 server.py
```

* Open in Chromium browser on the Pi: **`http://localhost:8000`**
* Or access from any device on your local network: **`http://<PI_IP>:8000`**

#### First-Run Web UI Walkthrough:
1. **Stats Tab:** Click **RESET DATABASE** to initialize a clean biometric vault.
2. **Enroll Tab:** Enter a username, follow the 5-second countdown timer across 6 postures (flat, tilt left, tilt right, higher, wider), and click **SAVE TO DATABASE**.
3. **Scan Tab:** Select verification intent (**Palm Pay Auth**, **Door Access**, or **Identity Verify**) and start the 3-second palm scan.

---

### Mode 2: Interactive Terminal Diagnostic (`cam_test.py`)

A standalone OpenCV terminal utility for hardware diagnostics, camera alignment, and verification without running the web server:

```bash
python3 cam_test.py
```

#### Key Controls (in OpenCV Window):
| Key | Action | Description |
|---|---|---|
| **`N`** | **Enroll User** | Guides through 6 samples with 5s countdowns, CLAHE display, and consistency check. |
| **`S`** | **Scan & Match** | 3s countdown, real-time ROI extraction, MNHD identification, and timing report. |
| **`L`** | **List Users** | Displays all enrolled identities and template counts in terminal. |
| **`Q` / `ESC`** | **Quit** | Gracefully shuts down camera and worker pools. |

*Captures and enhanced ROIs are automatically saved to `captures/` and `roi_clahe/`.*

---

### Mode 3: Raw Dataset Sample Collector (`collect_samples.py`)

For researchers collecting high-resolution raw palm captures without ROI cropping or Gabor encoding (for model training, fine-tuning, or data analysis):

```bash
python3 collect_samples.py
```

#### Key Controls:
| Key | Action | Description |
|---|---|---|
| **`SPACE` / `C`** | **Instant Snap** | Snaps raw uncompressed frame with green border feedback. |
| **`B`** | **Guided Batch** | Captures 6 guided samples with 4-second countdowns. |
| **`H`** | **Toggle Hand** | Switches active hand label (`RIGHT` ⟷ `LEFT`). |
| **`U`** | **Change User** | Changes active subject ID in terminal. |
| **`[` / `]`** | **Exposure Tuning** | Decreases / Increases camera shutter speed in 1000µs increments. |
| **`L`** | **List Stats** | Displays total sample count summary. |
| **`Q` / `ESC`** | **Quit** | Clean exit. |

*Saves raw high-res images to `dataset/<subject_name>/` and metadata to `dataset/dataset_log.csv`.*

---

## 📁 Repository Structure

```text
vein-detection1/
├── server.py                       # FastAPI application & camera locking server
├── gabor.py                        # 2D Gabor wavelet filter bank & shift-tolerant MNHD
├── mediapipe_img.py                # MediaPipe landmarking & Ma et al. (2017) ROI extraction
├── search_engine.py                # Two-layer search engine (RAM Euclidean + parallel MNHD)
├── db_manager.py                   # SQLite database manager with zlib compression
├── cam_test.py                     # Direct OpenCV terminal testing tool
├── collect_samples.py              # Raw dataset sample collection engine
├── hand_landmarker.task            # MediaPipe hand landmark model (~8MB)
├── requirements.txt                # Production Python dependencies
├── README.md                       # Main deployment and usage documentation
├── SYSTEM_ARCHITECTURE_OVERVIEW.md # Comprehensive engineering & architectural deep-dive
├── static/                         # Production-built React frontend assets
└── web/                            # React 18 + TypeScript + Tailwind source code
    ├── src/
    │   ├── App.tsx                 # Neobrutalism application shell (1,130 LOC)
    │   ├── index.css               # Design tokens, polka dots, and mechanical button CSS
    │   └── main.tsx                # React DOM mount point
    ├── vite.config.ts              # Vite configuration (outputs directly to ../static/)
    └── package.json                # Web build scripts and dependencies
```

---

## 📡 REST API Reference

| Method | Endpoint | Response Model | Description |
|---|---|---|---|
| `GET` | `/health` | `{"ok": true}` | Edge liveness and container readiness probe. |
| `GET` | `/api/status` | `StatusResponse` | Camera driver name, user count, total templates, threshold. |
| `GET` | `/api/video_feed` | `StreamingResponse` | Live MJPEG continuous video stream with target reticle. |
| `POST` | `/api/scan` | `ScanResponse` | Captures palm frame, computes VeinCode, and identifies user. |
| `POST` | `/api/enroll/sample` | `SampleResponse` | Captures and caches 1 of 6 multi-angle enrollment samples. |
| `POST` | `/api/enroll/save` | `SaveResponse` | Commits cached samples ($\ge 3$) to SQLite and updates RAM index. |
| `POST` | `/api/enroll/cancel` | `{"cleared": bool}` | Purges active uncommitted enrollment session from cache. |
| `GET` | `/api/users` | `UsersResponse` | Lists all enrolled users and sample counts. |
| `DELETE` | `/api/users/{username}`| `DeleteResponse` | Removes user profile and updates search index. |
| `DELETE` | `/api/database/reset` | `ResetResponse` | Wipes all biometric records for clean re-provisioning. |
| `GET` | `/api/report` | `ReportResponse` | Intra-user self-matches & inter-user cross-match analytics. |

---

## 🛡️ Engineering Safeguards & Best Practices

1. **Hardware Exposure Stability:** `Picamera2` is initialized with auto-exposure disabled (`AeEnable: False`) and exposure fixed at `5000µs`. This prevents illuminance oscillation and guarantees consistent vascular contrast.
2. **Non-Blocking Threadpool Offloading:** All heavy CPU-bound OpenCV and Gabor algorithms run via `run_in_threadpool`, keeping the FastAPI asynchronous event loop responsive.
3. **Pipeline Timeouts:** `/api/scan` and `/api/enroll/sample` enforce a strict **15.0-second timeout** on MediaPipe/Gabor processing and a **10.0-second timeout** on parallel matching to prevent hangs from obstructed camera views.
4. **Session Bleed Prevention:** Switching away from the Enroll tab automatically invokes `/api/enroll/cancel`, clearing uncommitted templates from server memory.
5. **Biometric Privacy:** Only binary phase VeinCodes are stored in SQLite; raw palm images and templates never leave the local edge environment.

---

## 📜 License & Acknowledgments
* **Architecture:** Based on the biometric feature extraction framework proposed by *Ma et al. (2017), IET Biometrics*.
* **Computer Vision:** Powered by Google MediaPipe and OpenCV.
* **Interface:** Designed following the modern Neobrutalism design movement.
