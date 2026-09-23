# Raspberry Pi 5 Runtime Setup & Hardware Deployment Guide
================================================================

> **IMPORTANT NOTICE:**  
> **RUN MANUALLY ON RASPBERRY PI.**  
> These instructions are prepared on the development laptop and must be executed directly in the Raspberry Pi terminal.

---

## 1. System Requirements & Architecture Check

The AMPVNet ONNX biometric inference engine requires a **64-bit ARM architecture (`aarch64`)**.

```bash
# RUN MANUALLY ON RASPBERRY PI
# 1. Check OS architecture (MUST report 'aarch64')
uname -m

# 2. Check Python version (Python 3.10, 3.11, 3.12, or 3.13 supported)
python3 --version
```

*Expected output:*
* `uname -m` $\rightarrow$ `aarch64` *(If it outputs `armv7l`, you have a 32-bit OS installed and must reflash with 64-bit Raspberry Pi OS).*
* `python3 --version` $\rightarrow$ `Python 3.11.x`, `Python 3.12.x`, or `Python 3.13.x`.

---

## 2. Install System APT Packages (Picamera2 & OpenCV)

Raspberry Pi OS (Debian 12 Bookworm) provides `picamera2` and libcamera hardware bindings exclusively via system APT packages.

```bash
# RUN MANUALLY ON RASPBERRY PI
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-opencv python3-pip python3-venv libatlas-base-dev jq curl
```

---

## 3. Create Python Virtual Environment with System Site Packages

> **CRITICAL:** Always use `--system-site-packages` when creating the virtual environment on Raspberry Pi OS. This allows Python in the venv to access the APT-installed `picamera2` and system `cv2` libraries without needing to compile them from source.

```bash
# RUN MANUALLY ON RASPBERRY PI
cd ~/vein-detection  # (or your repo directory)

# If an isolated venv was already created without system-site-packages, recreate it:
deactivate 2>/dev/null
rm -rf .venv
python3 -m venv --system-site-packages .venv

# Activate the virtual environment
source .venv/bin/activate

# Verify that picamera2 is now visible inside the venv:
python3 -c "import picamera2; print('[+] Picamera2 is successfully imported inside venv!')"
```

---

## 4. Install Runtime Dependencies via Pip

Install only the edge runtime packages (no PyTorch, no heavy training libraries):

```bash
# RUN MANUALLY ON RASPBERRY PI (inside activated .venv)
pip install --upgrade pip

# Install minimal edge runtime requirements
pip install -r requirements.txt
```

*Note on `onnxruntime`:* Official precompiled `aarch64` wheels are available on PyPI for Raspberry Pi 5. If `pip install onnxruntime` fails due to wheel resolution, run:
```bash
pip install --prefer-binary onnxruntime
```

---

## 5. Verify ONNX Runtime & CPUExecutionProvider

Verify that ONNX Runtime is installed and detects the CPU provider:

```bash
# RUN MANUALLY ON RASPBERRY PI
python3 -c "import onnxruntime as ort; print('[+] ONNX Runtime:', ort.__version__); print('[+] Available Providers:', ort.get_available_providers())"
```

*Expected output:*
```text
[+] ONNX Runtime: 1.19.x (or 1.20.x)
[+] Available Providers: ['CPUExecutionProvider']
```

---

## 6. Verify Model File Existence

Confirm that the fine-tuned ONNX model exists in the `models/` directory:

```bash
# RUN MANUALLY ON RASPBERRY PI
ls -lh models/ampvnet_finetuned.onnx
```

*Expected output:*
```text
-rw-r--r-- 1 pi pi 6.2M models/ampvnet_finetuned.onnx
```

---

## 7. Run the Standalone Runtime Smoke Test

Execute the runtime smoke test tool created specifically for edge verification:

```bash
# RUN MANUALLY ON RASPBERRY PI
python3 tools/pi_runtime_smoke_test.py
```

*Expected output:*
```text
====================================================================
  RASPBERRY PI RUNTIME ENVIRONMENT SMOKE TEST
====================================================================
Platform : Linux-6.6.x-v8-16k+-aarch64
Target   : Raspberry Pi 5 / Debian 12 Bookworm (ARM64)

--- Core Runtime Dependencies ---
  [✓ PASS]  Python Version               Python 3.11.x (>= 3.10 required)
  [✓ PASS]  System Architecture          aarch64 (64-bit OS)
  [✓ PASS]  OpenCV (cv2)                 v4.x.x
  [✓ PASS]  NumPy                        v1.26.x
  [✓ PASS]  MediaPipe Package            v0.10.x
  [✓ PASS]  Picamera2 Hardware Driver    Available (installed via apt)
  [✓ PASS]  ONNX Runtime Package         v1.19.x
  [✓ PASS]  CPUExecutionProvider         Available (['CPUExecutionProvider'])

--- Biometric Model & Inference Pipeline ---
  [✓ PASS]  ONNX Model File              ampvnet_finetuned.onnx (6.14 MB)
  [✓ PASS]  Model Session Load           InferenceSession initialized on CPU
  [✓ PASS]  512-D Embedding Extraction   Shape: (512,), Dtype: float32
  [✓ PASS]  L2 Unit Normalization        ||v||_2 = 1.000000

====================================================================
  [✓] SUCCESS: Raspberry Pi runtime environment is fully operational!
====================================================================
```

---

## 8. Physical Camera Hardware Detection & Verification

### 8.1. Check CSI Camera Detection via libcamera
```bash
# RUN MANUALLY ON RASPBERRY PI
rpicam-hello --list-cameras
```

*Expected output:*
```text
Available cameras:
0 : ov5647 [2592x1944 10-bit] (/base/axi/pcie@120000/...)
    Modes: 'SRGGB10_CSI2P' : 640x480 [...], 1280x720 [...], 1920x1080 [...]
```

### 8.2. Capture a 2-second Still Test Frame
```bash
# RUN MANUALLY ON RASPBERRY PI
rpicam-still -t 2000 -o test_camera.jpg
ls -lh test_camera.jpg
```

### 8.3. Run the Comprehensive Optical & Signal Diagnostic Tool
```bash
# RUN MANUALLY ON RASPBERRY PI
python3 tools/pi_camera_diagnostic.py --save-frame diag_frame.png
```

---

## 9. Optical Focus & Capture Quality Calibration Guide

### Physical Camera Specifications:
* **Sensor:** 5 MP (OmniVision OV5647 or similar)
* **Resolution:** 1080p still / 640×480 preview
* **Lens:** 3.6 mm M12 screw-threaded lens barrel
* **Lighting:** Dual 850 nm high-power NIR LEDs
* **Focus:** **MANUAL FOCUS** (screw barrel, NOT motorized autofocus)

### Analysis of the 9 Optical & Failure Vectors:

| Failure Vector | Cause & Physical Mechanism | Calibration Action |
| :--- | :--- | :--- |
| **1. Defocus / Soft Preview** | Factory focus set to $\infty$ (CCTV mode); macro depth of field at 12 cm is $< 1.5$ cm. | Loosen knurled lock ring. Rotate lens barrel **counter-clockwise 1/4 to 1/2 turn** until `tools/pi_camera_diagnostic.py` reports Sharpness $\ge 45.0$. |
| **2. Exposure / Motion Blur** | Auto-Exposure in dim NIR environment increases shutter to 33,000 µs (1/30 s). Hand tremor causes blur. | Shutter is locked to **$4000\text{--}5000\mu\text{s}$** (`ExposureTime: 5000`, `AnalogueGain: 1.0`) in software. |
| **3. IR LED Center Saturation** | Dual IR LEDs follow inverse-square law ($I \propto 1/d^2$). Holding hand $< 10$ cm blows out palm center (pixels = 255). | Maintain palm distance at **$12\text{--}15\text{ cm}$**. The diagnostic tool will alert if `sat_pct > 3.0%`. |
| **4. Hand Tremor / Stability** | Free-floating palm without hand guide causes orientation drift. | Rest wrist on a fixed base or maintain palm parallel to lens for 1 second. |
| **5. Sensor Resolution** | 640×480 preview bins pixels; still capture uses native sensor mode. | Preview is used for positioning; enrollment samples crop from full-resolution arrays. |
| **6. Preview Aspect Ratio** | Kiosk 5-inch screen scaling 4:3 preview into 16:9 box causes perceived distortion. | CSS `object-fit: cover` and fixed guide brackets enforce accurate centering. |
| **7. `HAND_TOO_CLOSE` (159/333)** | Palm held $<8$ cm exceeds 53° FOV of 3.6mm lens. Wrist & fingertips are clipped outside sensor borders. | Keep palm at **10–15 cm** distance. Pre-landmark heuristic immediately instructs user: *"Move hand farther from lens"*. |
| **8. MediaPipe Rejection** | MediaPipe requires 4 knuckle valley landmarks (base of index, middle, ring, pinky). Curled fingers prevent detection. | Keep fingers **naturally spread and flat** parallel to sensor plane. |
| **9. Low Vascular Contrast** | Cold hands or insufficient NIR absorption. | CLAHE enhancement ($8\times 8$ grid, clip limit 2.5) normalizes contrast dynamically. |

---

## 10. Start the Production Backend Server

Once all smoke tests and camera diagnostics pass:

```bash
# RUN MANUALLY ON RASPBERRY PI
cd ~/vein-detection
source .venv/bin/activate

# Launch FastAPI server bound to all network interfaces on port 8000
python3 -m app.server
```

Or using uvicorn directly:
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

---

## 11. Verify Server Health & Status Endpoints

Open a second terminal on the Raspberry Pi (or query from your laptop via Pi's IP address):

```bash
# RUN MANUALLY ON RASPBERRY PI (or from laptop replacing localhost with Pi IP)
# 1. Check health endpoint
curl -s http://localhost:8000/health | jq .

# 2. Check API status endpoint
curl -s http://localhost:8000/api/status | jq .
```

*Expected `/health` response:*
```json
{
  "status": "healthy",
  "engine": "v2",
  "model_loaded": true,
  "model_error": null,
  "camera_available": true
}
```

*Expected `/api/status` response:*
```json
{
  "camera_available": true,
  "camera_type": "picamera2",
  "camera_device": "CSI",
  "camera_error": null,
  "model_loaded": true,
  "model_error": null,
  "enrolled_users_count": 0,
  "total_templates": 0,
  "match_threshold": 0.2226,
  "biometric_engine": "v2"
}
```

---

## 12. Troubleshooting Matrix

| Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| `No module named 'onnxruntime'` | ONNX Runtime not installed in active environment. | `pip install onnxruntime` |
| `No module named 'picamera2'` | Virtualenv created without access to system packages. | Recreate venv with: `python3 -m venv --system-site-packages .venv` |
| `Camera UNAVAILABLE` | CSI ribbon cable loose or camera disabled in raspi-config. | Run `sudo raspi-config` -> Interface Options -> Camera. Check physical ribbon cable orientation. |
| `model_loaded: false` | `models/ampvnet_finetuned.onnx` missing or wrong path. | Run `git pull origin main` and check `ls -lh models/ampvnet_finetuned.onnx`. |
| `HAND_TOO_CLOSE` on scan | User holding palm too close ($<10\text{ cm}$). | Back hand away to $12\text{--}15\text{ cm}$ until corner guide brackets frame the palm. |
| `503 Biometric model not loaded` | Model initialization failed at startup. | Inspect error message returned in HTTP 503 response body; it now provides exact diagnostic details. |
