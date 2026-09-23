# Raspberry Pi Live Demo Deployment & Hardening Guide
======================================================

> **MANUAL PI EXECUTION INSTRUCTIONS:**  
> This guide is prepared on the development laptop. All commands below must be executed manually on the Raspberry Pi terminal.  
> Do not assume the laptop executed or verified hardware on the Pi.

---

## 1. System Overview & Verification

The Palm-Vein Biometrics system operates as a Category B Working Prototype with the following frozen specifications:
- **Biometric Model:** AMPVNet Fine-Tuned ONNX (`models/ampvnet_finetuned.onnx`)
- **Embedding:** 512-D float32, L2-normalized
- **Matching Threshold:** `0.2226` (Frozen; cosine similarity $\ge 0.2226$ accepted)
- **Landmark Detector:** MediaPipe Hand Landmarker (`models/hand_landmarker.task`)
- **Capture Resilience:** 5-frame burst capture (~0.4s window) with structured diagnostic logging

```bash
# RUN MANUALLY ON RASPBERRY PI
# 1. Verify 64-bit architecture
uname -m
# -> Expected: aarch64

# 2. Check Python version
python3 --version
# -> Python 3.11, 3.12, or 3.13 supported
```

---

## 2. Environment & Runtime Setup

Raspberry Pi OS Bookworm utilizes APT for `picamera2` and hardware libcamera bindings. The virtual environment must be configured with `--system-site-packages`.

```bash
# RUN MANUALLY ON RASPBERRY PI
cd ~/vein-detection  # (replace with your actual workspace path)

# 1. Install system prerequisites
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-opencv python3-pip python3-venv jq curl chromium-browser

# 2. Recreate virtual environment with system site packages
deactivate 2>/dev/null
rm -rf .venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 3. Upgrade pip and install runtime dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify Picamera2 and ONNX Runtime
python3 -c "import picamera2; import onnxruntime; print('[+] Pi runtime imports successful!')"
```

---

## 3. Verify Model Assets

Verify that both required neural network assets exist locally:

```bash
# RUN MANUALLY ON RASPBERRY PI
ls -lh models/ampvnet_finetuned.onnx models/hand_landmarker.task
```

*Expected sizes:*
- `ampvnet_finetuned.onnx`: ~132 MB
- `hand_landmarker.task`: ~9.5 MB

---

## 4. Clean Demo Database Management & Reset Procedure

Before commencing a live presentation or evaluation session, purge all synthetic development records and establish a clean database state:

```bash
# RUN MANUALLY ON RASPBERRY PI (inside .venv)

# 1. Check current database state
python3 tools/demo_data_manager.py --status

# 2. Purge synthetic test users (if any) while preserving genuine users:
python3 tools/demo_data_manager.py --clean-test-users

# 3. OPTIONAL: Pristine reset before a fresh demo (creates timestamped backup in data/backups/)
python3 tools/demo_data_manager.py --reset --force

# 4. Verify template storage and math integrity:
python3 tools/demo_data_manager.py --verify
```

---

## 5. Starting the Backend Service

### Option A: Manual Interactive Launch (Best for Debugging)

```bash
# RUN MANUALLY ON RASPBERRY PI
cd ~/vein-detection
source .venv/bin/activate

# Start Uvicorn on 0.0.0.0:8000
python3 -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --log-level info
```

### Option B: Production Autostart via Systemd (Best for Kiosk / Standalone)

Create a systemd unit file to automatically boot the backend on system power-on:

```bash
# RUN MANUALLY ON RASPBERRY PI
sudo tee /etc/systemd/system/palm-vein.service > /dev/null << 'EOF'
[Unit]
Description=Palm-Vein Biometric Kiosk Backend
After=network.target

[Service]
Type=simple
User=yesh
WorkingDirectory=/home/yesh/vein-detection
Environment="PATH=/home/yesh/vein-detection/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
ExecStart=/home/yesh/vein-detection/.venv/bin/python3 -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --log-level info
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Reload and enable service
sudo systemctl daemon-reload
sudo systemctl enable palm-vein.service
sudo systemctl start palm-vein.service

# Check service status
sudo systemctl status palm-vein.service
```

---

## 6. Health & Hardware Verification Check

Once the server is running, verify the backend health and CSI camera detection:

```bash
# RUN MANUALLY ON RASPBERRY PI
# 1. Test /health endpoint
curl -s http://127.0.0.1:8000/health | jq .

# 2. Test /api/status endpoint
curl -s http://127.0.0.1:8000/api/status | jq .
```

*Expected JSON output:*
```json
{
  "camera_available": true,
  "camera_type": "picamera2",
  "camera_device": "CSI (OV5647/IMX219/IMX477)",
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

## 7. Display Orientation Configuration (Portrait Mode)

The physical kiosk display is mounted vertically. Configure display rotation for either Wayland (Wayfire/Labwc) or X11.

### Option 1: Wayland Desktop (Raspberry Pi OS Default with Wayfire)

Edit `~/.config/wayfire.ini`:

```ini
[output:HDMI-A-1]
mode = 720x1280@60000
transform = 90
```
*(If your physical screen is mounted 270° inverted, set `transform = 270`)*.

For a 480×800 panel:
```ini
[output:DSI-1]
mode = 480x800@60000
transform = 90
```

### Option 2: X11 / Xrandr Display Server

```bash
# Rotate display 90 degrees clockwise
DISPLAY=:0 xrandr --output HDMI-1 --rotate right

# Or 90 degrees counter-clockwise (inverted)
DISPLAY=:0 xrandr --output HDMI-1 --rotate left
```

---

## 8. Fullscreen Chromium Kiosk Launch

Launch Chromium in dedicated kiosk mode targeting the portrait viewport.

Create a kiosk launcher script `scripts/launch_kiosk.sh`:

```bash
#!/usr/bin/env bash
# scripts/launch_kiosk.sh — Fullscreen Portrait Kiosk Launcher

# Wait for backend to be fully responsive
until curl -s http://127.0.0.1:8000/health | grep -q "healthy"; do
    echo "[*] Waiting for palm-vein backend to come online..."
    sleep 1
done

# Target logical resolution: 720x1280 (or 480x800)
# Flags disable info bars, translation prompts, zoom gestures, and crash bubbles
chromium-browser \
    --kiosk \
    --app=http://127.0.0.1:8000 \
    --window-size=720,1280 \
    --window-position=0,0 \
    --noerrdialogs \
    --disable-infobars \
    --check-for-update-interval=31536000 \
    --disable-pinch \
    --overscroll-history-navigation=0 \
    --disable-features=TranslateUI \
    --user-data-dir=/tmp/chromium-kiosk-profile
```

Make it executable:
```bash
chmod +x scripts/launch_kiosk.sh
```

---

## 9. Live Demo Execution Checklist

Follow this exact sequence during your live demo:

### Step 1: Pre-Demo Status Check
```bash
python3 tools/demo_data_manager.py --status
```
Verify:
- `v2 Templates`: 0 (or known enrolled count)
- `Camera Available`: True

### Step 2: Live Enrollment
1. Open the kiosk interface on the portrait display.
2. Enter the user name (e.g. `Presenter_Left`).
3. Press **Start Enrollment**.
4. Hold the palm flat ~10–15 cm above the lens.
5. The 5-frame burst capture will automatically select the sharpest valid frame for each sample.
6. The UI will guide the presenter through all 6 sample angles.
7. Upon sample #6, the backend computes the mean aggregate template and writes the 512-D embedding to SQLite.

### Step 3: Verification (Genuine Acceptance)
1. Navigate to **Verify / Scan**.
2. Present the enrolled hand.
3. System extracts the ROI in <50 ms, computes cosine similarity against enrolled templates, and displays:
   - **Access Granted** (Green banner)
   - Cosine Similarity Score (typically `0.45` to `0.85` for genuine palms vs threshold `0.2226`).

### Step 4: Impostor Test (Rejection)
1. Invite an unenrolled attendee or present the opposite (unenrolled) hand.
2. The system evaluates the probe.
3. Cosine similarity will fall below the threshold (typically `< 0.20`).
4. System displays **Access Denied** (Red banner).

---

## 10. Diagnostics & Troubleshooting

If a frame capture fails or the user sees positioning guidance on the kiosk:

### Inspecting Real-Time Capture Diagnostics
The backend logs detailed JSON diagnostic events on every failed capture to `logs/capture_diagnostics.jsonl`:

```bash
# RUN MANUALLY ON RASPBERRY PI
tail -f logs/capture_diagnostics.jsonl | jq .
```

Each log entry includes:
- `timestamp`: UTC ISO-8601 timestamp
- `stage`: Pipeline failure stage (`positioning`, `mediapipe`, `valleys`, `roi`, `quality`)
- `error_code`: Canonical error code (e.g. `HAND_TOO_CLOSE`, `QUALITY_LOW_CONTRAST`)
- `occupancy`: Estimated hand occupancy percentage
- `border_touches`: Number of image borders touched by the hand mask
- `contrast`: Standard deviation of pixel intensities across the ROI
- `padding`: Fraction of the ROI derived from replicate border padding
- `burst_attempts`: Total frames attempted before failing

### Common Error Codes & Physical Fixes

| Canonical Error Code | Diagnostic Root Cause | Recommended Action |
| :--- | :--- | :--- |
| `HAND_TOO_CLOSE` | Hand mask occupies >54% of frame or touches $\ge 3$ borders | Move hand 3–5 cm further from camera lens |
| `HAND_TOO_FAR` | Hand mask occupies <20% of frame | Move hand closer (~10–15 cm above lens) |
| `HAND_OUTSIDE_FRAME` | Occupancy <8% or mean brightness <20 | Center palm directly above sensor aperture |
| `MEDIAPIPE_NO_LANDMARKS`| Landmark detector confidence <0.4 | Hold palm steady; ensure fingers are gently spread |
| `VALLEY_EXTRACTION_FAILED` | Finger valleys occluded or fingers pressed together | Spread fingers slightly to reveal webbing |
| `QUALITY_LOW_CONTRAST` | Standard deviation <12 across palm vessels | Adjust NIR LED angle or ambient room lighting |
| `QUALITY_EXCESSIVE_PADDING`| Extracted square ROI extends outside frame (>35%) | Center palm in the middle of camera view |
