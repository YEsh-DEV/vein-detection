# Palm Vein Authentication System
Raspberry Pi 5 + NoIR Camera + FastAPI + React

## Deploying to Raspberry Pi 5

### 1. Transfer the package
Copy `palm_vein_pi5.zip` to the Pi via USB drive or SCP:
```bash
scp palm_vein_pi5.zip pi@<PI_IP>:~/
```

### 2. Extract on Pi
```bash
cd ~/
unzip palm_vein_pi5.zip -d palm_vein
cd palm_vein
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### 4. Install picamera2 (if not already installed)
```bash
sudo apt install python3-picamera2 -y
```

### 5. Verify camera is detected
```bash
libcamera-hello --list-cameras
```

---

## ⚡ Direct Live Terminal Test (`cam_test.py`)

To verify the camera hardware, MediaPipe landmarking, Gabor filtering, enrollment, and identification logic without running the web server:

```bash
python3 cam_test.py
```

### Key Controls in Camera Window:
- **`N`**: Enroll new user (6 samples, 5s countdown HUD, live CLAHE preview, consistency analysis)
- **`S`**: Scan & identify (3s countdown HUD, CLAHE preview, score & timing report)
- **`L`**: List all enrolled users and sample counts
- **`Q`** or **`ESC`**: Quit and release camera

---

## 🌐 Web Server & UI Mode (`server.py`)

```bash
# Start FastAPI backend & UI
python3 server.py

# Open in Chromium on Pi (or browse from LAN at http://<PI_IP>:8000)
http://localhost:8000
```

### First Run Checklist (Web UI)
- **Stats tab:** click RESET DATABASE (clears any old test data)
- **Enroll tab:** confirm live camera feed is visible
- Enroll yourself with 3–6 samples
- **Scan tab:** test authentication

---

## Project Structure

| File | Purpose |
|------|---------|
| `cam_test.py` | Direct live camera terminal test (Enroll, Scan, Verification) |
| `server.py` | FastAPI backend & Web UI server |
| `gabor.py` | Adaptive Gabor filter feature extraction (Ma et al. 2017) |
| `mediapipe_img.py` | MediaPipe hand landmark ROI extraction |
| `search_engine.py` | Two-layer MNHD biometric search engine |
| `db_manager.py` | SQLite template storage |
| `hand_landmarker.task` | MediaPipe hand landmark model (required at runtime) |
| `static/` | Built React frontend (served by FastAPI) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/api/status` | Camera status + user count |
| GET | `/api/users` | List enrolled users |
| POST | `/api/scan` | Authenticate by palm scan |
| POST | `/api/enroll/sample` | Capture one enrollment sample |
| POST | `/api/enroll/save` | Commit samples to database |
| POST | `/api/enroll/cancel` | Clear partial enrollment session |
| DELETE | `/api/users/{username}` | Remove a user |
| DELETE | `/api/database/reset` | Wipe all biometric data |
| GET | `/api/report` | Self-match and cross-match quality report |

---

## Notes

- **picamera2** is pre-installed on Raspberry Pi OS. If missing: `sudo apt install python3-picamera2`
- **Pipeline timeouts:** scan has a 15-second MediaPipe/Gabor timeout and a 10-second matching timeout. Poor lighting will time out before hanging the UI.
- **Enrollment rules:** username must be 2–30 chars (letters, numbers, hyphens, underscores). Minimum 3 samples required, maximum 6 allowed.
- The database (`data/palm_vein.db`) and capture directories are created automatically on first run.
