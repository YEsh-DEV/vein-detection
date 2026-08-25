# Palm Vein Authentication System
Raspberry Pi 5 + NoIR Camera + FastAPI + React

## Quick Start (Raspberry Pi 5)

```bash
# Install Python dependencies
pip install -r requirements.txt --break-system-packages

# picamera2 is pre-installed on Pi OS — no pip needed

# Start the server
python3 server.py

# Open browser on Pi
http://localhost:8000
```

## Project Structure

- `server.py` — FastAPI backend, entry point, run this
- `gabor.py` — Adaptive Gabor filter feature extraction (Ma et al. 2017)
- `mediapipe_img.py` — MediaPipe hand landmark ROI extraction
- `search_engine.py` — Two-layer MNHD biometric search engine
- `db_manager.py` — SQLite template storage
- `test_offline.py` — Offline pipeline test (no camera needed)
- `web/` — React + Tailwind source (edit here)
- `static/` — Built frontend (served by FastAPI, do not edit directly)

## Rebuild Frontend

```bash
cd web
npm install
npm run build
```

## picamera2 Note
picamera2 is installed by default on Raspberry Pi OS.
If missing: `sudo apt install python3-picamera2`
