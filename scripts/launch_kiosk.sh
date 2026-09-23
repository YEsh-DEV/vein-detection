#!/usr/bin/env bash
# scripts/launch_kiosk.sh
# =============================================================================
# Fullscreen Portrait Kiosk Launcher for Raspberry Pi
# Supports logical viewports such as 720x1280 and 480x800.
# =============================================================================

set -e

PORT=${1:-8000}
URL="http://127.0.0.1:${PORT}"

echo "[*] Launching Palm-Vein Biometrics Portrait Kiosk on ${URL}..."

# Wait for backend health endpoint to respond
MAX_WAIT=30
WAITED=0
until curl -s "${URL}/health" | grep -q "healthy"; do
    echo "    Waiting for backend to be online... (${WAITED}s/${MAX_WAIT}s)"
    sleep 1
    WAITED=$((WAITED + 1))
    if [ ${WAITED} -ge ${MAX_WAIT} ]; then
        echo "[!] ERROR: Backend did not start within ${MAX_WAIT} seconds."
        exit 1
    fi
done

echo "[+] Backend is healthy. Starting Chromium in portrait kiosk mode..."

# Chromium flags:
# - Fullscreen kiosk without address bar or tabs
# - Target portrait resolution 720x1280 (or 480x800 depending on screen)
# - Disable touch zoom/pinch, infobars, crash bubbles, and translation prompts
# - Dedicated temporary profile to prevent session restore popups
chromium-browser \
    --kiosk \
    --app="${URL}" \
    --window-size=720,1280 \
    --window-position=0,0 \
    --noerrdialogs \
    --disable-infobars \
    --check-for-update-interval=31536000 \
    --disable-pinch \
    --overscroll-history-navigation=0 \
    --disable-features=TranslateUI \
    --user-data-dir=/tmp/chromium-kiosk-profile \
    "$@"
