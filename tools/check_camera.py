#!/usr/bin/env python3
"""
tools/check_camera.py — Diagnostic & Troubleshooting Utility for Raspberry Pi 5 Camera
Run this on your Raspberry Pi 5 terminal:
    python3 tools/check_camera.py
"""

import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

# Suppress noisy OpenCV backend warnings during multi-index probe
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_cmd(cmd_list):
    """Run a system command and return (returncode, stdout, stderr)."""
    try:
        res = subprocess.run(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8
        )
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except FileNotFoundError:
        return -1, "", "Command not found"
    except Exception as e:
        return -1, "", str(e)


def check_system_info():
    print_banner("1. SYSTEM & PLATFORM")
    # Model
    model = "Unknown Device"
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            model = model_path.read_text().strip().replace('\x00', '')
        except Exception:
            pass
    print(f"[*] Hardware Model : {model}")

    # Kernel & OS
    _, kernel, _ = run_cmd(["uname", "-r"])
    print(f"[*] Kernel Version : {kernel or 'Unknown'}")

    os_info = "Unknown Linux"
    os_release = Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_info = line.split("=", 1)[1].strip('"')
                break
    print(f"[*] OS Distro      : {os_info}")

    # User groups
    import grp
    user = os.environ.get("USER", "current_user")
    user_groups = []
    try:
        for g in grp.getgrall():
            if user in g.gr_mem:
                user_groups.append(g.gr_name)
    except Exception:
        pass
    print(f"[*] User '{user}' groups : {', '.join(user_groups) if user_groups else 'Standard'}")
    if "video" not in user_groups and os.geteuid() != 0:
        print("    [!] Warning: User is not in 'video' group. You may need:")
        print(f"        sudo usermod -a -G video,render {user}")


def check_boot_config():
    print_banner("2. BOOT CONFIGURATION (/boot/firmware/config.txt)")
    config_paths = [
        Path("/boot/firmware/config.txt"),  # Debian Bookworm / Pi 5 standard
        Path("/boot/config.txt")            # Legacy Bullseye
    ]
    cfg_file = None
    for p in config_paths:
        if p.exists():
            cfg_file = p
            break

    if not cfg_file:
        print("[-] config.txt not found in standard paths.")
        return

    print(f"[*] Located config file: {cfg_file}")
    content = cfg_file.read_text()

    auto_detect = None
    overlays = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "camera_auto_detect" in line:
            auto_detect = line
        if "dtoverlay" in line and any(k in line.lower() for k in ["cam", "imx", "ov", "arducam"]):
            overlays.append(line)

    print(f"[*] camera_auto_detect setting : {auto_detect or 'Default (1 / enabled)'}")
    if overlays:
        print("[*] Active camera dtoverlays:")
        for ov in overlays:
            print(f"    - {ov}")
    else:
        print("[*] No custom camera dtoverlays configured.")
        print("    Note: OV5647 (standard IR camera) is NOT auto-detected on Pi 5 and requires:")
        print("    dtoverlay=ov5647,cam1   (or cam0)")


def check_hardware_csi():
    print_banner("3. HARDWARE CSI CAMERA (rpicam-apps / libcamera)")
    # Check tool availability
    tool = None
    for t in ["rpicam-hello", "libcamera-hello"]:
        if shutil.which(t):
            tool = t
            break

    if not tool:
        print("[-] Neither 'rpicam-hello' nor 'libcamera-hello' found in PATH.")
        print("    Install via: sudo apt install -y rpicam-apps")
        return False, "rpicam-apps not installed"

    print(f"[*] Testing sensor detection via '{tool} --list-cameras'...")
    rc, stdout, stderr = run_cmd([tool, "--list-cameras"])
    output = f"{stdout}\n{stderr}".strip()

    if "No cameras available" in output or not stdout:
        print("[-] RESULT: 'No cameras available'")
        print("    -> Raspberry Pi 5 kernel does not detect any camera on CSI ports.")
        print("    -> Causes:")
        print("       1. Ribbon cable inserted backwards or into wrong port.")
        print("          (Pi 5 has 22-pin miniature CAM0 and CAM1. Contacts must face correctly).")
        print("       2. Camera sensor (like OV5647 NoIR) requires explicit overlay in config.txt.")
        print("       3. Ribbon cable damaged or not seated firmly in the connector latch.")
        return False, "No CSI cameras detected by libcamera"
    else:
        print("[+] RESULT: Camera sensor detected by kernel!")
        for line in stdout.splitlines():
            print(f"    {line}")
        return True, stdout


def check_v4l2_and_usb():
    print_banner("4. V4L2 DEVICE NODES & USB DEVICES")
    # /dev/video*
    nodes = sorted(glob.glob("/dev/video*"))
    if nodes:
        print(f"[*] Found {len(nodes)} V4L2 device nodes:")
        for node in nodes:
            readable = os.access(node, os.R_OK)
            writable = os.access(node, os.W_OK)
            perms = f"R:{readable} W:{writable}"
            print(f"    - {node} ({perms})")
    else:
        print("[-] No /dev/video* nodes found.")

    # lsusb
    if shutil.which("lsusb"):
        rc, stdout, _ = run_cmd(["lsusb"])
        if rc == 0 and stdout:
            print("\n[*] USB Bus devices:")
            for line in stdout.splitlines():
                if any(kw in line.lower() for kw in ["cam", "video", "imaging", "sony", "logitech"]):
                    print(f"    [+] Possible Camera: {line}")
                else:
                    print(f"    - {line}")

    # v4l2-ctl
    if shutil.which("v4l2-ctl"):
        rc, stdout, _ = run_cmd(["v4l2-ctl", "--list-devices"])
        if rc == 0 and stdout:
            print("\n[*] v4l2-ctl device mapping:")
            for line in stdout.splitlines():
                print(f"    {line}")


def check_python_environment():
    print_banner("5. PYTHON ENVIRONMENT & LIBRARIES")
    in_venv = sys.prefix != sys.base_prefix
    print(f"[*] Python Executable : {sys.executable}")
    print(f"[*] Virtualenv Active : {in_venv} (Prefix: {sys.prefix})")

    # Check system site-packages
    system_packages_accessible = any("dist-packages" in p for p in sys.path)
    print(f"[*] System dist-packages accessible: {system_packages_accessible}")
    if in_venv and not system_packages_accessible:
        print("    [!] CRITICAL: Your virtual environment was created WITHOUT --system-site-packages.")
        print("        Raspberry Pi 5 camera bindings ('picamera2') are installed via system APT.")
        print("        Isolated virtualenvs cannot see 'picamera2' unless created with:")
        print("        python3 -m venv --system-site-packages .venv")

    # Picamera2
    print("\n[*] Testing 'import picamera2'...")
    picam2_ok = False
    try:
        from picamera2 import Picamera2
        print("[+] 'import picamera2' succeeded.")
        try:
            p = Picamera2()
            print("[+] Picamera2 instance created successfully!")
            p.close()
            picam2_ok = True
        except Exception as e:
            print(f"[-] Picamera2() failed to initialize hardware: {e}")
    except ImportError as e:
        print(f"[-] Failed to import picamera2: {e}")

    # OpenCV
    print("\n[*] Testing 'import cv2'...")
    cv2_ok = False
    try:
        import cv2
        try:
            cv2.setLogLevel(0)
        except Exception:
            pass
        print(f"[+] OpenCV version {cv2.__version__} loaded.")
        cv2_ok = True

        # Probe video nodes
        print("[*] Probing OpenCV VideoCapture on indices 0 through 7...")
        working_indices = []
        for idx in range(8):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    mean_val = frame.mean()
                    print(f"    [+] /dev/video{idx}: Readable ({w}x{h}, mean brightness: {mean_val:.1f}/255)")
                    working_indices.append(idx)
                else:
                    print(f"    [-] /dev/video{idx}: Opened but could not read frame")
                cap.release()
            else:
                pass
        if not working_indices:
            print("    [-] No OpenCV VideoCapture device was able to read a frame.")
    except ImportError as e:
        print(f"[-] Failed to import cv2: {e}")

    return picam2_ok, cv2_ok


def print_summary_and_next_steps(csi_detected, picam2_ok):
    print_banner("6. DIAGNOSTIC SUMMARY & ACTIONABLE FIXES")

    if picam2_ok:
        print("[✓] SUCCESS: Picamera2 is operational!")
        print("    You can start the server with: python3 -m uvicorn app.server:app --host 0.0.0.0 --port 8000")
        print("    Or test live stream with:      python3 tools/cam_test.py --view")
        return

    print("[!] ACTION REQUIRED TO FIX CAMERA:\n")

    if csi_detected and not picam2_ok:
        print("--- SITUATION A: Hardware is detected by OS, but Python cannot use it ---")
        print("Your Raspberry Pi OS sees the camera, but your Python environment lacks Picamera2.")
        print("Run these exact commands to fix your virtual environment:")
        print("  deactivate 2>/dev/null || true")
        print("  sudo apt update && sudo apt install -y python3-picamera2 python3-libcamera")
        print("  rm -rf .venv")
        print("  python3 -m venv --system-site-packages .venv")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt\n")

    elif not csi_detected:
        print("--- SITUATION B: Kernel does NOT see the camera (Hardware / Config issue) ---")
        print("1. Check the ribbon cable:")
        print("   - Pi 5 uses 22-pin miniature connectors (CAM/DISP0 and CAM/DISP1).")
        print("   - Make sure the contacts face the correct direction and the latch is locked.")
        print("   - Try plugging into CAM1 if currently in CAM0, or vice versa.")
        print("2. For OV5647 (standard night-vision IR cameras with LEDs):")
        print("   - Raspberry Pi 5 does NOT auto-detect OV5647 sensors.")
        print("   - Edit config: sudo nano /boot/firmware/config.txt")
        print("   - Add at the bottom:")
        print("       dtoverlay=ov5647,cam1")
        print("     (or dtoverlay=ov5647,cam0 if using port 0)")
        print("   - Save (Ctrl+O, Enter, Ctrl+X) and reboot: sudo reboot")
        print("3. For Raspberry Pi Camera Module 2 (IMX219) or Module 3 (IMX708):")
        print("   - If auto-detect fails, add:")
        print("       dtoverlay=imx219,cam1   # (for Camera V2)")
        print("       dtoverlay=imx708,cam1   # (for Camera V3)")
        print("   - Reboot: sudo reboot\n")

    print("After applying the fix, run this diagnostic again:")
    print("    python3 tools/check_camera.py")
    print("=" * 65 + "\n")


def main():
    check_system_info()
    check_boot_config()
    csi_detected, _ = check_hardware_csi()
    check_v4l2_and_usb()
    picam2_ok, _ = check_python_environment()
    print_summary_and_next_steps(csi_detected, picam2_ok)


if __name__ == "__main__":
    main()
