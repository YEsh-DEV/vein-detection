#!/usr/bin/env python3
"""
Forwarding wrapper for tools/real_data_analysis.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_SCRIPT = os.path.join(PROJECT_ROOT, "tools", "real_data_analysis.py")

if __name__ == "__main__":
    import runpy
    sys.argv[0] = TARGET_SCRIPT
    runpy.run_path(TARGET_SCRIPT, run_name="__main__")
