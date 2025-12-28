#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import os

os.chdir(r'c:\Users\王照旭\Desktop\CODE\archon_algorithm')

cmd = [
    sys.executable,
    'scripts/test_dual_stream.py',
    '--rgb-video', 'data/test_videos/rgb_test.mp4',
    '--thermal-video', 'data/test_videos/thermal_test.mp4',
    '--rgb-model', 'weights/yolo11n.pt',
    '--thermal-model', 'weights/yolo11n.pt'
]

print("Starting dual-stream PTZ test...")
print(f"Command: {' '.join(cmd)}\n")
subprocess.run(cmd)
