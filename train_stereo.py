#!/usr/bin/env python3
"""
Quick start script for training Stereo I-JEPA on FlyingThings3D
Single GPU setup
"""

import subprocess
import sys

# Training command
cmd = [
    sys.executable,  # Python executable
    "main.py",
    "--fname", "configs/flyingthings3d_stereo_naive.yaml",
    "--devices", "cuda:0"  # Single GPU
]

print("=" * 60)
print("Starting Stereo I-JEPA Training")
print("=" * 60)
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

# Run training
subprocess.run(cmd, cwd=".")
