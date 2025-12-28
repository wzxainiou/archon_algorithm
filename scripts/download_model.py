#!/usr/bin/env python3
"""
Download YOLO11n model weights.

This script downloads the yolo11n.pt model from Ultralytics
and saves it to the weights/ directory.

Usage:
    python scripts/download_model.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        print("Run: pip install ultralytics")
        sys.exit(1)

    # Create weights directory
    weights_dir = Path(__file__).parent.parent / "weights"
    weights_dir.mkdir(exist_ok=True)

    target_path = weights_dir / "yolo11n.pt"

    print("Downloading YOLO11n model...")
    print(f"Target: {target_path}")

    # Download model (Ultralytics will cache it)
    model = YOLO("yolo11n.pt")

    # Get the actual downloaded path
    import shutil
    source_path = Path(model.ckpt_path)

    if source_path.exists():
        # Copy to weights directory if not already there
        if not target_path.exists() or source_path != target_path:
            shutil.copy2(source_path, target_path)
            print(f"Model copied to: {target_path}")
        else:
            print(f"Model already exists at: {target_path}")

        # Verify
        print(f"\nModel info:")
        print(f"  Path: {target_path}")
        print(f"  Size: {target_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  Task: {model.task}")

        print("\n✅ Download complete!")
        print("\nYou can now run:")
        print(f"  python scripts/test_inference_visual.py --mode rgb --source <video> --model {target_path}")
    else:
        print(f"Error: Could not find downloaded model at {source_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
