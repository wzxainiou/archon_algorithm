#!/usr/bin/env python3
"""
Download KAIST RGB-Thermal sample and convert to video.

KAIST Multispectral Pedestrian Dataset:
- 20 FPS, 640x480 resolution
- RGB + Thermal paired images
- Pedestrian detection scenario (similar to wildlife)

This script downloads a sample subset and creates paired test videos.

Usage:
    python scripts/download_kaist_sample.py
"""

import sys
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        from datasets import load_dataset
    except ImportError:
        missing.append("datasets")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("Missing dependencies. Please install:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def download_kaist_sample(num_frames: int = 200):
    """
    Download KAIST sample from Hugging Face and convert to video.

    Args:
        num_frames: Number of frames to include (default: 200 = 10 seconds at 20 FPS)
    """
    import cv2
    from datasets import load_dataset
    from PIL import Image
    import numpy as np

    print("=" * 60)
    print("KAIST RGB-Thermal Dataset Downloader")
    print("=" * 60)
    print(f"\nDownloading {num_frames} paired frames...")
    print("This may take a few minutes on first run.\n")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "test_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_video_path = output_dir / "rgb_test.mp4"
    thermal_video_path = output_dir / "thermal_test.mp4"

    # Load dataset from Hugging Face
    print("Loading KAIST dataset from Hugging Face...")
    print("(First download: ~4.29 GB, cached for future use)\n")

    try:
        dataset = load_dataset(
            "richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset",
            split="train",
            streaming=True  # Stream to avoid downloading entire dataset
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://huggingface.co/datasets/richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset")
        return False

    # Collect frames
    print("Collecting frames...")
    rgb_frames = []
    thermal_frames = []

    for i, sample in enumerate(dataset):
        if i >= num_frames:
            break

        # Get images from sample
        rgb_img = sample.get("visible_image") or sample.get("image")
        thermal_img = sample.get("thermal_image") or sample.get("lwir_image")

        if rgb_img is None or thermal_img is None:
            # Try alternative field names
            if "visible" in sample:
                rgb_img = sample["visible"]
            if "lwir" in sample:
                thermal_img = sample["lwir"]

        if rgb_img is None or thermal_img is None:
            print(f"  Skipping sample {i}: missing image fields")
            continue

        # Convert PIL Image to numpy array
        if isinstance(rgb_img, Image.Image):
            rgb_np = np.array(rgb_img)
        else:
            rgb_np = rgb_img

        if isinstance(thermal_img, Image.Image):
            thermal_np = np.array(thermal_img)
        else:
            thermal_np = thermal_img

        # Convert RGB to BGR for OpenCV
        if len(rgb_np.shape) == 3 and rgb_np.shape[2] == 3:
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        # Convert thermal to 3-channel if grayscale
        if len(thermal_np.shape) == 2:
            thermal_np = cv2.cvtColor(thermal_np, cv2.COLOR_GRAY2BGR)
        elif thermal_np.shape[2] == 3:
            thermal_np = cv2.cvtColor(thermal_np, cv2.COLOR_RGB2BGR)

        rgb_frames.append(rgb_np)
        thermal_frames.append(thermal_np)

        if (i + 1) % 50 == 0:
            print(f"  Collected {i + 1}/{num_frames} frames...")

    if not rgb_frames or not thermal_frames:
        print("Error: No frames collected!")
        print("\nDataset field names may have changed.")
        print("Please check the dataset structure on Hugging Face.")
        return False

    print(f"\nCollected {len(rgb_frames)} paired frames.")

    # Get frame dimensions
    h, w = rgb_frames[0].shape[:2]
    fps = 20  # KAIST native FPS

    print(f"\nFrame info: {w}x{h} @ {fps} FPS")
    print(f"Video duration: {len(rgb_frames) / fps:.1f} seconds")

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print(f"\nWriting RGB video: {rgb_video_path}")
    rgb_writer = cv2.VideoWriter(str(rgb_video_path), fourcc, fps, (w, h))

    print(f"Writing Thermal video: {thermal_video_path}")
    thermal_writer = cv2.VideoWriter(str(thermal_video_path), fourcc, fps, (w, h))

    # Write frames
    for i, (rgb_frame, thermal_frame) in enumerate(zip(rgb_frames, thermal_frames)):
        # Ensure same size
        if rgb_frame.shape[:2] != (h, w):
            rgb_frame = cv2.resize(rgb_frame, (w, h))
        if thermal_frame.shape[:2] != (h, w):
            thermal_frame = cv2.resize(thermal_frame, (w, h))

        rgb_writer.write(rgb_frame)
        thermal_writer.write(thermal_frame)

    rgb_writer.release()
    thermal_writer.release()

    # Verify outputs
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)

    for path in [rgb_video_path, thermal_video_path]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {path.name}: {size_mb:.2f} MB")
        else:
            print(f"  {path.name}: FAILED")

    print("\nYou can now run:")
    print(f"  python scripts/test_inference_visual.py --mode dual \\")
    print(f"      --rgb_source {rgb_video_path} \\")
    print(f"      --thermal_source {thermal_video_path} \\")
    print(f"      --model weights/yolo11n.pt")

    return True


def create_sample_from_single_video(video_path: str, num_frames: int = 200):
    """
    Alternative: Create test videos from a single video file.

    This is useful when you don't have access to Hugging Face
    or want to quickly test the code logic.

    Args:
        video_path: Path to source video
        num_frames: Number of frames to extract
    """
    import cv2

    output_dir = Path(__file__).parent.parent / "data" / "test_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Source: {w}x{h} @ {fps:.1f} FPS")

    # Create writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rgb_writer = cv2.VideoWriter(
        str(output_dir / "rgb_test.mp4"), fourcc, fps, (w, h)
    )
    thermal_writer = cv2.VideoWriter(
        str(output_dir / "thermal_test.mp4"), fourcc, fps, (w, h)
    )

    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Write same frame as RGB
        rgb_writer.write(frame)

        # Create pseudo-thermal (grayscale colormap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        thermal_writer.write(thermal)

        frame_count += 1

    cap.release()
    rgb_writer.release()
    thermal_writer.release()

    print(f"\nCreated {frame_count} frame test videos from: {video_path}")
    print(f"  RGB: {output_dir / 'rgb_test.mp4'}")
    print(f"  Thermal: {output_dir / 'thermal_test.mp4'}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download KAIST RGB-Thermal sample videos"
    )
    parser.add_argument(
        "--frames", type=int, default=200,
        help="Number of frames to download (default: 200 = 10 sec at 20 FPS)"
    )
    parser.add_argument(
        "--from-video", type=str, default=None,
        help="Alternative: Create test videos from existing video file"
    )

    args = parser.parse_args()

    if args.from_video:
        # Alternative: Use existing video
        if not Path(args.from_video).exists():
            print(f"Error: Video not found: {args.from_video}")
            sys.exit(1)
        success = create_sample_from_single_video(args.from_video, args.frames)
    else:
        # Main: Download from Hugging Face
        if not check_dependencies():
            sys.exit(1)
        success = download_kaist_sample(args.frames)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
