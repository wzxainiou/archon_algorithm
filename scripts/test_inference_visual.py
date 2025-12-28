#!/usr/bin/env python3
"""
Visual inference test script with real-time display.

Supports three processing modes:
- rgb: RGB stream only
- thermal: Thermal stream only
- dual: Both streams side-by-side

Usage:
    # RGB mode
    python scripts/test_inference_visual.py --mode rgb \
        --source data/test_videos/rgb_test.mp4 \
        --model weights/yolo11n.pt

    # Thermal mode
    python scripts/test_inference_visual.py --mode thermal \
        --source data/test_videos/thermal_test.mp4 \
        --model weights/yolo11n.pt

    # Dual mode (side-by-side display)
    python scripts/test_inference_visual.py --mode dual \
        --rgb_source data/test_videos/rgb_test.mp4 \
        --thermal_source data/test_videos/thermal_test.mp4 \
        --model weights/yolo11n.pt

Press 'q' to quit during playback.
"""

import argparse
import cv2
import time
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def draw_detections(frame, results, color=(0, 255, 0), label_prefix=""):
    """
    Draw detection boxes on frame.

    Args:
        frame: Input frame (modified in-place)
        results: YOLO inference results
        color: Box color (B, G, R)
        label_prefix: Optional prefix for labels

    Returns:
        Modified frame with drawn detections
    """
    if len(results) == 0 or results[0].boxes is None:
        return frame

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = results[0].names[cls]
        label = f"{label_prefix}{class_name} {conf:.2f}"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def draw_info_overlay(frame, fps, latency_ms, num_detections, mode_label):
    """
    Draw information overlay on frame.

    Args:
        frame: Input frame
        fps: Frames per second
        latency_ms: Inference latency in milliseconds
        num_detections: Number of detections
        mode_label: Mode label to display
    """
    h, w = frame.shape[:2]

    # Semi-transparent background for info
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw text
    cv2.putText(frame, f"Mode: {mode_label}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {num_detections}", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def create_side_by_side(rgb_frame, thermal_frame):
    """
    Create side-by-side view of RGB and Thermal frames.

    Args:
        rgb_frame: RGB camera frame
        thermal_frame: Thermal camera frame

    Returns:
        Combined frame with both views side by side
    """
    h1, w1 = rgb_frame.shape[:2]
    h2, w2 = thermal_frame.shape[:2]

    # Resize to same height
    target_h = max(h1, h2)
    if h1 != target_h:
        scale = target_h / h1
        rgb_frame = cv2.resize(rgb_frame, (int(w1 * scale), target_h))
    if h2 != target_h:
        scale = target_h / h2
        thermal_frame = cv2.resize(thermal_frame, (int(w2 * scale), target_h))

    # Draw stream labels
    cv2.putText(rgb_frame, "RGB", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(thermal_frame, "THERMAL", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Add vertical separator line
    rgb_frame[:, -2:] = [255, 255, 255]

    # Concatenate horizontally
    combined = np.hstack([rgb_frame, thermal_frame])
    return combined


def run_rgb_mode(source: str, model_path: str, conf: float = 0.25):
    """
    Run RGB-only mode with visualization.

    Args:
        source: Video file path or camera index
        model_path: Path to YOLO model weights
        conf: Confidence threshold
    """
    print(f"\n[RGB Mode] Loading model: {model_path}")
    model = YOLO(model_path)

    # Open video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source: {source}")
        return

    # Get video info
    fps_native = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {width}x{height} @ {fps_native:.1f} FPS")

    cv2.namedWindow("RGB Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB Inference", 1280, 720)

    frame_count = 0
    fps_smooth = 0

    print("Press 'q' to quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        start = time.time()
        results = model(frame, conf=conf, verbose=False)
        latency = (time.time() - start) * 1000

        # Count detections
        num_det = len(results[0].boxes) if results[0].boxes is not None else 0

        # Draw detections (green for RGB)
        frame = draw_detections(frame, results, color=(0, 255, 0))

        # Calculate smoothed FPS
        fps = 1000 / latency if latency > 0 else 0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        # Draw info overlay
        draw_info_overlay(frame, fps_smooth, latency, num_det, "RGB")

        cv2.imshow("RGB Inference", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User quit")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


def run_thermal_mode(source: str, model_path: str, conf: float = 0.25):
    """
    Run Thermal-only mode with visualization.

    Args:
        source: Video file path or camera index
        model_path: Path to YOLO model weights
        conf: Confidence threshold
    """
    print(f"\n[Thermal Mode] Loading model: {model_path}")
    model = YOLO(model_path)

    # Open video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source: {source}")
        return

    fps_native = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {width}x{height} @ {fps_native:.1f} FPS")

    cv2.namedWindow("Thermal Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Thermal Inference", 1280, 720)

    frame_count = 0
    fps_smooth = 0

    print("Press 'q' to quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        start = time.time()
        results = model(frame, conf=conf, verbose=False)
        latency = (time.time() - start) * 1000

        num_det = len(results[0].boxes) if results[0].boxes is not None else 0

        # Draw detections (red/orange for Thermal)
        frame = draw_detections(frame, results, color=(0, 100, 255))

        fps = 1000 / latency if latency > 0 else 0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        draw_info_overlay(frame, fps_smooth, latency, num_det, "THERMAL")

        cv2.imshow("Thermal Inference", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User quit")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


def run_dual_mode(rgb_source: str, thermal_source: str, model_path: str, conf: float = 0.25):
    """
    Run dual-stream mode with side-by-side visualization.

    Args:
        rgb_source: RGB video file path
        thermal_source: Thermal video file path
        model_path: Path to YOLO model weights
        conf: Confidence threshold
    """
    print(f"\n[Dual Mode] Loading model: {model_path}")
    model = YOLO(model_path)

    # Open both video sources
    rgb_cap = cv2.VideoCapture(rgb_source)
    thermal_cap = cv2.VideoCapture(thermal_source)

    if not rgb_cap.isOpened():
        print(f"Error: Cannot open RGB video: {rgb_source}")
        return
    if not thermal_cap.isOpened():
        print(f"Error: Cannot open Thermal video: {thermal_source}")
        rgb_cap.release()
        return

    # Get video info
    rgb_w = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    thermal_w = int(thermal_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    thermal_h = int(thermal_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"RGB Source: {rgb_w}x{rgb_h}")
    print(f"Thermal Source: {thermal_w}x{thermal_h}")

    cv2.namedWindow("Dual Stream Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dual Stream Inference", 1920, 540)

    frame_count = 0
    fps_smooth = 0

    print("Press 'q' to quit\n")

    while rgb_cap.isOpened() and thermal_cap.isOpened():
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_thermal, thermal_frame = thermal_cap.read()

        if not (ret_rgb and ret_thermal):
            print("End of video(s)")
            break

        start = time.time()

        # Infer on both frames
        rgb_results = model(rgb_frame, conf=conf, verbose=False)
        thermal_results = model(thermal_frame, conf=conf, verbose=False)

        latency = (time.time() - start) * 1000

        # Count detections
        rgb_det = len(rgb_results[0].boxes) if rgb_results[0].boxes is not None else 0
        thermal_det = len(thermal_results[0].boxes) if thermal_results[0].boxes is not None else 0

        # Draw detections
        rgb_display = draw_detections(rgb_frame.copy(), rgb_results, color=(0, 255, 0))
        thermal_display = draw_detections(thermal_frame.copy(), thermal_results, color=(0, 100, 255))

        # Create side-by-side view
        combined = create_side_by_side(rgb_display, thermal_display)

        # Calculate smoothed FPS
        fps = 1000 / latency if latency > 0 else 0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        # Draw combined info at bottom
        h = combined.shape[0]
        info_text = f"DUAL MODE | FPS: {fps_smooth:.1f} | Latency: {latency:.1f}ms | RGB Det: {rgb_det} | Thermal Det: {thermal_det}"

        # Semi-transparent background
        overlay = combined.copy()
        cv2.rectangle(overlay, (5, h - 35), (len(info_text) * 12, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, combined, 0.5, 0, combined)

        cv2.putText(combined, info_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Dual Stream Inference", combined)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User quit")
            break

    rgb_cap.release()
    thermal_cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frame pairs")


def main():
    parser = argparse.ArgumentParser(
        description="Visual inference test with real-time display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RGB mode
  python scripts/test_inference_visual.py --mode rgb \\
      --source data/test_videos/rgb_test.mp4 \\
      --model weights/yolo11n.pt

  # Thermal mode
  python scripts/test_inference_visual.py --mode thermal \\
      --source data/test_videos/thermal_test.mp4 \\
      --model weights/yolo11n.pt

  # Dual mode (side-by-side)
  python scripts/test_inference_visual.py --mode dual \\
      --rgb_source data/test_videos/rgb_test.mp4 \\
      --thermal_source data/test_videos/thermal_test.mp4 \\
      --model weights/yolo11n.pt

  # Use camera
  python scripts/test_inference_visual.py --mode rgb --source 0 --model weights/yolo11n.pt
        """
    )

    parser.add_argument("--mode", choices=["rgb", "thermal", "dual"],
                       required=True, help="Processing mode")
    parser.add_argument("--source", help="Video source for rgb/thermal mode (file path or camera index)")
    parser.add_argument("--rgb_source", help="RGB video for dual mode")
    parser.add_argument("--thermal_source", help="Thermal video for dual mode")
    parser.add_argument("--model", required=True, help="Model weights path (.pt, .onnx, or .engine)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Validate arguments based on mode
    if args.mode == "dual":
        if not args.rgb_source or not args.thermal_source:
            parser.error("--mode dual requires --rgb_source and --thermal_source")

        if not Path(args.rgb_source).exists():
            print(f"Error: RGB video not found: {args.rgb_source}")
            sys.exit(1)
        if not Path(args.thermal_source).exists():
            print(f"Error: Thermal video not found: {args.thermal_source}")
            sys.exit(1)

        run_dual_mode(args.rgb_source, args.thermal_source, args.model, args.conf)

    else:
        if not args.source:
            parser.error(f"--mode {args.mode} requires --source")

        # Check if source is a file (not camera index)
        if not args.source.isdigit() and not Path(args.source).exists():
            print(f"Error: Video file not found: {args.source}")
            sys.exit(1)

        if args.mode == "rgb":
            run_rgb_mode(args.source, args.model, args.conf)
        else:
            run_thermal_mode(args.source, args.model, args.conf)


if __name__ == "__main__":
    main()
