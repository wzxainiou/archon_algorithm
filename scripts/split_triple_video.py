#!/usr/bin/env python3
"""
Split triple-panel video into separate RGB and Thermal videos.

Input: Video with 3 side-by-side panels [RGB | Thermal | Grayscale]
Output: Two separate videos - rgb_test.mp4 and thermal_test.mp4

Usage:
    python scripts/split_triple_video.py data/test_videos/2022-05-08-11-23-59.mp4

    # Custom output names
    python scripts/split_triple_video.py input.mp4 --rgb_out rgb.mp4 --thermal_out thermal.mp4
"""

import argparse
import cv2
import sys
from pathlib import Path


def split_triple_video(
    input_path: str,
    rgb_output: str = None,
    thermal_output: str = None,
    show_preview: bool = False
):
    """
    Split a triple-panel video into RGB and Thermal videos.

    Args:
        input_path: Path to input video with 3 side-by-side panels
        rgb_output: Output path for RGB video (default: same dir as input, rgb_test.mp4)
        thermal_output: Output path for Thermal video (default: same dir as input, thermal_test.mp4)
        show_preview: Show preview window while processing
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    # Default output paths
    output_dir = input_path.parent
    if rgb_output is None:
        rgb_output = output_dir / "rgb_test.mp4"
    if thermal_output is None:
        thermal_output = output_dir / "thermal_test.mp4"

    rgb_output = Path(rgb_output)
    thermal_output = Path(thermal_output)

    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate panel width (3 equal panels)
    panel_width = total_width // 3

    print("=" * 60)
    print("Triple Video Splitter")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Total size: {total_width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames / fps:.1f} seconds")
    print(f"\nPanel width: {panel_width} pixels each")
    print(f"  Left (RGB):     0 - {panel_width}")
    print(f"  Center (Thermal): {panel_width} - {panel_width * 2}")
    print(f"  Right (Discard):  {panel_width * 2} - {total_width}")
    print(f"\nOutput:")
    print(f"  RGB:     {rgb_output}")
    print(f"  Thermal: {thermal_output}")
    print("=" * 60)

    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rgb_writer = cv2.VideoWriter(str(rgb_output), fourcc, fps, (panel_width, height))
    thermal_writer = cv2.VideoWriter(str(thermal_output), fourcc, fps, (panel_width, height))

    if not rgb_writer.isOpened() or not thermal_writer.isOpened():
        print("Error: Cannot create output video writers")
        return False

    if show_preview:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 1280, 360)

    frame_count = 0
    print("\nProcessing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Split frame into 3 panels
        rgb_panel = frame[:, 0:panel_width]
        thermal_panel = frame[:, panel_width:panel_width * 2]
        # grayscale_panel = frame[:, panel_width * 2:]  # Discarded

        # Write to output videos
        rgb_writer.write(rgb_panel)
        thermal_writer.write(thermal_panel)

        frame_count += 1

        # Progress update
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")

        # Preview
        if show_preview:
            preview = cv2.hconcat([rgb_panel, thermal_panel])
            cv2.putText(preview, "RGB", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(preview, "THERMAL", (panel_width + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Preview", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser interrupted")
                break

    # Cleanup
    cap.release()
    rgb_writer.release()
    thermal_writer.release()

    if show_preview:
        cv2.destroyAllWindows()

    # Verify outputs
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

    for path, name in [(rgb_output, "RGB"), (thermal_output, "Thermal")]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {name}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  {name}: FAILED")

    print(f"\nProcessed {frame_count} frames")
    print(f"Output resolution: {panel_width}x{height} @ {fps:.1f} FPS")

    print("\nYou can now run:")
    print(f"  python scripts/test_inference_visual.py --mode dual \\")
    print(f"      --rgb_source {rgb_output} \\")
    print(f"      --thermal_source {thermal_output} \\")
    print(f"      --model weights/yolo11n.pt")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split triple-panel video into RGB and Thermal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/split_triple_video.py data/test_videos/2022-05-08-11-23-59.mp4

The input video should have 3 side-by-side panels:
    [RGB | Thermal | Grayscale]

Only RGB (left) and Thermal (center) are kept.
        """
    )

    parser.add_argument("input", help="Input video path (triple-panel format)")
    parser.add_argument("--rgb_out", help="Output path for RGB video")
    parser.add_argument("--thermal_out", help="Output path for Thermal video")
    parser.add_argument("--preview", action="store_true",
                       help="Show preview window while processing")

    args = parser.parse_args()

    success = split_triple_video(
        args.input,
        args.rgb_out,
        args.thermal_out,
        args.preview
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
