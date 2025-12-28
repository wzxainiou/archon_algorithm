"""
Dual-stream video/camera loader for synchronized RGB and Thermal input.
"""

import cv2
import time
from pathlib import Path
from typing import Literal, Iterator, Dict
import logging

logger = logging.getLogger(__name__)


class DualSourceLoader:
    """Dual-stream loader for RGB and Thermal sources (cameras or videos)."""

    def __init__(self,
                 source_type: Literal["camera", "video"] = "camera",
                 rgb_source: str = "0",
                 thermal_source: str = "1",
                 target_fps: float = 5.0,
                 max_frames: int = 300,
                 sync_tolerance_ms: float = 100.0):
        """
        Initialize dual source loader.

        Args:
            source_type: "camera" or "video"
            rgb_source:
                - If source_type="camera": device ID (e.g., "0")
                - If source_type="video": path to RGB video file
            thermal_source:
                - If source_type="camera": device ID (e.g., "1")
                - If source_type="video": path to thermal video file
            target_fps: Desired FPS for processing (FIXED at 5.0)
            max_frames: Maximum frames to process
            sync_tolerance_ms: Max allowed time difference between streams
        """
        self.source_type = source_type
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.sync_tolerance_ms = sync_tolerance_ms
        self.frame_interval = 1.0 / target_fps  # ~0.2s for 5 FPS

        # Open sources
        if source_type == "camera":
            # Open camera devices
            self.rgb_cap = cv2.VideoCapture(int(rgb_source))
            self.thermal_cap = cv2.VideoCapture(int(thermal_source))
            logger.info(f"Opened cameras: RGB={rgb_source}, Thermal={thermal_source}")
        elif source_type == "video":
            # Open video files
            rgb_path = Path(rgb_source)
            thermal_path = Path(thermal_source)

            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB video not found: {rgb_source}")
            if not thermal_path.exists():
                raise FileNotFoundError(f"Thermal video not found: {thermal_source}")

            self.rgb_cap = cv2.VideoCapture(str(rgb_path))
            self.thermal_cap = cv2.VideoCapture(str(thermal_path))
            logger.info(f"Opened videos: RGB={rgb_source}, Thermal={thermal_source}")
        else:
            raise ValueError(f"Invalid source_type: {source_type}")

        # Check if opened successfully
        if not self.rgb_cap.isOpened():
            raise RuntimeError(f"Failed to open RGB source: {rgb_source}")
        if not self.thermal_cap.isOpened():
            raise RuntimeError(f"Failed to open thermal source: {thermal_source}")

        # Get native FPS
        self.rgb_native_fps = self.rgb_cap.get(cv2.CAP_PROP_FPS)
        self.thermal_native_fps = self.thermal_cap.get(cv2.CAP_PROP_FPS)

        # For video files, use frame skipping strategy
        if source_type == "video":
            self.rgb_frame_skip = max(1, int(self.rgb_native_fps / target_fps))
            self.thermal_frame_skip = max(1, int(self.thermal_native_fps / target_fps))
            logger.info(f"Video mode: RGB skip={self.rgb_frame_skip} frames, Thermal skip={self.thermal_frame_skip} frames")
            logger.info(f"  RGB FPS: {self.rgb_native_fps:.1f} -> {target_fps} FPS")
            logger.info(f"  Thermal FPS: {self.thermal_native_fps:.1f} -> {target_fps} FPS")
        else:
            # For cameras, use time-based sampling
            self.rgb_frame_skip = None
            self.thermal_frame_skip = None
            logger.info(f"Camera mode: time-based sampling at {target_fps} FPS")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release resources."""
        if self.rgb_cap is not None:
            self.rgb_cap.release()
        if self.thermal_cap is not None:
            self.thermal_cap.release()
        logger.info("Released video/camera resources")

    def __iter__(self) -> Iterator[Dict]:
        """Yield synchronized frame pairs."""
        frame_count = 0
        last_process_time = 0
        rgb_skip_counter = 0
        thermal_skip_counter = 0

        while self.max_frames is None or frame_count < self.max_frames:
            if self.source_type == "camera":
                # Time-based sampling for cameras
                current_time = time.time()

                if current_time - last_process_time < self.frame_interval:
                    # Skip frames to maintain target FPS
                    self.rgb_cap.grab()
                    self.thermal_cap.grab()
                    continue

                # Read both frames
                ret_rgb, rgb_frame = self.rgb_cap.read()
                ret_thermal, thermal_frame = self.thermal_cap.read()

                last_process_time = current_time

            else:  # source_type == "video"
                # Frame-skipping strategy for video files with independent counters
                rgb_skip_counter += 1
                thermal_skip_counter += 1

                # Determine if we should read this frame for each stream
                should_read_rgb = rgb_skip_counter >= self.rgb_frame_skip
                should_read_thermal = thermal_skip_counter >= self.thermal_frame_skip

                # Only yield when BOTH streams are ready to read
                if not (should_read_rgb and should_read_thermal):
                    # Skip frames for streams that need skipping
                    if not should_read_rgb:
                        ret_rgb_skip = self.rgb_cap.grab()
                        if not ret_rgb_skip:
                            logger.info(f"RGB stream ended at frame {frame_count}")
                            break
                    if not should_read_thermal:
                        ret_thermal_skip = self.thermal_cap.grab()
                        if not ret_thermal_skip:
                            logger.info(f"Thermal stream ended at frame {frame_count}")
                            break
                    continue

                # Reset counters
                rgb_skip_counter = 0
                thermal_skip_counter = 0

                # Read both frames - decode them synchronously
                ret_rgb, rgb_frame = self.rgb_cap.read()
                ret_thermal, thermal_frame = self.thermal_cap.read()

            # Check if both frames read successfully
            if not (ret_rgb and ret_thermal):
                logger.info(f"End of stream reached at frame {frame_count}")
                break

            frame_id = f"frame_{frame_count:06d}"
            frame_count += 1

            yield {
                "rgb_frame": rgb_frame,
                "thermal_frame": thermal_frame,
                "frame_id": frame_id,
                "timestamp": time.time(),
            }

    def get_source_info(self) -> dict:
        """Get information about the sources."""
        return {
            "source_type": self.source_type,
            "rgb_fps": self.rgb_native_fps,
            "thermal_fps": self.thermal_native_fps,
            "target_fps": self.target_fps,
            "rgb_width": int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "rgb_height": int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "thermal_width": int(self.thermal_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "thermal_height": int(self.thermal_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
