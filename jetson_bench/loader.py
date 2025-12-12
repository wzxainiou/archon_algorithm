"""
Input source loader for images, videos, and camera streams.
"""

import cv2
from pathlib import Path
from typing import Iterator, Tuple, Optional
import numpy as np


class SourceLoader:
    """Unified loader for different input sources."""

    def __init__(self, source_type: str, source_path: Optional[str], max_frames: int = 300):
        self.source_type = source_type
        self.source_path = source_path
        self.max_frames = max_frames
        self.frame_count = 0
        self._cap = None
        self._image_files = []
        self._image_idx = 0

    def __enter__(self):
        """Initialize the source."""
        if self.source_type == "image_dir":
            path = Path(self.source_path)
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            self._image_files = sorted([
                f for f in path.iterdir()
                if f.suffix.lower() in valid_exts
            ])
            if not self._image_files:
                raise ValueError(f"No valid images found in {self.source_path}")

        elif self.source_type == "video":
            self._cap = cv2.VideoCapture(self.source_path)
            if not self._cap.isOpened():
                raise ValueError(f"Failed to open video: {self.source_path}")

        elif self.source_type == "camera":
            camera_idx = int(self.source_path) if self.source_path else 0
            self._cap = cv2.VideoCapture(camera_idx)
            if not self._cap.isOpened():
                raise ValueError(f"Failed to open camera: {camera_idx}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        if self._cap is not None:
            self._cap.release()

    def __iter__(self) -> Iterator[Tuple[np.ndarray, str]]:
        """Iterate over frames."""
        return self

    def __next__(self) -> Tuple[np.ndarray, str]:
        """Get next frame."""
        if self.frame_count >= self.max_frames:
            raise StopIteration

        if self.source_type == "image_dir":
            if self._image_idx >= len(self._image_files):
                raise StopIteration

            img_path = self._image_files[self._image_idx]
            frame = cv2.imread(str(img_path))
            if frame is None:
                # Skip corrupted images
                self._image_idx += 1
                return self.__next__()

            self._image_idx += 1
            self.frame_count += 1
            return frame, img_path.name

        elif self.source_type in ["video", "camera"]:
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration

            self.frame_count += 1
            frame_name = f"frame_{self.frame_count:06d}"
            return frame, frame_name

        raise StopIteration

    def get_total_frames(self) -> Optional[int]:
        """Get total number of frames if known."""
        if self.source_type == "image_dir":
            return min(len(self._image_files), self.max_frames)
        elif self.source_type == "video":
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return min(total, self.max_frames)
        else:
            # Camera stream - unknown
            return self.max_frames

    def get_source_info(self) -> dict:
        """Get information about the source."""
        info = {
            "type": self.source_type,
            "path": self.source_path,
            "max_frames": self.max_frames,
        }

        if self.source_type == "image_dir":
            info["total_images"] = len(self._image_files)
        elif self.source_type == "video" and self._cap:
            info["fps"] = self._cap.get(cv2.CAP_PROP_FPS)
            info["width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["total_frames"] = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif self.source_type == "camera" and self._cap:
            info["fps"] = self._cap.get(cv2.CAP_PROP_FPS)
            info["width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return info
