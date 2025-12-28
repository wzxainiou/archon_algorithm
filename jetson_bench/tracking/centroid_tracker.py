"""
Centroid-based object tracker for hunting camera.

Tracks the largest animal by bounding box area.
"""

from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LargestAnimalTracker:
    """Tracks the largest animal across frames using centroid tracking."""

    def __init__(self, max_disappeared: int = 10):
        """
        Initialize tracker.

        Args:
            max_disappeared: Max frames to keep track when object disappears
        """
        self.tracked_object: Optional[Dict] = None  # {id, bbox, centroid, area, class}
        self.disappeared_count = 0
        self.max_disappeared = max_disappeared
        self.next_id = 0

    def update(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List[Dict] from fuse_detections() with fields:
                - class: str
                - confidence: float
                - bbox: [x1, y1, x2, y2]
                - source: str
                - area: float

        Returns:
            Dict or None: Tracked object info with fields:
                - id: int
                - bbox: [x1, y1, x2, y2]
                - centroid: (cx, cy)
                - area: float
                - class: str
                - confidence: float
        """
        if len(detections) == 0:
            # No detections - increment disappeared count
            if self.tracked_object:
                self.disappeared_count += 1
                if self.disappeared_count > self.max_disappeared:
                    logger.info(f"Lost track of object ID {self.tracked_object['id']} "
                              f"after {self.disappeared_count} frames")
                    self.tracked_object = None
            return self.tracked_object

        # Find largest detection by bbox area
        largest_det = max(detections, key=lambda d: d["area"])
        centroid = self._compute_centroid(largest_det["bbox"])

        # If no current track, start new track
        if self.tracked_object is None:
            self.tracked_object = {
                "id": self.next_id,
                "bbox": largest_det["bbox"],
                "centroid": centroid,
                "area": largest_det["area"],
                "class": largest_det["class"],
                "confidence": largest_det["confidence"],
            }
            logger.info(f"Started tracking object ID {self.next_id}: "
                       f"{largest_det['class']} (area={largest_det['area']:.1f})")
            self.next_id += 1
            self.disappeared_count = 0
            return self.tracked_object

        # Check if largest detection matches current track
        prev_centroid = self.tracked_object["centroid"]
        distance = self._euclidean_distance(centroid, prev_centroid)

        # If distance is reasonable, update track
        if distance < 100:  # pixels (adjust based on image size)
            self.tracked_object.update({
                "bbox": largest_det["bbox"],
                "centroid": centroid,
                "area": largest_det["area"],
                "class": largest_det["class"],
                "confidence": largest_det["confidence"],
            })
            self.disappeared_count = 0
            logger.debug(f"Updated track ID {self.tracked_object['id']}: "
                        f"moved {distance:.1f}px")
        else:
            # Object moved too far or new object appeared
            # Start tracking new largest object
            logger.info(f"Object moved too far ({distance:.1f}px). "
                       f"Starting new track for {largest_det['class']}")
            self.tracked_object = {
                "id": self.next_id,
                "bbox": largest_det["bbox"],
                "centroid": centroid,
                "area": largest_det["area"],
                "class": largest_det["class"],
                "confidence": largest_det["confidence"],
            }
            self.next_id += 1
            self.disappeared_count = 0

        return self.tracked_object

    @staticmethod
    def _compute_centroid(bbox: List[float]) -> Tuple[float, float]:
        """
        Compute centroid of bounding box.

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            (cx, cy) centroid coordinates
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (cx, cy)

    @staticmethod
    def _euclidean_distance(point1: Tuple[float, float],
                           point2: Tuple[float, float]) -> float:
        """
        Compute Euclidean distance between two points.

        Args:
            point1: (x1, y1)
            point2: (x2, y2)

        Returns:
            Distance in pixels
        """
        import math
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
