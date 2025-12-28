"""
Detection fusion module for combining RGB and Thermal detections.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU value (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_bbox_area(bbox: List[float]) -> float:
    """
    Compute area of bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def fuse_detections(rgb_detections: List[Dict],
                   thermal_detections: List[Dict],
                   iou_threshold: float = 0.5) -> List[Dict]:
    """
    Fuse detections from RGB and thermal streams.

    Strategy:
    1. Match RGB and thermal detections with same class and high IoU
    2. Average their confidences for matched detections
    3. Keep unmatched detections from both streams

    Args:
        rgb_detections: List of RGB detections
        thermal_detections: List of thermal detections
        iou_threshold: Minimum IoU to consider a match

    Returns:
        List[Dict]: Fused detections with format:
            {
                "class": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],
                "source": "both" | "rgb_only" | "thermal_only",
                "area": float
            }
    """
    fused = []
    matched_thermal_indices = set()

    # Match RGB detections with thermal
    for rgb_det in rgb_detections:
        best_match = None
        best_iou = iou_threshold

        for i, thermal_det in enumerate(thermal_detections):
            if i in matched_thermal_indices:
                continue
            if rgb_det["class"] != thermal_det["class"]:
                continue

            iou = compute_iou(rgb_det["bbox"], thermal_det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = (i, thermal_det)

        if best_match:
            i, thermal_det = best_match
            matched_thermal_indices.add(i)
            # Average confidence
            fused.append({
                "class": rgb_det["class"],
                "confidence": (rgb_det["confidence"] + thermal_det["confidence"]) / 2,
                "bbox": rgb_det["bbox"],  # Use RGB bbox (usually more accurate)
                "source": "both",
                "area": compute_bbox_area(rgb_det["bbox"])
            })
        else:
            # RGB only
            fused.append({
                "class": rgb_det["class"],
                "confidence": rgb_det["confidence"],
                "bbox": rgb_det["bbox"],
                "source": "rgb_only",
                "area": compute_bbox_area(rgb_det["bbox"])
            })

    # Add unmatched thermal detections
    for i, thermal_det in enumerate(thermal_detections):
        if i not in matched_thermal_indices:
            fused.append({
                "class": thermal_det["class"],
                "confidence": thermal_det["confidence"],
                "bbox": thermal_det["bbox"],
                "source": "thermal_only",
                "area": compute_bbox_area(thermal_det["bbox"])
            })

    logger.debug(f"Fused {len(rgb_detections)} RGB + {len(thermal_detections)} thermal "
                f"= {len(fused)} total detections")

    return fused
