"""
YOLO model wrapper using Ultralytics.

Provides a unified interface for YOLO inference with performance tracking.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from ultralytics import YOLO

from .backend import detect_backend, get_backend_info

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Single frame inference result."""
    frame_id: str
    latency_ms: float
    num_detections: int
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    class_names: List[str] = field(default_factory=list)  # NEW: Human-readable class names


@dataclass
class ModelPerformance:
    """Performance statistics for a model."""
    model_name: str
    backend: str
    total_frames: int = 0
    total_time_ms: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    detection_counts: List[int] = field(default_factory=list)

    def add_result(self, result: InferenceResult):
        """Add inference result to statistics."""
        self.total_frames += 1
        self.total_time_ms += result.latency_ms
        self.latencies_ms.append(result.latency_ms)
        self.detection_counts.append(result.num_detections)

    def compute_metrics(self) -> Dict:
        """Compute performance metrics."""
        if not self.latencies_ms:
            return {
                "model_name": self.model_name,
                "backend": self.backend,
                "status": "no_data",
            }

        latencies = np.array(self.latencies_ms)

        metrics = {
            "model_name": self.model_name,
            "backend": self.backend,
            "total_frames": self.total_frames,
            "fps": 1000.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0,
            "latency_ms": {
                "mean": float(np.mean(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p90": float(np.percentile(latencies, 90)),
                "p99": float(np.percentile(latencies, 99)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
            },
            "detections_per_frame": {
                "mean": float(np.mean(self.detection_counts)),
                "min": int(np.min(self.detection_counts)),
                "max": int(np.max(self.detection_counts)),
            },
        }

        return metrics


class YOLOInference:
    """YOLO model inference wrapper."""

    def __init__(self, model_name: str, weight_path: str, imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45, gpu_mem_limit_gb: float = 8.0,
                 tensorrt_workspace_size: Optional[int] = None):
        """
        Initialize YOLO model.

        Args:
            model_name: Name of the model
            weight_path: Path to model weights
            imgsz: Input image size
            conf: Confidence threshold
            iou: IOU threshold for NMS
            gpu_mem_limit_gb: GPU memory limit in GB
            tensorrt_workspace_size: TensorRT workspace size in bytes (optional)
        """
        self.model_name = model_name
        self.weight_path = weight_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.gpu_mem_limit_gb = gpu_mem_limit_gb

        # Detect backend
        self.backend = detect_backend(weight_path)
        self.backend_info = get_backend_info(weight_path)

        # Initialize model with memory constraints
        try:
            self.model = YOLO(weight_path)
            self.model.overrides['conf'] = conf
            self.model.overrides['iou'] = iou
            self.model.overrides['imgsz'] = imgsz
            self.model.overrides['verbose'] = False

            # Force batch size = 1 for memory control
            self.model.overrides['batch'] = 1

            # Apply TensorRT workspace limit if applicable
            if self.backend == "tensorrt" and tensorrt_workspace_size:
                self.model.overrides['workspace'] = tensorrt_workspace_size
                logger.info(f"TensorRT workspace size: {tensorrt_workspace_size / (1024**3):.2f}GB")

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        # Log backend and memory settings
        logger.info(f"Model {model_name}: backend={self.backend}, batch=1, GPU_limit={gpu_mem_limit_gb}GB")

        # Performance tracking
        self.performance = ModelPerformance(
            model_name=model_name,
            backend=self.backend
        )

    def infer(self, frame: np.ndarray, frame_id: str) -> InferenceResult:
        """
        Run inference on a single frame.

        Args:
            frame: Input image (BGR format)
            frame_id: Identifier for the frame

        Returns:
            InferenceResult containing detections and timing
        """
        start_time = time.perf_counter()

        # Run inference
        results = self.model(frame, verbose=False)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract results
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            # NEW: Convert class IDs to human-readable names
            class_names = [self.model.names[int(cls)] for cls in classes]
        else:
            boxes = np.array([])
            scores = np.array([])
            classes = np.array([])
            class_names = []

        result = InferenceResult(
            frame_id=frame_id,
            latency_ms=latency_ms,
            num_detections=len(boxes),
            boxes=boxes,
            scores=scores,
            classes=classes,
            class_names=class_names,
        )

        # Update performance stats
        self.performance.add_result(result)

        return result

    def get_performance(self) -> Dict:
        """Get performance metrics."""
        return self.performance.compute_metrics()

    def warmup(self, num_iterations: int = 10):
        """Warmup the model with dummy inputs."""
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(num_iterations):
            self.model(dummy_frame, verbose=False)

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": self.model_name,
            "weight_path": self.weight_path,
            "backend": self.backend,
            "backend_info": self.backend_info,
            "config": {
                "imgsz": self.imgsz,
                "conf": self.conf,
                "iou": self.iou,
            },
        }
