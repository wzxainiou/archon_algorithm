"""
Configuration management for Jetson benchmarking.

This module defines the exact 4 model slots and runtime configuration.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal


@dataclass
class ModelConfig:
    """Configuration for a single model slot."""
    name: str
    weight_path: Optional[str] = None
    skip_reason: Optional[str] = None

    def validate(self) -> bool:
        """Validate model configuration and set skip reason if invalid."""
        if not self.weight_path:
            self.skip_reason = "No weight path provided"
            return False

        path = Path(self.weight_path)
        if not path.exists():
            self.skip_reason = f"Weight file not found: {self.weight_path}"
            return False

        if path.suffix not in ['.engine', '.onnx', '.pt']:
            self.skip_reason = f"Unsupported weight format: {path.suffix}"
            return False

        return True


@dataclass
class BenchConfig:
    """Main benchmark configuration with exactly 4 model slots."""

    # Exactly 4 model slots (cannot be changed)
    models: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig(name="yolo11n_rgb"),
        ModelConfig(name="yolo11s_rgb"),
        ModelConfig(name="yolo11n_thermal"),
        ModelConfig(name="yolov8n_thermal"),
    ])

    # Input source configuration
    source_type: Literal["image_dir", "video", "camera"] = "image_dir"
    source_path: Optional[str] = None

    # Inference parameters
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    max_frames: int = 300

    # Execution mode
    parallel: bool = False

    # Metrics collection
    metrics_interval: float = 0.5  # seconds

    # GPU memory limitation (GB)
    gpu_mem_limit_gb: float = 8.0  # Maximum GPU memory usage

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self):
        """Ensure exactly 4 models and validate GPU memory limit."""
        if len(self.models) != 4:
            raise ValueError(f"Must have exactly 4 models, got {len(self.models)}")

        # Enforce 8GB GPU memory limit
        if self.gpu_mem_limit_gb > 8.0:
            print(f"⚠️  Warning: GPU memory limit {self.gpu_mem_limit_gb}GB exceeds maximum 8GB")
            print(f"   Clamping to 8GB (local GPU constraint)")
            self.gpu_mem_limit_gb = 8.0

    def set_model_weights(self, model_idx: int, weight_path: str):
        """Set weight path for a specific model slot."""
        if model_idx < 0 or model_idx >= 4:
            raise ValueError(f"Model index must be 0-3, got {model_idx}")
        self.models[model_idx].weight_path = weight_path

    def validate_source(self) -> bool:
        """Validate input source configuration."""
        if self.source_type == "image_dir":
            if not self.source_path:
                print("Error: image_dir source requires a path", file=sys.stderr)
                return False
            path = Path(self.source_path)
            if not path.exists():
                print(f"Error: Image directory not found: {self.source_path}", file=sys.stderr)
                return False
            if not path.is_dir():
                print(f"Error: Path is not a directory: {self.source_path}", file=sys.stderr)
                return False
            # Check for valid images
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = [f for f in path.iterdir() if f.suffix.lower() in valid_exts]
            if not images:
                print(f"Error: No valid images found in: {self.source_path}", file=sys.stderr)
                return False

        elif self.source_type == "video":
            if not self.source_path:
                print("Error: video source requires a path", file=sys.stderr)
                return False
            path = Path(self.source_path)
            if not path.exists():
                print(f"Error: Video file not found: {self.source_path}", file=sys.stderr)
                return False

        elif self.source_type == "camera":
            # Camera index validation happens at runtime
            pass

        return True

    def validate_models(self):
        """Validate all 4 model slots."""
        for model in self.models:
            model.validate()

    def get_active_models(self) -> List[ModelConfig]:
        """Get list of models that are not skipped."""
        return [m for m in self.models if m.skip_reason is None]

    def get_skipped_models(self) -> List[ModelConfig]:
        """Get list of models that are skipped."""
        return [m for m in self.models if m.skip_reason is not None]


def verify_environment():
    """Verify the runtime environment is suitable."""
    errors = []
    warnings = []

    # Check Python version
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}")

    # Check if running on Jetson
    jetson_release = Path("/etc/nv_tegra_release")
    if not jetson_release.exists():
        warnings.append("Not running on Jetson (no /etc/nv_tegra_release found)")

    # Check for tegrastats
    import shutil
    if not shutil.which("tegrastats"):
        warnings.append("tegrastats not found - GPU metrics will be limited")

    # Check required packages
    try:
        import cv2
    except ImportError:
        errors.append("opencv-python not installed. Install with: pip install opencv-python")

    try:
        import psutil
    except ImportError:
        errors.append("psutil not installed. Install with: pip install psutil")

    try:
        from ultralytics import YOLO
    except ImportError:
        errors.append("ultralytics not installed. Install with: pip install ultralytics")

    # Print results
    if warnings:
        print("⚠️  Warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n❌ Errors:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("✅ Environment validation passed")
    return True
