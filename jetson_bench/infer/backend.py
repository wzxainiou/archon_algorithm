"""
Backend detection and inference engine management.

Supports TensorRT (.engine), ONNX (.onnx), and PyTorch (.pt) backends.
"""

from pathlib import Path
from typing import Optional, Literal
import warnings


BackendType = Literal["tensorrt", "onnx", "pytorch", "unknown"]


def detect_backend(weight_path: str) -> BackendType:
    """
    Detect inference backend based on weight file extension.

    Priority: .engine (TensorRT) > .onnx (ONNX Runtime) > .pt (PyTorch)
    """
    path = Path(weight_path)
    suffix = path.suffix.lower()

    backend_map = {
        ".engine": "tensorrt",
        ".onnx": "onnx",
        ".pt": "pytorch",
    }

    return backend_map.get(suffix, "unknown")


def verify_backend_compatibility(weight_path: str) -> tuple[bool, Optional[str]]:
    """
    Verify that the backend is compatible with the current system.

    Returns:
        (is_valid, error_message)
    """
    backend = detect_backend(weight_path)

    if backend == "unknown":
        return False, f"Unknown weight format: {Path(weight_path).suffix}"

    if backend == "tensorrt":
        # TensorRT engines are platform-specific
        # We can't easily verify without trying to load it
        return True, None

    if backend == "onnx":
        try:
            import onnxruntime
            return True, None
        except ImportError:
            return False, "ONNX Runtime not installed. Install with: pip install onnxruntime-gpu"

    if backend == "pytorch":
        try:
            import torch
            return True, None
        except ImportError:
            return False, "PyTorch not installed. Install with: pip install torch"

    return True, None


def get_backend_info(weight_path: str) -> dict:
    """Get information about the inference backend."""
    backend = detect_backend(weight_path)
    is_valid, error = verify_backend_compatibility(weight_path)

    info = {
        "backend": backend,
        "weight_path": weight_path,
        "is_valid": is_valid,
        "error": error,
    }

    # Add backend-specific information
    if backend == "tensorrt":
        try:
            import tensorrt as trt
            info["tensorrt_version"] = trt.__version__
        except ImportError:
            info["tensorrt_version"] = "not_installed"

    elif backend == "onnx":
        try:
            import onnxruntime
            info["onnxruntime_version"] = onnxruntime.__version__
        except ImportError:
            info["onnxruntime_version"] = "not_installed"

    elif backend == "pytorch":
        try:
            import torch
            info["pytorch_version"] = torch.__version__
        except ImportError:
            info["pytorch_version"] = "not_installed"

    return info


def suggest_backend_optimization(weight_path: str) -> Optional[str]:
    """Suggest optimization if a better backend is available."""
    backend = detect_backend(weight_path)

    if backend == "pytorch":
        return "Consider exporting to ONNX or TensorRT for better performance"
    elif backend == "onnx":
        return "Consider exporting to TensorRT for optimal Jetson performance"

    return None
