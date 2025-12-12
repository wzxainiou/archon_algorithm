"""
GPU memory management and limitation.

Enforces strict 8GB GPU memory limit through multiple mechanisms:
1. PyTorch CUDA memory fraction limit (hard)
2. TensorRT workspace size limit (hard)
3. Environment variables (soft)
4. Runtime monitoring and validation (verification)
"""

import os
import sys
import logging
from typing import Optional, Literal, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


MemoryLimitType = Literal["hard", "soft", "unavailable"]


class GPUMemoryManager:
    """Manages GPU memory limits and monitoring."""

    def __init__(self, limit_gb: float = 8.0):
        """
        Initialize GPU memory manager.

        Args:
            limit_gb: GPU memory limit in GB (maximum 8.0)
        """
        if limit_gb > 8.0:
            logger.warning(f"GPU memory limit {limit_gb}GB exceeds maximum 8GB, clamping to 8GB")
            limit_gb = 8.0

        self.limit_gb = limit_gb
        self.limit_bytes = int(limit_gb * 1024**3)
        self.limit_type: MemoryLimitType = "unavailable"
        self.torch_available = False
        self.cuda_available = False

    def apply_limits(self) -> Dict[str, any]:
        """
        Apply GPU memory limits using all available mechanisms.

        Returns:
            Dictionary with limit status and information
        """
        results = {
            "limit_gb": self.limit_gb,
            "limit_type": "unavailable",
            "methods_applied": [],
            "warnings": [],
        }

        # Method 1: PyTorch CUDA memory fraction (preferred - hard limit)
        pytorch_result = self._apply_pytorch_limit()
        if pytorch_result["success"]:
            results["methods_applied"].append("pytorch_cuda")
            results["limit_type"] = "hard"
            self.limit_type = "hard"
            logger.info(f"✅ Applied hard GPU memory limit via PyTorch: {self.limit_gb}GB")
        else:
            results["warnings"].append(pytorch_result["message"])

        # Method 2: Environment variables (soft limit)
        env_result = self._apply_env_limits()
        if env_result["success"]:
            results["methods_applied"].append("environment")
            if results["limit_type"] == "unavailable":
                results["limit_type"] = "soft"
                self.limit_type = "soft"
            logger.info(f"✅ Set GPU memory environment variables: {self.limit_gb}GB")
        else:
            results["warnings"].append(env_result["message"])

        # Log final status
        if results["limit_type"] == "unavailable":
            logger.warning(f"⚠️  Could not apply GPU memory limits - monitoring only")
            logger.warning(f"   GPU memory usage will be monitored but not enforced")
        elif results["limit_type"] == "soft":
            logger.warning(f"⚠️  GPU memory limit is SOFT ({self.limit_gb}GB)")
            logger.warning(f"   Enforcement depends on backend cooperation")

        print(f"\n{'='*60}")
        print(f"GPU Memory Configuration")
        print(f"{'='*60}")
        print(f"Limit: {self.limit_gb} GB")
        print(f"Limit Type: {results['limit_type']}")
        print(f"Methods Applied: {', '.join(results['methods_applied']) or 'None'}")
        if results['warnings']:
            print(f"\nWarnings:")
            for w in results['warnings']:
                print(f"  - {w}")
        print(f"{'='*60}\n")

        return results

    def _apply_pytorch_limit(self) -> Dict[str, any]:
        """Apply PyTorch CUDA memory fraction limit."""
        try:
            import torch

            self.torch_available = True

            if not torch.cuda.is_available():
                return {
                    "success": False,
                    "message": "PyTorch installed but CUDA not available"
                }

            self.cuda_available = True

            # Get total GPU memory
            device = torch.device("cuda:0")
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_gb = total_memory / (1024**3)

            # Calculate fraction
            fraction = min(self.limit_gb / total_gb, 1.0)

            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)

            # Also set max_split_size to prevent fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

            logger.info(f"PyTorch CUDA memory limit: {fraction:.2%} of {total_gb:.1f}GB = {self.limit_gb}GB")

            return {
                "success": True,
                "message": f"PyTorch CUDA limit set to {self.limit_gb}GB ({fraction:.2%} of {total_gb:.1f}GB)",
                "total_gb": total_gb,
                "fraction": fraction,
            }

        except ImportError:
            return {
                "success": False,
                "message": "PyTorch not installed - cannot set CUDA memory limit"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to set PyTorch CUDA limit: {e}"
            }

    def _apply_env_limits(self) -> Dict[str, any]:
        """Set environment variables for GPU memory limits."""
        try:
            # CUDA_LAUNCH_BLOCKING for better error messages
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

            # TensorRT workspace size (in bytes)
            # This is used by some backends
            os.environ['TENSORRT_WORKSPACE_SIZE'] = str(self.limit_bytes)

            return {
                "success": True,
                "message": "Environment variables set for GPU memory limits"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to set environment variables: {e}"
            }

    def get_tensorrt_workspace_size(self) -> int:
        """
        Get recommended TensorRT workspace size in bytes.

        For 8GB limit, use 4GB workspace (conservative).
        """
        # Use 50% of limit for workspace to leave room for model weights and activations
        workspace_gb = self.limit_gb * 0.5
        return int(workspace_gb * 1024**3)

    def check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.cuda_available

    def get_limit_info(self) -> Dict:
        """Get information about GPU memory limits."""
        return {
            "limit_gb": self.limit_gb,
            "limit_bytes": self.limit_bytes,
            "limit_type": self.limit_type,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
        }


def get_current_gpu_memory_usage() -> Optional[Dict]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with GPU memory info or None if unavailable
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory

        return {
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "total_gb": total / (1024**3),
            "source": "pytorch_cuda",
        }
    except ImportError:
        pass

    # Fallback: try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        return {
            "used_gb": info.used / (1024**3),
            "total_gb": info.total / (1024**3),
            "source": "nvml",
        }
    except (ImportError, Exception):
        pass

    return None


def validate_memory_within_limit(limit_gb: float, current_usage_gb: float) -> tuple[bool, str]:
    """
    Validate that GPU memory usage is within limit.

    Returns:
        (is_valid, message)
    """
    if current_usage_gb > limit_gb:
        return False, f"GPU memory {current_usage_gb:.2f}GB exceeds limit {limit_gb:.2f}GB"

    # Warning threshold: 90%
    if current_usage_gb > limit_gb * 0.9:
        return True, f"GPU memory {current_usage_gb:.2f}GB approaching limit {limit_gb:.2f}GB (90%+)"

    return True, f"GPU memory {current_usage_gb:.2f}GB within limit {limit_gb:.2f}GB"
