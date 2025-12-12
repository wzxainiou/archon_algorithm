"""
Metrics aggregation - combines tegrastats and system metrics.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .tegrastats import TegrastatsMonitor
from .sysmetrics import SystemMonitor

logger = logging.getLogger(__name__)


@dataclass
class GPUMemorySnapshot:
    """GPU memory usage snapshot."""
    timestamp: float
    used_gb: float
    source: str  # pytorch_cuda, nvml, tegrastats


class MetricsAggregator:
    """Aggregates metrics from multiple sources."""

    def __init__(self, interval_sec: float = 0.5, gpu_mem_limit_gb: float = 8.0):
        """
        Initialize metrics aggregator.

        Args:
            interval_sec: Sampling interval in seconds
            gpu_mem_limit_gb: GPU memory limit in GB for validation
        """
        self.interval_sec = interval_sec
        self.gpu_mem_limit_gb = gpu_mem_limit_gb
        self.system_monitor = SystemMonitor(interval_sec=interval_sec)
        self.tegra_monitor = TegrastatsMonitor(interval_ms=int(interval_sec * 1000))
        self.start_time = None
        self.end_time = None

        # GPU memory tracking
        self.gpu_memory_history: List[GPUMemorySnapshot] = []
        self.memory_violations: List[Dict] = []

    def start(self):
        """Start all monitors."""
        self.start_time = time.time()
        self.system_monitor.start()

        if self.tegra_monitor.is_available():
            self.tegra_monitor.start()
        else:
            print("⚠️  Tegrastats not available - GPU metrics will be limited")

        # Give monitors time to start collecting
        time.sleep(1.0)

    def stop(self):
        """Stop all monitors."""
        self.end_time = time.time()
        self.system_monitor.stop()
        self.tegra_monitor.stop()

    def collect_gpu_memory(self):
        """Collect current GPU memory usage and check for violations."""
        from ..gpu_memory import get_current_gpu_memory_usage, validate_memory_within_limit

        usage = get_current_gpu_memory_usage()
        if not usage:
            return

        # Determine GPU memory from available sources
        gpu_mem_gb = None
        source = usage.get("source", "unknown")

        if "allocated_gb" in usage:
            gpu_mem_gb = usage["allocated_gb"]
        elif "used_gb" in usage:
            gpu_mem_gb = usage["used_gb"]

        if gpu_mem_gb is not None:
            # Record snapshot
            snapshot = GPUMemorySnapshot(
                timestamp=time.time(),
                used_gb=gpu_mem_gb,
                source=source
            )
            self.gpu_memory_history.append(snapshot)

            # Check for violation
            is_valid, message = validate_memory_within_limit(self.gpu_mem_limit_gb, gpu_mem_gb)
            if not is_valid:
                logger.error(f"❌ GPU MEMORY VIOLATION: {message}")
                self.memory_violations.append({
                    "timestamp": snapshot.timestamp,
                    "used_gb": gpu_mem_gb,
                    "limit_gb": self.gpu_mem_limit_gb,
                    "message": message,
                })

    def get_summary(self) -> Dict:
        """Get combined metrics summary."""
        duration = self.end_time - self.start_time if self.end_time else 0

        summary = {
            "collection": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                "duration_sec": duration,
                "interval_sec": self.interval_sec,
            },
            "system": self.system_monitor.get_metrics_summary(),
            "jetson": self.tegra_monitor.get_metrics_summary(),
            "gpu_memory": self._get_gpu_memory_summary(),
        }

        return summary

    def _get_gpu_memory_summary(self) -> Dict:
        """Get GPU memory usage summary."""
        if not self.gpu_memory_history:
            return {
                "available": False,
                "reason": "No GPU memory data collected"
            }

        import numpy as np

        memory_values = [s.used_gb for s in self.gpu_memory_history]
        peak_gb = max(memory_values)
        mean_gb = np.mean(memory_values)
        within_limit = peak_gb <= self.gpu_mem_limit_gb

        summary = {
            "available": True,
            "limit_gb": self.gpu_mem_limit_gb,
            "peak_gb": float(peak_gb),
            "mean_gb": float(mean_gb),
            "min_gb": float(min(memory_values)),
            "within_limit": within_limit,
            "source": self.gpu_memory_history[0].source if self.gpu_memory_history else "unknown",
            "violations": len(self.memory_violations),
        }

        if self.memory_violations:
            summary["violation_details"] = self.memory_violations

        return summary

    def print_current(self, model_name: Optional[str] = None):
        """Print current metrics to console."""
        prefix = f"[{model_name}] " if model_name else ""

        # System metrics
        self.system_monitor.print_current()

        # Tegra metrics (GPU)
        if self.tegra_monitor.is_available():
            tegra = self.tegra_monitor.get_current_metrics()
            if tegra and tegra.gpu_util_percent is not None:
                print(f" | GPU: {tegra.gpu_util_percent:5.1f}%", end="")

        # GPU memory
        if self.gpu_memory_history:
            latest = self.gpu_memory_history[-1]
            print(f" | GPU_MEM: {latest.used_gb:.2f}/{self.gpu_mem_limit_gb:.1f}GB", end="")

        print()  # Newline

    def save_timeseries(self, output_path: Path):
        """Save time-series metrics to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            # Write system metrics
            for metrics in self.system_monitor.metrics_history:
                entry = {
                    "source": "system",
                    "timestamp": metrics.timestamp,
                    "cpu_percent": metrics.cpu_percent,
                    "memory_used_mb": metrics.memory_used_mb,
                    "memory_percent": metrics.memory_percent,
                }
                f.write(json.dumps(entry) + '\n')

            # Write tegra metrics
            for metrics in self.tegra_monitor.metrics_history:
                entry = {
                    "source": "tegra",
                    "timestamp": metrics.timestamp,
                }
                if metrics.ram_used_mb is not None:
                    entry["ram_used_mb"] = metrics.ram_used_mb
                if metrics.gpu_util_percent is not None:
                    entry["gpu_util_percent"] = metrics.gpu_util_percent
                if metrics.cpu_util_percent is not None:
                    entry["cpu_util_percent"] = metrics.cpu_util_percent
                if metrics.temp_gpu is not None:
                    entry["temp_gpu_c"] = metrics.temp_gpu

                f.write(json.dumps(entry) + '\n')

            # Write GPU memory metrics
            for snapshot in self.gpu_memory_history:
                entry = {
                    "source": "gpu_memory",
                    "timestamp": snapshot.timestamp,
                    "used_gb": snapshot.used_gb,
                    "limit_gb": self.gpu_mem_limit_gb,
                    "data_source": snapshot.source,
                }
                f.write(json.dumps(entry) + '\n')
