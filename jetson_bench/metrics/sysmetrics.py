"""
System metrics collection using psutil (CPU, memory, etc).
"""

import psutil
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    memory_total_mb: int
    memory_used_mb: int
    memory_available_mb: int
    memory_percent: float
    swap_total_mb: int
    swap_used_mb: int
    swap_percent: float


class SystemMonitor:
    """System resource monitor using psutil."""

    def __init__(self, interval_sec: float = 0.5):
        """
        Initialize system monitor.

        Args:
            interval_sec: Sampling interval in seconds
        """
        self.interval_sec = interval_sec
        self.metrics_history: List[SystemMetrics] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=None),
            cpu_per_core=psutil.cpu_percent(interval=None, percpu=True),
            memory_total_mb=mem.total // (1024 * 1024),
            memory_used_mb=mem.used // (1024 * 1024),
            memory_available_mb=mem.available // (1024 * 1024),
            memory_percent=mem.percent,
            swap_total_mb=swap.total // (1024 * 1024),
            swap_used_mb=swap.used // (1024 * 1024),
            swap_percent=swap.percent,
        )

        return metrics

    def _monitor_thread(self):
        """Background thread to collect system metrics."""
        try:
            # Initial call to initialize cpu_percent
            psutil.cpu_percent(interval=None)

            while not self._stop_event.is_set():
                try:
                    metrics = self._collect_metrics()
                    self.metrics_history.append(metrics)
                except Exception as e:
                    warnings.warn(f"Error collecting system metrics: {e}")

                self._stop_event.wait(self.interval_sec)

        except Exception as e:
            warnings.warn(f"System monitoring error: {e}")

    def start(self):
        """Start monitoring."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=5)

    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of collected metrics."""
        if not self.metrics_history:
            return {
                "available": False,
                "reason": "No metrics collected"
            }

        import numpy as np

        def stats(values: List[float]) -> Dict:
            """Compute statistics."""
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p95": float(np.percentile(arr, 95)),
            }

        summary = {
            "available": True,
            "total_samples": len(self.metrics_history),
            "source": "psutil",
        }

        # CPU stats
        cpu_percent = [m.cpu_percent for m in self.metrics_history]
        summary["cpu_percent"] = stats(cpu_percent)

        # Per-core CPU (optional, can be verbose)
        # Uncomment if needed:
        # num_cores = len(self.metrics_history[0].cpu_per_core)
        # summary["cpu_per_core"] = {}
        # for i in range(num_cores):
        #     core_vals = [m.cpu_per_core[i] for m in self.metrics_history]
        #     summary["cpu_per_core"][f"core_{i}"] = stats(core_vals)

        # Memory stats
        memory_used = [m.memory_used_mb for m in self.metrics_history]
        memory_percent = [m.memory_percent for m in self.metrics_history]
        summary["memory_mb"] = stats(memory_used)
        summary["memory_mb"]["total"] = self.metrics_history[0].memory_total_mb
        summary["memory_percent"] = stats(memory_percent)

        # Swap stats (if swap exists)
        if self.metrics_history[0].swap_total_mb > 0:
            swap_used = [m.swap_used_mb for m in self.metrics_history]
            swap_percent = [m.swap_percent for m in self.metrics_history]
            summary["swap_mb"] = stats(swap_used)
            summary["swap_mb"]["total"] = self.metrics_history[0].swap_total_mb
            summary["swap_percent"] = stats(swap_percent)

        return summary

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def print_current(self):
        """Print current metrics to console."""
        metrics = self.get_current_metrics()
        if not metrics:
            return

        print(
            f"CPU: {metrics.cpu_percent:5.1f}% | "
            f"MEM: {metrics.memory_used_mb:5d}/{metrics.memory_total_mb:5d}MB "
            f"({metrics.memory_percent:4.1f}%)",
            end=""
        )

        if metrics.swap_total_mb > 0:
            print(f" | SWAP: {metrics.swap_used_mb:4d}/{metrics.swap_total_mb:4d}MB", end="")
