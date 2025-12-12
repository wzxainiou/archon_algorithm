"""
Tegrastats parser for Jetson GPU and system metrics.

Note: On Jetson, GPU memory and system RAM are shared (unified memory architecture).
"""

import re
import subprocess
import threading
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class TegraMetrics:
    """Metrics from tegrastats."""
    timestamp: float
    ram_used_mb: Optional[int] = None
    ram_total_mb: Optional[int] = None
    swap_used_mb: Optional[int] = None
    swap_total_mb: Optional[int] = None
    gpu_util_percent: Optional[float] = None
    cpu_util_percent: Optional[float] = None
    temp_ao: Optional[float] = None  # AO thermal zone
    temp_gpu: Optional[float] = None  # GPU thermal zone
    power_mw: Optional[int] = None


class TegrastatsMonitor:
    """Monitor for tegrastats output."""

    def __init__(self, interval_ms: int = 500):
        """
        Initialize tegrastats monitor.

        Args:
            interval_ms: Sampling interval in milliseconds
        """
        self.interval_ms = interval_ms
        self.metrics_history: List[TegraMetrics] = []
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if tegrastats is available."""
        try:
            result = subprocess.run(
                ["which", "tegrastats"],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if tegrastats monitoring is available."""
        return self._available

    def _parse_line(self, line: str) -> Optional[TegraMetrics]:
        """
        Parse a single line of tegrastats output.

        Example line:
        RAM 2505/7775MB (lfb 1429x4MB) SWAP 0/3887MB (cached 0MB) CPU [2%@2035,0%@2035,0%@2035,0%@2035,0%@2034,0%@2035]
        GR3D_FREQ 0% cpu@38.25C soc2@36.25C soc0@38.5C tj@38.25C soc1@37.25C gpu@36.5C
        """
        metrics = TegraMetrics(timestamp=time.time())

        try:
            # RAM: "RAM 2505/7775MB"
            ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
            if ram_match:
                metrics.ram_used_mb = int(ram_match.group(1))
                metrics.ram_total_mb = int(ram_match.group(2))

            # SWAP: "SWAP 0/3887MB"
            swap_match = re.search(r'SWAP\s+(\d+)/(\d+)MB', line)
            if swap_match:
                metrics.swap_used_mb = int(swap_match.group(1))
                metrics.swap_total_mb = int(swap_match.group(2))

            # GPU: "GR3D_FREQ 45%" or "GR3D 45%"
            gpu_match = re.search(r'GR3D[_FREQ]*\s+(\d+)%', line)
            if gpu_match:
                metrics.gpu_util_percent = float(gpu_match.group(1))

            # CPU: Average from all cores
            cpu_matches = re.findall(r'(\d+)%@', line)
            if cpu_matches:
                cpu_utils = [float(x) for x in cpu_matches]
                metrics.cpu_util_percent = sum(cpu_utils) / len(cpu_utils)

            # Temperature: Look for various thermal zones
            temp_matches = re.findall(r'(\w+)@([\d.]+)C', line)
            for zone, temp in temp_matches:
                temp_val = float(temp)
                if 'gpu' in zone.lower():
                    metrics.temp_gpu = temp_val
                elif 'ao' in zone.lower() or 'tj' in zone.lower():
                    metrics.temp_ao = temp_val

            # Power (if available): "POM_5V_IN 1234mW"
            power_match = re.search(r'(\d+)mW', line)
            if power_match:
                metrics.power_mw = int(power_match.group(1))

            return metrics

        except Exception as e:
            warnings.warn(f"Failed to parse tegrastats line: {e}")
            return None

    def _monitor_thread(self):
        """Background thread to collect tegrastats."""
        try:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            while not self._stop_event.is_set():
                if self._process.stdout:
                    line = self._process.stdout.readline()
                    if not line:
                        break

                    metrics = self._parse_line(line.strip())
                    if metrics:
                        self.metrics_history.append(metrics)

        except Exception as e:
            warnings.warn(f"Tegrastats monitoring error: {e}")
        finally:
            if self._process:
                self._process.terminate()
                self._process.wait(timeout=5)

    def start(self):
        """Start monitoring."""
        if not self._available:
            warnings.warn("tegrastats not available - skipping GPU metrics")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=5)

        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)

    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of collected metrics."""
        if not self.metrics_history:
            return {
                "available": False,
                "reason": "No metrics collected"
            }

        import numpy as np

        def safe_stats(values: List[float]) -> Dict:
            """Compute stats safely handling None values."""
            valid = [v for v in values if v is not None]
            if not valid:
                return {"available": False}
            arr = np.array(valid)
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p95": float(np.percentile(arr, 95)),
            }

        summary = {
            "available": True,
            "total_samples": len(self.metrics_history),
            "source": "tegrastats",
            "note": "GPU memory and system RAM are shared on Jetson (unified memory)",
        }

        # RAM stats
        ram_used = [m.ram_used_mb for m in self.metrics_history]
        summary["ram_mb"] = safe_stats(ram_used)
        if self.metrics_history[0].ram_total_mb:
            summary["ram_mb"]["total"] = self.metrics_history[0].ram_total_mb

        # GPU stats
        gpu_util = [m.gpu_util_percent for m in self.metrics_history]
        summary["gpu_util_percent"] = safe_stats(gpu_util)

        # CPU stats
        cpu_util = [m.cpu_util_percent for m in self.metrics_history]
        summary["cpu_util_percent"] = safe_stats(cpu_util)

        # Temperature
        if any(m.temp_gpu for m in self.metrics_history):
            temps = [m.temp_gpu for m in self.metrics_history]
            summary["temperature_gpu_c"] = safe_stats(temps)

        if any(m.temp_ao for m in self.metrics_history):
            temps = [m.temp_ao for m in self.metrics_history]
            summary["temperature_ao_c"] = safe_stats(temps)

        # Power
        if any(m.power_mw for m in self.metrics_history):
            power = [m.power_mw for m in self.metrics_history]
            summary["power_mw"] = safe_stats(power)

        return summary

    def get_current_metrics(self) -> Optional[TegraMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
