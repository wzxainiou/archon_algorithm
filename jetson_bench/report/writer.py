"""
Report writer - generates JSON and Markdown reports.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class ReportWriter:
    """Generates benchmark reports in multiple formats."""

    def __init__(self, output_dir: Path):
        """
        Initialize report writer.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, data: Dict, filename: str = "report.json"):
        """Write JSON report."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"📄 JSON report: {output_path}")

    def write_markdown(self, data: Dict, filename: str = "report.md"):
        """Write Markdown report."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Jetson Orin Nano Multi-Model Benchmark Report\n\n")

            # Memory Safety Summary (NEW)
            if "memory_safety" in data:
                self._write_memory_safety_section(f, data["memory_safety"])

            # Metadata
            f.write("## Run Information\n\n")
            if "metadata" in data:
                meta = data["metadata"]
                f.write(f"- **Timestamp**: {meta.get('timestamp', 'N/A')}\n")
                f.write(f"- **Platform**: {meta.get('platform', 'N/A')}\n")
                f.write(f"- **Python Version**: {meta.get('python_version', 'N/A')}\n")
                f.write(f"- **GPU Memory Limit**: {meta.get('gpu_mem_limit_gb', 8.0)}GB\n")
                f.write("\n")

            # Input source
            if "input_source" in data:
                src = data["input_source"]
                f.write("## Input Source\n\n")
                f.write(f"- **Type**: {src.get('type', 'N/A')}\n")
                f.write(f"- **Path**: {src.get('path', 'N/A')}\n")
                if "total_frames_processed" in src:
                    f.write(f"- **Frames Processed**: {src['total_frames_processed']}\n")
                f.write("\n")

            # Model performance
            if "models" in data:
                f.write("## Model Performance\n\n")
                for model in data["models"]:
                    self._write_model_section(f, model)

            # System metrics
            if "system_metrics" in data:
                f.write("## System Resource Usage\n\n")
                self._write_metrics_section(f, data["system_metrics"])

            # Summary
            if "summary" in data:
                f.write("## Summary\n\n")
                summary = data["summary"]
                f.write(f"- **Total Models**: 2 (required)\n")
                f.write(f"- **Models Executed**: {summary.get('models_executed', 0)}\n")
                f.write(f"- **Models Skipped**: {summary.get('models_skipped', 0)}\n")
                if summary.get('models_skipped', 0) > 0:
                    f.write("\n### Skipped Models\n\n")
                    for skip in summary.get('skip_reasons', []):
                        f.write(f"- **{skip['model']}**: {skip['reason']}\n")

        print(f"📄 Markdown report: {output_path}")

    def _write_model_section(self, f, model: Dict):
        """Write a single model's performance section."""
        name = model.get('name', 'Unknown')
        status = model.get('status', 'unknown')

        f.write(f"### {name}\n\n")

        if status == "skipped":
            f.write(f"⚠️ **Status**: Skipped\n")
            f.write(f"**Reason**: {model.get('skip_reason', 'Unknown')}\n\n")
            return

        if status == "failed":
            f.write(f"❌ **Status**: Failed\n")
            f.write(f"**Error**: {model.get('error', 'Unknown error')}\n\n")
            return

        # Performance metrics
        if "performance" in model:
            perf = model["performance"]
            f.write(f"✅ **Status**: Completed\n\n")
            f.write(f"- **Backend**: {perf.get('backend', 'N/A')}\n")
            f.write(f"- **Frames Processed**: {perf.get('total_frames', 0)}\n")
            f.write(f"- **FPS**: {perf.get('fps', 0):.2f}\n")

            if "latency_ms" in perf:
                lat = perf["latency_ms"]
                f.write(f"- **Latency (ms)**:\n")
                f.write(f"  - Mean: {lat.get('mean', 0):.2f}\n")
                f.write(f"  - P50: {lat.get('p50', 0):.2f}\n")
                f.write(f"  - P90: {lat.get('p90', 0):.2f}\n")
                f.write(f"  - P99: {lat.get('p99', 0):.2f}\n")

            if "detections_per_frame" in perf:
                det = perf["detections_per_frame"]
                f.write(f"- **Detections per Frame**:\n")
                f.write(f"  - Mean: {det.get('mean', 0):.2f}\n")
                f.write(f"  - Min: {det.get('min', 0)}\n")
                f.write(f"  - Max: {det.get('max', 0)}\n")

        # GPU Memory info
        if "gpu_memory" in model:
            gpu_mem = model["gpu_memory"]
            within_limit = "✅" if gpu_mem.get("within_limit", True) else "❌"
            f.write(f"- **GPU Memory** {within_limit}:\n")
            f.write(f"  - Limit: {gpu_mem.get('limit_gb', 8.0):.1f}GB\n")
            f.write(f"  - Peak: {gpu_mem.get('peak_gb', 0):.2f}GB\n")
            f.write(f"  - Mean: {gpu_mem.get('mean_gb', 0):.2f}GB\n")
            f.write(f"  - Within Limit: {gpu_mem.get('within_limit', 'Unknown')}\n")
            if gpu_mem.get('limit_type'):
                f.write(f"  - Limit Type: {gpu_mem.get('limit_type')}\n")

        f.write("\n")

        # NEW: Detection results section
        if "detection_summary" in model:
            self._write_detection_section(f, model["detection_summary"])

    def _write_detection_section(self, f, detection_summary: Dict):
        """Write detection results section."""
        total_detections = detection_summary.get("total_detections", 0)
        class_distribution = detection_summary.get("class_distribution", {})
        frame_detections = detection_summary.get("frame_detections", [])

        f.write(f"#### 📸 检测结果统计\n\n")

        # Class distribution
        if class_distribution:
            f.write(f"**类别分布** (共检测到 {total_detections} 个对象):\n\n")
            # Sort by count descending
            sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes:
                percentage = (count / total_detections * 100) if total_detections > 0 else 0
                # Add emoji for common classes
                emoji = self._get_class_emoji(class_name)
                f.write(f"- {emoji} **{class_name}**: {count}次 ({percentage:.1f}%)\n")
            f.write("\n")

        # Frame-by-frame detections
        if frame_detections:
            f.write(f"**详细检测结果** (按帧):\n\n")
            for frame_det in frame_detections:
                frame_name = frame_det.get("frame_name", "unknown")
                num_detections = frame_det.get("num_detections", 0)
                detections = frame_det.get("detections", [])

                f.write(f"**Frame: {frame_name}** ({num_detections} 个对象)\n\n")

                if detections:
                    for i, det in enumerate(detections, 1):
                        class_name = det.get("class", "unknown")
                        confidence = det.get("confidence", 0)
                        bbox = det.get("bbox", [0, 0, 0, 0])
                        emoji = self._get_class_emoji(class_name)

                        f.write(f"  {i}. {emoji} **{class_name}** "
                               f"(置信度: {confidence:.2f}) - "
                               f"位置: [x1={bbox[0]:.0f}, y1={bbox[1]:.0f}, "
                               f"x2={bbox[2]:.0f}, y2={bbox[3]:.0f}]\n")
                else:
                    f.write(f"  未检测到对象\n")

                f.write("\n")

        f.write("\n")

    def _get_class_emoji(self, class_name: str) -> str:
        """Get emoji for common object classes."""
        emoji_map = {
            "person": "👤",
            "bicycle": "🚲",
            "car": "🚗",
            "motorcycle": "🏍️",
            "airplane": "✈️",
            "bus": "🚌",
            "train": "🚆",
            "truck": "🚚",
            "boat": "⛵",
            "traffic light": "🚦",
            "fire hydrant": "🚰",
            "stop sign": "🛑",
            "bench": "🪑",
            "bird": "🐦",
            "cat": "🐱",
            "dog": "🐶",
            "horse": "🐴",
            "sheep": "🐑",
            "cow": "🐄",
            "elephant": "🐘",
            "bear": "🐻",
            "zebra": "🦓",
            "giraffe": "🦒",
            "backpack": "🎒",
            "umbrella": "☂️",
            "handbag": "👜",
            "tie": "👔",
            "suitcase": "🧳",
            "sports ball": "⚽",
            "kite": "🪁",
            "baseball bat": "⚾",
            "skateboard": "🛹",
            "surfboard": "🏄",
            "tennis racket": "🎾",
            "bottle": "🍾",
            "wine glass": "🍷",
            "cup": "☕",
            "fork": "🍴",
            "knife": "🔪",
            "spoon": "🥄",
            "bowl": "🥣",
            "banana": "🍌",
            "apple": "🍎",
            "sandwich": "🥪",
            "orange": "🍊",
            "broccoli": "🥦",
            "carrot": "🥕",
            "pizza": "🍕",
            "donut": "🍩",
            "cake": "🍰",
            "chair": "🪑",
            "couch": "🛋️",
            "bed": "🛏️",
            "toilet": "🚽",
            "tv": "📺",
            "laptop": "💻",
            "mouse": "🖱️",
            "remote": "📱",
            "keyboard": "⌨️",
            "cell phone": "📱",
            "microwave": "📟",
            "oven": "🔥",
            "toaster": "🍞",
            "refrigerator": "🧊",
            "book": "📖",
            "clock": "🕐",
            "vase": "🏺",
            "scissors": "✂️",
            "teddy bear": "🧸",
            "hair drier": "💨",
            "toothbrush": "🪥",
        }
        return emoji_map.get(class_name, "🔸")

    def _write_metrics_section(self, f, metrics: Dict):
        """Write system metrics section."""
        # System metrics from psutil
        if "system" in metrics and metrics["system"].get("available"):
            sys_metrics = metrics["system"]
            f.write("### CPU & Memory (psutil)\n\n")

            if "cpu_percent" in sys_metrics:
                cpu = sys_metrics["cpu_percent"]
                f.write(f"- **CPU Utilization**:\n")
                f.write(f"  - Mean: {cpu.get('mean', 0):.1f}%\n")
                f.write(f"  - Max: {cpu.get('max', 0):.1f}%\n")
                f.write(f"  - P95: {cpu.get('p95', 0):.1f}%\n")

            if "memory_mb" in sys_metrics:
                mem = sys_metrics["memory_mb"]
                f.write(f"- **Memory Usage**:\n")
                f.write(f"  - Total: {mem.get('total', 0)} MB\n")
                f.write(f"  - Mean: {mem.get('mean', 0):.0f} MB\n")
                f.write(f"  - Peak: {mem.get('max', 0):.0f} MB\n")

            f.write("\n")

        # Jetson metrics from tegrastats
        if "jetson" in metrics and metrics["jetson"].get("available"):
            jetson = metrics["jetson"]
            f.write("### GPU & Jetson Metrics (tegrastats)\n\n")
            f.write(f"ℹ️ *{jetson.get('note', '')}*\n\n")

            if "gpu_util_percent" in jetson:
                gpu = jetson["gpu_util_percent"]
                f.write(f"- **GPU Utilization**:\n")
                f.write(f"  - Mean: {gpu.get('mean', 0):.1f}%\n")
                f.write(f"  - Max: {gpu.get('max', 0):.1f}%\n")
                f.write(f"  - P95: {gpu.get('p95', 0):.1f}%\n")

            if "ram_mb" in jetson:
                ram = jetson["ram_mb"]
                f.write(f"- **RAM Usage** (from tegrastats):\n")
                f.write(f"  - Total: {ram.get('total', 0)} MB\n")
                f.write(f"  - Mean: {ram.get('mean', 0):.0f} MB\n")
                f.write(f"  - Peak: {ram.get('max', 0):.0f} MB\n")

            if "temperature_gpu_c" in jetson:
                temp = jetson["temperature_gpu_c"]
                f.write(f"- **GPU Temperature**:\n")
                f.write(f"  - Mean: {temp.get('mean', 0):.1f}°C\n")
                f.write(f"  - Max: {temp.get('max', 0):.1f}°C\n")

            if "power_mw" in jetson:
                power = jetson["power_mw"]
                f.write(f"- **Power Consumption**:\n")
                f.write(f"  - Mean: {power.get('mean', 0) / 1000:.2f}W\n")
                f.write(f"  - Peak: {power.get('max', 0) / 1000:.2f}W\n")

            f.write("\n")

        # GPU Memory metrics (NEW)
        if "gpu_memory" in metrics and metrics["gpu_memory"].get("available"):
            gpu_mem = metrics["gpu_memory"]
            f.write("### GPU Memory Usage\n\n")
            within_limit = "✅" if gpu_mem.get("within_limit", True) else "❌"
            f.write(f"**Status**: {within_limit} {'Within Limit' if gpu_mem.get('within_limit') else 'EXCEEDED LIMIT'}\n\n")
            f.write(f"- **Limit**: {gpu_mem.get('limit_gb', 8.0):.1f}GB\n")
            f.write(f"- **Peak Usage**: {gpu_mem.get('peak_gb', 0):.2f}GB\n")
            f.write(f"- **Mean Usage**: {gpu_mem.get('mean_gb', 0):.2f}GB\n")
            f.write(f"- **Min Usage**: {gpu_mem.get('min_gb', 0):.2f}GB\n")
            f.write(f"- **Data Source**: {gpu_mem.get('source', 'unknown')}\n")
            if gpu_mem.get('violations', 0) > 0:
                f.write(f"- **⚠️ Memory Violations**: {gpu_mem['violations']}\n")
            f.write("\n")

    def _write_memory_safety_section(self, f, memory_safety: Dict):
        """Write memory safety summary section."""
        f.write("## 🔒 Memory Safety Summary\n\n")

        all_within_limit = memory_safety.get("all_within_limit", False)
        status_emoji = "✅" if all_within_limit else "❌"

        f.write(f"{status_emoji} **Overall Status**: ")
        if all_within_limit:
            f.write("All models operated within 8GB GPU memory limit\n\n")
        else:
            f.write("⚠️ **MEMORY LIMIT EXCEEDED**\n\n")

        f.write(f"- **GPU Memory Limit**: {memory_safety.get('limit_gb', 8.0)}GB\n")
        f.write(f"- **Limit Type**: {memory_safety.get('limit_type', 'unknown')}\n")
        f.write(f"- **Total Violations**: {memory_safety.get('total_violations', 0)}\n")

        if "top_memory_models" in memory_safety:
            f.write(f"\n**Models Closest to Memory Limit**:\n\n")
            for i, model_mem in enumerate(memory_safety["top_memory_models"], 1):
                f.write(f"{i}. **{model_mem['model']}**: {model_mem['peak_gb']:.2f}GB ")
                f.write(f"({model_mem['peak_gb']/memory_safety.get('limit_gb', 8.0)*100:.1f}% of limit)\n")

        f.write("\n")

    def create_full_report(self, models_results: List[Dict], metrics_summary: Dict,
                          source_info: Dict, config: Dict, memory_limit_type: str = "soft") -> Dict:
        """
        Create complete report structure.

        Args:
            models_results: List of model results
            metrics_summary: System metrics summary
            source_info: Input source information
            config: Benchmark configuration
            memory_limit_type: Type of memory limit (hard/soft/unavailable)

        Returns:
            Complete report dictionary
        """
        # Create memory safety summary
        memory_safety = self._create_memory_safety_summary(
            models_results,
            metrics_summary.get("gpu_memory", {}),
            config.get("gpu_mem_limit_gb", 8.0),
            memory_limit_type
        )

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "platform": self._get_platform_info(),
                "python_version": self._get_python_version(),
                "gpu_mem_limit_gb": config.get("gpu_mem_limit_gb", 8.0),
                "memory_limit_type": memory_limit_type,
            },
            "config": {
                "imgsz": config.get("imgsz", 640),
                "conf": config.get("conf", 0.25),
                "iou": config.get("iou", 0.45),
            },
            "input_source": source_info,
            "models": models_results,
            "system_metrics": metrics_summary,
            "memory_safety": memory_safety,
            "summary": self._create_summary(models_results),
        }

        return report

    def _get_platform_info(self) -> str:
        """Get platform information."""
        import platform
        jetson_release = Path("/etc/nv_tegra_release")
        if jetson_release.exists():
            try:
                with open(jetson_release) as f:
                    return f.read().strip()
            except Exception:
                pass
        return platform.platform()

    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _create_summary(self, models_results: List[Dict]) -> Dict:
        """Create summary statistics."""
        executed = sum(1 for m in models_results if m.get("status") == "completed")
        skipped = sum(1 for m in models_results if m.get("status") == "skipped")
        failed = sum(1 for m in models_results if m.get("status") == "failed")

        skip_reasons = [
            {"model": m["name"], "reason": m.get("skip_reason", "Unknown")}
            for m in models_results if m.get("status") == "skipped"
        ]

        return {
            "total_models": 4,
            "models_executed": executed,
            "models_skipped": skipped,
            "models_failed": failed,
            "skip_reasons": skip_reasons,
        }

    def _create_memory_safety_summary(self, models_results: List[Dict],
                                      gpu_memory_metrics: Dict, limit_gb: float,
                                      limit_type: str) -> Dict:
        """Create memory safety summary."""
        # Collect GPU memory info from all models
        model_memory_usage = []
        total_violations = 0

        for model in models_results:
            if "gpu_memory" in model and model.get("status") == "completed":
                gpu_mem = model["gpu_memory"]
                model_memory_usage.append({
                    "model": model["name"],
                    "peak_gb": gpu_mem.get("peak_gb", 0),
                    "mean_gb": gpu_mem.get("mean_gb", 0),
                    "within_limit": gpu_mem.get("within_limit", True),
                })
                if not gpu_mem.get("within_limit", True):
                    total_violations += 1

        # Sort by peak usage (descending)
        model_memory_usage.sort(key=lambda x: x["peak_gb"], reverse=True)

        # Check if all models are within limit
        all_within_limit = all(m["within_limit"] for m in model_memory_usage) if model_memory_usage else True

        # Also check global GPU memory metrics
        if gpu_memory_metrics.get("available"):
            global_within_limit = gpu_memory_metrics.get("within_limit", True)
            global_violations = gpu_memory_metrics.get("violations", 0)
            all_within_limit = all_within_limit and global_within_limit
            total_violations = max(total_violations, global_violations)

        return {
            "all_within_limit": all_within_limit,
            "limit_gb": limit_gb,
            "limit_type": limit_type,
            "total_violations": total_violations,
            "top_memory_models": model_memory_usage,  # All 2 models
        }
