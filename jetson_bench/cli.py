"""
Command-line interface for Jetson benchmarking.

Main entry point for running multi-model inference benchmarks.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from .config import BenchConfig, ModelConfig, verify_environment
from .loader import SourceLoader
from .infer.yoloultralytics import YOLOInference
from .metrics.aggregator import MetricsAggregator
from .report.writer import ReportWriter
from .gpu_memory import GPUMemoryManager


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Jetson Orin Nano Multi-Model Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on image directory
  python -m jetson_bench.cli --source image_dir=/path/to/images \\
      --model0 /path/to/yolo11n_rgb.engine \\
      --model1 /path/to/yolo11s_rgb.engine

  # Run on video with max frames
  python -m jetson_bench.cli --source video=/path/to/video.mp4 --max_frames 300

  # Run on camera
  python -m jetson_bench.cli --source camera=0 --max_frames 100
        """
    )

    # Input source (mutually exclusive)
    parser.add_argument(
        "--source",
        required=True,
        help="Input source: image_dir=/path, video=/path/to/video.mp4, or camera=0"
    )

    # Model weights (4 slots)
    parser.add_argument("--model0", help="Weight path for yolo11n_rgb")
    parser.add_argument("--model1", help="Weight path for yolo11s_rgb")
    parser.add_argument("--model2", help="Weight path for yolo11n_thermal")
    parser.add_argument("--model3", help="Weight path for yolov8n_thermal")

    # Inference parameters
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold (default: 0.45)")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames to process (default: 300)")

    # Execution mode
    parser.add_argument("--parallel", action="store_true", help="Run models in parallel (default: sequential)")

    # Metrics
    parser.add_argument("--metrics_interval", type=float, default=0.5,
                       help="Metrics collection interval in seconds (default: 0.5)")

    # GPU Memory (NEW)
    parser.add_argument("--gpu_mem_limit_gb", type=float, default=8.0,
                       help="GPU memory limit in GB (default: 8.0, max: 8.0)")

    # Output
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"),
                       help="Output directory (default: outputs)")

    # Misc
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def parse_source(source_str: str) -> tuple[str, str]:
    """
    Parse source string.

    Examples:
        image_dir=/path/to/images -> ("image_dir", "/path/to/images")
        video=/path/to/video.mp4 -> ("video", "/path/to/video.mp4")
        camera=0 -> ("camera", "0")
    """
    if "=" not in source_str:
        print(f"Error: Invalid source format: {source_str}", file=sys.stderr)
        print("Expected format: image_dir=/path, video=/path, or camera=0", file=sys.stderr)
        sys.exit(1)

    source_type, source_path = source_str.split("=", 1)

    valid_types = ["image_dir", "video", "camera"]
    if source_type not in valid_types:
        print(f"Error: Invalid source type: {source_type}", file=sys.stderr)
        print(f"Valid types: {', '.join(valid_types)}", file=sys.stderr)
        sys.exit(1)

    return source_type, source_path


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_model_inference(model_config: ModelConfig, config: BenchConfig,
                       loader: SourceLoader, metrics: MetricsAggregator,
                       gpu_mem_manager: GPUMemoryManager) -> Dict:
    """
    Run inference for a single model.

    Returns:
        Result dictionary with performance metrics or error information
    """
    model_name = model_config.name

    # Check if model should be skipped
    if model_config.skip_reason:
        print(f"\n⚠️  Skipping {model_name}: {model_config.skip_reason}")
        return {
            "name": model_name,
            "status": "skipped",
            "skip_reason": model_config.skip_reason,
        }

    print(f"\n{'='*60}")
    print(f"Running inference: {model_name}")
    print(f"Weight: {model_config.weight_path}")
    print(f"{'='*60}")

    # Track GPU memory for this model
    model_gpu_memory_start = len(metrics.gpu_memory_history)

    try:
        # Initialize model with GPU memory constraints
        model = YOLOInference(
            model_name=model_name,
            weight_path=model_config.weight_path,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            gpu_mem_limit_gb=config.gpu_mem_limit_gb,
            tensorrt_workspace_size=gpu_mem_manager.get_tensorrt_workspace_size(),
        )

        print(f"Backend: {model.backend}")
        print("Warming up model...")
        model.warmup(num_iterations=5)

        # Run inference on all frames
        print("Running inference...")
        frame_count = 0

        with SourceLoader(config.source_type, config.source_path, config.max_frames) as source:
            for frame, frame_id in source:
                result = model.infer(frame, frame_id)
                frame_count += 1

                # Collect GPU memory every frame
                metrics.collect_gpu_memory()

                # Print progress every 50 frames
                if frame_count % 50 == 0:
                    metrics.print_current(model_name)
                    print(f"  Frames: {frame_count}, FPS: {1000/result.latency_ms:.1f}, "
                          f"Detections: {result.num_detections}")

        # Get performance metrics
        performance = model.get_performance()
        print(f"\n✅ Completed {model_name}: {frame_count} frames processed")
        print(f"   FPS: {performance['fps']:.2f}")
        print(f"   Latency P50/P90/P99: {performance['latency_ms']['p50']:.1f}/"
              f"{performance['latency_ms']['p90']:.1f}/{performance['latency_ms']['p99']:.1f} ms")

        # Get GPU memory stats for this model
        model_gpu_memory = _get_model_gpu_memory_stats(
            metrics.gpu_memory_history[model_gpu_memory_start:],
            config.gpu_mem_limit_gb,
            gpu_mem_manager.limit_type
        )

        return {
            "name": model_name,
            "status": "completed",
            "performance": performance,
            "model_info": model.get_model_info(),
            "gpu_memory": model_gpu_memory,
        }

    except Exception as e:
        print(f"\n❌ Error running {model_name}: {e}")
        logging.exception(f"Error in model {model_name}")
        return {
            "name": model_name,
            "status": "failed",
            "error": str(e),
        }


def _get_model_gpu_memory_stats(gpu_snapshots: List, limit_gb: float, limit_type: str) -> Dict:
    """Get GPU memory statistics for a model."""
    if not gpu_snapshots:
        return {
            "limit_gb": limit_gb,
            "peak_gb": 0,
            "mean_gb": 0,
            "within_limit": True,
            "limit_type": limit_type,
        }

    import numpy as np
    memory_values = [s.used_gb for s in gpu_snapshots]
    peak_gb = max(memory_values)
    mean_gb = np.mean(memory_values)

    return {
        "limit_gb": limit_gb,
        "peak_gb": float(peak_gb),
        "mean_gb": float(mean_gb),
        "within_limit": peak_gb <= limit_gb,
        "limit_type": limit_type,
    }


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    print("=" * 60)
    print("Jetson Orin Nano Multi-Model Benchmark")
    print("=" * 60)

    # Verify environment
    print("\n[1/7] Verifying environment...")
    if not verify_environment():
        print("\n❌ Environment verification failed")
        sys.exit(1)

    # Initialize GPU memory manager (NEW)
    print("\n[2/7] Initializing GPU memory limits...")
    gpu_mem_manager = GPUMemoryManager(limit_gb=args.gpu_mem_limit_gb)
    gpu_limit_result = gpu_mem_manager.apply_limits()

    # Parse source
    source_type, source_path = parse_source(args.source)

    # Create configuration
    config = BenchConfig(
        source_type=source_type,
        source_path=source_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_frames=args.max_frames,
        parallel=args.parallel,
        metrics_interval=args.metrics_interval,
        gpu_mem_limit_gb=args.gpu_mem_limit_gb,
        output_dir=args.output_dir,
    )

    # Set model weights
    if args.model0:
        config.set_model_weights(0, args.model0)
    if args.model1:
        config.set_model_weights(1, args.model1)
    if args.model2:
        config.set_model_weights(2, args.model2)
    if args.model3:
        config.set_model_weights(3, args.model3)

    # Validate configuration
    print("\n[3/7] Validating configuration...")
    if not config.validate_source():
        print("\n❌ Source validation failed")
        sys.exit(1)

    config.validate_models()

    print(f"Source: {config.source_type} = {config.source_path}")
    print(f"Models:")
    for i, model in enumerate(config.models):
        status = "✓" if model.skip_reason is None else "⚠"
        print(f"  {status} [{i}] {model.name}: {model.weight_path or 'NOT SET'}")
        if model.skip_reason:
            print(f"      Reason: {model.skip_reason}")

    active_models = config.get_active_models()
    if len(active_models) == 0:
        print("\n❌ No models available to run (all 4 are skipped)")
        print("Please provide at least one valid model weight path using --model0, --model1, --model2, or --model3")
        sys.exit(1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = config.output_dir / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {run_output_dir}")

    # Setup logging to file
    log_file = run_output_dir / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Start metrics collection
    print("\n[4/7] Starting metrics collection...")
    metrics = MetricsAggregator(interval_sec=config.metrics_interval, gpu_mem_limit_gb=config.gpu_mem_limit_gb)
    metrics.start()

    # Get source info
    print("\n[5/7] Loading input source...")
    with SourceLoader(config.source_type, config.source_path, config.max_frames) as source:
        source_info = source.get_source_info()
        source_info["total_frames_processed"] = 0

    print(f"Source info: {source_info}")

    # Run inference on all models
    print("\n[6/7] Running inference...")
    results = []

    if config.parallel:
        print("⚠️  Parallel mode not yet implemented - running sequentially")

    # Sequential execution
    for model_config in config.models:
        result = run_model_inference(model_config, config, loader=None, metrics=metrics, gpu_mem_manager=gpu_mem_manager)
        results.append(result)

        # Update total frames processed
        if result["status"] == "completed":
            source_info["total_frames_processed"] = result["performance"]["total_frames"]

    # Stop metrics collection
    print("\n[7/7] Generating reports...")
    metrics.stop()

    # Generate reports
    writer = ReportWriter(run_output_dir)

    metrics_summary = metrics.get_summary()

    report_data = writer.create_full_report(
        models_results=results,
        metrics_summary=metrics_summary,
        source_info=source_info,
        config={
            "imgsz": config.imgsz,
            "conf": config.conf,
            "iou": config.iou,
            "gpu_mem_limit_gb": config.gpu_mem_limit_gb,
        },
        memory_limit_type=gpu_mem_manager.limit_type
    )

    writer.write_json(report_data)
    writer.write_markdown(report_data)

    # Save time-series metrics
    metrics.save_timeseries(run_output_dir / "metrics.jsonl")
    print(f"📄 Metrics time-series: {run_output_dir / 'metrics.jsonl'}")

    print("\n" + "=" * 60)
    print("✅ Benchmark completed successfully!")
    print(f"📁 Results: {run_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
