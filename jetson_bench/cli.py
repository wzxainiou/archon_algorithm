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
from .dual_stream.dual_loader import DualSourceLoader
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

    # Processing mode
    parser.add_argument(
        "--mode",
        choices=["rgb", "thermal", "dual"],
        default="rgb",
        help="Processing mode: rgb (RGB only), thermal (Thermal only), dual (both streams)"
    )

    # Input source (for single stream modes: rgb or thermal)
    parser.add_argument(
        "--source",
        help="Input source for single mode: image_dir=/path, video=/path/to/video.mp4, or camera=0"
    )

    # Dual stream sources (for dual mode)
    parser.add_argument("--rgb_source", help="RGB video/camera source (for dual mode)")
    parser.add_argument("--thermal_source", help="Thermal video/camera source (for dual mode)")

    # Model weights (2 slots)
    parser.add_argument("--model0", help="Weight path for yolo11n_rgb (RGB)")
    parser.add_argument("--model1", help="Weight path for yolo11n_thermal (Thermal)")

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
        frame_detections = []  # NEW: Store per-frame detection details

        with SourceLoader(config.source_type, config.source_path, config.max_frames) as source:
            for frame, frame_id in source:
                result = model.infer(frame, frame_id)
                frame_count += 1

                # NEW: Collect detection details for this frame
                detections = []
                for i in range(result.num_detections):
                    detections.append({
                        "class": result.class_names[i],
                        "confidence": float(result.scores[i]),
                        "bbox": result.boxes[i].tolist(),  # [x1, y1, x2, y2]
                    })

                frame_detections.append({
                    "frame_name": frame_id,
                    "num_detections": result.num_detections,
                    "detections": detections,
                })

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

        # NEW: Compute class distribution
        from collections import Counter
        all_classes = []
        for frame_det in frame_detections:
            for det in frame_det["detections"]:
                all_classes.append(det["class"])

        class_distribution = dict(Counter(all_classes))
        total_detections = len(all_classes)

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
            "detection_summary": {  # NEW
                "total_detections": total_detections,
                "class_distribution": class_distribution,
                "frame_detections": frame_detections,
            },
        }

    except Exception as e:
        print(f"\n❌ Error running {model_name}: {e}")
        logging.exception(f"Error in model {model_name}")
        return {
            "name": model_name,
            "status": "failed",
            "error": str(e),
        }


def run_dual_stream_inference(rgb_model_config: ModelConfig, thermal_model_config: ModelConfig,
                               config: BenchConfig, dual_loader: DualSourceLoader,
                               metrics: MetricsAggregator, gpu_mem_manager: GPUMemoryManager) -> List[Dict]:
    """
    Run dual-stream inference with synchronized RGB and Thermal frames.

    Returns:
        List of two result dictionaries [rgb_result, thermal_result]
    """
    from collections import Counter

    # Check if models should be skipped
    rgb_skip = rgb_model_config.skip_reason
    thermal_skip = thermal_model_config.skip_reason

    if rgb_skip and thermal_skip:
        print("\n❌ Both models are skipped - cannot run dual stream mode")
        return [
            {"name": rgb_model_config.name, "status": "skipped", "skip_reason": rgb_skip},
            {"name": thermal_model_config.name, "status": "skipped", "skip_reason": thermal_skip},
        ]

    print(f"\n{'='*60}")
    print(f"Running dual-stream inference")
    print(f"RGB Model: {rgb_model_config.weight_path or 'SKIPPED'}")
    print(f"Thermal Model: {thermal_model_config.weight_path or 'SKIPPED'}")
    print(f"{'='*60}")

    # Track GPU memory
    model_gpu_memory_start = len(metrics.gpu_memory_history)

    try:
        # Initialize models
        rgb_model = None
        thermal_model = None

        if not rgb_skip:
            rgb_model = YOLOInference(
                model_name=rgb_model_config.name,
                weight_path=rgb_model_config.weight_path,
                imgsz=config.imgsz,
                conf=config.conf,
                iou=config.iou,
                gpu_mem_limit_gb=config.gpu_mem_limit_gb,
                tensorrt_workspace_size=gpu_mem_manager.get_tensorrt_workspace_size(),
            )
            print(f"RGB Backend: {rgb_model.backend}")
            rgb_model.warmup(num_iterations=3)

        if not thermal_skip:
            thermal_model = YOLOInference(
                model_name=thermal_model_config.name,
                weight_path=thermal_model_config.weight_path,
                imgsz=config.imgsz,
                conf=config.conf,
                iou=config.iou,
                gpu_mem_limit_gb=config.gpu_mem_limit_gb,
                tensorrt_workspace_size=gpu_mem_manager.get_tensorrt_workspace_size(),
            )
            print(f"Thermal Backend: {thermal_model.backend}")
            thermal_model.warmup(num_iterations=3)

        print("Running dual-stream inference...")
        frame_count = 0
        rgb_frame_detections = []
        thermal_frame_detections = []

        for frame_pair in dual_loader:
            rgb_frame = frame_pair["rgb_frame"]
            thermal_frame = frame_pair["thermal_frame"]
            frame_id = frame_pair["frame_id"]

            # Run inference on both streams
            if rgb_model:
                rgb_result = rgb_model.infer(rgb_frame, frame_id)
                rgb_detections = []
                for i in range(rgb_result.num_detections):
                    rgb_detections.append({
                        "class": rgb_result.class_names[i],
                        "confidence": float(rgb_result.scores[i]),
                        "bbox": rgb_result.boxes[i].tolist(),
                    })
                rgb_frame_detections.append({
                    "frame_name": frame_id,
                    "num_detections": rgb_result.num_detections,
                    "detections": rgb_detections,
                })

            if thermal_model:
                thermal_result = thermal_model.infer(thermal_frame, frame_id)
                thermal_detections = []
                for i in range(thermal_result.num_detections):
                    thermal_detections.append({
                        "class": thermal_result.class_names[i],
                        "confidence": float(thermal_result.scores[i]),
                        "bbox": thermal_result.boxes[i].tolist(),
                    })
                thermal_frame_detections.append({
                    "frame_name": frame_id,
                    "num_detections": thermal_result.num_detections,
                    "detections": thermal_detections,
                })

            frame_count += 1
            metrics.collect_gpu_memory()

            # Print progress every 50 frames
            if frame_count % 50 == 0:
                metrics.print_current("dual")
                print(f"  Frames: {frame_count}")

        print(f"\n✅ Completed dual-stream: {frame_count} frame pairs processed")

        # Build results
        results = []

        # RGB result
        if rgb_model:
            rgb_performance = rgb_model.get_performance()
            rgb_all_classes = []
            for fd in rgb_frame_detections:
                for det in fd["detections"]:
                    rgb_all_classes.append(det["class"])
            rgb_gpu_memory = _get_model_gpu_memory_stats(
                metrics.gpu_memory_history[model_gpu_memory_start:],
                config.gpu_mem_limit_gb,
                gpu_mem_manager.limit_type
            )
            results.append({
                "name": rgb_model_config.name,
                "status": "completed",
                "performance": rgb_performance,
                "model_info": rgb_model.get_model_info(),
                "gpu_memory": rgb_gpu_memory,
                "detection_summary": {
                    "total_detections": len(rgb_all_classes),
                    "class_distribution": dict(Counter(rgb_all_classes)),
                    "frame_detections": rgb_frame_detections,
                },
            })
        else:
            results.append({
                "name": rgb_model_config.name,
                "status": "skipped",
                "skip_reason": rgb_skip,
            })

        # Thermal result
        if thermal_model:
            thermal_performance = thermal_model.get_performance()
            thermal_all_classes = []
            for fd in thermal_frame_detections:
                for det in fd["detections"]:
                    thermal_all_classes.append(det["class"])
            thermal_gpu_memory = _get_model_gpu_memory_stats(
                metrics.gpu_memory_history[model_gpu_memory_start:],
                config.gpu_mem_limit_gb,
                gpu_mem_manager.limit_type
            )
            results.append({
                "name": thermal_model_config.name,
                "status": "completed",
                "performance": thermal_performance,
                "model_info": thermal_model.get_model_info(),
                "gpu_memory": thermal_gpu_memory,
                "detection_summary": {
                    "total_detections": len(thermal_all_classes),
                    "class_distribution": dict(Counter(thermal_all_classes)),
                    "frame_detections": thermal_frame_detections,
                },
            })
        else:
            results.append({
                "name": thermal_model_config.name,
                "status": "skipped",
                "skip_reason": thermal_skip,
            })

        return results

    except Exception as e:
        print(f"\n❌ Error in dual-stream inference: {e}")
        logging.exception("Error in dual-stream inference")
        return [
            {"name": rgb_model_config.name, "status": "failed", "error": str(e)},
            {"name": thermal_model_config.name, "status": "failed", "error": str(e)},
        ]


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
    print(f"Mode: {args.mode.upper()}")
    print("=" * 60)

    # Validate mode-specific arguments
    if args.mode == "dual":
        if not args.rgb_source or not args.thermal_source:
            print("\n❌ Dual mode requires --rgb_source and --thermal_source")
            sys.exit(1)
    else:
        if not args.source:
            print(f"\n❌ {args.mode} mode requires --source")
            sys.exit(1)

    # Verify environment
    print("\n[1/7] Verifying environment...")
    if not verify_environment():
        print("\n❌ Environment verification failed")
        sys.exit(1)

    # Initialize GPU memory manager
    print("\n[2/7] Initializing GPU memory limits...")
    gpu_mem_manager = GPUMemoryManager(limit_gb=args.gpu_mem_limit_gb)
    gpu_limit_result = gpu_mem_manager.apply_limits()

    # Create configuration based on mode
    if args.mode == "dual":
        # Dual mode - use DualSourceLoader
        config = BenchConfig(
            mode="dual",
            rgb_source=args.rgb_source,
            thermal_source=args.thermal_source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_frames=args.max_frames,
            metrics_interval=args.metrics_interval,
            gpu_mem_limit_gb=args.gpu_mem_limit_gb,
            output_dir=args.output_dir,
        )
    else:
        # Single mode (rgb or thermal)
        source_type, source_path = parse_source(args.source)
        config = BenchConfig(
            mode=args.mode,
            source_type=source_type,
            source_path=source_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_frames=args.max_frames,
            metrics_interval=args.metrics_interval,
            gpu_mem_limit_gb=args.gpu_mem_limit_gb,
            output_dir=args.output_dir,
        )

    # Set model weights
    if args.model0:
        config.set_model_weights(0, args.model0)
    if args.model1:
        config.set_model_weights(1, args.model1)

    # Validate configuration
    print("\n[3/7] Validating configuration...")

    # Validate source based on mode
    if args.mode == "dual":
        # Validate dual sources exist
        from pathlib import Path
        if not Path(args.rgb_source).exists():
            print(f"\n❌ RGB source not found: {args.rgb_source}")
            sys.exit(1)
        if not Path(args.thermal_source).exists():
            print(f"\n❌ Thermal source not found: {args.thermal_source}")
            sys.exit(1)
        print(f"RGB Source: {args.rgb_source}")
        print(f"Thermal Source: {args.thermal_source}")
    else:
        if not config.validate_source():
            print("\n❌ Source validation failed")
            sys.exit(1)
        print(f"Source: {config.source_type} = {config.source_path}")

    config.validate_models()

    print(f"Mode: {config.mode}")
    print(f"Models:")
    for i, model in enumerate(config.models):
        status = "✓" if model.skip_reason is None else "⚠"
        print(f"  {status} [{i}] {model.name}: {model.weight_path or 'NOT SET'}")
        if model.skip_reason:
            print(f"      Reason: {model.skip_reason}")

    # Check model requirements based on mode
    if args.mode == "rgb" and config.models[0].skip_reason:
        print("\n❌ RGB mode requires --model0")
        sys.exit(1)
    elif args.mode == "thermal" and config.models[1].skip_reason:
        print("\n❌ Thermal mode requires --model1")
        sys.exit(1)
    elif args.mode == "dual":
        if config.models[0].skip_reason and config.models[1].skip_reason:
            print("\n❌ Dual mode requires at least one model (--model0 or --model1)")
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

    # Get source info and run inference based on mode
    print("\n[5/7] Loading input source...")

    results = []
    source_info = {"mode": args.mode, "total_frames_processed": 0}

    if args.mode == "dual":
        # Dual stream mode
        with DualSourceLoader(
            source_type="video",
            rgb_source=args.rgb_source,
            thermal_source=args.thermal_source,
            target_fps=5.0,
            max_frames=config.max_frames,
        ) as dual_loader:
            source_info.update(dual_loader.get_source_info())
            print(f"Source info: {source_info}")

            print("\n[6/7] Running dual-stream inference...")
            results = run_dual_stream_inference(
                config.models[0], config.models[1],
                config, dual_loader, metrics, gpu_mem_manager
            )

            # Update frames processed
            for r in results:
                if r.get("status") == "completed":
                    source_info["total_frames_processed"] = r.get("performance", {}).get("total_frames", 0)
                    break

    else:
        # Single stream mode (rgb or thermal)
        with SourceLoader(config.source_type, config.source_path, config.max_frames) as source:
            source_info.update(source.get_source_info())
            print(f"Source info: {source_info}")

        print("\n[6/7] Running inference...")

        if args.mode == "rgb":
            # Only run RGB model (model0)
            result = run_model_inference(config.models[0], config, loader=None, metrics=metrics, gpu_mem_manager=gpu_mem_manager)
            results.append(result)
            # Add skipped thermal
            results.append({
                "name": config.models[1].name,
                "status": "skipped",
                "skip_reason": "Not used in RGB mode",
            })
        else:  # thermal mode
            # Add skipped RGB
            results.append({
                "name": config.models[0].name,
                "status": "skipped",
                "skip_reason": "Not used in Thermal mode",
            })
            # Only run Thermal model (model1)
            result = run_model_inference(config.models[1], config, loader=None, metrics=metrics, gpu_mem_manager=gpu_mem_manager)
            results.append(result)

        # Update total frames processed
        for r in results:
            if r.get("status") == "completed":
                source_info["total_frames_processed"] = r.get("performance", {}).get("total_frames", 0)
                break

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
            "mode": config.mode,
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
