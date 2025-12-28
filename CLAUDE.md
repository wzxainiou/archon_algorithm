# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Jetson Orin Nano multi-model inference benchmarking suite** for YOLO object detection models. The project evaluates exactly 2 predefined model slots (RGB and Thermal) with comprehensive performance and system resource metrics.

**Key Constraints**:
1. The architecture is fixed to exactly 2 model slots. This cannot be changed.
2. **🔒 GPU memory is strictly limited to 8GB maximum**. This is enforced at runtime.

## Running the Benchmark

### Processing Modes

The benchmark supports three processing modes via `--mode`:

| Mode | Description | Required Args |
|------|-------------|---------------|
| `rgb` | Process RGB stream only | `--source`, `--model0` |
| `thermal` | Process Thermal stream only | `--source`, `--model1` |
| `dual` | Process both streams simultaneously | `--rgb_source`, `--thermal_source`, `--model0`/`--model1` |

### Common Examples

```bash
# RGB mode - single stream
python3 -m jetson_bench.cli --mode rgb \
    --source video=/path/to/rgb.mp4 \
    --model0 yolo11n.pt

# Thermal mode - single stream
python3 -m jetson_bench.cli --mode thermal \
    --source video=/path/to/thermal.mp4 \
    --model1 yolo11n.pt

# Dual mode - synchronized dual streams
python3 -m jetson_bench.cli --mode dual \
    --rgb_source /path/to/rgb.mp4 \
    --thermal_source /path/to/thermal.mp4 \
    --model0 yolo11n.pt \
    --model1 yolo11n.pt

# Image directory (RGB mode)
python3 -m jetson_bench.cli --mode rgb \
    --source image_dir=/path/to/images \
    --model0 /path/to/model.engine
```

### Running Tests
```bash
python3 -m pytest tests/ -v
```

### Exporting to TensorRT
```bash
./scripts/export_trt.sh /path/to/model.pt 640
# Or manually:
yolo export model=/path/to/model.pt format=engine imgsz=640 device=0 half=True
```

## Architecture

### Critical Design Constraints

1. **Exactly 2 Model Slots** (IMMUTABLE):
   - Slot 0: `yolo11n_rgb` (RGB camera input)
   - Slot 1: `yolo11n_thermal` (Thermal camera input)

   Users can skip models (by not providing weights) but cannot add/remove slots.

2. **Model Backend Priority**:
   - TensorRT (.engine) → ONNX (.onnx) → PyTorch (.pt)
   - TensorRT engines are platform-specific and must be exported on target Jetson

3. **🔒 GPU Memory Limitation** (NEW - CRITICAL):
   - **Hard limit of 8GB GPU memory** (cannot exceed, clamped in config)
   - Enforced via `PyTorch CUDA memory fraction` (preferred, hard)
   - TensorRT workspace limited to 4GB (50% of 8GB limit)
   - Batch size forced to 1
   - Real-time monitoring with violation tracking
   - All violations logged and reported

4. **Jetson-Specific Behavior**:
   - GPU memory and system RAM are unified (shared physical memory)
   - Metrics are collected via `tegrastats` (Jetson-specific) + `psutil` (cross-platform)
   - Tegrastats requires appropriate permissions (video group or sudo)

### Module Organization

- **[jetson_bench/config.py](jetson_bench/config.py)**: Configuration with 2-model validation, source validation, **GPU memory limit**
- **[jetson_bench/gpu_memory.py](jetson_bench/gpu_memory.py)**: **GPU memory management and enforcement** (NEW)
- **[jetson_bench/loader.py](jetson_bench/loader.py)**: Unified input source loader (images/video/camera)
- **[jetson_bench/infer/](jetson_bench/infer/)**:
  - `backend.py`: Backend detection and compatibility checking
  - `yoloultralytics.py`: YOLO inference wrapper with performance tracking, **memory constraints**
- **[jetson_bench/metrics/](jetson_bench/metrics/)**:
  - `tegrastats.py`: Jetson GPU metrics parser (parses tegrastats output)
  - `sysmetrics.py`: System metrics via psutil
  - `aggregator.py`: Combines metrics + **GPU memory tracking with violation detection** (NEW)
- **[jetson_bench/report/](jetson_bench/report/)**:
  - `writer.py`: Generates JSON and Markdown reports with **Memory Safety Summary** (NEW)
- **[jetson_bench/cli.py](jetson_bench/cli.py)**: Main entry point, orchestrates benchmark, **initializes GPU memory manager**

### Key Data Flow

1. CLI parses arguments → creates `BenchConfig` (with GPU memory limit)
2. **Initialize `GPUMemoryManager` and apply limits** (NEW)
   - PyTorch CUDA memory fraction set
   - TensorRT workspace size determined
   - Limit type determined (hard/soft/unavailable)
3. Config validates source and all 2 models
4. Metrics aggregator starts background monitoring (tegrastats + psutil + **GPU memory**)
5. For each active model:
   - Load model with appropriate backend **and memory constraints**
   - **TensorRT workspace limit applied if TensorRT backend**
   - Run inference on all frames from source
   - **Collect GPU memory usage every frame**
   - Collect per-frame latency and detection counts
   - **Check for memory violations in real-time**
6. Stop metrics collection
7. Generate JSON + Markdown reports with all metrics + **Memory Safety Summary**

## Important Implementation Details

### Error Handling Philosophy

- **Models**: Individual model failures do not stop execution
  - Skipped models (no weights) → status: "skipped"
  - Failed models (load/inference error) → status: "failed"
  - All 2 models appear in final report with their status

- **Metrics**: Metric collection failures generate warnings but don't crash
  - Missing tegrastats → warning + limited GPU metrics
  - Metric parsing errors → logged but continue

### Tegrastats Parsing

The tegrastats parser ([jetson_bench/metrics/tegrastats.py:61](jetson_bench/metrics/tegrastats.py#L61)) uses regex to extract:
- RAM: `RAM 2505/7775MB`
- GPU: `GR3D_FREQ 45%` or `GR3D 45%`
- CPU: Average of `[2%@2035,0%@2035,...]`
- Temperature: `gpu@36.5C`, `tj@38.25C`
- Power: `1234mW`

Tegrastats output format varies by JetPack version, so the parser is designed to be flexible.

### Input Source Handling

The `SourceLoader` class provides a unified interface:
- Image directory: Sorts files, filters by extension (.jpg/.png/.bmp)
- Video: Uses cv2.VideoCapture with frame limit
- Camera: Live stream with frame limit

All sources implement iterator protocol for consistent usage.

## Common Development Tasks

### Adding Support for New Model Architectures

**DO NOT** add a 3rd model slot. The 2-slot design is intentional.

To support different model architectures within the existing slots:
1. Ensure Ultralytics supports the model format
2. Update backend detection if new format (unlikely)
3. Test with existing CLI - should work automatically

### Modifying Metrics Collection

When adding new metrics:
1. Add parsing logic to `tegrastats.py` or `sysmetrics.py`
2. Update `aggregator.py` to include in summary
3. Update `writer.py` to display in reports
4. Add tests for parsing logic

### Changing Report Format

To modify reports:
- JSON: Edit `ReportWriter.create_full_report()` structure
- Markdown: Edit `ReportWriter._write_model_section()` and `_write_metrics_section()`

Both formats should remain in sync with same data.

## 🔒 GPU Memory Management (NEW - CRITICAL)

### Implementation Overview

GPU memory is strictly limited to **8GB maximum** through a multi-layered approach:

1. **Hard Enforcement** ([gpu_memory.py:43](jetson_bench/gpu_memory.py#L43)):
   ```python
   torch.cuda.set_per_process_memory_fraction(limit_gb / total_gb, device=0)
   ```
   - Sets GPU driver-level memory limit
   - Most reliable method when CUDA is available

2. **TensorRT Workspace Limit** ([gpu_memory.py:118](jetson_bench/gpu_memory.py#L118)):
   - Workspace size = 4GB (50% of 8GB limit)
   - Prevents TensorRT from over-allocating

3. **Batch Size = 1** ([yoloultralytics.py:119](jetson_bench/infer/yoloultralytics.py#L119)):
   - Forces single-frame inference
   - Prevents batch-related memory spikes

4. **Real-Time Monitoring** ([aggregator.py:68](jetson_bench/metrics/aggregator.py#L68)):
   - Collects GPU memory every frame
   - Logs ERROR if > 8GB detected
   - Tracks violations for reporting

### Key Files for GPU Memory

- **[jetson_bench/gpu_memory.py](jetson_bench/gpu_memory.py)**: Core GPU memory management
  - `GPUMemoryManager.apply_limits()`: Applies all enforcement methods
  - `get_current_gpu_memory_usage()`: Reads current GPU memory (PyTorch/NVML)
  - `validate_memory_within_limit()`: Checks compliance

- **[jetson_bench/metrics/aggregator.py](jetson_bench/metrics/aggregator.py)**: GPU memory tracking
  - `collect_gpu_memory()`: Called every frame during inference
  - `_get_gpu_memory_summary()`: Computes peak/mean/violations

- **[jetson_bench/report/writer.py](jetson_bench/report/writer.py)**: Memory reporting
  - `_write_memory_safety_section()`: Top-level memory summary
  - `_create_memory_safety_summary()`: Aggregates model memory stats

### Memory Limit Types

When `GPUMemoryManager.apply_limits()` runs, it determines the limit type:

- **"hard"**: PyTorch CUDA limit successfully applied
  - Memory enforced by GPU driver
  - Most reliable

- **"soft"**: Only environment variables set
  - Depends on backend cooperation
  - Less reliable

- **"unavailable"**: No enforcement possible
  - PyTorch/CUDA not available
  - Monitoring only

### Handling Memory Violations

**Detection** ([aggregator.py:95](jetson_bench/metrics/aggregator.py#L95)):
```python
if gpu_mem_gb > self.gpu_mem_limit_gb:
    logger.error(f"❌ GPU MEMORY VIOLATION: {gpu_mem_gb:.2f}GB exceeds {limit_gb:.2f}GB")
    self.memory_violations.append({...})
```

**Reporting**:
- Violations logged to console + `run.log`
- Model marked with ❌ in report
- Memory Safety Summary shows total violations
- Per-model GPU memory section shows peak/mean

**Philosophy**:
- DO NOT crash on violation (allow benchmark to complete)
- DO log at ERROR level for visibility
- DO report prominently in outputs

### Adding GPU Memory to New Features

If adding new inference paths:

1. Pass `gpu_mem_limit_gb` to model initialization
2. Pass `tensorrt_workspace_size` from `GPUMemoryManager`
3. Call `metrics.collect_gpu_memory()` during inference
4. Include `gpu_memory` dict in result

Example:
```python
model = YOLOInference(
    ...,
    gpu_mem_limit_gb=config.gpu_mem_limit_gb,
    tensorrt_workspace_size=gpu_mem_manager.get_tensorrt_workspace_size(),
)

# During inference loop
metrics.collect_gpu_memory()

# In result
result["gpu_memory"] = {
    "limit_gb": limit_gb,
    "peak_gb": peak,
    "mean_gb": mean,
    "within_limit": peak <= limit_gb,
    "limit_type": limit_type,
}
```

## Platform-Specific Notes

### Jetson vs Non-Jetson

The code detects Jetson by checking `/etc/nv_tegra_release`:
- **On Jetson**: Full tegrastats monitoring with GPU/temp/power metrics
- **Non-Jetson**: Warning + psutil-only metrics (no GPU data)

TensorRT export script warns if not on Jetson but allows export (for cross-compilation scenarios).

### OpenCV on Jetson

Jetson's JetPack includes OpenCV with CUDA support pre-installed. The requirements.txt includes `opencv-python` as fallback, but Jetson users should use the system OpenCV:
```bash
# Check system OpenCV
python3 -c "import cv2; print(cv2.getBuildInformation())"
```

## Testing Strategy

- **Unit tests**: Config validation, tegrastats parsing
- **Integration tests**: (Not yet implemented) End-to-end with dummy models
- **Manual testing**: Run on actual Jetson with real models

When adding features, add corresponding tests in `tests/`.

## Performance Considerations

### Memory Management

- Models are loaded sequentially (not kept in memory simultaneously)
- Frame-by-frame processing (no batch accumulation)
- Metrics stored as list (grows with time) - acceptable for typical runs

### Optimization Opportunities

If performance issues arise:
1. Enable batch inference (requires loader changes)
2. Implement true parallel model execution
3. Add option to limit metrics history size
4. Use multiprocessing for metrics collection

## Debugging Tips

### Enable Verbose Logging
```bash
python3 -m jetson_bench.cli --verbose ...
```

### Check Tegrastats Manually
```bash
tegrastats --interval 500
```

### Verify Model Loading
```python
from ultralytics import YOLO
model = YOLO("/path/to/model.engine")
print(model.model)
```

### Check Output Files
All outputs go to timestamped directory:
- `report.json`: Machine-readable results
- `report.md`: Human-readable summary
- `metrics.jsonl`: Time-series data for analysis
- `run.log`: Full execution log
