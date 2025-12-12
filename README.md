# Jetson Orin Nano Multi-Model Inference Benchmark

A comprehensive benchmarking tool for evaluating YOLO models on NVIDIA Jetson Orin Nano, with detailed system resource monitoring and performance metrics.

## Features

- **Fixed 4-model architecture**: Benchmarks exactly 4 predefined model slots
- **Multi-backend support**: TensorRT (.engine) > ONNX (.onnx) > PyTorch (.pt)
- **🔒 GPU Memory Limitation**: Strict 8GB GPU memory limit with real-time monitoring
- **Comprehensive metrics**: FPS, latency (P50/P90/P99), detections per frame, GPU memory usage
- **System monitoring**: CPU, memory, GPU, temperature, power via tegrastats and psutil
- **Flexible input**: Images, videos, or camera streams
- **Robust error handling**: Models can fail/skip without affecting others
- **Detailed reports**: JSON (machine-readable) + Markdown (human-readable) with memory safety summary

## System Requirements

- **Hardware**: NVIDIA Jetson Orin Nano Super Developer Kit (8GB RAM)
- **OS**: JetPack 5.x or later
- **Python**: 3.10+
- **Dependencies**: See [requirements.txt](requirements.txt)

## Installation

### 1. On Jetson Orin Nano

```bash
# Clone the repository
git clone <repository-url>
cd jetson_bench

# Install system dependencies (if needed)
sudo apt-get update
sudo apt-get install python3-pip python3-dev

# Install Python dependencies
pip3 install -r requirements.txt

# Note: On Jetson, OpenCV is typically pre-installed with CUDA support
# If you need to install it manually:
# sudo apt-get install python3-opencv
```

### 2. Verify Installation

```bash
python3 -m jetson_bench.cli --help
```

You should see the help message with all available options.

## Quick Start

### Minimal Example (Image Directory)

```bash
python3 -m jetson_bench.cli \
    --source image_dir=/path/to/images \
    --model0 /path/to/yolo11n_rgb.engine
```

This will:
1. Run inference on the first model slot (yolo11n_rgb)
2. Skip the other 3 models (no weights provided)
3. Process images from the directory
4. Generate reports in `outputs/YYYY-MM-DD_HH-MM-SS/`

### Full Example (All 4 Models)

```bash
python3 -m jetson_bench.cli \
    --source image_dir=/path/to/rgb_images \
    --model0 /path/to/yolo11n_rgb.engine \
    --model1 /path/to/yolo11s_rgb.engine \
    --model2 /path/to/yolo11n_thermal.engine \
    --model3 /path/to/yolov8n_thermal.engine \
    --imgsz 640 \
    --conf 0.25 \
    --max_frames 300
```

### Video Input

```bash
python3 -m jetson_bench.cli \
    --source video=/path/to/video.mp4 \
    --model0 /path/to/model.engine \
    --max_frames 300
```

### Camera Input

```bash
python3 -m jetson_bench.cli \
    --source camera=0 \
    --model0 /path/to/model.engine \
    --max_frames 100
```

## Command-Line Options

### Required

- `--source`: Input source (one of):
  - `image_dir=/path/to/images` - Directory of images
  - `video=/path/to/video.mp4` - Video file
  - `camera=0` - Camera device index

### Model Weights (Optional)

- `--model0`: Weight path for `yolo11n_rgb`
- `--model1`: Weight path for `yolo11s_rgb`
- `--model2`: Weight path for `yolo11n_thermal`
- `--model3`: Weight path for `yolov8n_thermal`

**Note**: At least one model weight must be provided. Models without weights will be skipped.

### Inference Parameters

- `--imgsz`: Input image size (default: 640)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IOU threshold for NMS (default: 0.45)
- `--max_frames`: Maximum frames to process (default: 300)

### GPU Memory (NEW)

- `--gpu_mem_limit_gb`: GPU memory limit in GB (default: 8.0, max: 8.0)
  - Values > 8.0 are automatically clamped to 8.0
  - Enforces strict memory constraints on local GPU

### Other Options

- `--metrics_interval`: Metrics collection interval in seconds (default: 0.5)
- `--output_dir`: Output directory (default: `outputs`)
- `--parallel`: Run models in parallel (not yet implemented, runs sequentially)
- `--verbose`: Enable verbose logging

## 🔒 GPU Memory Limitation

This project enforces a **strict 8GB GPU memory limit** to ensure models can run on local GPU hardware with limited VRAM.

### How It Works

The GPU memory limitation is enforced through multiple mechanisms:

1. **PyTorch CUDA Memory Fraction** (Hard Limit - Preferred)
   ```python
   torch.cuda.set_per_process_memory_fraction(fraction=limit_gb/total_gb)
   ```
   - Sets a hard limit on GPU memory allocation
   - Most effective method when PyTorch with CUDA is available

2. **TensorRT Workspace Size** (Hard Limit)
   - Limits TensorRT engine workspace to 50% of memory limit (4GB default)
   - Prevents TensorRT from allocating excessive memory

3. **Batch Size = 1** (Hard Constraint)
   - Forces single-frame inference to minimize memory footprint
   - Prevents dynamic batch expansion

4. **Real-Time Monitoring** (Verification)
   - Continuously tracks GPU memory usage during inference
   - Logs ERROR-level messages if limit is exceeded
   - Reports violations in final output

### Memory Limit Types

- **hard**: PyTorch CUDA limit successfully applied - memory enforced by GPU driver
- **soft**: Only environment variables set - depends on backend cooperation
- **unavailable**: No enforcement possible - monitoring only

### Jetson vs. Local GPU

**Important Difference**:

- **Jetson Orin Nano**: Uses **unified memory architecture**
  - GPU and system RAM share the same physical 8GB memory
  - "GPU memory" and "system memory" are the same resource
  - Tegrastats reports combined RAM usage

- **Local GPU (e.g., RTX 3080)**: Has **dedicated VRAM**
  - GPU has separate VRAM (e.g., 10GB, 12GB, 24GB)
  - System RAM is separate
  - This project limits GPU VRAM to 8GB maximum

### Verifying Memory Compliance

Every benchmark report includes a **Memory Safety Summary**:

```markdown
## 🔒 Memory Safety Summary

✅ **Overall Status**: All models operated within 8GB GPU memory limit

- **GPU Memory Limit**: 8.0GB
- **Limit Type**: hard
- **Total Violations**: 0

**Models Closest to Memory Limit**:
1. **yolo11s_rgb**: 5.23GB (65.4% of limit)
2. **yolo11n_rgb**: 3.87GB (48.4% of limit)
```

Each model's section also includes GPU memory metrics:

```markdown
- **GPU Memory** ✅:
  - Limit: 8.0GB
  - Peak: 5.23GB
  - Mean: 4.98GB
  - Within Limit: True
  - Limit Type: hard
```

### What Happens If Limit Is Exceeded?

If GPU memory exceeds 8GB during inference:

1. **Logging**: ERROR-level message in console and `run.log`
   ```
   ❌ GPU MEMORY VIOLATION: GPU memory 8.24GB exceeds limit 8.00GB
   ```

2. **Report Marking**: Model marked with violation in report
   ```markdown
   - **GPU Memory** ❌:
     - Peak: 8.24GB
     - Within Limit: False
   ```

3. **Memory Safety Summary**: Shows total violations
   ```markdown
   ❌ **Overall Status**: ⚠️ **MEMORY LIMIT EXCEEDED**
   - **Total Violations**: 3
   ```

4. **No Crash**: Program continues (unless actual OOM occurs)
   - Violations are recorded but don't halt execution
   - Allows completing benchmark even with memory issues

### Optimizing for 8GB Limit

If models exceed the 8GB limit:

1. **Reduce Input Size**: `--imgsz 480` or `--imgsz 320`
2. **Use Smaller Models**: nano instead of small
3. **Export to TensorRT**: Most memory-efficient backend
4. **Reduce Max Frames**: Lower `--max_frames` (doesn't reduce peak but helps overall)

## Exporting Models to TensorRT

For optimal performance on Jetson, convert your PyTorch models to TensorRT:

```bash
# Using the provided script
./scripts/export_trt.sh /path/to/yolo11n.pt 640

# Or manually with Ultralytics CLI
yolo export model=/path/to/yolo11n.pt format=engine imgsz=640 device=0 half=True
```

**Important**: TensorRT engines are platform-specific and must be exported on the target Jetson device.

## Output Files

Each run creates a timestamped directory under `outputs/`:

```
outputs/2025-12-13_10-30-00/
├── report.json          # Machine-readable results (JSON)
├── report.md            # Human-readable report (Markdown)
├── metrics.jsonl        # Time-series metrics (JSONL)
└── run.log              # Execution log
```

### Report Contents

- **🔒 Memory Safety Summary** (NEW):
  - Overall compliance status
  - GPU memory limit and type (hard/soft/unavailable)
  - Total violations
  - Top memory-consuming models
- **Metadata**: Timestamp, platform, Python version, GPU memory limit
- **Input Source**: Type, path, frames processed
- **Model Performance**:
  - FPS
  - Latency (P50, P90, P99)
  - Detections per frame (mean, min, max)
  - Backend used (TensorRT/ONNX/PyTorch)
  - **GPU Memory** (NEW): Peak, mean, within limit status
- **System Metrics**:
  - CPU utilization (%)
  - Memory usage (MB, %)
  - GPU utilization (%) - from tegrastats
  - **GPU Memory Usage** (NEW): Peak, mean, violations
  - Temperature (°C)
  - Power consumption (W)

## Architecture

### The 4 Model Slots

The benchmark is designed for exactly 4 model slots:

1. **yolo11n_rgb** - YOLO11 nano for RGB images
2. **yolo11s_rgb** - YOLO11 small for RGB images
3. **yolo11n_thermal** - YOLO11 nano for thermal images
4. **yolov8n_thermal** - YOLOv8 nano for thermal images

You cannot add or remove slots, but you can:
- Skip any slot by not providing weights
- Replace weights for any slot with your own model

### Backend Priority

When loading models, the system prefers:

1. **TensorRT** (.engine) - Fastest, Jetson-optimized
2. **ONNX** (.onnx) - Portable, good performance
3. **PyTorch** (.pt) - Most compatible, slower

### Metrics Collection

- **psutil**: CPU, RAM, swap usage
- **tegrastats**: GPU utilization, temperature, power (Jetson-specific)

**Note**: On Jetson, GPU memory and system RAM share the same physical memory (unified memory architecture).

## Troubleshooting

### TensorRT Engine Not Compatible

**Problem**: `Error loading TensorRT engine`

**Solution**: TensorRT engines are platform-specific. Re-export on the target Jetson:
```bash
./scripts/export_trt.sh /path/to/model.pt 640
```

### Tegrastats Permission Denied

**Problem**: `tegrastats: permission denied`

**Solution**: Run with sudo or add user to appropriate group:
```bash
sudo usermod -a -G video $USER
# Log out and back in
```

### Camera Not Opening

**Problem**: `Failed to open camera: 0`

**Solutions**:
- Check camera connection: `ls /dev/video*`
- Try different camera index: `--source camera=1`
- Check permissions: `sudo usermod -a -G video $USER`
- Test with: `v4l2-ctl --list-devices`

### Out of Memory

**Problem**: Process killed due to OOM

**Solutions**:
- Reduce `--imgsz` (e.g., try 320 or 480)
- Reduce `--max_frames`
- Use smaller models (e.g., nano instead of small)
- Run models sequentially instead of parallel
- Close other applications

### No Images Found

**Problem**: `No valid images found in: /path`

**Solutions**:
- Check path exists and is correct
- Ensure images have valid extensions (.jpg, .jpeg, .png, .bmp)
- Check file permissions: `ls -la /path/to/images`

## Testing

Run the test suite:

```bash
# Install pytest
pip3 install pytest

# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_config.py -v
```

## Project Structure

```
jetson_bench/
├── README.md
├── requirements.txt
├── jetson_bench/
│   ├── __init__.py
│   ├── cli.py              # Main entry point
│   ├── config.py           # Configuration and validation
│   ├── loader.py           # Input source loading
│   ├── infer/
│   │   ├── __init__.py
│   │   ├── backend.py      # Backend detection
│   │   └── yoloultralytics.py  # YOLO inference wrapper
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── tegrastats.py   # Jetson GPU metrics
│   │   ├── sysmetrics.py   # System metrics (psutil)
│   │   └── aggregator.py   # Metrics aggregation
│   └── report/
│       ├── __init__.py
│       └── writer.py       # Report generation
├── scripts/
│   └── export_trt.sh       # TensorRT export helper
├── tests/
│   ├── test_config.py
│   └── test_tegrastats_parse.py
└── outputs/                # Generated reports
```

## License

[Your License Here]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Acknowledgments

- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Designed for NVIDIA Jetson Orin Nano
- Uses psutil for system monitoring
- Uses tegrastats for Jetson-specific metrics
