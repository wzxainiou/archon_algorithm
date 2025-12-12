# Project Structure

Complete file tree for the Jetson Orin Nano Multi-Model Benchmark project.

```
.
├── CLAUDE.md                           # Guidance for Claude Code
├── README.md                            # User documentation
├── PROJECT_STRUCTURE.md                 # This file
├── requirements.txt                     # Python dependencies
│
├── jetson_bench/                        # Main package
│   ├── __init__.py                      # Package initialization
│   ├── cli.py                           # Command-line interface (main entry point)
│   ├── config.py                        # Configuration and validation
│   ├── loader.py                        # Input source loading
│   │
│   ├── infer/                           # Inference module
│   │   ├── __init__.py
│   │   ├── backend.py                   # Backend detection (TensorRT/ONNX/PyTorch)
│   │   └── yoloultralytics.py          # YOLO inference wrapper
│   │
│   ├── metrics/                         # Metrics collection
│   │   ├── __init__.py
│   │   ├── aggregator.py                # Metrics aggregation
│   │   ├── sysmetrics.py                # System metrics (psutil)
│   │   └── tegrastats.py                # Jetson GPU metrics parser
│   │
│   └── report/                          # Report generation
│       ├── __init__.py
│       └── writer.py                    # JSON and Markdown report writer
│
├── scripts/                             # Helper scripts
│   └── export_trt.sh                    # TensorRT export script
│
├── tests/                               # Test suite
│   ├── __init__.py
│   ├── test_config.py                   # Configuration tests
│   └── test_tegrastats_parse.py         # Tegrastats parser tests
│
└── outputs/                             # Generated benchmark results (created at runtime)
    └── YYYY-MM-DD_HH-MM-SS/            # Timestamped run directory
        ├── report.json                  # Machine-readable results
        ├── report.md                    # Human-readable report
        ├── metrics.jsonl                # Time-series metrics
        └── run.log                      # Execution log
```

## File Descriptions

### Root Files

- **CLAUDE.md**: Comprehensive guide for Claude Code instances working on this project
  - Architecture overview
  - Common development tasks
  - Platform-specific notes
  - Debugging tips

- **README.md**: User-facing documentation
  - Installation instructions
  - Quick start examples
  - Command-line reference
  - Troubleshooting guide

- **requirements.txt**: Python package dependencies
  - ultralytics (YOLO)
  - opencv-python
  - psutil
  - numpy

### Core Package (`jetson_bench/`)

#### [cli.py](jetson_bench/cli.py)
Main entry point for the benchmark. Orchestrates:
- Argument parsing
- Environment verification
- Model loading and inference
- Metrics collection
- Report generation

**Run**: `python3 -m jetson_bench.cli --source <SOURCE> [OPTIONS]`

#### [config.py](jetson_bench/config.py)
Configuration management with validation:
- `ModelConfig`: Individual model configuration (name, weight_path, skip_reason)
- `BenchConfig`: Main config with exactly 4 model slots
- `verify_environment()`: Checks Python version, Jetson platform, dependencies

**Key constraint**: Enforces exactly 4 models (cannot be changed).

#### [loader.py](jetson_bench/loader.py)
Unified input source loader:
- `SourceLoader`: Iterator for images/video/camera
- Supports: `.jpg`, `.png`, `.bmp` images
- Frame limiting via `max_frames`
- Provides source metadata

### Inference Module (`jetson_bench/infer/`)

#### [backend.py](jetson_bench/infer/backend.py)
Backend detection and compatibility:
- `detect_backend()`: Detects TensorRT/ONNX/PyTorch from file extension
- `verify_backend_compatibility()`: Checks if backend dependencies are available
- `suggest_backend_optimization()`: Recommends better backends

**Priority**: .engine (TensorRT) > .onnx (ONNX) > .pt (PyTorch)

#### [yoloultralytics.py](jetson_bench/infer/yoloultralytics.py)
YOLO inference wrapper:
- `YOLOInference`: Manages model loading and inference
- `InferenceResult`: Per-frame results (latency, detections)
- `ModelPerformance`: Aggregated statistics (FPS, P50/P90/P99 latency)

### Metrics Module (`jetson_bench/metrics/`)

#### [tegrastats.py](jetson_bench/metrics/tegrastats.py)
Jetson-specific metrics via tegrastats:
- `TegrastatsMonitor`: Parses tegrastats output in background thread
- `TegraMetrics`: RAM, GPU utilization, temperature, power
- Regex-based parsing of tegrastats format

**Note**: Requires tegrastats binary (Jetson-specific).

#### [sysmetrics.py](jetson_bench/metrics/sysmetrics.py)
Cross-platform system metrics via psutil:
- `SystemMonitor`: Collects CPU and memory metrics
- `SystemMetrics`: CPU%, memory usage, swap
- Background thread with configurable interval

#### [aggregator.py](jetson_bench/metrics/aggregator.py)
Combines tegrastats and system metrics:
- `MetricsAggregator`: Orchestrates both monitors
- `get_summary()`: Computes statistics (mean, min, max, P95)
- `save_timeseries()`: Exports JSONL time-series data

### Report Module (`jetson_bench/report/`)

#### [writer.py](jetson_bench/report/writer.py)
Report generation in multiple formats:
- `ReportWriter`: Creates JSON and Markdown reports
- `create_full_report()`: Assembles complete report structure
- Includes: metadata, model performance, system metrics, summary

### Scripts (`scripts/`)

#### [export_trt.sh](scripts/export_trt.sh)
Helper script for TensorRT export:
- Exports PyTorch models to TensorRT .engine format
- Validates Jetson platform
- Uses Ultralytics CLI
- Warns about platform-specific nature of TensorRT engines

**Usage**: `./scripts/export_trt.sh /path/to/model.pt 640`

### Tests (`tests/`)

#### [test_config.py](tests/test_config.py)
Configuration validation tests:
- Model config validation (paths, formats)
- 4-model enforcement
- Model weight setting
- Active/skipped model filtering

#### [test_tegrastats_parse.py](tests/test_tegrastats_parse.py)
Tegrastats parser tests:
- Parsing complete tegrastats lines
- Parsing minimal output
- Handling malformed input

**Run tests**: `python3 -m pytest tests/ -v`

## Key Design Decisions

### 1. Fixed 4-Model Architecture
The project is intentionally designed for exactly 4 model slots. This constraint:
- Ensures consistent benchmarking across runs
- Simplifies configuration
- Matches the specific use case (RGB + thermal, nano + small variants)

### 2. Backend Priority System
Automatic selection of best available backend:
- TensorRT for maximum Jetson performance
- ONNX for portability
- PyTorch as fallback

### 3. Fail-Safe Execution
Individual model failures don't stop the entire benchmark:
- Missing weights → model skipped
- Load errors → model marked failed
- All 4 models appear in final report

### 4. Dual Metrics Collection
Two parallel metrics sources for comprehensive monitoring:
- **tegrastats**: Jetson-specific GPU, temperature, power
- **psutil**: Cross-platform CPU, memory

### 5. Multiple Report Formats
- JSON: Machine-readable for automation/analysis
- Markdown: Human-readable for review
- JSONL: Time-series for detailed analysis

## Development Workflow

### Adding New Features
1. Implement in appropriate module
2. Add tests in `tests/`
3. Update [CLAUDE.md](CLAUDE.md) with usage notes
4. Update [README.md](README.md) if user-facing

### Testing Changes
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_config.py::test_bench_config_must_have_4_models -v

# Run with coverage
python3 -m pytest tests/ --cov=jetson_bench --cov-report=html
```

### Typical Usage Flow
```bash
# 1. Export model to TensorRT (on Jetson)
./scripts/export_trt.sh models/yolo11n.pt 640

# 2. Run benchmark
python3 -m jetson_bench.cli \
    --source image_dir=data/images \
    --model0 models/yolo11n_640.engine \
    --imgsz 640 \
    --max_frames 100

# 3. Review results
cat outputs/*/report.md
```

## Dependencies

### Required
- Python ≥ 3.10
- ultralytics ≥ 8.0.0
- opencv-python ≥ 4.8.0
- psutil ≥ 5.9.0
- numpy ≥ 1.24.0

### Platform-Specific
- **Jetson**: tegrastats (pre-installed with JetPack)
- **TensorRT**: Pre-installed on Jetson with JetPack

### Optional
- pytest (for testing)
- onnxruntime-gpu (for ONNX backend)

## Troubleshooting Resources

See [README.md](README.md) "Troubleshooting" section for:
- TensorRT engine compatibility issues
- Tegrastats permission problems
- Camera access issues
- Out of memory errors
- Input source validation errors
