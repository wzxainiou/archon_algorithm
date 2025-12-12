@echo off
REM Jetson Benchmark Quick Test Script
REM Usage: run_test.bat

echo ============================================================
echo Jetson Orin Nano Benchmark - Quick Test
echo ============================================================
echo.

REM Check if model exists
if not exist "yolo11n.pt" (
    echo Error: Model file yolo11n.pt not found!
    echo Please run: python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
    pause
    exit /b 1
)

REM Check if test data exists
if not exist "test_data\images" (
    echo Error: Test data directory not found!
    echo Please create test_data\images and add some images
    pause
    exit /b 1
)

echo Running benchmark with default settings...
echo.

python -m jetson_bench.cli ^
    --source image_dir=test_data/images ^
    --model0 yolo11n.pt ^
    --max_frames 10 ^
    --gpu_mem_limit_gb 8.0

echo.
echo ============================================================
echo Benchmark completed!
echo Check outputs\ directory for results
echo ============================================================
echo.

pause
