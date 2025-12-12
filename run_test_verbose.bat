@echo off
REM Jetson Benchmark Verbose Test Script
REM Shows detailed logging

echo ============================================================
echo Jetson Orin Nano Benchmark - Verbose Mode
echo ============================================================
echo.

python -m jetson_bench.cli ^
    --source image_dir=test_data/images ^
    --model0 yolo11n.pt ^
    --max_frames 10 ^
    --gpu_mem_limit_gb 8.0 ^
    --verbose

echo.
pause
