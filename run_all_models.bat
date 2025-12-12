@echo off
REM Jetson Benchmark - Complete 4-Model Test
REM Tests all 4 model slots in a single run with unified report

echo ============================================================
echo Jetson Orin Nano - Complete 4-Model Benchmark
echo ============================================================
echo.
echo This will test all 4 model slots:
echo   [0] yolo11n_rgb   - YOLO11 Nano (RGB)
echo   [1] yolo11s_rgb   - YOLO11 Small (RGB)
echo   [2] yolo11n_thermal - YOLO11 Nano (Thermal)
echo   [3] yolov8n_thermal - YOLOv8 Nano (Thermal)
echo.
echo GPU Memory Limit: 8.0 GB
echo Input Source: test_data/images (2 images)
echo Max Frames: 50
echo.
echo ============================================================
echo.

python -m jetson_bench.cli ^
    --source image_dir=test_data/images ^
    --model0 yolo11n.pt ^
    --model1 yolo11s.pt ^
    --model2 yolo11n.pt ^
    --model3 yolov8n.pt ^
    --max_frames 50 ^
    --gpu_mem_limit_gb 8.0 ^
    --verbose

echo.
echo ============================================================
echo Test completed! Check the outputs/ directory for reports.
echo ============================================================
echo.
pause
