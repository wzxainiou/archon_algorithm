#!/bin/bash
#
# Export YOLO models to TensorRT for optimal Jetson performance
#
# Usage:
#   ./scripts/export_trt.sh /path/to/model.pt
#
# This script exports a PyTorch YOLO model to TensorRT .engine format
# optimized for Jetson Orin Nano.

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_model.pt> [imgsz]"
    echo "Example: $0 yolo11n.pt 640"
    exit 1
fi

MODEL_PATH="$1"
IMGSZ="${2:-640}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: Not running on Jetson - TensorRT engine may not be compatible"
    echo "   TensorRT engines are platform-specific and should be exported on the target device"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "================================================================"
echo "Exporting YOLO model to TensorRT"
echo "================================================================"
echo "Model: $MODEL_PATH"
echo "Image size: $IMGSZ"
echo ""

# Get base filename
BASENAME=$(basename "$MODEL_PATH" .pt)
DIRNAME=$(dirname "$MODEL_PATH")
OUTPUT_ENGINE="${DIRNAME}/${BASENAME}_${IMGSZ}.engine"

echo "Output: $OUTPUT_ENGINE"
echo ""

# Export using Ultralytics CLI
echo "Running export..."
yolo export model="$MODEL_PATH" format=engine imgsz="$IMGSZ" device=0 half=True

echo ""
echo "✅ Export completed!"
echo ""
echo "Generated file: $OUTPUT_ENGINE"
echo ""
echo "You can now use this engine file with jetson_bench:"
echo "  python -m jetson_bench.cli --source image_dir=/path/to/images \\"
echo "    --model0 $OUTPUT_ENGINE"
echo ""
echo "⚠️  Important: TensorRT engines are platform-specific."
echo "   This engine will only work on devices with the same:"
echo "   - GPU architecture"
echo "   - TensorRT version"
echo "   - CUDA version"
