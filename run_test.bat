@echo off
cd /d "c:\Users\王照旭\Desktop\CODE\archon_algorithm"
python scripts\test_dual_stream.py --rgb-video "data\test_videos\rgb_test.mp4" --thermal-video "data\test_videos\thermal_test.mp4" --rgb-model "weights\yolo11n.pt" --thermal-model "weights\yolo11n.pt"
pause
