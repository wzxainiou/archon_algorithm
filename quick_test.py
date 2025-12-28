import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run test
from scripts.test_dual_stream import test_dual_stream

print("Starting dual-stream PTZ test...")
test_dual_stream(
    rgb_video=str(project_root / "data" / "test_videos" / "rgb_test.mp4"),
    thermal_video=str(project_root / "data" / "test_videos" / "thermal_test.mp4"),
    rgb_model_path=str(project_root / "weights" / "yolo11n.pt"),
    thermal_model_path=str(project_root / "weights" / "yolo11n.pt"),
    max_frames=50  # Process 50 frames for quick test
)
