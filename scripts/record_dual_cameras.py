#!/usr/bin/env python3
"""
录制双摄像头测试视频
用于开发和测试双流追踪系统
"""
import cv2
import time
import argparse
from pathlib import Path


def record_dual_cameras(rgb_device=0, thermal_device=1, duration=30, output_dir="test_videos"):
    """
    同时录制RGB和热成像摄像头视频

    Args:
        rgb_device: RGB摄像头设备ID
        thermal_device: 热成像摄像头设备ID
        duration: 录制时长（秒）
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    rgb_output = output_path / f"rgb_{timestamp}.mp4"
    thermal_output = output_path / f"thermal_{timestamp}.mp4"

    # 打开摄像头
    print(f"📹 Opening cameras...")
    rgb_cap = cv2.VideoCapture(rgb_device)
    thermal_cap = cv2.VideoCapture(thermal_device)

    if not rgb_cap.isOpened():
        print(f"❌ Failed to open RGB camera {rgb_device}")
        return
    if not thermal_cap.isOpened():
        print(f"❌ Failed to open thermal camera {thermal_device}")
        return

    # 获取摄像头参数
    fps = 30
    rgb_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    thermal_width = int(thermal_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    thermal_height = int(thermal_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📐 RGB: {rgb_width}x{rgb_height} @ {fps} FPS")
    print(f"📐 Thermal: {thermal_width}x{thermal_height} @ {fps} FPS")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rgb_writer = cv2.VideoWriter(str(rgb_output), fourcc, fps, (rgb_width, rgb_height))
    thermal_writer = cv2.VideoWriter(str(thermal_output), fourcc, fps, (thermal_width, thermal_height))

    # 录制视频
    total_frames = fps * duration
    print(f"\n🎬 Recording for {duration} seconds ({total_frames} frames)...")
    print("Press ESC to stop early")

    frame_count = 0
    start_time = time.time()

    while frame_count < total_frames:
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_thermal, thermal_frame = thermal_cap.read()

        if not (ret_rgb and ret_thermal):
            print(f"⚠️  Frame read failed at frame {frame_count}")
            break

        # 写入视频
        rgb_writer.write(rgb_frame)
        thermal_writer.write(thermal_frame)

        # 显示预览
        cv2.imshow('RGB Preview', rgb_frame)
        cv2.imshow('Thermal Preview', thermal_frame)

        frame_count += 1

        # 显示进度
        if frame_count % fps == 0:
            elapsed = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames, {elapsed:.1f}s)")

        # ESC键退出
        if cv2.waitKey(1) & 0xFF == 27:
            print("\n⏹️  Recording stopped by user")
            break

    # 清理资源
    rgb_cap.release()
    thermal_cap.release()
    rgb_writer.release()
    thermal_writer.release()
    cv2.destroyAllWindows()

    # 输出结果
    elapsed = time.time() - start_time
    print(f"\n✅ Recording complete!")
    print(f"   Duration: {elapsed:.1f} seconds")
    print(f"   Frames recorded: {frame_count}")
    print(f"   RGB video: {rgb_output}")
    print(f"   Thermal video: {thermal_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record dual camera test videos")
    parser.add_argument("--rgb-device", type=int, default=0, help="RGB camera device ID")
    parser.add_argument("--thermal-device", type=int, default=1, help="Thermal camera device ID")
    parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")
    parser.add_argument("--output-dir", type=str, default="test_videos", help="Output directory")

    args = parser.parse_args()

    record_dual_cameras(
        rgb_device=args.rgb_device,
        thermal_device=args.thermal_device,
        duration=args.duration,
        output_dir=args.output_dir
    )
