#!/usr/bin/env python3
"""
测试双流追踪系统（使用本地视频文件）
无需实际硬件，用于开发和调试
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from jetson_bench.dual_stream.dual_loader import DualSourceLoader
# from jetson_bench.dual_stream.fusion import fuse_detections  # 不再使用融合逻辑
from jetson_bench.tracking.centroid_tracker import LargestAnimalTracker
from jetson_bench.camera_control.ptz_controller import PTZController
from jetson_bench.camera_control.ptz_aggregator import PTZAggregator
from jetson_bench.visualization.overlay import HuntingCameraDisplay
from jetson_bench.infer.yoloultralytics import YOLOInference
from jetson_bench.config import ModelConfig
import cv2
import argparse


def test_dual_stream(rgb_video, thermal_video, rgb_model_path, thermal_model_path, max_frames=150):
    """
    测试双流追踪系统

    Args:
        rgb_video: RGB视频文件路径
        thermal_video: 热成像视频文件路径
        rgb_model_path: RGB YOLO模型路径
        thermal_model_path: 热成像 YOLO模型路径
        max_frames: 最大处理帧数
    """
    print("=" * 60)
    print("🎯 Dual-Stream Hunting Camera Test")
    print("=" * 60)

    # 1. 初始化加载器
    print("\n[1/6] Initializing dual source loader...")
    loader = DualSourceLoader(
        source_type="video",
        rgb_source=rgb_video,
        thermal_source=thermal_video,
        target_fps=5.0,  # 固定 5 FPS
        max_frames=None,  # 不限制帧数，播放完整视频
    )

    source_info = loader.get_source_info()
    print(f"  ✓ RGB: {source_info['rgb_width']}x{source_info['rgb_height']} @ {source_info['rgb_fps']} FPS")
    print(f"  ✓ Thermal: {source_info['thermal_width']}x{source_info['thermal_height']} @ {source_info['thermal_fps']} FPS")
    print(f"  ✓ Target processing rate: {source_info['target_fps']} FPS")

    # 2. 加载YOLO模型
    print("\n[2/6] Loading YOLO models...")
    rgb_model_config = ModelConfig(name="yolo11n_rgb", weight_path=rgb_model_path)
    thermal_model_config = ModelConfig(name="yolo11n_thermal", weight_path=thermal_model_path)

    rgb_model = YOLOInference(
        model_name="yolo11n_rgb",
        weight_path=rgb_model_path,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        gpu_mem_limit_gb=8.0,
    )
    print(f"  ✓ RGB model loaded: {rgb_model.backend}")

    thermal_model = YOLOInference(
        model_name="yolo11n_thermal",
        weight_path=thermal_model_path,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        gpu_mem_limit_gb=8.0,
    )
    print(f"  ✓ Thermal model loaded: {thermal_model.backend}")

    # 3. 初始化追踪器（RGB和Thermal各自独立）
    print("\n[3/6] Initializing independent trackers...")
    rgb_tracker = LargestAnimalTracker(max_disappeared=10)
    thermal_tracker = LargestAnimalTracker(max_disappeared=10)
    print("  ✓ RGB tracker ready")
    print("  ✓ Thermal tracker ready")

    # 4. 初始化PTZ控制器（RGB和Thermal各自独立）
    print("\n[4/6] Initializing independent PTZ controllers...")
    rgb_ptz = PTZController(
        serial_port="/dev/ttyUSB0",  # 会自动进入模拟模式
        frame_width=source_info['rgb_width'],
        frame_height=source_info['rgb_height'],
    )
    thermal_ptz = PTZController(
        serial_port="/dev/ttyUSB1",  # 独立控制器
        frame_width=source_info['thermal_width'],
        frame_height=source_info['thermal_height'],
    )
    print("  ✓ RGB PTZ controller ready (simulation mode)")
    print("  ✓ Thermal PTZ controller ready (simulation mode)")

    # 初始化PTZ聚合器（RGB和Thermal各自独立，双重阈值：速度100px/s + 距离80%边界）
    rgb_ptz_aggregator = PTZAggregator(
        smooth_window=0.3,
        velocity_threshold=100.0,
        distance_threshold_percent=0.8,
        frame_width=source_info['rgb_width'],
        frame_height=source_info['rgb_height']
    )
    thermal_ptz_aggregator = PTZAggregator(
        smooth_window=0.3,
        velocity_threshold=100.0,
        distance_threshold_percent=0.8,
        frame_width=source_info['thermal_width'],
        frame_height=source_info['thermal_height']
    )
    print("  ✓ RGB PTZ aggregator ready:")
    print(f"      - Frame size: {source_info['rgb_width']}x{source_info['rgb_height']}")
    print("      - Window: 0.3s, Velocity: 100px/s, Distance: 80% boundary")
    print("  ✓ Thermal PTZ aggregator ready:")
    print(f"      - Frame size: {source_info['thermal_width']}x{source_info['thermal_height']}")
    print("      - Window: 0.3s, Velocity: 100px/s, Distance: 80% boundary")

    # 5. 初始化显示
    print("\n[5/6] Initializing display...")
    display = HuntingCameraDisplay(
        frame_skip=loader.frame_skip if hasattr(loader, 'frame_skip') and loader.frame_skip else 1,
        target_fps=loader.target_fps if hasattr(loader, 'target_fps') else None,
        frame_width=source_info['rgb_width'],
        frame_height=source_info['rgb_height']
    )
    print("  ✓ Display windows created")

    # 6. 主处理循环（循环播放模式）
    print("\n[6/6] Processing frames...")
    print("  🔁 Loop mode: Video will restart automatically")
    print("  Press ESC to quit\n")

    frame_count = 0
    rgb_detection_count = 0
    thermal_detection_count = 0
    rgb_tracking_count = 0
    thermal_tracking_count = 0
    loop_count = 0
    user_quit = False

    try:
        while not user_quit:
            loop_count += 1
            print(f"\n{'='*60}")
            print(f"🔄 Loop #{loop_count}")
            print(f"{'='*60}\n")

            # 重新创建加载器以重置视频
            loop_loader = DualSourceLoader(
                source_type="video",
                rgb_source=rgb_video,
                thermal_source=thermal_video,
                target_fps=5.0,
                max_frames=None,  # 不限制帧数，播放完整视频
            )

            with loop_loader:
                for frame_data in loop_loader:
                    frame_count += 1

                    # ============ RGB 流处理（独立） ============
                    rgb_result = rgb_model.infer(
                        frame_data["rgb_frame"],
                        frame_data["frame_id"]
                    )
                    rgb_dets = rgb_result.to_dict_list()
                    rgb_detection_count += len(rgb_dets)

                    # RGB 追踪
                    rgb_tracked_obj = rgb_tracker.update(rgb_dets)
                    if rgb_tracked_obj:
                        rgb_tracking_count += 1

                    # RGB PTZ控制
                    rgb_ptz_status = None
                    if rgb_tracked_obj:
                        rgb_ptz_status = rgb_ptz.calculate_movement(rgb_tracked_obj["centroid"])
                        rgb_ptz_aggregator.add_movement(rgb_ptz_status)

                        # 检查RGB聚合输出
                        rgb_aggregated = rgb_ptz_aggregator.get_aggregated_output()
                        if rgb_aggregated:
                            rgb_ptz_status["aggregated_output"] = rgb_aggregated

                            # RGB触发类型
                            if rgb_aggregated['trigger'] == 'emergency_velocity':
                                emoji = "🚀"
                                reason = "FAST MOTION"
                            elif rgb_aggregated['trigger'] == 'emergency_distance':
                                emoji = "⚠️"
                                reason = "TOO FAR"
                            else:
                                emoji = "✓"
                                reason = "REGULAR"

                            print(f"  [RGB] {emoji} Motor Output [{reason}]: "
                                  f"({rgb_aggregated['output_x']:+.1f}, {rgb_aggregated['output_y']:+.1f})px, "
                                  f"velocity={rgb_aggregated['velocity']:.1f}px/s, "
                                  f"distance={rgb_aggregated['pixel_distance_total']:.0f}px")

                    # ============ Thermal 流处理（独立） ============
                    thermal_result = thermal_model.infer(
                        frame_data["thermal_frame"],
                        frame_data["frame_id"]
                    )
                    thermal_dets = thermal_result.to_dict_list()
                    thermal_detection_count += len(thermal_dets)

                    # Thermal 追踪
                    thermal_tracked_obj = thermal_tracker.update(thermal_dets)
                    if thermal_tracked_obj:
                        thermal_tracking_count += 1

                    # Thermal PTZ控制
                    thermal_ptz_status = None
                    if thermal_tracked_obj:
                        thermal_ptz_status = thermal_ptz.calculate_movement(thermal_tracked_obj["centroid"])
                        thermal_ptz_aggregator.add_movement(thermal_ptz_status)

                        # 检查Thermal聚合输出
                        thermal_aggregated = thermal_ptz_aggregator.get_aggregated_output()
                        if thermal_aggregated:
                            thermal_ptz_status["aggregated_output"] = thermal_aggregated

                            # Thermal触发类型
                            if thermal_aggregated['trigger'] == 'emergency_velocity':
                                emoji = "🚀"
                                reason = "FAST MOTION"
                            elif thermal_aggregated['trigger'] == 'emergency_distance':
                                emoji = "⚠️"
                                reason = "TOO FAR"
                            else:
                                emoji = "✓"
                                reason = "REGULAR"

                            print(f"  [THERMAL] {emoji} Motor Output [{reason}]: "
                                  f"({thermal_aggregated['output_x']:+.1f}, {thermal_aggregated['output_y']:+.1f})px, "
                                  f"velocity={thermal_aggregated['velocity']:.1f}px/s, "
                                  f"distance={thermal_aggregated['pixel_distance_total']:.0f}px")

                    # 计算原始帧位置（基于抽帧逻辑）
                    if hasattr(loop_loader, 'frame_skip') and loop_loader.frame_skip:
                        # Video mode: 估算原始帧号
                        native_frame_id = (frame_count - 1) * loop_loader.frame_skip + 1
                    else:
                        # Camera mode: 无原始帧号概念
                        native_frame_id = frame_count

                    # DEBUG: Print frame info on first few frames
                    if frame_count <= 3:
                        print(f"\n[DEBUG Frame {frame_count}]")
                        print(f"  RGB frame: {frame_data['rgb_frame'].shape}")
                        print(f"  Thermal frame: {frame_data['thermal_frame'].shape}")
                        if thermal_dets:
                            print(f"  Thermal detections: {len(thermal_dets)}")
                            for i, det in enumerate(thermal_dets):
                                print(f"    Det {i}: bbox={det['bbox']}, class={det['class']}")
                        if thermal_tracked_obj:
                            print(f"  Thermal tracked bbox: {thermal_tracked_obj['bbox']}")
                            print(f"  Thermal tracked centroid: {thermal_tracked_obj['centroid']}")

                    # 可视化（三窗口：RGB独立、Thermal独立、Fusion融合视觉）
                    display.draw_dual_stream_frame(
                        frame_data["rgb_frame"],
                        frame_data["thermal_frame"],
                        rgb_dets,  # RGB检测
                        thermal_dets,  # Thermal检测
                        rgb_tracked_obj,  # RGB追踪
                        thermal_tracked_obj,  # Thermal追踪
                        rgb_ptz_status,  # RGB PTZ状态
                        thermal_ptz_status,  # Thermal PTZ状态
                        frame_id=frame_count,
                        native_frame_id=native_frame_id
                    )

                    # 进度显示（分别显示两个流的检测数量）
                    if frame_count % 10 == 0:
                        print(f"  Frame {frame_count} (Loop {loop_count}): "
                              f"RGB: {len(rgb_dets)} dets, Thermal: {len(thermal_dets)} dets, "
                              f"RGB Tracking: {'Yes' if rgb_tracked_obj else 'No'}, "
                              f"Thermal Tracking: {'Yes' if thermal_tracked_obj else 'No'}")

                    # ESC键退出
                    # Use 5ms wait to ensure all three windows refresh synchronously
                    # This is critical for real-time camera scenarios
                    if cv2.waitKey(5) & 0xFF == 27:
                        print("\n⏹️  Stopped by user")
                        user_quit = True
                        break

            # 如果用户没有退出，准备下一轮循环
            if not user_quit:
                print(f"\n✅ Loop #{loop_count} completed, restarting...\n")

    finally:
        rgb_ptz.close()
        thermal_ptz.close()
        display.close()

    # 统计信息（分别显示RGB和Thermal）
    print("\n" + "=" * 80)
    print("📊 Test Summary - Independent Dual Stream")
    print("=" * 80)
    print(f"  Total frames processed: {frame_count}")

    print(f"\n  ═══ RGB Stream ═══")
    print(f"    Detections: {rgb_detection_count} total")
    print(f"    Frames with tracking: {rgb_tracking_count} ({rgb_tracking_count/frame_count*100:.1f}%)")
    print(f"    Average detections per frame: {rgb_detection_count/frame_count:.2f}")

    rgb_ptz_stats = rgb_ptz_aggregator.get_stats()
    print(f"    PTZ Outputs: {rgb_ptz_stats['total_outputs']} total")
    print(f"      - Regular (✓): {rgb_ptz_stats['regular_outputs']}")
    print(f"      - Velocity emergency (🚀): {rgb_ptz_stats['velocity_emergency_outputs']}")
    print(f"      - Distance emergency (⚠️): {rgb_ptz_stats['distance_emergency_outputs']}")

    print(f"\n  ═══ Thermal Stream ═══")
    print(f"    Detections: {thermal_detection_count} total")
    print(f"    Frames with tracking: {thermal_tracking_count} ({thermal_tracking_count/frame_count*100:.1f}%)")
    print(f"    Average detections per frame: {thermal_detection_count/frame_count:.2f}")

    thermal_ptz_stats = thermal_ptz_aggregator.get_stats()
    print(f"    PTZ Outputs: {thermal_ptz_stats['total_outputs']} total")
    print(f"      - Regular (✓): {thermal_ptz_stats['regular_outputs']}")
    print(f"      - Velocity emergency (🚀): {thermal_ptz_stats['velocity_emergency_outputs']}")
    print(f"      - Distance emergency (⚠️): {thermal_ptz_stats['distance_emergency_outputs']}")

    print("\n✅ Test complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dual-stream tracking with video files")
    parser.add_argument("--rgb-video", required=True, help="RGB video file path")
    parser.add_argument("--thermal-video", required=True, help="Thermal video file path")
    parser.add_argument("--rgb-model", required=True, help="RGB YOLO model path (.pt, .onnx, or .engine)")
    parser.add_argument("--thermal-model", required=True, help="Thermal YOLO model path (.pt, .onnx, or .engine)")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum frames to process")

    args = parser.parse_args()

    # 验证文件存在
    for path_arg, path_val in [("rgb-video", args.rgb_video),
                                ("thermal-video", args.thermal_video),
                                ("rgb-model", args.rgb_model),
                                ("thermal-model", args.thermal_model)]:
        if not Path(path_val).exists():
            print(f"❌ Error: {path_arg} file not found: {path_val}")
            sys.exit(1)

    test_dual_stream(
        rgb_video=args.rgb_video,
        thermal_video=args.thermal_video,
        rgb_model_path=args.rgb_model,
        thermal_model_path=args.thermal_model,
        max_frames=args.max_frames
    )
