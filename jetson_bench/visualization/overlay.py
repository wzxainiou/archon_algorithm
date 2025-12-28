"""
Visualization and display module for hunting camera.

Draws detections and tracking info on BOTH RGB and Thermal frames.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HuntingCameraDisplay:
    """Real-time display with tracking and control info on dual windows."""

    def __init__(self, frame_skip: int = 1, target_fps: float = None,
                 frame_width: int = None, frame_height: int = None):
        """
        Initialize display windows.

        Args:
            frame_skip: Number of frames being skipped (for sampling mode visualization)
            target_fps: Target processing FPS (for sampling mode visualization)
            frame_width: Frame width for PTZ zone calculation (optional)
            frame_height: Frame height for PTZ zone calculation (optional)
        """
        self.rgb_window = "RGB Camera"
        self.thermal_window = "Thermal Camera"
        self.fusion_window = "Fusion Vision (RGB + Thermal)"
        self.frame_skip = frame_skip
        self.target_fps = target_fps
        self.frame_count = 0
        self.start_time = time.time()

        # PTZ zones initialization
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zones_initialized = False

        # If frame dimensions provided, pre-calculate PTZ zones
        if frame_width and frame_height:
            self._init_ptz_zones()

        cv2.namedWindow(self.rgb_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.thermal_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.fusion_window, cv2.WINDOW_NORMAL)
        logger.info("Initialized display windows (RGB, Thermal, Fusion)")

    def _draw_frame_info(self, frame: np.ndarray, frame_id: int, native_frame_id: int,
                         num_detections: int, stream_label: str) -> None:
        """
        Draw frame information panel at top of frame.

        Args:
            frame: Frame to draw on (modified in-place)
            frame_id: Processed frame number (1, 2, 3...)
            native_frame_id: Original video frame number (1, 6, 11... if sampling)
            num_detections: Number of detections in this frame
            stream_label: "RGB" or "THERMAL"
        """
        h, w = frame.shape[:2]

        # Dynamic parameters based on stream_label
        if stream_label == "THERMAL":
            # Thermal窗口：更小的背景和字体（进一步优化）
            panel_height = 65        # 进一步减小背景高度（从80改为65）
            font_scale_1 = 0.4       # 第1、2行字体（从0.5改为0.4）
            font_scale_2 = 0.35      # 第3行字体（从0.4改为0.35）
            font_thickness = 1       # 字体线宽保持1
            line1_y = 15             # 第1行y坐标（从20改为15）
            line2_y = 33             # 第2行y坐标（从42改为33）
            line3_y = 51             # 第3行y坐标（从64改为51）
        else:  # RGB窗口
            # RGB窗口：保持原有大小
            panel_height = 120
            font_scale_1 = 0.7
            font_scale_2 = 0.6
            font_thickness = 2
            line1_y = 25
            line2_y = 55
            line3_y = 85

        # Semi-transparent background for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Line 1: Stream label + Mode indicator
        if self.frame_skip > 1:
            mode_text = f"SAMPLING: Every {self.frame_skip} frames ({self.target_fps:.1f} FPS)"
        else:
            mode_text = "CONTINUOUS: All frames"

        cv2.putText(frame, f"{stream_label} | {mode_text}", (10, line1_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_1, (0, 255, 255), font_thickness)

        # Line 2: Frame numbers
        if self.frame_skip > 1:
            frame_text = f"Processed Frame: {frame_id} | Native Frame: {native_frame_id}"
        else:
            frame_text = f"Frame: {frame_id}"

        cv2.putText(frame, frame_text, (10, line2_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_1, (255, 255, 255), font_thickness)

        # Line 3: Runtime statistics
        elapsed = time.time() - self.start_time
        fps = frame_id / elapsed if elapsed > 0 else 0
        stats_text = f"Runtime: {elapsed:.1f}s | Processing FPS: {fps:.1f} | Detections: {num_detections}"

        cv2.putText(frame, stats_text, (10, line3_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_2, (0, 255, 0), font_thickness)

    def draw_dual_stream_frame(self,
                               rgb_frame: np.ndarray,
                               thermal_frame: np.ndarray,
                               rgb_detections: List[Dict],
                               thermal_detections: List[Dict],
                               rgb_tracked_object: Optional[Dict],
                               thermal_tracked_object: Optional[Dict],
                               rgb_ptz_status: Optional[Dict],
                               thermal_ptz_status: Optional[Dict],
                               frame_id: int = 0,
                               native_frame_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw visualization for independent dual-stream processing with fusion view.

        Args:
            rgb_frame: RGB camera frame
            thermal_frame: Thermal camera frame
            rgb_detections: RGB detections only
            thermal_detections: Thermal detections only
            rgb_tracked_object: RGB tracked object or None
            thermal_tracked_object: Thermal tracked object or None
            rgb_ptz_status: RGB PTZ status or None
            thermal_ptz_status: Thermal PTZ status or None
            frame_id: Processed frame number (default 0)
            native_frame_id: Original video frame number (default 0)

        Returns:
            Tuple of (rgb_display, thermal_display, fusion_display) frames with overlays
        """
        # Create copies for drawing
        rgb_display = rgb_frame.copy()
        thermal_display = thermal_frame.copy()

        # Draw RGB stream (independent)
        self._draw_stream_overlay(
            rgb_display, rgb_detections, rgb_tracked_object, rgb_ptz_status,
            frame_id, native_frame_id, "RGB", len(rgb_detections)
        )

        # Draw Thermal stream (independent)
        self._draw_stream_overlay(
            thermal_display, thermal_detections, thermal_tracked_object, thermal_ptz_status,
            frame_id, native_frame_id, "THERMAL", len(thermal_detections)
        )

        # Create fusion visualization (RGB + Thermal overlay, showing RGB detections)
        fusion_display = self._create_fusion_view(
            rgb_frame, thermal_frame, rgb_detections, rgb_tracked_object,
            frame_id, native_frame_id
        )

        # Show all three windows
        cv2.imshow(self.rgb_window, rgb_display)
        cv2.imshow(self.thermal_window, thermal_display)
        cv2.imshow(self.fusion_window, fusion_display)

        return rgb_display, thermal_display, fusion_display

    def _draw_stream_overlay(self,
                             frame: np.ndarray,
                             detections: List[Dict],
                             tracked_object: Optional[Dict],
                             ptz_status: Optional[Dict],
                             frame_id: int,
                             native_frame_id: int,
                             stream_label: str,
                             num_detections: int) -> None:
        """
        Draw overlays on a single stream.

        Args:
            frame: Frame to draw on (modified in-place)
            detections: List of detections for this stream
            tracked_object: Tracked object or None
            ptz_status: PTZ status or None
            frame_id: Frame number
            native_frame_id: Native frame number
            stream_label: "RGB" or "THERMAL"
            num_detections: Number of detections
        """
        h, w = frame.shape[:2]

        # 根据stream_label动态调整显示参数
        if stream_label == "THERMAL":
            # Thermal窗口：细线条、小字体
            det_line_width = 1          # 普通检测框线宽（从2改为1）
            track_line_width = 2        # 追踪框线宽（从4改为2）
            centroid_radius = 5         # 质心半径（从8改为5）
            label_font_scale = 0.5      # 字体大小（从0.8改为0.5）
            label_thickness = 1         # 字体线宽（从2改为1）
            ptz_line_width = 1          # PTZ状态文字线宽（从2改为1）
            ptz_font_scale = 0.4        # PTZ状态字体（从0.6改为0.4）
        else:  # RGB窗口
            # RGB窗口：保持原有大小
            det_line_width = 2
            track_line_width = 4
            centroid_radius = 8
            label_font_scale = 0.8
            label_thickness = 2
            ptz_line_width = 2
            ptz_font_scale = 0.6

        # Draw frame info panel
        if frame_id > 0:
            self._draw_frame_info(frame, frame_id, native_frame_id, num_detections, stream_label)

        # Draw all detections (faint boxes)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]  # Keep float precision
            color = (100, 100, 100)  # Gray for non-tracked
            # Convert to int only when drawing
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, det_line_width)

        # Draw tracked object (prominent)
        if tracked_object:
            x1, y1, x2, y2 = tracked_object["bbox"]  # Keep float precision
            cx, cy = tracked_object["centroid"]      # Keep float precision

            # Thick green box for tracked animal (convert to int only when drawing)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), track_line_width)

            # Centroid marker
            cv2.circle(frame, (int(cx), int(cy)), centroid_radius, (0, 0, 255), -1)

            # Label
            label = f"ID:{tracked_object['id']} {tracked_object['class']} {tracked_object['confidence']:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 255, 0), label_thickness)

        # Draw center zones
        self._draw_zones(frame, w, h, stream_label)

        # Draw PTZ status
        if ptz_status:
            ptz_y_pos = 140 if frame_id > 0 else 40

            # Line 1: Zone and pan/tilt angles
            status_line1 = f"PTZ Zone: [{ptz_status['zone'].upper()}] | PAN:{ptz_status['pan']:+.2f} TILT:{ptz_status['tilt']:+.2f}"
            cv2.putText(frame, status_line1, (10, ptz_y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, ptz_font_scale, (255, 255, 0), ptz_line_width)

            # Line 2: Pixel distance to center
            pd = ptz_status['pixel_distance']
            status_line2 = f"Distance: ({pd['x']:+.0f}, {pd['y']:+.0f})px | Total: {pd['total']:.0f}px"
            cv2.putText(frame, status_line2, (10, ptz_y_pos + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, ptz_font_scale, (255, 255, 0), ptz_line_width)

            # Line 3: Weighted distance
            wd = ptz_status['weighted_distance']
            status_line3 = f"Weight: {ptz_status['weight']:.1f} | Weighted: ({wd['x']:+.1f}, {wd['y']:+.1f})px"
            cv2.putText(frame, status_line3, (10, ptz_y_pos + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, ptz_font_scale, (0, 255, 255), ptz_line_width)

            # Line 4: Aggregated output (if available) - keep slightly larger font for important info
            if 'aggregated_output' in ptz_status and ptz_status['aggregated_output']:
                agg = ptz_status['aggregated_output']
                trigger_color = (0, 0, 255) if agg['trigger'] == 'emergency' else (0, 255, 0)
                status_line4 = f"Output [{agg['trigger'].upper()}]: ({agg['output_x']:+.1f}, {agg['output_y']:+.1f})px"
                cv2.putText(frame, status_line4, (10, ptz_y_pos + 90),
                           cv2.FONT_HERSHEY_SIMPLEX, ptz_font_scale * 1.17, trigger_color, ptz_line_width)

    def _create_fusion_view(self,
                            rgb_frame: np.ndarray,
                            thermal_frame: np.ndarray,
                            rgb_detections: List[Dict],
                            rgb_tracked_object: Optional[Dict],
                            frame_id: int,
                            native_frame_id: int) -> np.ndarray:
        """
        Create fusion visualization: RGB as base + Thermal overlay (showing RGB detections).

        Args:
            rgb_frame: RGB camera frame
            thermal_frame: Thermal camera frame
            rgb_detections: RGB detections to display
            rgb_tracked_object: RGB tracked object or None
            frame_id: Frame number
            native_frame_id: Native frame number

        Returns:
            Fusion display frame
        """
        # Resize thermal to match RGB dimensions
        h_rgb, w_rgb = rgb_frame.shape[:2]
        thermal_resized = cv2.resize(thermal_frame, (w_rgb, h_rgb))

        # Convert thermal to BGR if grayscale
        if len(thermal_resized.shape) == 2:
            thermal_resized = cv2.cvtColor(thermal_resized, cv2.COLOR_GRAY2BGR)

        # Create weighted blend (RGB dominant, thermal overlay with transparency)
        fusion = cv2.addWeighted(rgb_frame, 0.6, thermal_resized, 0.4, 0)

        # Draw frame info
        if frame_id > 0:
            self._draw_frame_info(fusion, frame_id, native_frame_id,
                                  len(rgb_detections), "FUSION (RGB Detections)")

        # Draw RGB detections on fusion view (already in RGB coordinate space)
        # Draw all RGB detections (yellow boxes)
        for det in rgb_detections:
            x1, y1, x2, y2 = det["bbox"]
            # No scaling needed - already in RGB coordinates
            cv2.rectangle(fusion, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # Draw RGB tracked object (prominent cyan box)
        if rgb_tracked_object:
            x1, y1, x2, y2 = rgb_tracked_object["bbox"]
            cx, cy = rgb_tracked_object["centroid"]

            # No scaling needed - already in RGB coordinates
            cv2.rectangle(fusion, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 4)

            # Centroid marker
            cv2.circle(fusion, (int(cx), int(cy)), 8, (0, 0, 255), -1)

            # Label
            label = f"[RGB] ID:{rgb_tracked_object['id']} {rgb_tracked_object['class']} {rgb_tracked_object['confidence']:.2f}"
            cv2.putText(fusion, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return fusion

    def draw_frame(self,
                   rgb_frame: np.ndarray,
                   thermal_frame: np.ndarray,
                   tracked_object: Optional[Dict],
                   ptz_status: Optional[Dict],
                   fused_detections: List[Dict],
                   frame_id: int = 0,
                   native_frame_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw complete visualization on BOTH RGB and Thermal frames.
        (Legacy method for backward compatibility)

        Args:
            rgb_frame: RGB camera frame
            thermal_frame: Thermal camera frame
            tracked_object: Dict from tracker or None
            ptz_status: Dict from PTZ controller or None
            fused_detections: List of all detections
            frame_id: Processed frame number (default 0)
            native_frame_id: Original video frame number (default 0)

        Returns:
            Tuple of (rgb_display, thermal_display) frames with overlays
        """
        # Create copies for drawing
        rgb_display = rgb_frame.copy()
        thermal_display = thermal_frame.copy()

        h, w = rgb_display.shape[:2]

        # Count detections per stream (if source information available)
        rgb_det_count = len([d for d in fused_detections if d.get('source') == 'rgb'])
        thermal_det_count = len([d for d in fused_detections if d.get('source') == 'thermal'])

        # If source info not available, use total count for both
        if rgb_det_count == 0 and thermal_det_count == 0 and len(fused_detections) > 0:
            rgb_det_count = thermal_det_count = len(fused_detections)

        # Draw frame info panels at the top
        if frame_id > 0:  # Only draw if frame_id provided
            self._draw_frame_info(rgb_display, frame_id, native_frame_id,
                                  rgb_det_count, "RGB")
            self._draw_frame_info(thermal_display, frame_id, native_frame_id,
                                  thermal_det_count, "THERMAL")

        # Draw on BOTH frames
        for frame_display in [rgb_display, thermal_display]:
            # Draw all detections (faint boxes)
            for det in fused_detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                color = (100, 100, 100)  # Gray for non-tracked
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)

            # Draw tracked object (prominent)
            if tracked_object:
                x1, y1, x2, y2 = map(int, tracked_object["bbox"])
                cx, cy = map(int, tracked_object["centroid"])

                # Thick green box for tracked animal
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 4)

                # Centroid marker
                cv2.circle(frame_display, (cx, cy), 8, (0, 0, 255), -1)

                # Label
                label = f"ID:{tracked_object['id']} {tracked_object['class']} {tracked_object['confidence']:.2f}"
                cv2.putText(frame_display, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw center zones
            self._draw_zones(frame_display, w, h)

            # Draw PTZ status with detailed distance information
            if ptz_status:
                ptz_y_pos = 140 if frame_id > 0 else 40  # Move down if frame info is shown

                # Line 1: Zone and pan/tilt angles
                status_line1 = f"PTZ Zone: [{ptz_status['zone'].upper()}] | PAN:{ptz_status['pan']:+.2f} TILT:{ptz_status['tilt']:+.2f}"
                cv2.putText(frame_display, status_line1, (10, ptz_y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Line 2: Pixel distance to center
                pd = ptz_status['pixel_distance']
                status_line2 = f"Distance: ({pd['x']:+.0f}, {pd['y']:+.0f})px | Total: {pd['total']:.0f}px"
                cv2.putText(frame_display, status_line2, (10, ptz_y_pos + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Line 3: Weighted distance
                wd = ptz_status['weighted_distance']
                status_line3 = f"Weight: {ptz_status['weight']:.1f} | Weighted: ({wd['x']:+.1f}, {wd['y']:+.1f})px"
                cv2.putText(frame_display, status_line3, (10, ptz_y_pos + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Line 4: Aggregated output (if available)
                if 'aggregated_output' in ptz_status and ptz_status['aggregated_output']:
                    agg = ptz_status['aggregated_output']
                    trigger_color = (0, 0, 255) if agg['trigger'] == 'emergency' else (0, 255, 0)
                    status_line4 = f"Output [{agg['trigger'].upper()}]: ({agg['output_x']:+.1f}, {agg['output_y']:+.1f})px"
                    cv2.putText(frame_display, status_line4, (10, ptz_y_pos + 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, trigger_color, 2)

        # Show both frames in separate windows
        cv2.imshow(self.rgb_window, rgb_display)
        cv2.imshow(self.thermal_window, thermal_display)

        return rgb_display, thermal_display

    def _init_ptz_zones(self):
        """Pre-calculate PTZ control zone rectangle coordinates (maintain aspect ratio)."""
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2

        # Dead zone: 10% of frame dimensions (maintain aspect ratio)
        dead_half_width = int(self.frame_width * 0.10)
        dead_half_height = int(self.frame_height * 0.10)
        self.dead_zone_rect = (
            center_x - dead_half_width,
            center_y - dead_half_height,
            center_x + dead_half_width,
            center_y + dead_half_height
        )

        # Slow zone: 30% of frame dimensions (maintain aspect ratio)
        slow_half_width = int(self.frame_width * 0.30)
        slow_half_height = int(self.frame_height * 0.30)
        self.slow_zone_rect = (
            center_x - slow_half_width,
            center_y - slow_half_height,
            center_x + slow_half_width,
            center_y + slow_half_height
        )

        # Emergency distance boundary: 80% of frame dimensions from center (orange zone)
        # This represents the threshold where emergency distance trigger activates
        emergency_half_width = int(self.frame_width * 0.40)  # 80% from center = 40% on each side
        emergency_half_height = int(self.frame_height * 0.40)
        self.emergency_boundary_rect = (
            center_x - emergency_half_width,
            center_y - emergency_half_height,
            center_x + emergency_half_width,
            center_y + emergency_half_height
        )

        self.zones_initialized = True

    def _draw_zones(self, frame: np.ndarray, w: int, h: int, stream_label: str = "RGB"):
        """
        Draw PTZ control zones: emergency boundary, slow zone, and dead zone.

        Args:
            frame: Frame to draw on
            w: Frame width
            h: Frame height
            stream_label: "RGB" or "THERMAL" (for dynamic line widths)
        """
        # Re-initialize zones if frame dimensions changed (RGB vs Thermal have different sizes)
        if not hasattr(self, 'zones_initialized') or not self.zones_initialized or \
           self.frame_width != w or self.frame_height != h:
            self.frame_width = w
            self.frame_height = h
            self._init_ptz_zones()

        # 根据stream_label动态调整PTZ区域框线宽
        if stream_label == "THERMAL":
            emergency_width, slow_width, dead_width = 2, 1, 1
        else:
            emergency_width, slow_width, dead_width = 3, 2, 2

        # Draw emergency distance boundary (outermost, orange rectangle)
        # This zone triggers emergency distance output when object reaches 80% from center
        x1, y1, x2, y2 = self.emergency_boundary_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), emergency_width)  # Orange (BGR)

        # Draw slow zone (yellow rectangle)
        x1, y1, x2, y2 = self.slow_zone_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), slow_width)

        # Draw dead zone (green rectangle)
        x1, y1, x2, y2 = self.dead_zone_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), dead_width)

    def close(self):
        """Close display windows."""
        cv2.destroyWindow(self.rgb_window)
        cv2.destroyWindow(self.thermal_window)
        cv2.destroyWindow(self.fusion_window)
        cv2.destroyAllWindows()
        logger.info("Closed display windows (RGB, Thermal, Fusion)")
