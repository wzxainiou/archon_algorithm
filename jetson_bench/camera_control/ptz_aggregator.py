"""
PTZ movement aggregator for time-windowed output control.

This module implements the "direct output + dual emergency threshold" strategy:
- Normal mode: Output current state every 0.3 seconds
- Emergency mode 1: Output immediately when velocity exceeds threshold (fast motion)
- Emergency mode 2: Output immediately when distance exceeds 80% boundary (too far)
- No accumulation: Always output current target position
"""

import time
import math
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PTZAggregator:
    """
    Aggregates PTZ movements with time-windowed output control.

    Strategy: Direct output of current state with dual emergency thresholds.
    - Monitors every frame but only outputs periodically
    - Outputs current target position (not accumulated history)
    - Dual emergency triggers: velocity (fast motion) + distance (too far)
    """

    def __init__(self,
                 smooth_window: float = 0.3,
                 velocity_threshold: float = 100.0,
                 distance_threshold_percent: float = 0.8,
                 frame_width: int = 1280,
                 frame_height: int = 720):
        """
        Initialize PTZ aggregator with dual emergency thresholds.

        Args:
            smooth_window: Time interval for regular output (seconds)
            velocity_threshold: Velocity threshold for fast motion trigger (px/s)
            distance_threshold_percent: Distance threshold as percentage of max distance (0.8 = 80% boundary)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.smooth_window = smooth_window
        self.velocity_threshold = velocity_threshold
        self.distance_threshold_percent = distance_threshold_percent

        # Calculate 80% boundary distance threshold (from center to edge)
        max_distance_from_center = math.sqrt(
            (frame_width / 2) ** 2 + (frame_height / 2) ** 2
        )
        self.distance_threshold_px = max_distance_from_center * distance_threshold_percent

        # Position tracking (for velocity calculation)
        self.last_position = None
        self.last_timestamp = None

        # Current state (not accumulated)
        self.current_weighted_x = 0.0
        self.current_weighted_y = 0.0
        self.current_pixel_distance_total = 0.0
        self.current_velocity = 0.0

        self.last_output_time = time.time()

        # Statistics
        self.total_outputs = 0
        self.velocity_emergency_outputs = 0
        self.distance_emergency_outputs = 0
        self.regular_outputs = 0

        logger.info(f"PTZ Aggregator initialized:")
        logger.info(f"  - Window: {smooth_window}s")
        logger.info(f"  - Velocity threshold: {velocity_threshold}px/s")
        logger.info(f"  - Distance threshold: {distance_threshold_percent * 100}% boundary ({self.distance_threshold_px:.1f}px)")

    def add_movement(self, ptz_status: Dict) -> None:
        """
        Update current state and calculate velocity (replace, not accumulate).

        Args:
            ptz_status: PTZ status dict with weighted_distance and pixel_distance fields
        """
        current_time = time.time()

        # Update weighted distance (direct replacement)
        self.current_weighted_x = ptz_status["weighted_distance"]["x"]
        self.current_weighted_y = ptz_status["weighted_distance"]["y"]
        self.current_pixel_distance_total = ptz_status["pixel_distance"]["total"]

        # Calculate velocity (based on weighted distance change)
        if self.last_position is not None and self.last_timestamp is not None:
            dt = current_time - self.last_timestamp
            if dt > 0:
                dx = self.current_weighted_x - self.last_position[0]
                dy = self.current_weighted_y - self.last_position[1]
                displacement = math.sqrt(dx**2 + dy**2)
                self.current_velocity = displacement / dt  # px/s

        # Save current position and timestamp
        self.last_position = (self.current_weighted_x, self.current_weighted_y)
        self.last_timestamp = current_time

    def get_aggregated_output(self) -> Optional[Dict]:
        """
        Get output if conditions are met (three trigger conditions).

        Trigger conditions (priority order):
        1. Velocity emergency: velocity exceeds threshold (fast motion)
        2. Distance emergency: target exceeds 80% boundary (too far)
        3. Regular trigger: smooth_window seconds elapsed

        Returns:
            Output dict if triggered, None otherwise
        """
        current_time = time.time()

        # Calculate current weighted total distance
        current_weighted_total = math.sqrt(
            self.current_weighted_x**2 + self.current_weighted_y**2
        )

        time_elapsed = current_time - self.last_output_time
        should_output = False
        trigger_reason = None

        # Condition 1: Velocity emergency threshold (fast motion)
        if self.current_velocity >= self.velocity_threshold:
            should_output = True
            trigger_reason = "emergency_velocity"
            self.velocity_emergency_outputs += 1
            logger.warning(f"🚀 Velocity emergency: {self.current_velocity:.1f}px/s >= {self.velocity_threshold}px/s")

        # Condition 2: Distance emergency threshold (80% boundary)
        elif self.current_pixel_distance_total >= self.distance_threshold_px:
            should_output = True
            trigger_reason = "emergency_distance"
            self.distance_emergency_outputs += 1
            logger.warning(f"⚠️  Distance emergency: {self.current_pixel_distance_total:.1f}px >= {self.distance_threshold_px:.1f}px (80% boundary)")

        # Condition 3: Regular timer
        elif time_elapsed >= self.smooth_window:
            # Only output if there's meaningful offset
            if current_weighted_total > 0.1:
                should_output = True
                trigger_reason = "regular"
                self.regular_outputs += 1

        if should_output:
            output = {
                "output_x": round(self.current_weighted_x, 2),
                "output_y": round(self.current_weighted_y, 2),
                "output_total": round(current_weighted_total, 2),
                "trigger": trigger_reason,
                "velocity": round(self.current_velocity, 2),
                "pixel_distance_total": round(self.current_pixel_distance_total, 2),
                "time_since_last": round(time_elapsed, 3),
            }

            self.last_output_time = current_time
            self.total_outputs += 1

            logger.debug(f"PTZ Output: ({output['output_x']:+.1f}, {output['output_y']:+.1f})px, "
                        f"trigger={trigger_reason}, velocity={self.current_velocity:.1f}px/s, "
                        f"distance={self.current_pixel_distance_total:.0f}px, elapsed={time_elapsed:.2f}s")

            return output

        return None

    def get_stats(self) -> Dict:
        """Get aggregator statistics with dual threshold breakdown."""
        return {
            "total_outputs": self.total_outputs,
            "velocity_emergency_outputs": self.velocity_emergency_outputs,
            "distance_emergency_outputs": self.distance_emergency_outputs,
            "regular_outputs": self.regular_outputs,
            "current_velocity": round(self.current_velocity, 2),
            "current_weighted_x": round(self.current_weighted_x, 2),
            "current_weighted_y": round(self.current_weighted_y, 2),
        }
