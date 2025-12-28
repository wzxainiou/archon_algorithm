"""
Pan-Tilt-Zoom camera controller for hunting camera system.

Implements 3-zone movement strategy:
- Center zone (±10%): No movement
- Slow zone (±30%): Slow movement
- Fast zone (>30%): Fast movement
"""

from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PTZController:
    """Pan-Tilt-Zoom camera controller with 3-zone movement strategy."""

    def __init__(self,
                 serial_port: str = "/dev/ttyUSB0",
                 baudrate: int = 9600,
                 frame_width: int = 640,
                 frame_height: int = 480,
                 fov_horizontal: float = 60.0,  # degrees
                 fov_vertical: float = 45.0):   # degrees
        """
        Initialize PTZ controller.

        Args:
            serial_port: Serial port path (e.g., "/dev/ttyUSB0")
            baudrate: Serial communication baud rate
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            fov_horizontal: Camera horizontal field of view in degrees
            fov_vertical: Camera vertical field of view in degrees
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fov_h = fov_horizontal
        self.fov_v = fov_vertical

        # Zone thresholds (percentage of frame size)
        self.dead_zone = 0.10   # ±10% of center
        self.slow_zone = 0.30   # ±30% of center

        # Movement speeds (degrees per update)
        self.fast_speed = 5.0
        self.slow_speed = 1.0

        # Serial connection
        self.serial = None
        try:
            import serial as pyserial
            self.serial = pyserial.Serial(serial_port, baudrate, timeout=1)
            logger.info(f"✅ PTZ Controller connected to {serial_port}")
        except ImportError:
            logger.warning("⚠️  pyserial not installed - running in simulation mode")
        except Exception as e:
            logger.warning(f"⚠️  Could not connect to PTZ: {e} - running in simulation mode")

    def calculate_movement(self, centroid: Tuple[float, float]) -> Dict:
        """
        Calculate pan/tilt angles and weighted distances based on object centroid.

        Args:
            centroid: (cx, cy) in pixels

        Returns:
            Dict: {
                "pan": float (degrees),
                "tilt": float (degrees),
                "zone": str ("center"|"slow"|"fast"),
                "pixel_distance": {x, y, total},
                "weight": float,
                "weighted_distance": {x, y, total},
                "timestamp": float
            }
        """
        import time
        import math

        cx, cy = centroid

        # Calculate pixel distance to center
        center_x = self.frame_width / 2
        center_y = self.frame_height / 2
        pixel_distance_x = cx - center_x
        pixel_distance_y = cy - center_y
        pixel_distance_total = math.sqrt(pixel_distance_x**2 + pixel_distance_y**2)

        # Normalize to [-0.5, 0.5] relative to center
        dx = pixel_distance_x / self.frame_width
        dy = pixel_distance_y / self.frame_height

        # Determine zone (rectangular: both x and y must be within zone)
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx < self.dead_zone and abs_dy < self.dead_zone:
            zone = "center"
            speed = 0.0
            weight = 0.0
        elif abs_dx < self.slow_zone and abs_dy < self.slow_zone:
            zone = "slow"
            speed = self.slow_speed
            weight = 0.1
        else:
            zone = "fast"
            speed = self.fast_speed
            weight = 0.2

        # Calculate weighted distances
        weighted_distance_x = pixel_distance_x * weight
        weighted_distance_y = pixel_distance_y * weight
        weighted_distance_total = pixel_distance_total * weight

        # Convert to angles (keep existing logic)
        pan_angle = dx * self.fov_h * speed
        tilt_angle = -dy * self.fov_v * speed  # Negative: up is positive tilt

        return {
            "pan": round(pan_angle, 2),
            "tilt": round(tilt_angle, 2),
            "zone": zone,
            "pixel_distance": {
                "x": round(pixel_distance_x, 2),
                "y": round(pixel_distance_y, 2),
                "total": round(pixel_distance_total, 2),
            },
            "weight": weight,
            "weighted_distance": {
                "x": round(weighted_distance_x, 2),
                "y": round(weighted_distance_y, 2),
                "total": round(weighted_distance_total, 2),
            },
            "timestamp": time.time(),
        }

    def send_command(self, pan: float, tilt: float) -> bool:
        """
        Send PTZ command via serial port.

        Args:
            pan: Pan angle in degrees
            tilt: Tilt angle in degrees

        Returns:
            bool: True if command sent successfully

        Command format: "PAN:+5.20,TILT:-3.10\n"
        """
        command = f"PAN:{pan:+.2f},TILT:{tilt:+.2f}\n"

        if self.serial and self.serial.is_open:
            try:
                self.serial.write(command.encode('ascii'))
                logger.debug(f"📡 PTZ Command: {command.strip()}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to send PTZ command: {e}")
                return False
        else:
            # Simulation mode (no hardware)
            logger.info(f"🎮 [SIMULATION] PTZ Command: {command.strip()}")
            return True

    def close(self):
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Closed PTZ serial connection")
