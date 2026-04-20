"""Simulation-facing driver node that follows official ranger_ros2 semantics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from geometry_msgs.msg import Twist
from std_msgs.msg import UInt8

from .base_vehicle_controller import BaseVehicleController


@dataclass
class RangerDriverCommand:
    motion_mode: int
    linear_speed: float
    lateral_speed: float
    steering_angle: float
    angular_speed: float
    received_time_sec: float


@dataclass(frozen=True)
class _RobotParams:
    track: float
    wheelbase: float
    max_linear_speed: float
    max_angular_speed: float
    max_steer_angle_parallel: float
    min_turn_radius: float
    max_steer_angle_ackermann: float


class RangerROS2Driver(BaseVehicleController):
    """Direct Python port of AgileX `ranger_ros2` cmd_vel handling logic."""

    MOTION_MODE_DUAL_ACKERMAN = 0
    MOTION_MODE_PARALLEL = 1
    MOTION_MODE_SPINNING = 2
    MOTION_MODE_SIDE_SLIP = 3

    _MODEL_PARAMS = {
        "ranger": _RobotParams(
            track=0.56,
            wheelbase=0.90,
            max_linear_speed=2.7,
            max_angular_speed=0.7853,
            max_steer_angle_parallel=1.570,
            min_turn_radius=0.810330349,
            max_steer_angle_ackermann=0.6981,
        ),
        "ranger_mini_v1": _RobotParams(
            track=0.36,
            wheelbase=0.36,
            max_linear_speed=1.5,
            max_angular_speed=0.3,
            max_steer_angle_parallel=0.6981,
            min_turn_radius=0.536,
            max_steer_angle_ackermann=0.6981,
        ),
        "ranger_mini_v2": _RobotParams(
            track=0.364,
            wheelbase=0.494,
            max_linear_speed=1.5,
            max_angular_speed=4.8,
            max_steer_angle_parallel=1.570,
            min_turn_radius=0.4764,
            max_steer_angle_ackermann=0.6981,
        ),
        "ranger_mini_v3": _RobotParams(
            track=0.364,
            wheelbase=0.494,
            max_linear_speed=1.5,
            max_angular_speed=4.8,
            max_steer_angle_parallel=1.570,
            min_turn_radius=0.47644,
            max_steer_angle_ackermann=0.6981,
        ),
    }

    def __init__(self, base_cfg: dict, node_name: str = "ranger_ros2_driver"):
        ros_cfg = base_cfg.get("ros", {})
        self._publish_driver_topics = bool(ros_cfg.get("publish_driver_topics", False))
        self._robot_model = str(ros_cfg.get("ranger_model", "ranger_mini_v3")).lower()
        default_params = self._MODEL_PARAMS.get(self._robot_model, self._MODEL_PARAMS["ranger"])
        self._robot_params = _RobotParams(
            track=float(ros_cfg.get("track", default_params.track)),
            wheelbase=float(ros_cfg.get("wheelbase", default_params.wheelbase)),
            max_linear_speed=float(ros_cfg.get("max_linear_speed", default_params.max_linear_speed)),
            max_angular_speed=float(ros_cfg.get("max_angular_speed", default_params.max_angular_speed)),
            max_steer_angle_parallel=float(
                ros_cfg.get("max_steer_angle_parallel", default_params.max_steer_angle_parallel)
            ),
            min_turn_radius=float(ros_cfg.get("min_turn_radius", default_params.min_turn_radius)),
            max_steer_angle_ackermann=float(
                ros_cfg.get("max_steer_angle_ackermann", default_params.max_steer_angle_ackermann)
            ),
        )
        self._allow_spinning = bool(ros_cfg.get("allow_spinning", True))
        self._spinning_requires_zero_linear = bool(ros_cfg.get("spinning_requires_zero_linear", True))
        self._parking_mode = False
        self._last_nonzero_x = 1.0

        self._received_cmd_vel_count = 0
        self._published_command_count = 0
        self._last_received_cmd_vel = {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "angular_z": 0.0,
        }
        history_size = max(int(base_cfg.get("debug_history_size", 256)), 1)
        self._debug_cmd_vel_history = deque(maxlen=history_size)
        self._last_published_motion_mode = None
        self._latest_command = RangerDriverCommand(
            motion_mode=self.MOTION_MODE_DUAL_ACKERMAN,
            linear_speed=0.0,
            lateral_speed=0.0,
            steering_angle=0.0,
            angular_speed=0.0,
            received_time_sec=0.0,
        )
        super().__init__(base_cfg=base_cfg, node_name=node_name)
        self._latest_command = RangerDriverCommand(
            motion_mode=self.MOTION_MODE_DUAL_ACKERMAN,
            linear_speed=0.0,
            lateral_speed=0.0,
            steering_angle=0.0,
            angular_speed=0.0,
            received_time_sec=self._now_sec(),
        )

    @property
    def required_ros_fields(self) -> tuple[str, ...]:
        if self._publish_driver_topics:
            return ("command_topic", "motion_mode_topic")
        return ()

    def _setup_publishers(self):
        self._command_pub = None
        self._motion_mode_pub = None
        if self._publish_driver_topics:
            self._command_pub = self.node.create_publisher(Twist, self.ros_cfg["command_topic"], 10)
            self._motion_mode_pub = self.node.create_publisher(UInt8, self.ros_cfg["motion_mode_topic"], 10)

    def _handle_cmd_vel(self, msg: Twist):
        receive_time_sec = self._now_sec()
        self._received_cmd_vel_count += 1
        self._last_received_cmd_vel = {
            "linear_x": float(msg.linear.x),
            "linear_y": float(msg.linear.y),
            "angular_z": float(msg.angular.z),
            "received_time_sec": float(receive_time_sec),
        }
        command = self._resolve_command(msg, received_time_sec=receive_time_sec)
        self._latest_command = command
        self._published_command_count += 1
        self._last_published_motion_mode = int(command.motion_mode)
        self._debug_cmd_vel_history.append(
            {
                "received_time_sec": float(receive_time_sec),
                "cmd_vel": {
                    "linear_x": float(msg.linear.x),
                    "linear_y": float(msg.linear.y),
                    "angular_z": float(msg.angular.z),
                },
                "resolved_command": {
                    "motion_mode": int(command.motion_mode),
                    "linear_speed": float(command.linear_speed),
                    "lateral_speed": float(command.lateral_speed),
                    "steering_angle": float(command.steering_angle),
                    "angular_speed": float(command.angular_speed),
                },
            }
        )
        if self._publish_driver_topics:
            motion_mode_msg = UInt8()
            motion_mode_msg.data = int(command.motion_mode)
            self._motion_mode_pub.publish(motion_mode_msg)

            driver_msg = Twist()
            driver_msg.linear.x = float(command.linear_speed)
            driver_msg.linear.y = float(command.lateral_speed)
            driver_msg.angular.x = float(command.steering_angle)
            driver_msg.angular.z = float(command.angular_speed)
            self._command_pub.publish(driver_msg)

    def get_active_command(self) -> RangerDriverCommand:
        return RangerDriverCommand(
            motion_mode=int(self._latest_command.motion_mode),
            linear_speed=float(self._latest_command.linear_speed),
            lateral_speed=float(self._latest_command.lateral_speed),
            steering_angle=float(self._latest_command.steering_angle),
            angular_speed=float(self._latest_command.angular_speed),
            received_time_sec=float(self._latest_command.received_time_sec),
        )

    def _resolve_command(self, msg: Twist, *, received_time_sec: float | None = None) -> RangerDriverCommand:
        if self._parking_mode and self._robot_model == "ranger_mini_v2":
            return self.get_active_command()

        linear_x = float(msg.linear.x)
        linear_y = float(msg.linear.y)
        angular_z = float(msg.angular.z)

        if linear_y != 0.0:
            if linear_x == 0.0 and self._robot_model == "ranger_mini_v1":
                motion_mode = self.MOTION_MODE_SIDE_SLIP
            else:
                motion_mode = self.MOTION_MODE_PARALLEL
        else:
            steer_cmd, radius = self._calculate_steering_angle(msg)
            pure_rotation_cmd = abs(linear_x) <= 1.0e-4 and abs(linear_y) <= 1.0e-4 and abs(angular_z) > 1.0e-6
            spinning_allowed = self._allow_spinning and (
                not self._spinning_requires_zero_linear or pure_rotation_cmd
            )
            if spinning_allowed and radius < self._robot_params.min_turn_radius:
                motion_mode = self.MOTION_MODE_SPINNING
            else:
                motion_mode = self.MOTION_MODE_DUAL_ACKERMAN

        steering_angle = 0.0
        resolved_linear_speed = 0.0
        resolved_lateral_speed = 0.0
        resolved_angular_speed = 0.0

        if motion_mode == self.MOTION_MODE_DUAL_ACKERMAN:
            steer_cmd, _ = self._calculate_steering_angle(msg)
            steering_angle = self._clip(
                steer_cmd,
                -self._robot_params.max_steer_angle_ackermann,
                self._robot_params.max_steer_angle_ackermann,
            )
            resolved_linear_speed = self._clip(
                linear_x,
                -self._robot_params.max_linear_speed,
                self._robot_params.max_linear_speed,
            )
        elif motion_mode == self.MOTION_MODE_PARALLEL:
            steering_angle = math.atan2(linear_y, linear_x if linear_x != 0.0 else 0.0)
            if linear_x != 0.0:
                self._last_nonzero_x = linear_x
            if math.copysign(1.0, linear_x) < 0.0:
                steering_angle = -steering_angle
            steering_angle = self._clip(
                steering_angle,
                -self._robot_params.max_steer_angle_parallel,
                self._robot_params.max_steer_angle_parallel,
            )

            if linear_x == 0.0 and linear_y != 0.0:
                if math.copysign(1.0, self._last_nonzero_x) < 0.0:
                    steering_angle = -abs(steering_angle)
                else:
                    steering_angle = abs(steering_angle)
                vel_sign = 1.0 if linear_y >= 0.0 else -1.0
            else:
                vel_sign = 1.0 if linear_x >= 0.0 else -1.0

            resolved_linear_speed = self._clip(
                vel_sign * math.sqrt(linear_x * linear_x + linear_y * linear_y),
                -self._robot_params.max_linear_speed,
                self._robot_params.max_linear_speed,
            )
        elif motion_mode == self.MOTION_MODE_SPINNING:
            resolved_angular_speed = self._clip(
                angular_z,
                -self._robot_params.max_angular_speed,
                self._robot_params.max_angular_speed,
            )
        elif motion_mode == self.MOTION_MODE_SIDE_SLIP:
            resolved_lateral_speed = self._clip(
                linear_y,
                -self._robot_params.max_linear_speed,
                self._robot_params.max_linear_speed,
            )

        return RangerDriverCommand(
            motion_mode=motion_mode,
            linear_speed=resolved_linear_speed,
            lateral_speed=resolved_lateral_speed,
            steering_angle=steering_angle,
            angular_speed=resolved_angular_speed,
            received_time_sec=float(self._now_sec() if received_time_sec is None else received_time_sec),
        )

    def _calculate_steering_angle(self, msg: Twist) -> tuple[float, float]:
        linear = abs(float(msg.linear.x))
        angular = abs(float(msg.angular.z))
        if angular < 1.0e-6:
            return 0.0, float("inf")

        radius = linear / angular
        sign = 1.0 if (float(msg.angular.z) * float(msg.linear.x)) >= 0.0 else -1.0
        phi_i = math.atan((self._robot_params.wheelbase * 0.5) / max(radius, 1.0e-8))
        phi_i = min(phi_i, math.radians(40.0))
        return sign * phi_i, radius

    def _now_sec(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _clip(value: float, low: float, high: float):
        return max(low, min(high, value))


# Keep the old class name as a compatibility alias for existing imports.
RangerMiniV3Controller = RangerROS2Driver
