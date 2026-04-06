"""Ranger Mini V3 cmd_vel controller built on the generic mobile base class."""

import math

from geometry_msgs.msg import Twist
from std_msgs.msg import UInt8

from .base_vehicle_controller import BaseVehicleController


class RangerMiniV3Controller(BaseVehicleController):
    """Convert /cmd_vel into Ranger-driver-like simulation command topics."""

    MOTION_MODE_DUAL_ACKERMAN = 0
    MOTION_MODE_PARALLEL = 1
    MOTION_MODE_SPINNING = 2
    MOTION_MODE_SIDE_SLIP = 3

    def __init__(self, base_cfg: dict, node_name: str = "ranger_mini_v3_controller"):
        ros_cfg = base_cfg.get("ros", {})

        self._robot_model = str(ros_cfg.get("ranger_model", "ranger")).lower()
        self._steering_limit = float(base_cfg["steering_limit"])
        self._max_steer_angle_ackermann = float(ros_cfg.get("max_steer_angle_ackermann", self._steering_limit))
        self._wheel_base = float(base_cfg["wheel_base"])
        self._track_width = float(base_cfg["track_width"])
        self._wheel_radius = float(base_cfg["wheel_radius"])
        self._wheel_velocity_limit = float(base_cfg["wheel_velocity_limit"])

        if self._wheel_base <= 0.0 or self._track_width <= 0.0:
            raise ValueError("wheel_base and track_width must be positive")
        if self._wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")

        self._max_linear_speed = self._wheel_velocity_limit * self._wheel_radius
        lever_arm = 0.5 * math.hypot(self._wheel_base, self._track_width)
        self._max_angular_speed = self._max_linear_speed / max(lever_arm, 1e-6)

        if abs(math.tan(self._max_steer_angle_ackermann)) < 1e-6:
            self._min_turn_radius = float("inf")
        else:
            self._min_turn_radius = abs((self._wheel_base * 0.5) / math.tan(self._max_steer_angle_ackermann))

        self._last_nonzero_x = 1.0

        super().__init__(base_cfg=base_cfg, node_name=node_name)

    @property
    def required_ros_fields(self) -> tuple[str, ...]:
        return ("command_topic", "motion_mode_topic")

    def _setup_publishers(self):
        self._base_command_pub = self.node.create_publisher(Twist, self.ros_cfg["command_topic"], 10)
        self._motion_mode_pub = self.node.create_publisher(UInt8, self.ros_cfg["motion_mode_topic"], 10)

    def _handle_cmd_vel(self, msg: Twist):
        motion_mode, sim_command = self._convert_cmd_vel(msg)

        mode_msg = UInt8()
        mode_msg.data = int(motion_mode)
        self._motion_mode_pub.publish(mode_msg)
        self._base_command_pub.publish(sim_command)

    def _convert_cmd_vel(self, msg: Twist):
        if abs(msg.linear.y) > 1e-8:
            if abs(msg.linear.x) <= 1e-8 and self._robot_model == "ranger_mini_v1":
                motion_mode = self.MOTION_MODE_SIDE_SLIP
            else:
                motion_mode = self.MOTION_MODE_PARALLEL
        else:
            steer_cmd, radius = self._calculate_steering_angle(msg)
            if radius < self._min_turn_radius:
                motion_mode = self.MOTION_MODE_SPINNING
            else:
                motion_mode = self.MOTION_MODE_DUAL_ACKERMAN

        command = Twist()
        if motion_mode == self.MOTION_MODE_DUAL_ACKERMAN:
            steer_cmd, _ = self._calculate_steering_angle(msg)
            steer_cmd = self._clip(steer_cmd, -self._max_steer_angle_ackermann, self._max_steer_angle_ackermann)
            command.linear.x = self._clip(msg.linear.x, -self._max_linear_speed, self._max_linear_speed)
            command.angular.x = steer_cmd
        elif motion_mode == self.MOTION_MODE_PARALLEL:
            base_x = msg.linear.x if abs(msg.linear.x) > 1e-8 else self._last_nonzero_x
            if abs(msg.linear.x) > 1e-8:
                self._last_nonzero_x = msg.linear.x

            steer_cmd = math.atan2(msg.linear.y, base_x)
            if math.copysign(1.0, msg.linear.x if abs(msg.linear.x) > 1e-8 else base_x) < 0:
                steer_cmd = -steer_cmd

            steer_cmd = self._clip(steer_cmd, -self._steering_limit, self._steering_limit)

            if abs(msg.linear.x) <= 1e-8 and abs(msg.linear.y) > 1e-8:
                if self._last_nonzero_x < 0:
                    steer_cmd = -abs(steer_cmd)
                else:
                    steer_cmd = abs(steer_cmd)
                vel_sign = 1.0 if msg.linear.y >= 0.0 else -1.0
            else:
                vel_sign = 1.0 if msg.linear.x >= 0.0 else -1.0

            speed = vel_sign * math.sqrt(msg.linear.x * msg.linear.x + msg.linear.y * msg.linear.y)
            command.linear.x = self._clip(speed, -self._max_linear_speed, self._max_linear_speed)
            command.angular.x = steer_cmd
        elif motion_mode == self.MOTION_MODE_SPINNING:
            command.angular.z = self._clip(msg.angular.z, -self._max_angular_speed, self._max_angular_speed)
        elif motion_mode == self.MOTION_MODE_SIDE_SLIP:
            command.linear.y = self._clip(msg.linear.y, -self._max_linear_speed, self._max_linear_speed)

        return motion_mode, command

    def _calculate_steering_angle(self, msg: Twist):
        linear = abs(msg.linear.x)
        angular = abs(msg.angular.z)

        if angular < 1e-8:
            return 0.0, float("inf")

        radius = linear / angular
        steer = math.atan((self._wheel_base * 0.5) / max(radius, 1e-8))
        steer = min(steer, self._max_steer_angle_ackermann)
        sign = 1.0 if (msg.angular.z * msg.linear.x) >= 0.0 else -1.0
        return sign * steer, radius

    @staticmethod
    def _clip(value: float, low: float, high: float):
        return max(low, min(high, value))
