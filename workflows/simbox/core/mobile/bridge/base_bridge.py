"""移动底盘 Isaac Bridge 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from .types import BaseCommand


class BaseBridge(ABC):
    """将标准 /cmd_vel 直接桥接为 Isaac articulation 控制。"""

    def __init__(self, robot, node_name: str, driver=None):
        if driver is not None:
            raise ValueError("Internal driver translation is disabled; publish /cmd_vel directly to Isaac bridge.")

        self.robot = robot
        self.base_interface = robot.get_base_interface()
        self.base_cfg = self.base_interface["base_cfg"]
        self.ros_cfg = self.base_cfg["ros"]

        required_ros_fields = ["cmd_vel_topic", "joint_state_topic", "odom_topic", "base_frame", "odom_frame"]
        missing_fields = [field for field in required_ros_fields if field not in self.ros_cfg]
        if missing_fields:
            raise KeyError(f"Missing ROS base bridge config fields: {missing_fields}")

        virtual_odom_cfg = self.ros_cfg.get("virtual_odom", {})
        if not isinstance(virtual_odom_cfg, dict):
            raise TypeError("ros.virtual_odom must be a dict when present")
        if bool(virtual_odom_cfg.get("enabled", False)):
            raise ValueError("virtual_odom is not supported for the direct /cmd_vel 4WIS bridge")

        self._command_timeout = float(self.base_cfg["command_timeout"])
        self._steering_limit = float(self.base_cfg["steering_limit"])
        self._steering_rate_limit = float(self.base_cfg["steering_rate_limit"])
        self._wheel_velocity_limit = float(self.base_cfg["wheel_velocity_limit"])
        self._wheel_base = float(self.base_cfg["wheel_base"])
        self._track_width = float(self.base_cfg["track_width"])
        self._wheel_radius = float(self.base_cfg["wheel_radius"])
        self._steering_command_sign = float(
            self.base_cfg.get("steering_command_sign", self.ros_cfg.get("steering_command_sign", 1.0))
        )
        if abs(self._steering_command_sign) <= 1.0e-6:
            raise ValueError("steering_command_sign must be non-zero")
        if self._wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")

        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        self._validate_bridge_configuration(steering_count=steering_count, wheel_count=wheel_count)

        self._owns_rclpy_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        self.node = Node(node_name)
        self._tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)
        self._joint_state_pub = self.node.create_publisher(JointState, self.ros_cfg["joint_state_topic"], 10)
        self._odom_pub = self.node.create_publisher(Odometry, self.ros_cfg["odom_topic"], 10)
        self._cmd_vel_sub = self.node.create_subscription(
            Twist,
            self.ros_cfg["cmd_vel_topic"],
            self._on_cmd_vel,
            10,
        )

        now_sec = self._now_sec()
        self._command = BaseCommand.zero(received_time_sec=now_sec)
        self._last_applied_steering = np.zeros(steering_count, dtype=np.float32)
        self._last_requested_steering = np.zeros(steering_count, dtype=np.float32)
        self._last_requested_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_step_time_sec = now_sec
        self._last_step_dt = 1e-3

        history_size = max(int(self.base_cfg.get("debug_history_size", 256)), 1)
        self._received_cmd_vel_count = 0
        self._driver_command_message_count = 0
        self._pending_driver_command_count = 0
        self._applied_driver_command_count = 0
        self._motion_mode_message_count = 0
        self._has_motion_mode = False
        self._latest_motion_mode = 0
        self._last_received_cmd_vel = {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "angular_z": 0.0,
            "received_time_sec": float(now_sec),
        }
        self._debug_cmd_vel_history = deque(maxlen=history_size)
        self._debug_command_history = deque(maxlen=history_size)
        self._last_published_pose_debug = {}

        translation, orientation = self._get_robot_base_pose()
        self._last_actual_translation = np.array(translation, dtype=np.float32)
        self._last_actual_yaw = float(self._yaw_from_wxyz(orientation))
        self._last_actual_linear_velocity_world = np.zeros(3, dtype=np.float32)
        self._last_actual_angular_velocity_world = np.zeros(3, dtype=np.float32)

    @abstractmethod
    def _validate_bridge_configuration(self, *, steering_count: int, wheel_count: int):
        """校验桥接器与目标底盘的配置是否匹配。"""

    @abstractmethod
    def _map_command(self, command: BaseCommand) -> tuple[np.ndarray, np.ndarray]:
        """将车体级命令映射为关节转向角和轮角速度。"""

    def destroy(self):
        self.node.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()

    def step(self, step_dt: float | None = None):
        self._spin_available_callbacks()
        now_sec = self._now_sec()
        if step_dt is None:
            dt = max(now_sec - self._last_step_time_sec, 1e-3)
        else:
            dt = max(float(step_dt), 1e-3)
        self._last_step_time_sec = now_sec
        self._last_step_dt = dt

        command = self._resolve_active_command(now_sec)
        requested_steering, wheel_velocities = self._map_command(command)
        self._last_requested_steering = requested_steering.astype(np.float32).copy()
        self._last_requested_wheel_velocities = wheel_velocities.astype(np.float32).copy()
        steering_positions = self._apply_steering_limits(requested_steering, dt)
        self.robot.apply_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
        )
        self._publish_joint_state()
        self._publish_odometry()
        self._record_debug_history(
            command=command,
            requested_steering=requested_steering,
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
            now_sec=now_sec,
            dt=dt,
        )
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def _on_cmd_vel(self, msg: Twist):
        received_time_sec = self._now_sec()
        self._received_cmd_vel_count += 1
        self._driver_command_message_count += 1
        self._applied_driver_command_count += 1
        self._last_received_cmd_vel = {
            "linear_x": float(msg.linear.x),
            "linear_y": float(msg.linear.y),
            "angular_z": float(msg.angular.z),
            "received_time_sec": float(received_time_sec),
        }
        command = BaseCommand.from_twist_message(msg, received_time_sec=received_time_sec)
        self._command = command
        self._debug_cmd_vel_history.append(
            {
                "received_time_sec": float(received_time_sec),
                "cmd_vel": {
                    "linear_x": float(msg.linear.x),
                    "linear_y": float(msg.linear.y),
                    "angular_z": float(msg.angular.z),
                },
                "resolved_command": {
                    "vx_body": float(command.vx_body),
                    "vy_body": float(command.vy_body),
                    "wz_body": float(command.wz_body),
                },
            }
        )

    def _resolve_active_command(self, now_sec: float) -> BaseCommand:
        if now_sec - self._command.received_time_sec <= self._command_timeout:
            return self._command
        return BaseCommand.zero(received_time_sec=self._command.received_time_sec)

    def _apply_steering_limits(self, requested_positions: np.ndarray, dt: float):
        requested_positions = np.clip(requested_positions, -self._steering_limit, self._steering_limit)
        max_delta = self._steering_rate_limit * dt
        delta = requested_positions - self._last_applied_steering
        delta = np.clip(delta, -max_delta, max_delta)
        limited = self._last_applied_steering + delta
        self._last_applied_steering = limited.astype(np.float32)
        return self._last_applied_steering.copy()

    def _publish_joint_state(self):
        joint_state = self.robot.get_base_joint_state()
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.base_interface["steering_joint_names"] + self.base_interface["wheel_joint_names"]
        msg.position = (
            list(joint_state["steering_positions"].astype(float)) + list(joint_state["wheel_positions"].astype(float))
        )
        msg.velocity = (
            list(joint_state["steering_velocities"].astype(float)) + list(joint_state["wheel_velocities"].astype(float))
        )
        self._joint_state_pub.publish(msg)

    def _publish_odometry(self):
        translation, orientation = self._get_robot_base_pose()
        linear_velocity, angular_velocity = self._get_actual_base_twist(translation, orientation)
        linear_velocity_body = self._world_linear_velocity_to_body(linear_velocity, orientation)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.node.get_clock().now().to_msg()
        odom_msg.header.frame_id = self.ros_cfg["odom_frame"]
        odom_msg.child_frame_id = self.ros_cfg["base_frame"]
        odom_msg.pose.pose.position.x = float(translation[0])
        odom_msg.pose.pose.position.y = float(translation[1])
        odom_msg.pose.pose.position.z = float(translation[2])
        odom_msg.pose.pose.orientation.x = float(orientation[1])
        odom_msg.pose.pose.orientation.y = float(orientation[2])
        odom_msg.pose.pose.orientation.z = float(orientation[3])
        odom_msg.pose.pose.orientation.w = float(orientation[0])
        odom_msg.twist.twist.linear.x = float(linear_velocity_body[0])
        odom_msg.twist.twist.linear.y = float(linear_velocity_body[1])
        odom_msg.twist.twist.linear.z = float(linear_velocity_body[2])
        odom_msg.twist.twist.angular.x = float(angular_velocity[0])
        odom_msg.twist.twist.angular.y = float(angular_velocity[1])
        odom_msg.twist.twist.angular.z = float(angular_velocity[2])

        self._odom_pub.publish(odom_msg)
        self._last_published_pose_debug = {
            "x": float(translation[0]),
            "y": float(translation[1]),
            "z": float(translation[2]),
            "yaw": float(self._yaw_from_wxyz(orientation)),
            "linear_velocity_body": [float(v) for v in list(linear_velocity_body)],
            "angular_velocity_world": [float(v) for v in list(angular_velocity)],
        }

        if self.ros_cfg["tf_enabled"]:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = odom_msg.header.stamp
            tf_msg.header.frame_id = self.ros_cfg["odom_frame"]
            tf_msg.child_frame_id = self.ros_cfg["base_frame"]
            tf_msg.transform.translation.x = float(translation[0])
            tf_msg.transform.translation.y = float(translation[1])
            tf_msg.transform.translation.z = float(translation[2])
            tf_msg.transform.rotation.x = float(orientation[1])
            tf_msg.transform.rotation.y = float(orientation[2])
            tf_msg.transform.rotation.z = float(orientation[3])
            tf_msg.transform.rotation.w = float(orientation[0])
            self._tf_pub.publish(TFMessage(transforms=[tf_msg]))

    def _get_robot_base_pose(self):
        getter = getattr(self.robot, "get_mobile_base_pose", None)
        if callable(getter):
            return getter()
        return self.robot.get_world_pose()

    def _get_actual_base_twist(self, translation, orientation):
        translation = np.asarray(translation, dtype=np.float32)
        yaw = float(self._yaw_from_wxyz(orientation))
        dt = max(float(self._last_step_dt), 1e-3)

        linear_velocity = (translation - self._last_actual_translation) / dt
        yaw_delta = self._wrap_angle(yaw - self._last_actual_yaw)
        angular_velocity = np.array([0.0, 0.0, yaw_delta / dt], dtype=np.float32)

        self._last_actual_translation = translation.copy()
        self._last_actual_yaw = yaw
        self._last_actual_linear_velocity_world = linear_velocity.astype(np.float32)
        self._last_actual_angular_velocity_world = angular_velocity
        return self._last_actual_linear_velocity_world, self._last_actual_angular_velocity_world

    @staticmethod
    def _world_linear_velocity_to_body(linear_velocity_world, orientation_wxyz):
        linear_velocity_world = np.asarray(linear_velocity_world, dtype=np.float32)
        yaw = float(BaseBridge._yaw_from_wxyz(orientation_wxyz))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_world = float(linear_velocity_world[0])
        vy_world = float(linear_velocity_world[1])
        vz_world = float(linear_velocity_world[2])
        return np.array(
            [
                cos_yaw * vx_world + sin_yaw * vy_world,
                -sin_yaw * vx_world + cos_yaw * vy_world,
                vz_world,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _yaw_from_wxyz(q_wxyz):
        w = float(q_wxyz[0])
        x = float(q_wxyz[1])
        y = float(q_wxyz[2])
        z = float(q_wxyz[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap_angle(angle: float):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _record_debug_history(
        self,
        *,
        command: BaseCommand,
        requested_steering: np.ndarray,
        steering_positions: np.ndarray,
        wheel_velocities: np.ndarray,
        now_sec: float,
        dt: float,
    ):
        history_item = {
            "time_sec": float(now_sec),
            "dt": float(dt),
            "command": {
                "vx_body": float(command.vx_body),
                "vy_body": float(command.vy_body),
                "wz_body": float(command.wz_body),
            },
            "predicted_body_twist": {
                "vx": float(command.vx_body),
                "vy": float(command.vy_body),
                "wz": float(command.wz_body),
            },
            "requested_steering": [float(v) for v in list(np.asarray(requested_steering).reshape(-1))],
            "applied_steering": [float(v) for v in list(np.asarray(steering_positions).reshape(-1))],
            "wheel_velocities": [float(v) for v in list(np.asarray(wheel_velocities).reshape(-1))],
            "pose": dict(self._last_published_pose_debug),
        }
        self._debug_command_history.append(history_item)

    def _now_sec(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _spin_available_callbacks(self, max_callbacks: int = 8):
        callback_count = 0
        while callback_count < max(int(max_callbacks), 1):
            timeout_sec = 0.001 if callback_count == 0 else 0.0
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
            callback_count += 1
