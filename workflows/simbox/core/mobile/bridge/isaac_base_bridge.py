"""ROS to Isaac Sim bridge for the SplitAloha mobile base."""

from dataclasses import dataclass
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8
from tf2_msgs.msg import TFMessage


@dataclass
class _BaseCommand:
    motion_mode: int
    linear_speed: float
    lateral_speed: float
    steering_angle: float
    angular_speed: float
    received_time_sec: float


class SplitAlohaIsaacBaseBridge:
    """Bridge ROS base commands into Isaac Sim articulation commands."""

    MOTION_MODE_DUAL_ACKERMAN = 0
    MOTION_MODE_PARALLEL = 1
    MOTION_MODE_SPINNING = 2
    MOTION_MODE_SIDE_SLIP = 3

    def __init__(self, robot, node_name: str = "split_aloha_isaac_base_bridge"):
        self.robot = robot
        self.base_interface = robot.get_base_interface()
        self.base_cfg = self.base_interface["base_cfg"]
        self.ros_cfg = self.base_cfg["ros"]

        required_ros_fields = [
            "command_topic",
            "motion_mode_topic",
            "joint_state_topic",
            "odom_topic",
            "base_frame",
            "odom_frame",
        ]
        missing_fields = [field for field in required_ros_fields if field not in self.ros_cfg]
        if missing_fields:
            raise KeyError(f"Missing ROS base bridge config fields: {missing_fields}")

        self._command_type = str(self.ros_cfg.get("command_type", "")).lower()
        if self._command_type != "ranger_driver":
            raise ValueError(
                f"Unsupported command_type '{self._command_type}', only 'ranger_driver' is supported"
            )

        self._command_timeout = float(self.base_cfg["command_timeout"])
        self._steering_limit = float(self.base_cfg["steering_limit"])
        self._steering_rate_limit = float(self.base_cfg["steering_rate_limit"])
        self._wheel_velocity_limit = float(self.base_cfg["wheel_velocity_limit"])
        self._wheel_base = float(self.base_cfg["wheel_base"])
        self._track_width = float(self.base_cfg["track_width"])
        self._wheel_radius = float(self.base_cfg["wheel_radius"])
        if self._wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")
        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        if steering_count != 4 or wheel_count != 4:
            raise ValueError(
                f"SplitAloha Ranger-driver bridge expects 4 steering and 4 wheel joints, got {steering_count}/{wheel_count}"
            )

        self._owns_rclpy_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        self.node = Node(node_name)
        self._tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)
        self._joint_state_pub = self.node.create_publisher(JointState, self.ros_cfg["joint_state_topic"], 10)
        self._odom_pub = self.node.create_publisher(Odometry, self.ros_cfg["odom_topic"], 10)
        self._command_sub = self.node.create_subscription(
            Twist,
            self.ros_cfg["command_topic"],
            self._on_driver_command,
            10,
        )
        self._motion_mode_sub = self.node.create_subscription(
            UInt8,
            self.ros_cfg["motion_mode_topic"],
            self._on_motion_mode,
            10,
        )

        now_sec = self._now_sec()
        self._latest_motion_mode = self.MOTION_MODE_DUAL_ACKERMAN
        self._has_motion_mode = False
        self._command = _BaseCommand(
            motion_mode=self.MOTION_MODE_DUAL_ACKERMAN,
            linear_speed=0.0,
            lateral_speed=0.0,
            steering_angle=0.0,
            angular_speed=0.0,
            received_time_sec=now_sec,
        )
        self._last_applied_steering = np.zeros(steering_count, dtype=np.float32)
        self._last_step_time_sec = now_sec

    def destroy(self):
        self.node.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()

    def step(self):
        # A tiny timeout avoids starving DDS callback processing in tight simulation loops.
        rclpy.spin_once(self.node, timeout_sec=0.001)
        now_sec = self._now_sec()
        dt = max(now_sec - self._last_step_time_sec, 1e-3)
        self._last_step_time_sec = now_sec
        command = self._resolve_active_command(now_sec)
        requested_steering, wheel_velocities = self._map_ranger_command(command)
        steering_positions = self._apply_steering_limits(requested_steering, dt)
        self.robot.apply_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
        )
        self._publish_joint_state()
        self._publish_odometry()

    def _on_motion_mode(self, msg: UInt8):
        motion_mode = int(msg.data)
        if motion_mode not in {
            self.MOTION_MODE_DUAL_ACKERMAN,
            self.MOTION_MODE_PARALLEL,
            self.MOTION_MODE_SPINNING,
            self.MOTION_MODE_SIDE_SLIP,
        }:
            return
        self._latest_motion_mode = motion_mode
        self._has_motion_mode = True

    def _on_driver_command(self, msg: Twist):
        if not self._has_motion_mode:
            return

        self._command = _BaseCommand(
            motion_mode=self._latest_motion_mode,
            linear_speed=float(msg.linear.x),
            lateral_speed=float(msg.linear.y),
            steering_angle=float(msg.angular.x),
            angular_speed=float(msg.angular.z),
            received_time_sec=self._now_sec(),
        )

    def _resolve_active_command(self, now_sec: float):
        if not self._has_motion_mode:
            return _BaseCommand(
                motion_mode=self.MOTION_MODE_DUAL_ACKERMAN,
                linear_speed=0.0,
                lateral_speed=0.0,
                steering_angle=0.0,
                angular_speed=0.0,
                received_time_sec=self._command.received_time_sec,
            )
        if now_sec - self._command.received_time_sec <= self._command_timeout:
            return self._command
        return _BaseCommand(
            motion_mode=self._command.motion_mode,
            linear_speed=0.0,
            lateral_speed=0.0,
            steering_angle=0.0,
            angular_speed=0.0,
            received_time_sec=self._command.received_time_sec,
        )

    def _map_ranger_command(self, command: _BaseCommand):
        steering_positions = np.zeros(4, dtype=np.float32)
        wheel_linear = np.zeros(4, dtype=np.float32)

        if command.motion_mode == self.MOTION_MODE_DUAL_ACKERMAN:
            steer = float(np.clip(command.steering_angle, -self._steering_limit, self._steering_limit))
            steering_positions = np.array([steer, steer, -steer, -steer], dtype=np.float32)
            wheel_linear.fill(command.linear_speed)
        elif command.motion_mode == self.MOTION_MODE_PARALLEL:
            steer = float(np.clip(command.steering_angle, -self._steering_limit, self._steering_limit))
            steering_positions.fill(steer)
            wheel_linear.fill(command.linear_speed)
        elif command.motion_mode == self.MOTION_MODE_SPINNING:
            spin_steer = float(np.clip(np.arctan2(self._wheel_base, self._track_width), -self._steering_limit, self._steering_limit))
            steering_positions = np.array([spin_steer, -spin_steer, -spin_steer, spin_steer], dtype=np.float32)
            lever_arm = 0.5 * float(np.hypot(self._wheel_base, self._track_width))
            tangential = command.angular_speed * lever_arm
            wheel_linear = np.array([-tangential, tangential, -tangential, tangential], dtype=np.float32)
        elif command.motion_mode == self.MOTION_MODE_SIDE_SLIP:
            side_steer = float(np.clip(np.pi * 0.5, -self._steering_limit, self._steering_limit))
            steering_positions.fill(side_steer)
            wheel_linear.fill(command.lateral_speed)

        wheel_velocities = np.clip(wheel_linear / self._wheel_radius, -self._wheel_velocity_limit, self._wheel_velocity_limit)
        return steering_positions, wheel_velocities.astype(np.float32)

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
        translation, orientation = self.robot.get_world_pose()
        linear_velocity = self.robot.get_linear_velocity()
        angular_velocity = self.robot.get_angular_velocity()

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
        odom_msg.twist.twist.linear.x = float(linear_velocity[0])
        odom_msg.twist.twist.linear.y = float(linear_velocity[1])
        odom_msg.twist.twist.linear.z = float(linear_velocity[2])
        odom_msg.twist.twist.angular.x = float(angular_velocity[0])
        odom_msg.twist.twist.angular.y = float(angular_velocity[1])
        odom_msg.twist.twist.angular.z = float(angular_velocity[2])

        self._odom_pub.publish(odom_msg)

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

    def _now_sec(self):
        return self.node.get_clock().now().nanoseconds * 1e-9
