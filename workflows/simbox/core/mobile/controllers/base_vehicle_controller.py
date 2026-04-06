"""Base class for cmd_vel-driven mobile vehicle controllers."""

from abc import ABC, abstractmethod

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class BaseVehicleController(ABC):
    """Shared ROS cmd_vel lifecycle for all mobile vehicle controllers."""

    def __init__(self, base_cfg: dict, node_name: str = "base_vehicle_controller"):
        self.base_cfg = base_cfg
        self.ros_cfg = self.base_cfg.get("ros", {})
        if not isinstance(self.ros_cfg, dict):
            raise TypeError("base_cfg['ros'] must be a dict")

        required_fields = {"cmd_vel_topic"}
        required_fields.update(self.required_ros_fields)
        missing_fields = [field for field in sorted(required_fields) if field not in self.ros_cfg]
        if missing_fields:
            raise KeyError(f"Missing ROS vehicle controller config fields: {missing_fields}")

        self._owns_rclpy_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True

        self.node = Node(node_name)
        self._setup_publishers()
        self._cmd_vel_sub = self.node.create_subscription(
            Twist,
            self.ros_cfg["cmd_vel_topic"],
            self._on_cmd_vel,
            10,
        )

    @property
    def required_ros_fields(self) -> tuple[str, ...]:
        """Additional ROS fields required by a concrete controller."""
        return ()

    def destroy(self):
        self.node.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()

    def step(self):
        # A tiny timeout avoids starving DDS callback processing in tight simulation loops.
        rclpy.spin_once(self.node, timeout_sec=0.001)

    @abstractmethod
    def _setup_publishers(self):
        """Create publishers required by this controller."""

    @abstractmethod
    def _handle_cmd_vel(self, msg: Twist):
        """Convert/publish one cmd_vel message."""

    def _on_cmd_vel(self, msg: Twist):
        self._handle_cmd_vel(msg)
