"""Publish synthetic 2D LaserScan from Isaac Sim world ground-truth geometry."""

import math

import omni.physx as physx  # type: ignore[import-not-found]
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class IsaacWorldScanPublisher:
    """Generate planar LaserScan via PhysX raycasts and publish to ROS."""

    def __init__(self, robot, base_cfg: dict, node_name: str = "isaac_world_scan_publisher"):
        self.robot = robot
        self.base_cfg = base_cfg
        self.ros_cfg = self.base_cfg.get("ros", {})
        self.scan_cfg = self.ros_cfg.get("world_scan", {})
        if not isinstance(self.scan_cfg, dict):
            raise TypeError("base_cfg['ros']['world_scan'] must be a dict when present")
        if not bool(self.scan_cfg.get("enabled", False)):
            raise ValueError("IsaacWorldScanPublisher requires ros.world_scan.enabled=true")
        if "scan_topic" not in self.ros_cfg:
            raise KeyError("Missing ROS scan_topic in base_cfg['ros']")

        self._owns_rclpy_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True

        self.node = Node(node_name)
        self._scan_pub = self.node.create_publisher(LaserScan, self.ros_cfg["scan_topic"], 10)

        self._scene_query = physx.get_physx_scene_query_interface()

        self._scan_frame = str(self.scan_cfg.get("scan_frame", self.ros_cfg.get("base_frame", "base_link")))
        self._num_beams = max(8, int(self.scan_cfg.get("num_beams", 360)))
        self._angle_min = float(self.scan_cfg.get("angle_min", -math.pi))
        self._angle_max = float(self.scan_cfg.get("angle_max", math.pi))
        self._range_min = max(0.01, float(self.scan_cfg.get("range_min", 0.05)))
        self._range_max = max(self._range_min + 1e-3, float(self.scan_cfg.get("range_max", 12.0)))
        self._publish_rate_hz = max(0.5, float(self.scan_cfg.get("publish_rate_hz", 10.0)))
        self._both_sides = bool(self.scan_cfg.get("both_sides", False))

        sensor_offset = self.scan_cfg.get("sensor_offset", [0.0, 0.0, 0.25])
        if len(sensor_offset) != 3:
            raise ValueError("ros.world_scan.sensor_offset must have exactly 3 values")
        self._sensor_offset = [float(sensor_offset[0]), float(sensor_offset[1]), float(sensor_offset[2])]

        ignore_collisions = self.scan_cfg.get("ignore_collision_prefixes", [])
        if not isinstance(ignore_collisions, list):
            raise TypeError("ros.world_scan.ignore_collision_prefixes must be a list")
        self._ignore_collision_prefixes = [str(item) for item in ignore_collisions]

        self._ignore_robot_self = bool(self.scan_cfg.get("ignore_robot_self", True))
        self._robot_prim_path = str(getattr(self.robot, "robot_prim_path", ""))

        self._scan_period = 1.0 / self._publish_rate_hz
        self._last_publish_time_sec = -1e9

    def destroy(self):
        self.node.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()

    def step(self):
        rclpy.spin_once(self.node, timeout_sec=0.0)
        now_sec = self._now_sec()
        if now_sec - self._last_publish_time_sec < self._scan_period:
            return

        scan_msg = self._build_scan_message(now_sec)
        self._scan_pub.publish(scan_msg)
        self._last_publish_time_sec = now_sec

    def _build_scan_message(self, now_sec: float):
        translation, orientation_wxyz = self.robot.get_world_pose()
        yaw = self._yaw_from_wxyz(orientation_wxyz)

        ox, oy, oz = self._sensor_offset
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        origin = (
            float(translation[0] + ox * cos_yaw - oy * sin_yaw),
            float(translation[1] + ox * sin_yaw + oy * cos_yaw),
            float(translation[2] + oz),
        )

        angle_increment = (self._angle_max - self._angle_min) / float(self._num_beams - 1)
        ranges = []

        for beam_idx in range(self._num_beams):
            local_angle = self._angle_min + beam_idx * angle_increment
            world_angle = yaw + local_angle
            direction = (math.cos(world_angle), math.sin(world_angle), 0.0)

            hit = self._scene_query.raycast_closest(origin, direction, self._range_max, self._both_sides)
            distance = self._resolve_distance_from_hit(hit)
            ranges.append(distance)

        msg = LaserScan()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._scan_frame
        msg.angle_min = float(self._angle_min)
        msg.angle_max = float(self._angle_max)
        msg.angle_increment = float(angle_increment)
        msg.time_increment = float(self._scan_period / max(self._num_beams, 1))
        msg.scan_time = float(self._scan_period)
        msg.range_min = float(self._range_min)
        msg.range_max = float(self._range_max)
        msg.ranges = ranges
        return msg

    def _resolve_distance_from_hit(self, hit: dict) -> float:
        if not isinstance(hit, dict):
            return float(self._range_max)
        if not bool(hit.get("hit", False)):
            return float(self._range_max)

        collision_path = str(hit.get("collision", ""))
        rigid_body_path = str(hit.get("rigidBody", ""))

        if self._should_ignore_hit(collision_path, rigid_body_path):
            return float(self._range_max)

        distance = float(hit.get("distance", self._range_max))
        if not math.isfinite(distance):
            return float(self._range_max)
        return float(min(max(distance, self._range_min), self._range_max))

    def _should_ignore_hit(self, collision_path: str, rigid_body_path: str) -> bool:
        for prefix in self._ignore_collision_prefixes:
            if collision_path.startswith(prefix) or rigid_body_path.startswith(prefix):
                return True

        if self._ignore_robot_self and self._robot_prim_path:
            if collision_path.startswith(self._robot_prim_path) or rigid_body_path.startswith(self._robot_prim_path):
                return True

        return False

    def _now_sec(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _yaw_from_wxyz(q_wxyz):
        # Isaac orientation order is [w, x, y, z].
        w = float(q_wxyz[0])
        x = float(q_wxyz[1])
        y = float(q_wxyz[2])
        z = float(q_wxyz[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
