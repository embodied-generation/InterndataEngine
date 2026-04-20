"""Isaac-side Nav2 bridge client that only uses standard ROS message types."""

from __future__ import annotations

import importlib
import json
import logging
import math
import time
from typing import Any, Optional

from .isaac_ros_clock import ensure_isaac_ros2_bridge_ready


LOGGER = logging.getLogger("simbox.nav2_bridge_client")


def _load_ros_bridge_modules():
    ros2_imports = ensure_isaac_ros2_bridge_ready()
    rclpy = ros2_imports["rclpy"]
    node_module = importlib.import_module("rclpy.node")
    parameter_module = importlib.import_module("rclpy.parameter")
    qos_module = importlib.import_module("rclpy.qos")
    std_msgs_module = importlib.import_module("std_msgs.msg")
    geometry_msgs_module = importlib.import_module("geometry_msgs.msg")
    nav_msgs_module = importlib.import_module("nav_msgs.msg")
    return {
        "rclpy": rclpy,
        "Node": node_module.Node,
        "Parameter": parameter_module.Parameter,
        "QoSProfile": qos_module.QoSProfile,
        "QoSReliabilityPolicy": qos_module.QoSReliabilityPolicy,
        "QoSDurabilityPolicy": qos_module.QoSDurabilityPolicy,
        "String": std_msgs_module.String,
        "PoseStamped": geometry_msgs_module.PoseStamped,
        "Odometry": nav_msgs_module.Odometry,
    }


class Nav2BridgeClient:
    """Publish high-level navigation intents over standard ROS topics."""

    def __init__(self, robot, base_cfg: dict, node_name: str = "nav2_bridge_client"):
        self.robot = robot
        self.base_cfg = base_cfg
        self.ros_cfg = self.base_cfg.get("ros", {})
        self.nav2_cfg = self.ros_cfg.get("nav2", {})
        if not isinstance(self.nav2_cfg, dict):
            raise TypeError("base_cfg['ros']['nav2'] must be a dict when present")
        if not bool(self.nav2_cfg.get("enabled", False)):
            raise ValueError("Nav2BridgeClient requires ros.nav2.enabled=true")

        ros_modules = _load_ros_bridge_modules()
        self._rclpy = ros_modules["rclpy"]
        self._String = ros_modules["String"]
        self._PoseStamped = ros_modules["PoseStamped"]
        self._Odometry = ros_modules["Odometry"]
        control_qos = ros_modules["QoSProfile"](
            depth=10,
            reliability=ros_modules["QoSReliabilityPolicy"].RELIABLE,
            durability=ros_modules["QoSDurabilityPolicy"].TRANSIENT_LOCAL,
        )
        odom_qos = ros_modules["QoSProfile"](
            depth=10,
            reliability=ros_modules["QoSReliabilityPolicy"].RELIABLE,
            durability=ros_modules["QoSDurabilityPolicy"].VOLATILE,
        )

        self._owns_rclpy_context = False
        if not self._rclpy.ok():
            self._rclpy.init(args=None)
            self._owns_rclpy_context = True

        self.node = ros_modules["Node"](node_name)
        self.node.set_parameters([ros_modules["Parameter"]("use_sim_time", value=True)])

        self._robot_name = str(getattr(robot, "name", "robot"))
        self._global_frame = str(self.nav2_cfg.get("global_frame", "map"))
        self._odom_topic = str(self.ros_cfg.get("odom_topic", "/odom"))
        self._map_update_topic = str(self.nav2_cfg.get("bridge_map_update_topic", "/simbox/nav_bridge/map_update"))
        self._goal_topic = str(self.nav2_cfg.get("bridge_goal_topic", "/simbox/nav_bridge/goal"))
        self._cancel_topic = str(self.nav2_cfg.get("bridge_cancel_topic", "/simbox/nav_bridge/cancel"))
        self._status_topic = str(self.nav2_cfg.get("bridge_status_topic", "/simbox/nav_bridge/status"))
        self._result_topic = str(self.nav2_cfg.get("bridge_result_topic", "/simbox/nav_bridge/result"))
        self._bridge_alive_timeout_sec = float(self.nav2_cfg.get("bridge_alive_timeout_sec", 3.0))

        self._map_update_pub = self.node.create_publisher(self._String, self._map_update_topic, control_qos)
        self._goal_pub = self.node.create_publisher(self._String, self._goal_topic, control_qos)
        self._cancel_pub = self.node.create_publisher(self._String, self._cancel_topic, control_qos)
        self._status_sub = self.node.create_subscription(self._String, self._status_topic, self._on_status, control_qos)
        self._result_sub = self.node.create_subscription(self._String, self._result_topic, self._on_result, control_qos)
        self._odom_sub = self.node.create_subscription(self._Odometry, self._odom_topic, self._on_odom, odom_qos)

        self._sim_time_sec = 0.0
        self._last_wall_time_sec = time.monotonic()
        self._latest_status: dict[str, Any] = {}
        self._latest_result: dict[str, Any] = {}
        self._last_status_wall_time_sec = -1e9
        self._last_result_wall_time_sec = -1e9
        self._latest_odom_xy = None
        self._latest_odom_yaw = None
        self._odom_trace: list[dict[str, Any]] = []
        self._last_trace_append_wall_time_sec = -1e9

    def destroy(self):
        self.node.destroy_node()
        if self._owns_rclpy_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def reset_debug_trace(self):
        self._odom_trace = []
        self._last_trace_append_wall_time_sec = -1e9

    def step(self, step_dt: float | None = None):
        if step_dt is None:
            wall_now = time.monotonic()
            self._sim_time_sec += max(wall_now - self._last_wall_time_sec, 0.0)
            self._last_wall_time_sec = wall_now
        else:
            self._sim_time_sec += max(float(step_dt), 0.0)
            self._last_wall_time_sec = time.monotonic()
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    @property
    def bridge_online(self) -> bool:
        return (time.monotonic() - self._last_status_wall_time_sec) <= self._bridge_alive_timeout_sec

    @property
    def nav_stack_ready(self) -> bool:
        return bool(self._latest_status.get("stack_ready", False))

    @property
    def latest_status(self) -> dict[str, Any]:
        return dict(self._latest_status)

    @property
    def latest_result(self) -> dict[str, Any]:
        return dict(self._latest_result)

    @property
    def odom_trace(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._odom_trace]

    @property
    def latest_status_state(self) -> str:
        return str(self._latest_status.get("state", "")).strip().lower()

    def publish_map_update(self, *, request_id: str, stack_id: str, map_yaml_path: str, scene_name: str):
        payload = {
            "robot_name": self._robot_name,
            "request_id": str(request_id),
            "stack_id": str(stack_id),
            "scene_name": str(scene_name),
            "map_yaml_path": str(map_yaml_path),
            "sent_at": time.time(),
        }
        LOGGER.info(
            "publish map_update robot=%s request_id=%s stack_id=%s map=%s",
            self._robot_name,
            request_id,
            stack_id,
            map_yaml_path,
        )
        self._publish_json(self._map_update_pub, payload)

    def publish_goal(self, *, request_id: str, goal_x: float, goal_y: float, goal_yaw: float):
        payload = {
            "robot_name": self._robot_name,
            "request_id": str(request_id),
            "frame_id": self._global_frame,
            "goal": {
                "x": float(goal_x),
                "y": float(goal_y),
                "yaw": float(goal_yaw),
            },
            "sent_at": time.time(),
        }
        LOGGER.info(
            "publish goal robot=%s request_id=%s goal=(%.3f, %.3f, %.3f)",
            self._robot_name,
            request_id,
            goal_x,
            goal_y,
            goal_yaw,
        )
        self._publish_json(self._goal_pub, payload)

    def cancel_request(self, request_id: str = ""):
        payload = {
            "robot_name": self._robot_name,
            "request_id": str(request_id),
            "cancel_requested": True,
            "sent_at": time.time(),
        }
        LOGGER.info("publish cancel robot=%s request_id=%s", self._robot_name, request_id)
        self._publish_json(self._cancel_pub, payload)

    def request_status(self, request_id: str) -> dict[str, Any]:
        payload = self._latest_status
        if str(payload.get("request_id", "")).strip() != str(request_id).strip():
            return {}
        if str(payload.get("robot_name", self._robot_name)).strip() != self._robot_name:
            return {}
        return dict(payload)

    def request_result(self, request_id: str) -> dict[str, Any]:
        payload = self._latest_result
        if str(payload.get("request_id", "")).strip() != str(request_id).strip():
            return {}
        if str(payload.get("robot_name", self._robot_name)).strip() != self._robot_name:
            return {}
        return dict(payload)

    def get_current_pose_xy_yaw(self):
        if self._latest_odom_xy is not None and self._latest_odom_yaw is not None:
            return self._latest_odom_xy[0], self._latest_odom_xy[1], self._latest_odom_yaw
        translation, orientation = self._get_robot_base_pose()
        return float(translation[0]), float(translation[1]), self._yaw_from_wxyz(orientation)

    def _publish_json(self, publisher, payload: dict[str, Any]):
        msg = self._String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def _on_status(self, msg):
        payload = self._parse_json_message(msg.data)
        if str(payload.get("robot_name", self._robot_name)).strip() != self._robot_name:
            return
        self._latest_status = payload
        self._last_status_wall_time_sec = time.monotonic()
        state = str(payload.get("state", "")).strip()
        request_id = str(payload.get("request_id", "")).strip()
        if state:
            LOGGER.info("status update robot=%s request_id=%s state=%s", self._robot_name, request_id, state)

    def _on_result(self, msg):
        payload = self._parse_json_message(msg.data)
        if str(payload.get("robot_name", self._robot_name)).strip() != self._robot_name:
            return
        self._latest_result = payload
        self._last_result_wall_time_sec = time.monotonic()
        state = str(payload.get("state", "")).strip()
        request_id = str(payload.get("request_id", "")).strip()
        if state:
            LOGGER.info("result update robot=%s request_id=%s state=%s", self._robot_name, request_id, state)

    def _on_odom(self, msg):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = self._yaw_from_xyzw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self._latest_odom_xy = (x, y)
        self._latest_odom_yaw = yaw
        self._append_odom_trace(x=x, y=y, yaw=yaw)

    def _append_odom_trace(self, *, x: float, y: float, yaw: float):
        wall_now = time.monotonic()
        if not self._odom_trace:
            should_append = True
        else:
            last = self._odom_trace[-1]
            should_append = (
                math.hypot(float(last["x"]) - x, float(last["y"]) - y) >= 0.02
                or abs(self._angle_diff_rad(float(last["yaw"]), yaw)) >= 0.05
                or (wall_now - self._last_trace_append_wall_time_sec) >= 0.25
            )
        if not should_append:
            return
        self._odom_trace.append(
            {
                "x": float(x),
                "y": float(y),
                "yaw": float(yaw),
                "wall_time_sec": float(time.time()),
                "sim_time_sec": float(self._sim_time_sec),
            }
        )
        self._last_trace_append_wall_time_sec = wall_now

    def _get_robot_base_pose(self):
        getter = getattr(self.robot, "get_mobile_base_pose", None)
        if callable(getter):
            return getter()
        return self.robot.get_world_pose()

    @staticmethod
    def _parse_json_message(payload: str) -> dict[str, Any]:
        try:
            parsed = json.loads(str(payload))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _yaw_from_xyzw(x: float, y: float, z: float, w: float):
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _yaw_from_wxyz(q_wxyz):
        w = float(q_wxyz[0])
        x = float(q_wxyz[1])
        y = float(q_wxyz[2])
        z = float(q_wxyz[3])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _angle_diff_rad(current: float, target: float) -> float:
        return math.atan2(math.sin(float(target) - float(current)), math.cos(float(target) - float(current)))
