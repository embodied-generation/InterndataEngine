"""Nav2 client integration for cmd_vel-driven mobile robots."""

from __future__ import annotations

import ctypes
import glob
import math
import os
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.parameter import Parameter
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool


def _preload_nav2_typesupport_libs():
    ros_lib_dir = "/opt/ros/humble/lib"
    if not os.path.isdir(ros_lib_dir):
        return
    for path in sorted(glob.glob(os.path.join(ros_lib_dir, "libnav2_msgs__*.so"))):
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


_preload_nav2_typesupport_libs()
from nav2_msgs.action import NavigateToPose  # type: ignore[import-not-found]
from nav2_msgs.srv import ClearEntireCostmap, LoadMap  # type: ignore[import-not-found]

class Nav2Navigator:
    """Manage Nav2 goals through a real Nav2 action server."""

    RESULT_STATUS_SUCCEEDED = 4
    RESULT_STATUS_CANCELED = 5
    RESULT_STATUS_ABORTED = 6

    def __init__(self, robot, base_cfg: dict, node_name: str = "nav2_navigator"):
        self.robot = robot
        self.base_cfg = base_cfg
        self.ros_cfg = self.base_cfg.get("ros", {})
        self.nav2_cfg = self.ros_cfg.get("nav2", {})
        if not isinstance(self.nav2_cfg, dict):
            raise TypeError("base_cfg['ros']['nav2'] must be a dict when present")
        if not bool(self.nav2_cfg.get("enabled", False)):
            raise ValueError("Nav2Navigator requires ros.nav2.enabled=true")

        self._owns_rclpy_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True

        self.node = Node(node_name)
        self.node.set_parameters([Parameter("use_sim_time", value=True)])

        self._action_client_available = True

        self._action_name = str(self.nav2_cfg.get("action_name", "/navigate_to_pose"))
        self._global_frame = str(self.nav2_cfg.get("global_frame", "map"))
        self._base_frame = str(self.nav2_cfg.get("robot_base_frame", self.ros_cfg.get("base_frame", "base_link")))
        self._send_goal_timeout_sec = float(self.nav2_cfg.get("send_goal_timeout_sec", 0.2))
        self._server_wait_timeout_sec = max(
            self._send_goal_timeout_sec,
            float(self.nav2_cfg.get("server_wait_timeout_sec", 10.0)),
        )
        self._auto_goal_from_topic = bool(self.nav2_cfg.get("auto_goal_from_topic", True))
        self._auto_subscribe_plan = bool(self.nav2_cfg.get("auto_subscribe_plan", True))

        self._goal_topic = str(self.nav2_cfg.get("goal_topic", "/move_base_simple/goal"))
        self._cancel_topic = str(self.nav2_cfg.get("cancel_topic", "/nav2/cancel"))
        self._path_topic = str(self.nav2_cfg.get("path_topic", "/plan"))
        self._path_topics = self._resolve_plan_topics()
        self._load_map_service = str(self.nav2_cfg.get("load_map_service", "/map_server/load_map"))
        self._clear_global_costmap_service = str(
            self.nav2_cfg.get("clear_global_costmap_service", "/global_costmap/clear_entirely_global_costmap")
        )
        self._clear_local_costmap_service = str(
            self.nav2_cfg.get("clear_local_costmap_service", "/local_costmap/clear_entirely_local_costmap")
        )
        self._cmd_vel_topic = str(self.ros_cfg.get("cmd_vel_topic", "/cmd_vel"))
        self._odom_topic = str(self.ros_cfg.get("odom_topic", "/odom"))

        self._goal_pub = self.node.create_publisher(PoseStamped, self._goal_topic, 10)
        self._cmd_vel_pub = self.node.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._cancel_sub = self.node.create_subscription(Bool, self._cancel_topic, self._on_cancel, 10)
        self._odom_sub = self.node.create_subscription(Odometry, self._odom_topic, self._on_odom, 10)
        self._goal_sub = None
        if self._auto_goal_from_topic:
            self._goal_sub = self.node.create_subscription(PoseStamped, self._goal_topic, self._on_goal, 10)
        self._plan_subscriptions = []
        if self._auto_subscribe_plan:
            for topic in self._path_topics:
                self._plan_subscriptions.append(
                    self.node.create_subscription(
                        Path,
                        topic,
                        lambda msg, topic=topic: self._on_plan(msg, topic),
                        10,
                    )
                )

        self._action_client = ActionClient(self.node, NavigateToPose, self._action_name)
        self._load_map_client = self.node.create_client(LoadMap, self._load_map_service)
        self._clear_global_costmap_client = self.node.create_client(
            ClearEntireCostmap,
            self._clear_global_costmap_service,
        )
        self._clear_local_costmap_client = self.node.create_client(
            ClearEntireCostmap,
            self._clear_local_costmap_service,
        )

        self._pending_goal_pose: Optional[PoseStamped] = None
        self._pending_goal_created_time_sec = -1e9
        self._goal_handle = None
        self._goal_future = None
        self._result_future = None
        self._result_status = None
        self._goal_request_id = 0
        self._sim_time_sec = 0.0
        self._last_wall_time_sec = time.monotonic()

        self._latest_odom_xy = None
        self._latest_odom_yaw = None
        self._latest_odom_z = 0.0
        self._latest_odom_orientation = None

        self._local_goal_echo_filter_sec = float(self.nav2_cfg.get("local_goal_echo_filter_sec", 0.5))
        self._last_local_goal_signature = None
        self._last_local_goal_time_sec = -1e9
        self._latest_plan_xy: list[tuple[float, float]] = []
        self._latest_plan_frame_id = ""
        self._latest_plan_source_topic = ""
        self._latest_plan_received_time_sec = -1e9

    def destroy(self):
        self.cancel_active_goal()
        self.node.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()

    def step(self, step_dt: float | None = None):
        if step_dt is None:
            wall_now = time.monotonic()
            self._sim_time_sec += max(wall_now - self._last_wall_time_sec, 0.0)
            self._last_wall_time_sec = wall_now
        else:
            self._sim_time_sec += max(float(step_dt), 0.0)
            self._last_wall_time_sec = time.monotonic()

        rclpy.spin_once(self.node, timeout_sec=0.001)

        if self._pending_goal_pose is not None:
            self._drive_pending_goal()

    @property
    def has_active_goal(self) -> bool:
        if not self._action_client_available:
            return False
        if self._goal_handle is None:
            return False
        if self._result_future is None:
            return True
        return not self._result_future.done()

    @property
    def action_client_available(self) -> bool:
        return bool(self._action_client_available)

    @property
    def latest_result_status(self):
        return self._result_status

    @property
    def action_server_ready(self) -> bool:
        return bool(self._action_client is not None and self._action_client.server_is_ready())

    @property
    def nav_stack_ready(self) -> bool:
        return bool(
            self._probe_action_server()
            and self._probe_service(self._load_map_client)
            and self._probe_service(self._clear_global_costmap_client)
            and self._probe_service(self._clear_local_costmap_client)
        )

    @property
    def latest_plan_xy(self):
        return list(self._latest_plan_xy)

    @property
    def latest_plan_source_topic(self):
        return str(self._latest_plan_source_topic)

    def get_current_pose_xy_yaw(self):
        return self._current_pose_xy_yaw()

    def request_load_map(self, map_yaml_path: str):
        request = LoadMap.Request()
        request.map_url = str(map_yaml_path)
        if not self._probe_service(self._load_map_client):
            raise RuntimeError(f"service not ready: {self._load_map_service}")
        return self._load_map_client.call_async(request)

    def request_clear_all_costmaps(self):
        if not self._probe_service(self._clear_global_costmap_client):
            raise RuntimeError(f"service not ready: {self._clear_global_costmap_service}")
        if not self._probe_service(self._clear_local_costmap_client):
            raise RuntimeError(f"service not ready: {self._clear_local_costmap_service}")
        return (
            self._clear_global_costmap_client.call_async(ClearEntireCostmap.Request()),
            self._clear_local_costmap_client.call_async(ClearEntireCostmap.Request()),
        )

    def queue_goal(self, goal_pose: PoseStamped):
        self._pending_goal_pose = goal_pose
        self._pending_goal_created_time_sec = self._now_sec()
        self._result_status = None
        self._latest_plan_xy = []

    def send_goal(self, x: float, y: float, yaw: float, frame_id: Optional[str] = None):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = frame_id or self._global_frame
        goal_pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        goal_pose.pose.position.z = 0.0

        half_yaw = float(yaw) * 0.5
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = math.sin(half_yaw)
        goal_pose.pose.orientation.w = math.cos(half_yaw)
        self.queue_goal(goal_pose)

    def cancel_active_goal(self):
        self._pending_goal_pose = None
        if not self._action_client_available or self._goal_handle is None:
            self._result_status = self.RESULT_STATUS_CANCELED
            self._publish_stop_command()
            return
        self._goal_handle.cancel_goal_async()
        self._result_status = self.RESULT_STATUS_CANCELED
        self._publish_stop_command()

    def _drive_pending_goal(self):
        if self.has_active_goal:
            return

        if not self._action_client_available or self._action_client is None:
            self._result_status = "action_client_unavailable"
            self._pending_goal_pose = None
            self._publish_stop_command()
            return

        self._probe_action_server()

        if self._action_client.server_is_ready():
            goal = NavigateToPose.Goal()
            goal.pose = self._pending_goal_pose
            self._goal_request_id += 1
            request_id = int(self._goal_request_id)
            self._goal_future = self._action_client.send_goal_async(goal, feedback_callback=self._on_goal_feedback)
            self._goal_future.add_done_callback(
                lambda future, request_id=request_id: self._on_goal_response(future, request_id)
            )
            self._result_status = None
            self._pending_goal_pose = None
            return

        pending_age = self._now_sec() - self._pending_goal_created_time_sec
        if pending_age >= self._server_wait_timeout_sec:
            self._result_status = "action_server_unavailable"
            self._pending_goal_pose = None
            self._publish_stop_command()

    def _publish_goal_pose(self, goal_pose: PoseStamped):
        goal_pose.header.stamp = self.node.get_clock().now().to_msg()
        self._goal_pub.publish(goal_pose)
        self._remember_local_goal(goal_pose)

    def _publish_stop_command(self):
        self._cmd_vel_pub.publish(Twist())

    def _on_goal(self, msg: PoseStamped):
        if self._should_ignore_local_goal_echo(msg):
            return
        self.queue_goal(msg)

    def _on_cancel(self, msg: Bool):
        if bool(msg.data):
            self.cancel_active_goal()

    def _on_odom(self, msg: Odometry):
        self._latest_odom_xy = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))
        self._latest_odom_yaw = self._yaw_from_xyzw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self._latest_odom_z = float(msg.pose.pose.position.z)
        self._latest_odom_orientation = (
            float(msg.pose.pose.orientation.w),
            float(msg.pose.pose.orientation.x),
            float(msg.pose.pose.orientation.y),
            float(msg.pose.pose.orientation.z),
        )

    def _on_plan(self, msg: Path, source_topic: str = ""):
        self._latest_plan_frame_id = str(msg.header.frame_id)
        self._latest_plan_source_topic = str(source_topic or self._path_topic)
        self._latest_plan_xy = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in msg.poses
        ]
        self._latest_plan_received_time_sec = self._now_sec()

    def _resolve_plan_topics(self) -> list[str]:
        configured_topics = self.nav2_cfg.get("path_topics")
        candidates: list[str] = []
        if isinstance(configured_topics, list):
            candidates.extend(str(topic).strip() for topic in configured_topics if str(topic).strip())

        path_topic = str(self.nav2_cfg.get("path_topic", self._path_topic)).strip()
        if path_topic:
            candidates.append(path_topic)

        candidates.extend(
            [
                "/plan",
                "/planner_server/plan",
                "/unsmoothed_plan",
                "/smoothed_plan",
                "/plan_smoothed",
                "/received_global_plan",
            ]
        )

        deduped_topics = []
        for topic in candidates:
            if topic and topic not in deduped_topics:
                deduped_topics.append(topic)
        return deduped_topics

    def _on_goal_response(self, future, request_id: int):
        if int(request_id) != int(self._goal_request_id):
            return
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._goal_handle = None
            self._result_future = None
            self._result_status = "rejected"
            return

        self._goal_handle = goal_handle
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(
            lambda result_future, request_id=request_id: self._on_goal_result(result_future, request_id)
        )

    def _on_goal_result(self, future, request_id: int):
        if int(request_id) != int(self._goal_request_id):
            return
        result = future.result()
        self._result_status = int(result.status)
        self._goal_handle = None
        self._result_future = None

    def _on_goal_feedback(self, feedback_msg):
        del feedback_msg

    def _current_pose_xy_yaw(self):
        if self._latest_odom_xy is not None and self._latest_odom_yaw is not None:
            return self._latest_odom_xy[0], self._latest_odom_xy[1], self._latest_odom_yaw

        translation, orientation = self._get_robot_base_pose()
        return float(translation[0]), float(translation[1]), self._yaw_from_wxyz(orientation)

    def _get_robot_base_pose(self):
        getter = getattr(self.robot, "get_mobile_base_pose", None)
        if callable(getter):
            return getter()
        return self.robot.get_world_pose()

    def _pose_to_xy_yaw(self, pose: PoseStamped):
        return (
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            self._yaw_from_xyzw(
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ),
        )

    def _goal_signature(self, goal_pose: PoseStamped):
        return (
            str(goal_pose.header.frame_id),
            round(float(goal_pose.pose.position.x), 4),
            round(float(goal_pose.pose.position.y), 4),
            round(float(goal_pose.pose.position.z), 4),
            round(float(goal_pose.pose.orientation.x), 4),
            round(float(goal_pose.pose.orientation.y), 4),
            round(float(goal_pose.pose.orientation.z), 4),
            round(float(goal_pose.pose.orientation.w), 4),
        )

    def _remember_local_goal(self, goal_pose: PoseStamped):
        self._last_local_goal_signature = self._goal_signature(goal_pose)
        self._last_local_goal_time_sec = self._now_sec()

    def _should_ignore_local_goal_echo(self, goal_pose: PoseStamped):
        if self._last_local_goal_signature is None:
            return False
        if self._goal_signature(goal_pose) != self._last_local_goal_signature:
            return False
        return (self._now_sec() - self._last_local_goal_time_sec) <= self._local_goal_echo_filter_sec

    def _now_sec(self):
        return self._sim_time_sec

    def _probe_action_server(self) -> bool:
        if self._action_client is None:
            return False
        if not self._action_client.server_is_ready():
            self._action_client.wait_for_server(timeout_sec=self._send_goal_timeout_sec)
        return bool(self._action_client.server_is_ready())

    @staticmethod
    def _probe_service(client) -> bool:
        if client is None:
            return False
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=0.01)
        return bool(client.service_is_ready())

    @staticmethod
    def _wrap_angle(angle: float):
        return math.atan2(math.sin(angle), math.cos(angle))

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
