"""Nav2 client integration for cmd_vel-driven mobile robots."""

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool


class Nav2Navigator:
    """Manage Nav2 action goals and initial pose for a mobile base."""

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

        self._action_client = None
        self._navigate_action_type = None
        self._action_client_available = False

        # Import Nav2 action types lazily. Fallback to goal-topic mode when unavailable.
        try:
            from nav2_msgs.action import NavigateToPose  # pylint: disable=import-outside-toplevel  # type: ignore[import-not-found]

            self._navigate_action_type = NavigateToPose
            self._action_client_available = True
        except Exception:
            self._action_client_available = False

        self._action_name = str(self.nav2_cfg.get("action_name", "/navigate_to_pose"))
        self._global_frame = str(self.nav2_cfg.get("global_frame", "map"))
        self._base_frame = str(self.nav2_cfg.get("robot_base_frame", self.ros_cfg.get("base_frame", "base_link")))
        self._send_goal_timeout_sec = float(self.nav2_cfg.get("send_goal_timeout_sec", 0.2))
        self._auto_goal_from_topic = bool(self.nav2_cfg.get("auto_goal_from_topic", True))
        self._auto_initial_pose = bool(self.nav2_cfg.get("auto_initial_pose", True))

        self._initial_pose_topic = str(self.nav2_cfg.get("initialpose_topic", "/initialpose"))
        self._goal_topic = str(self.nav2_cfg.get("goal_topic", "/move_base_simple/goal"))
        self._cancel_topic = str(self.nav2_cfg.get("cancel_topic", "/nav2/cancel"))

        self._initial_pose_pub = self.node.create_publisher(PoseWithCovarianceStamped, self._initial_pose_topic, 10)
        self._goal_pub = self.node.create_publisher(PoseStamped, self._goal_topic, 10)
        self._cancel_sub = self.node.create_subscription(Bool, self._cancel_topic, self._on_cancel, 10)
        self._goal_sub = None
        if self._auto_goal_from_topic:
            self._goal_sub = self.node.create_subscription(PoseStamped, self._goal_topic, self._on_goal, 10)

        if self._action_client_available:
            self._action_client = ActionClient(self.node, self._navigate_action_type, self._action_name)

        self._pending_goal_pose: Optional[PoseStamped] = None
        self._goal_handle = None
        self._goal_future = None
        self._result_future = None
        self._result_status = None

        if self._auto_initial_pose:
            self.publish_initial_pose_from_robot()

    def destroy(self):
        try:
            self.cancel_active_goal()
        except Exception:
            pass

        self.node.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()

    def step(self):
        # Keep callbacks responsive and drive pending goal submission.
        rclpy.spin_once(self.node, timeout_sec=0.001)

        if self._pending_goal_pose is None:
            return

        if not self._action_client_available or self._action_client is None:
            self._goal_pub.publish(self._pending_goal_pose)
            self._result_status = "goal_topic_published"
            self._pending_goal_pose = None
            return

        if self.has_active_goal:
            return

        if not self._action_client.server_is_ready():
            self._action_client.wait_for_server(timeout_sec=self._send_goal_timeout_sec)
            if not self._action_client.server_is_ready():
                return

        goal = self._navigate_action_type.Goal()
        goal.pose = self._pending_goal_pose
        self._goal_future = self._action_client.send_goal_async(goal)
        self._goal_future.add_done_callback(self._on_goal_response)
        self._pending_goal_pose = None

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

    def queue_goal(self, goal_pose: PoseStamped):
        self._pending_goal_pose = goal_pose

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

    def publish_initial_pose_from_robot(self):
        translation, orientation = self.robot.get_world_pose()

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._global_frame
        msg.pose.pose.position.x = float(translation[0])
        msg.pose.pose.position.y = float(translation[1])
        msg.pose.pose.position.z = float(translation[2])
        msg.pose.pose.orientation.x = float(orientation[1])
        msg.pose.pose.orientation.y = float(orientation[2])
        msg.pose.pose.orientation.z = float(orientation[3])
        msg.pose.pose.orientation.w = float(orientation[0])

        covariance = self.nav2_cfg.get(
            "initial_pose_covariance",
            [
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.06853891945200942,
            ],
        )
        if len(covariance) != 36:
            raise ValueError("ros.nav2.initial_pose_covariance must contain 36 values")
        msg.pose.covariance = [float(value) for value in covariance]

        self._initial_pose_pub.publish(msg)

    def cancel_active_goal(self):
        if not self._action_client_available:
            return
        if self._goal_handle is None:
            return
        try:
            self._goal_handle.cancel_goal_async()
        except Exception:
            pass

    def _on_goal(self, msg: PoseStamped):
        self.queue_goal(msg)

    def _on_cancel(self, msg: Bool):
        if bool(msg.data):
            self.cancel_active_goal()

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._goal_handle = None
            self._result_future = None
            self._result_status = "rejected"
            return

        self._goal_handle = goal_handle
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        try:
            result = future.result()
        except Exception:
            self._result_status = "error"
            self._goal_handle = None
            self._result_future = None
            return

        self._result_status = int(result.status)
        self._goal_handle = None
        self._result_future = None
