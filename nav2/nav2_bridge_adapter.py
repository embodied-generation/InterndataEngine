#!/usr/bin/env python3
"""Nav2-side adapter that bridges standard ROS topics to Nav2 actions/services."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from typing import Any, Optional

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import ClearEntireCostmap, LoadMap
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


LOGGER = logging.getLogger("simbox.nav2_bridge_adapter")


class Nav2BridgeAdapter:
    def __init__(
        self,
        *,
        robot_name: str,
        map_update_topic: str,
        goal_topic: str,
        cancel_topic: str,
        status_topic: str,
        result_topic: str,
        odom_topic: str,
        action_name: str,
        planner_action_name: str,
        load_map_service: str,
        clear_global_costmap_service: str,
        clear_local_costmap_service: str,
        heartbeat_sec: float,
    ):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("interndata_nav2_bridge_adapter")
        self.node.set_parameters([Parameter("use_sim_time", value=True)])

        self.robot_name = str(robot_name)
        self.map_update_topic = str(map_update_topic)
        self.goal_topic = str(goal_topic)
        self.cancel_topic = str(cancel_topic)
        self.status_topic = str(status_topic)
        self.result_topic = str(result_topic)
        self.odom_topic = str(odom_topic)
        self.action_name = str(action_name)
        self.planner_action_name = str(planner_action_name)
        self.load_map_service = str(load_map_service)
        self.clear_global_costmap_service = str(clear_global_costmap_service)
        self.clear_local_costmap_service = str(clear_local_costmap_service)

        control_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        odom_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        tf_qos = QoSProfile(
            depth=100,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._status_pub = self.node.create_publisher(String, self.status_topic, control_qos)
        self._result_pub = self.node.create_publisher(String, self.result_topic, control_qos)
        self._tf_pub = self.node.create_publisher(TFMessage, "/tf", tf_qos)
        self.node.create_subscription(String, self.map_update_topic, self._on_map_update, control_qos)
        self.node.create_subscription(String, self.goal_topic, self._on_goal, control_qos)
        self.node.create_subscription(String, self.cancel_topic, self._on_cancel, control_qos)
        self.node.create_subscription(Odometry, self.odom_topic, self._on_odom, odom_qos)
        self.node.create_subscription(Costmap, "/global_costmap/costmap_raw", self._on_global_costmap, odom_qos)
        self.node.create_subscription(Costmap, "/local_costmap/costmap_raw", self._on_local_costmap, odom_qos)

        self._action_client = ActionClient(self.node, NavigateToPose, self.action_name)
        self._planner_action_client = ActionClient(self.node, ComputePathToPose, self.planner_action_name)
        self._load_map_client = self.node.create_client(LoadMap, self.load_map_service)
        self._clear_global_costmap_client = self.node.create_client(
            ClearEntireCostmap,
            self.clear_global_costmap_service,
        )
        self._clear_local_costmap_client = self.node.create_client(
            ClearEntireCostmap,
            self.clear_local_costmap_service,
        )

        self._state = "idle"
        self._detail = ""
        self._stack_id = ""
        self._request_id = ""
        self._request_generation = 0
        self._goal_handle = None
        self._result_future = None
        self._latest_odom_pose = {}
        self._latest_odom_frame_id = ""
        self._latest_base_frame_id = ""
        self._odom_tf_publish_count = 0
        self._last_status_publish_monotonic = -1e9
        self._heartbeat_sec = max(float(heartbeat_sec), 0.2)
        self._latest_planning_debug: dict[str, Any] = {}
        self._costmap_update_counts = {"global": 0, "local": 0}
        self._latest_costmap_debug: dict[str, dict[str, Any]] = {"global": {}, "local": {}}
        self._waiting_costmap_refresh = False
        self._costmap_refresh_request_id = ""
        self._costmap_refresh_generation = 0
        self._costmap_refresh_baseline = {"global": 0, "local": 0}
        self.node.create_timer(self._heartbeat_sec, self._publish_heartbeat)

    def run(self) -> int:
        try:
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.05)
        finally:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return 0

    def _on_map_update(self, msg: String):
        payload = self._parse_payload(msg.data)
        if not self._payload_matches_robot(payload):
            return

        request_id = str(payload.get("request_id", "")).strip()
        map_yaml_path = str(payload.get("map_yaml_path", "")).strip()
        if not request_id or not map_yaml_path:
            self._publish_status(
                state="failed",
                request_id=request_id,
                detail="map_update missing request_id or map_yaml_path",
            )
            return

        self._request_generation += 1
        generation = int(self._request_generation)
        self._request_id = request_id
        self._stack_id = str(payload.get("stack_id", "")).strip()
        self._detail = f"loading_map:{map_yaml_path}"
        self._state = "loading_map"
        self._latest_planning_debug = {}
        self._waiting_costmap_refresh = False
        self._costmap_refresh_request_id = ""
        self._costmap_refresh_generation = 0
        self._costmap_refresh_baseline = dict(self._costmap_update_counts)
        LOGGER.info("received map_update request_id=%s stack_id=%s map=%s", request_id, self._stack_id, map_yaml_path)
        self._publish_status(state=self._state, request_id=request_id, detail=self._detail)

        self._cancel_active_goal(request_id=request_id, detail="superseded_by_map_update", publish_terminal=False)

        if not self._wait_for_service(self._load_map_client, self.load_map_service):
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"service unavailable: {self.load_map_service}",
            )
            return

        request = LoadMap.Request()
        request.map_url = map_yaml_path
        future = self._load_map_client.call_async(request)
        future.add_done_callback(
            lambda future, request_id=request_id, generation=generation: self._on_load_map_done(
                future,
                request_id=request_id,
                generation=generation,
            )
        )

    def _on_goal(self, msg: String):
        payload = self._parse_payload(msg.data)
        if not self._payload_matches_robot(payload):
            return

        request_id = str(payload.get("request_id", "")).strip()
        if not request_id:
            self._publish_status(state="failed", request_id="", detail="goal missing request_id")
            return
        if request_id != self._request_id:
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"goal request_id mismatch: active={self._request_id}",
            )
            return
        if self._state not in {"ready", "accepted", "running"}:
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"goal rejected while adapter state={self._state}",
            )
            return

        LOGGER.info("received goal request_id=%s payload=%s", request_id, payload.get("goal", {}))
        goal = NavigateToPose.Goal()
        goal.pose = self._goal_pose_from_payload(payload)
        generation = int(self._request_generation)
        self._request_preflight_plan(goal.pose, request_id=request_id, generation=generation)

        if not self._wait_for_action_server():
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"action server unavailable: {self.action_name}",
            )
            return

        self._state = "goal_requested"
        self._detail = "waiting_for_goal_response"
        self._publish_status(state=self._state, request_id=request_id, detail=self._detail)
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(
            lambda future, request_id=request_id, generation=generation: self._on_goal_response(
                future,
                request_id=request_id,
                generation=generation,
            )
        )

    def _request_preflight_plan(self, goal_pose: PoseStamped, *, request_id: str, generation: int):
        planning_debug = {
            "source": "compute_path_to_pose",
            "action_name": self.planner_action_name,
            "requested_goal": self._pose_stamped_to_dict(goal_pose),
            "requested_at": time.time(),
            "state": "pending",
        }
        if not self._wait_for_planner_action_server():
            planning_debug["state"] = "planner_unavailable"
            self._latest_planning_debug = planning_debug
            return

        plan_goal = ComputePathToPose.Goal()
        plan_goal.goal = goal_pose
        plan_goal.planner_id = "GridBased"
        if self._latest_odom_pose:
            plan_goal.start = self._current_start_pose(goal_pose.header.frame_id)
            plan_goal.use_start = True
            planning_debug["requested_start"] = self._pose_stamped_to_dict(plan_goal.start)
        else:
            plan_goal.use_start = False
            planning_debug["requested_start"] = {}

        self._latest_planning_debug = planning_debug
        future = self._planner_action_client.send_goal_async(plan_goal)
        future.add_done_callback(
            lambda future, request_id=request_id, generation=generation: self._on_preflight_plan_goal_response(
                future,
                request_id=request_id,
                generation=generation,
            )
        )

    def _on_preflight_plan_goal_response(self, future, *, request_id: str, generation: int):
        if not self._is_current_request(request_id=request_id, generation=generation):
            return
        try:
            goal_handle = future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self._latest_planning_debug = {
                **dict(self._latest_planning_debug),
                "state": "send_goal_failed",
                "error": str(exc),
            }
            return
        if goal_handle is None or not goal_handle.accepted:
            self._latest_planning_debug = {
                **dict(self._latest_planning_debug),
                "state": "rejected",
            }
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future, request_id=request_id, generation=generation: self._on_preflight_plan_result(
                future,
                request_id=request_id,
                generation=generation,
            )
        )

    def _on_preflight_plan_result(self, future, *, request_id: str, generation: int):
        if not self._is_current_request(request_id=request_id, generation=generation):
            return
        try:
            result = future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self._latest_planning_debug = {
                **dict(self._latest_planning_debug),
                "state": "result_failed",
                "error": str(exc),
            }
            return

        status_code = int(getattr(result, "status", -1))
        wrapped_result = getattr(result, "result", None)
        planning_time = self._duration_to_sec(getattr(wrapped_result, "planning_time", None))
        path_msg = getattr(wrapped_result, "path", None)
        if status_code == 4 and path_msg is not None:
            self._latest_planning_debug = {
                **dict(self._latest_planning_debug),
                "state": "succeeded",
                "status_code": status_code,
                "planning_time_sec": planning_time,
                "path": self._path_to_dict(path_msg),
            }
        else:
            self._latest_planning_debug = {
                **dict(self._latest_planning_debug),
                "state": "failed",
                "status_code": status_code,
                "planning_time_sec": planning_time,
                "path": self._path_to_dict(path_msg) if path_msg is not None else {},
            }

    def _on_cancel(self, msg: String):
        payload = self._parse_payload(msg.data)
        if not self._payload_matches_robot(payload):
            return

        request_id = str(payload.get("request_id", "")).strip()
        if request_id and request_id != self._request_id:
            return
        LOGGER.info("received cancel request_id=%s active_request_id=%s", request_id, self._request_id)
        self._cancel_active_goal(
            request_id=self._request_id or request_id,
            detail="cancel_requested",
            publish_terminal=True,
        )

    def _on_odom(self, msg: Odometry):
        self._latest_odom_frame_id = str(msg.header.frame_id).strip()
        self._latest_base_frame_id = str(msg.child_frame_id).strip()
        self._latest_odom_pose = {
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "yaw": float(
                math.atan2(
                    2.0 * (
                        float(msg.pose.pose.orientation.w) * float(msg.pose.pose.orientation.z)
                        + float(msg.pose.pose.orientation.x) * float(msg.pose.pose.orientation.y)
                    ),
                    1.0
                    - 2.0
                    * (
                        float(msg.pose.pose.orientation.y) * float(msg.pose.pose.orientation.y)
                        + float(msg.pose.pose.orientation.z) * float(msg.pose.pose.orientation.z)
                    ),
                )
            ),
        }
        self._publish_odom_tf(msg)

    def _on_global_costmap(self, msg: Costmap):
        self._on_costmap("global", msg)

    def _on_local_costmap(self, msg: Costmap):
        self._on_costmap("local", msg)

    def _on_costmap(self, name: str, msg: Costmap):
        self._costmap_update_counts[name] = int(self._costmap_update_counts.get(name, 0)) + 1
        metadata = getattr(msg, "metadata", None)
        self._latest_costmap_debug[name] = {
            "count": int(self._costmap_update_counts[name]),
            "frame_id": str(getattr(getattr(msg, "header", None), "frame_id", "")),
            "size_x": int(getattr(metadata, "size_x", 0)) if metadata is not None else 0,
            "size_y": int(getattr(metadata, "size_y", 0)) if metadata is not None else 0,
            "resolution": float(getattr(metadata, "resolution", 0.0)) if metadata is not None else 0.0,
        }
        self._maybe_mark_costmaps_ready()

    def _publish_odom_tf(self, msg: Odometry):
        odom_frame_id = self._latest_odom_frame_id or "odom"
        base_frame_id = self._latest_base_frame_id or "base_link"

        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = odom_frame_id
        tf_msg.child_frame_id = base_frame_id
        tf_msg.transform.translation.x = float(msg.pose.pose.position.x)
        tf_msg.transform.translation.y = float(msg.pose.pose.position.y)
        tf_msg.transform.translation.z = float(msg.pose.pose.position.z)
        tf_msg.transform.rotation.x = float(msg.pose.pose.orientation.x)
        tf_msg.transform.rotation.y = float(msg.pose.pose.orientation.y)
        tf_msg.transform.rotation.z = float(msg.pose.pose.orientation.z)
        tf_msg.transform.rotation.w = float(msg.pose.pose.orientation.w)

        self._tf_pub.publish(TFMessage(transforms=[tf_msg]))
        self._odom_tf_publish_count += 1
        if self._odom_tf_publish_count in {1, 10}:
            LOGGER.info(
                "republished odom tf count=%s frame=%s->%s pose=(%.3f, %.3f)",
                self._odom_tf_publish_count,
                odom_frame_id,
                base_frame_id,
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
            )

    def _on_load_map_done(self, future, *, request_id: str, generation: int):
        if not self._is_current_request(request_id=request_id, generation=generation):
            return
        try:
            response = future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"load_map raised: {exc}",
            )
            return

        result_code = int(getattr(response, "result", -1))
        if result_code != 0:
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"load_map returned result={result_code}",
            )
            return

        if not self._wait_for_service(self._clear_global_costmap_client, self.clear_global_costmap_service):
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"service unavailable: {self.clear_global_costmap_service}",
            )
            return
        if not self._wait_for_service(self._clear_local_costmap_client, self.clear_local_costmap_service):
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"service unavailable: {self.clear_local_costmap_service}",
            )
            return

        self._state = "clearing_costmaps"
        self._detail = "waiting_for_clear_costmaps"
        self._publish_status(state=self._state, request_id=request_id, detail=self._detail)
        futures = [
            self._clear_global_costmap_client.call_async(ClearEntireCostmap.Request()),
            self._clear_local_costmap_client.call_async(ClearEntireCostmap.Request()),
        ]
        for future in futures:
            future.add_done_callback(
                lambda _future, request_id=request_id, generation=generation: self._on_clear_costmaps_progress(
                    request_id=request_id,
                    generation=generation,
                    futures=futures,
                )
            )

    def _on_clear_costmaps_progress(self, *, request_id: str, generation: int, futures: list):
        if not self._is_current_request(request_id=request_id, generation=generation):
            return
        if not all(future.done() for future in futures):
            return
        try:
            for future in futures:
                future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"clear_costmaps raised: {exc}",
            )
            return

        self._waiting_costmap_refresh = True
        self._costmap_refresh_request_id = str(request_id)
        self._costmap_refresh_generation = int(generation)
        self._costmap_refresh_baseline = dict(self._costmap_update_counts)
        self._state = "waiting_for_costmap_refresh"
        self._detail = "waiting_for_global_and_local_costmap_refresh"
        LOGGER.info(
            "costmaps cleared request_id=%s waiting for refresh baseline=%s",
            request_id,
            self._costmap_refresh_baseline,
        )
        self._publish_status(state=self._state, request_id=request_id, detail=self._detail)
        self._maybe_mark_costmaps_ready()

    def _maybe_mark_costmaps_ready(self):
        if not self._waiting_costmap_refresh:
            return
        if not self._is_current_request(
            request_id=self._costmap_refresh_request_id,
            generation=self._costmap_refresh_generation,
        ):
            self._waiting_costmap_refresh = False
            return
        if any(
            int(self._costmap_update_counts.get(name, 0)) <= int(self._costmap_refresh_baseline.get(name, 0))
            for name in ("global", "local")
        ):
            return

        request_id = str(self._costmap_refresh_request_id)
        self._waiting_costmap_refresh = False
        self._state = "ready"
        self._detail = "map_loaded_costmaps_refreshed"
        LOGGER.info(
            "map ready request_id=%s global_costmap=%s local_costmap=%s",
            request_id,
            self._latest_costmap_debug.get("global", {}),
            self._latest_costmap_debug.get("local", {}),
        )
        self._publish_status(state=self._state, request_id=request_id, detail=self._detail)

    def _on_goal_response(self, future, *, request_id: str, generation: int):
        if not self._is_current_request(request_id=request_id, generation=generation):
            return
        try:
            goal_handle = future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"goal response raised: {exc}",
            )
            return
        if goal_handle is None or not goal_handle.accepted:
            self._publish_terminal_failure(
                request_id=request_id,
                state="rejected",
                detail="NavigateToPose goal rejected",
            )
            return

        self._goal_handle = goal_handle
        self._state = "accepted"
        self._detail = "goal accepted"
        LOGGER.info("goal accepted request_id=%s", request_id)
        self._publish_status(state=self._state, request_id=request_id, detail=self._detail)
        self._state = "running"
        self._detail = "goal running"
        LOGGER.info("goal running request_id=%s", request_id)
        self._publish_status(state=self._state, request_id=request_id, detail=self._detail)
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(
            lambda future, request_id=request_id, generation=generation: self._on_goal_result(
                future,
                request_id=request_id,
                generation=generation,
            )
        )

    def _on_goal_result(self, future, *, request_id: str, generation: int):
        if not self._is_current_request(request_id=request_id, generation=generation):
            return
        try:
            result = future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self._publish_terminal_failure(
                request_id=request_id,
                state="failed",
                detail=f"goal result raised: {exc}",
            )
            return

        status_code = int(getattr(result, "status", -1))
        if status_code == 4:
            state = "succeeded"
        elif status_code == 5:
            state = "canceled"
        elif status_code == 6:
            state = "aborted"
        else:
            state = "failed"

        self._goal_handle = None
        self._result_future = None
        self._state = state
        self._detail = f"goal finished with status_code={status_code}"
        LOGGER.info("goal finished request_id=%s state=%s status_code=%s", request_id, state, status_code)
        self._publish_status(state=state, request_id=request_id, detail=self._detail)
        self._publish_result(
            request_id=request_id,
            state=state,
            detail=self._detail,
            status_code=status_code,
        )

    def _publish_heartbeat(self):
        now = time.monotonic()
        if now - self._last_status_publish_monotonic < self._heartbeat_sec * 0.8:
            return
        self._publish_status(state=self._state, request_id=self._request_id, detail=self._detail)

    def _publish_status(self, *, state: str, request_id: str, detail: str):
        self._state = str(state)
        self._detail = str(detail)
        payload = {
            "robot_name": self.robot_name,
            "request_id": str(request_id),
            "stack_id": self._stack_id,
            "state": self._state,
            "detail": self._detail,
            "stack_ready": self._stack_ready(),
            "reported_pose": dict(self._latest_odom_pose),
            "updated_at": time.time(),
        }
        self._publish_json(self._status_pub, payload)
        self._last_status_publish_monotonic = time.monotonic()

    def _publish_result(self, *, request_id: str, state: str, detail: str, status_code: int):
        payload = {
            "robot_name": self.robot_name,
            "request_id": str(request_id),
            "stack_id": self._stack_id,
            "state": str(state),
            "detail": str(detail),
            "status_code": int(status_code),
            "reported_pose": dict(self._latest_odom_pose),
            "planning": dict(self._latest_planning_debug),
            "updated_at": time.time(),
        }
        self._publish_json(self._result_pub, payload)

    def _publish_terminal_failure(self, *, request_id: str, state: str, detail: str):
        self._goal_handle = None
        self._result_future = None
        self._state = str(state)
        self._detail = str(detail)
        self._publish_status(state=self._state, request_id=request_id, detail=self._detail)
        self._publish_result(request_id=request_id, state=self._state, detail=self._detail, status_code=-1)

    def _cancel_active_goal(self, *, request_id: str, detail: str, publish_terminal: bool):
        if self._goal_handle is None:
            if publish_terminal and request_id:
                self._publish_status(state="canceled", request_id=request_id, detail=detail)
                self._publish_result(request_id=request_id, state="canceled", detail=detail, status_code=5)
            return
        self._goal_handle.cancel_goal_async()
        self._goal_handle = None
        self._result_future = None
        if publish_terminal and request_id:
            self._publish_status(state="canceled", request_id=request_id, detail=detail)
            self._publish_result(request_id=request_id, state="canceled", detail=detail, status_code=5)

    def _stack_ready(self) -> bool:
        if not self._action_client.server_is_ready():
            self._action_client.wait_for_server(timeout_sec=0.0)
        if not self._load_map_client.service_is_ready():
            self._load_map_client.wait_for_service(timeout_sec=0.0)
        if not self._clear_global_costmap_client.service_is_ready():
            self._clear_global_costmap_client.wait_for_service(timeout_sec=0.0)
        if not self._clear_local_costmap_client.service_is_ready():
            self._clear_local_costmap_client.wait_for_service(timeout_sec=0.0)
        return bool(
            self._action_client.server_is_ready()
            and self._load_map_client.service_is_ready()
            and self._clear_global_costmap_client.service_is_ready()
            and self._clear_local_costmap_client.service_is_ready()
        )

    def _wait_for_action_server(self) -> bool:
        if not self._action_client.server_is_ready():
            self._action_client.wait_for_server(timeout_sec=0.1)
        return bool(self._action_client.server_is_ready())

    def _wait_for_planner_action_server(self) -> bool:
        if not self._planner_action_client.server_is_ready():
            self._planner_action_client.wait_for_server(timeout_sec=0.1)
        return bool(self._planner_action_client.server_is_ready())

    @staticmethod
    def _wait_for_service(client, service_name: str) -> bool:
        del service_name
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=0.1)
        return bool(client.service_is_ready())

    def _current_start_pose(self, frame_id: str) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(frame_id or "map")
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = float(self._latest_odom_pose.get("x", 0.0))
        pose.pose.position.y = float(self._latest_odom_pose.get("y", 0.0))
        pose.pose.position.z = 0.0
        yaw = float(self._latest_odom_pose.get("yaw", 0.0))
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        return pose

    def _goal_pose_from_payload(self, payload: dict[str, Any]) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(payload.get("frame_id", "map"))
        pose.header.stamp = self.node.get_clock().now().to_msg()
        goal = dict(payload.get("goal", {}))
        yaw = float(goal.get("yaw", 0.0))
        pose.pose.position.x = float(goal.get("x", 0.0))
        pose.pose.position.y = float(goal.get("y", 0.0))
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        return pose

    def _publish_json(self, publisher, payload: dict[str, Any]):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)

    def _is_current_request(self, *, request_id: str, generation: int) -> bool:
        return request_id == self._request_id and int(generation) == int(self._request_generation)

    def _payload_matches_robot(self, payload: dict[str, Any]) -> bool:
        return str(payload.get("robot_name", self.robot_name)).strip() == self.robot_name

    @staticmethod
    def _parse_payload(payload: str) -> dict[str, Any]:
        try:
            parsed = json.loads(str(payload))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _pose_stamped_to_dict(pose: PoseStamped) -> dict[str, Any]:
        return {
            "frame_id": str(pose.header.frame_id),
            "x": float(pose.pose.position.x),
            "y": float(pose.pose.position.y),
            "z": float(pose.pose.position.z),
            "yaw": float(
                math.atan2(
                    2.0 * (
                        float(pose.pose.orientation.w) * float(pose.pose.orientation.z)
                        + float(pose.pose.orientation.x) * float(pose.pose.orientation.y)
                    ),
                    1.0
                    - 2.0
                    * (
                        float(pose.pose.orientation.y) * float(pose.pose.orientation.y)
                        + float(pose.pose.orientation.z) * float(pose.pose.orientation.z)
                    ),
                )
            ),
        }

    @staticmethod
    def _duration_to_sec(duration_msg) -> float:
        if duration_msg is None:
            return 0.0
        return float(getattr(duration_msg, "sec", 0)) + float(getattr(duration_msg, "nanosec", 0)) * 1.0e-9

    @staticmethod
    def _path_to_dict(path_msg) -> dict[str, Any]:
        if path_msg is None:
            return {}
        poses: list[dict[str, float]] = []
        total_length_m = 0.0
        previous_xy: Optional[tuple[float, float]] = None
        for pose_stamped in list(getattr(path_msg, "poses", [])):
            x = float(pose_stamped.pose.position.x)
            y = float(pose_stamped.pose.position.y)
            yaw = float(
                math.atan2(
                    2.0 * (
                        float(pose_stamped.pose.orientation.w) * float(pose_stamped.pose.orientation.z)
                        + float(pose_stamped.pose.orientation.x) * float(pose_stamped.pose.orientation.y)
                    ),
                    1.0
                    - 2.0
                    * (
                        float(pose_stamped.pose.orientation.y) * float(pose_stamped.pose.orientation.y)
                        + float(pose_stamped.pose.orientation.z) * float(pose_stamped.pose.orientation.z)
                    ),
                )
            )
            if previous_xy is not None:
                total_length_m += math.hypot(x - previous_xy[0], y - previous_xy[1])
            previous_xy = (x, y)
            poses.append({"x": x, "y": y, "yaw": yaw})
        return {
            "frame_id": str(getattr(getattr(path_msg, "header", None), "frame_id", "")),
            "num_poses": len(poses),
            "path_length_m": float(total_length_m),
            "poses": poses,
        }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-name", default="split_aloha")
    parser.add_argument("--map-update-topic", default="/simbox/nav_bridge/map_update")
    parser.add_argument("--goal-topic", default="/simbox/nav_bridge/goal")
    parser.add_argument("--cancel-topic", default="/simbox/nav_bridge/cancel")
    parser.add_argument("--status-topic", default="/simbox/nav_bridge/status")
    parser.add_argument("--result-topic", default="/simbox/nav_bridge/result")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--action-name", default="/navigate_to_pose")
    parser.add_argument("--planner-action-name", default="/compute_path_to_pose")
    parser.add_argument("--load-map-service", default="/map_server/load_map")
    parser.add_argument("--clear-global-costmap-service", default="/global_costmap/clear_entirely_global_costmap")
    parser.add_argument("--clear-local-costmap-service", default="/local_costmap/clear_entirely_local_costmap")
    parser.add_argument("--heartbeat-sec", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[nav2-bridge] %(asctime)s %(levelname)s %(message)s",
    )

    adapter = Nav2BridgeAdapter(
        robot_name=args.robot_name,
        map_update_topic=args.map_update_topic,
        goal_topic=args.goal_topic,
        cancel_topic=args.cancel_topic,
        status_topic=args.status_topic,
        result_topic=args.result_topic,
        odom_topic=args.odom_topic,
        action_name=args.action_name,
        planner_action_name=args.planner_action_name,
        load_map_service=args.load_map_service,
        clear_global_costmap_service=args.clear_global_costmap_service,
        clear_local_costmap_service=args.clear_local_costmap_service,
        heartbeat_sec=args.heartbeat_sec,
    )
    return adapter.run()


if __name__ == "__main__":
    raise SystemExit(main())
