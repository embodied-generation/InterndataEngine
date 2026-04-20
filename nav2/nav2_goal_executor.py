#!/usr/bin/env python3
"""Execute external Nav2 goal requests against a compose-managed Nav2 stack."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import math
import os
import signal
import sys
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nav2.protocol import (  # pylint: disable=wrong-import-position
    DEFAULT_EXTERNAL_NAV2_GOAL_REQUEST_ROOT,
    DEFAULT_EXTERNAL_NAV2_GOAL_RESULT_ROOT,
    DEFAULT_EXTERNAL_NAV2_GOAL_STATUS_ROOT,
    DEFAULT_EXTERNAL_NAV2_STACK_STATUS_ROOT,
    atomic_write_yaml,
    read_yaml_file,
    remove_file_if_exists,
)


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _yaw_from_xyzw(x: float, y: float, z: float, w: float) -> float:
    del x, y
    return math.atan2(2.0 * (w * z), 1.0 - 2.0 * (z * z))


def _status_payload(*, robot_name: str, stack_id: str, goal_id: str, state: str, detail: str = ""):
    return {
        "robot_name": robot_name,
        "stack_id": stack_id,
        "goal_id": goal_id,
        "state": state,
        "detail": detail,
        "updated_at": datetime.now().isoformat(),
    }


def _result_payload(
    *,
    robot_name: str,
    stack_id: str,
    goal_id: str,
    state: str,
    status_code: int | None,
    detail: str = "",
    reported_pose: dict | None = None,
):
    payload = {
        "robot_name": robot_name,
        "stack_id": stack_id,
        "goal_id": goal_id,
        "state": state,
        "updated_at": datetime.now().isoformat(),
    }
    if status_code is not None:
        payload["status_code"] = int(status_code)
    if detail:
        payload["detail"] = detail
    if reported_pose is not None:
        payload["reported_pose"] = reported_pose
    return payload


def _result_state(status_code: int) -> str:
    if int(status_code) == int(GoalStatus.STATUS_SUCCEEDED):
        return "succeeded"
    if int(status_code) == int(GoalStatus.STATUS_ABORTED):
        return "aborted"
    if int(status_code) == int(GoalStatus.STATUS_CANCELED):
        return "canceled"
    return "failed"


@dataclass
class GoalSession:
    robot_name: str
    stack_id: str
    goal_id: str
    action_name: str
    odom_topic: str
    status_path: Path
    result_path: Path
    request_path: Path
    goal_future: object | None = None
    goal_handle: object | None = None
    result_future: object | None = None
    cancel_future: object | None = None
    cancel_requested: bool = False


class Nav2GoalExecutor:
    def __init__(
        self,
        *,
        goal_request_root: Path,
        goal_status_root: Path,
        goal_result_root: Path,
        stack_status_root: Path,
        poll_interval: float,
    ):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("interndata_nav2_goal_executor")
        self.node.set_parameters([Parameter("use_sim_time", value=True)])
        self.goal_request_root = goal_request_root
        self.goal_status_root = goal_status_root
        self.goal_result_root = goal_result_root
        self.stack_status_root = stack_status_root
        self.poll_interval = max(float(poll_interval), 0.05)
        self._running = True
        self._clients: dict[str, ActionClient] = {}
        self._sessions: dict[str, GoalSession] = {}
        self._odom_subscriptions = {}
        self._latest_pose_by_topic = {}

    def stop(self):
        self._running = False

    def run(self) -> int:
        self.goal_request_root.mkdir(parents=True, exist_ok=True)
        self.goal_status_root.mkdir(parents=True, exist_ok=True)
        self.goal_result_root.mkdir(parents=True, exist_ok=True)
        try:
            while self._running:
                rclpy.spin_once(self.node, timeout_sec=0.05)
                self._sync_once()
                time.sleep(self.poll_interval)
        finally:
            for session in list(self._sessions.values()):
                self._cancel_session(session, reason="executor shutting down")
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return 0

    def _sync_once(self):
        active_robot_names: set[str] = set()
        request_paths = sorted(self.goal_request_root.glob("*.yaml"))
        for request_path in request_paths:
            request = read_yaml_file(str(request_path))
            robot_name = str(request.get("robot_name", request_path.stem)).strip() or request_path.stem
            active_robot_names.add(robot_name)
            goal_id = str(request.get("goal_id", "")).strip()
            stack_id = str(request.get("stack_id", "")).strip()
            enabled = bool(request.get("enabled", False))
            if not enabled or not goal_id or not stack_id:
                existing = self._sessions.pop(robot_name, None)
                if existing is not None:
                    self._cancel_session(existing, reason="request disabled")
                continue

            stack_status = read_yaml_file(str(self.stack_status_root / f"{robot_name}.yaml"))
            if str(stack_status.get("stack_id", "")).strip() != stack_id or str(stack_status.get("state", "")).strip().lower() != "ready":
                atomic_write_yaml(
                    str(self.goal_status_root / f"{robot_name}.yaml"),
                    _status_payload(
                        robot_name=robot_name,
                        stack_id=stack_id,
                        goal_id=goal_id,
                        state="waiting_for_stack",
                        detail=str(stack_status.get("state", "stack_not_ready")),
                    ),
                )
                continue

            action_name = str(request.get("action_name", "/navigate_to_pose")).strip() or "/navigate_to_pose"
            odom_topic = str(request.get("odom_topic", "/odom")).strip() or "/odom"
            self._ensure_odom_subscription(odom_topic)
            session = self._sessions.get(robot_name)
            if session is None or session.goal_id != goal_id:
                if session is not None:
                    self._cancel_session(session, reason="superseded by new goal")
                session = self._start_session(
                    robot_name=robot_name,
                    stack_id=stack_id,
                    goal_id=goal_id,
                    action_name=action_name,
                    odom_topic=odom_topic,
                    request=request,
                    request_path=request_path,
                )
                self._sessions[robot_name] = session

            session.cancel_requested = bool(request.get("cancel_requested", False))
            self._advance_session(session, request=request)

        stale_robot_names = [name for name in self._sessions.keys() if name not in active_robot_names]
        for robot_name in stale_robot_names:
            session = self._sessions.pop(robot_name)
            self._cancel_session(session, reason="request removed")
            remove_file_if_exists(str(self.goal_status_root / f"{robot_name}.yaml"))
            remove_file_if_exists(str(self.goal_result_root / f"{robot_name}.yaml"))

    def _client_for_action(self, action_name: str) -> ActionClient:
        client = self._clients.get(action_name)
        if client is None:
            client = ActionClient(self.node, NavigateToPose, action_name)
            self._clients[action_name] = client
        return client

    def _ensure_odom_subscription(self, odom_topic: str):
        if odom_topic in self._odom_subscriptions:
            return
        self._odom_subscriptions[odom_topic] = self.node.create_subscription(
            Odometry,
            odom_topic,
            lambda msg, odom_topic=odom_topic: self._on_odom(msg, odom_topic),
            10,
        )

    def _on_odom(self, msg: Odometry, odom_topic: str):
        self._latest_pose_by_topic[odom_topic] = {
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "yaw": float(
                _yaw_from_xyzw(
                    float(msg.pose.pose.orientation.x),
                    float(msg.pose.pose.orientation.y),
                    float(msg.pose.pose.orientation.z),
                    float(msg.pose.pose.orientation.w),
                )
            ),
        }

    def _start_session(
        self,
        *,
        robot_name: str,
        stack_id: str,
        goal_id: str,
        action_name: str,
        odom_topic: str,
        request: dict,
        request_path: Path,
    ) -> GoalSession:
        session = GoalSession(
            robot_name=robot_name,
            stack_id=stack_id,
            goal_id=goal_id,
            action_name=action_name,
            odom_topic=odom_topic,
            status_path=self.goal_status_root / f"{robot_name}.yaml",
            result_path=self.goal_result_root / f"{robot_name}.yaml",
            request_path=request_path,
        )
        client = self._client_for_action(action_name)
        if not client.server_is_ready():
            client.wait_for_server(timeout_sec=0.01)
        if not client.server_is_ready():
            atomic_write_yaml(
                str(session.status_path),
                _status_payload(
                    robot_name=robot_name,
                    stack_id=stack_id,
                    goal_id=goal_id,
                    state="waiting_for_action_server",
                    detail=action_name,
                ),
            )
            return session

        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = str(request.get("frame_id", "map"))
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = float(request.get("goal", {}).get("x", 0.0))
        pose.pose.position.y = float(request.get("goal", {}).get("y", 0.0))
        pose.pose.position.z = 0.0
        yaw = float(request.get("goal", {}).get("yaw", 0.0))
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        goal.pose = pose
        session.goal_future = client.send_goal_async(goal)
        atomic_write_yaml(
            str(session.status_path),
            _status_payload(
                robot_name=robot_name,
                stack_id=stack_id,
                goal_id=goal_id,
                state="accepting",
                detail=action_name,
            ),
        )
        return session

    def _advance_session(self, session: GoalSession, *, request: dict):
        client = self._client_for_action(session.action_name)
        if session.goal_future is None:
            if client.server_is_ready():
                refreshed = self._start_session(
                    robot_name=session.robot_name,
                    stack_id=session.stack_id,
                    goal_id=session.goal_id,
                    action_name=session.action_name,
                    odom_topic=session.odom_topic,
                    request=request,
                    request_path=session.request_path,
                )
                session.goal_future = refreshed.goal_future
            return

        if session.cancel_requested and session.goal_handle is not None and session.cancel_future is None:
            session.cancel_future = session.goal_handle.cancel_goal_async()
            atomic_write_yaml(
                str(session.status_path),
                _status_payload(
                    robot_name=session.robot_name,
                    stack_id=session.stack_id,
                    goal_id=session.goal_id,
                    state="canceling",
                    detail="cancel requested by workflow",
                ),
            )

        if session.goal_handle is None:
            if not session.goal_future.done():
                return
            session.goal_handle = session.goal_future.result()
            if session.goal_handle is None or not bool(getattr(session.goal_handle, "accepted", False)):
                atomic_write_yaml(
                    str(session.status_path),
                    _status_payload(
                        robot_name=session.robot_name,
                        stack_id=session.stack_id,
                        goal_id=session.goal_id,
                        state="rejected",
                        detail="goal rejected by action server",
                    ),
                )
                atomic_write_yaml(
                    str(session.result_path),
                    _result_payload(
                        robot_name=session.robot_name,
                        stack_id=session.stack_id,
                        goal_id=session.goal_id,
                        state="rejected",
                        status_code=None,
                    ),
                )
                self._sessions.pop(session.robot_name, None)
                return
            session.result_future = session.goal_handle.get_result_async()
            atomic_write_yaml(
                str(session.status_path),
                _status_payload(
                    robot_name=session.robot_name,
                    stack_id=session.stack_id,
                    goal_id=session.goal_id,
                    state="accepted",
                    detail="goal accepted by action server",
                ),
            )
            return

        if session.result_future is None:
            return

        if not session.result_future.done():
            atomic_write_yaml(
                str(session.status_path),
                _status_payload(
                    robot_name=session.robot_name,
                    stack_id=session.stack_id,
                    goal_id=session.goal_id,
                    state="running",
                    detail="goal executing",
                ),
            )
            return

        result = session.result_future.result()
        status_code = int(getattr(result, "status", GoalStatus.STATUS_UNKNOWN))
        state = _result_state(status_code)
        detail = f"status_code={status_code}"
        reported_pose = self._latest_pose_by_topic.get(session.odom_topic)
        atomic_write_yaml(
            str(session.status_path),
            _status_payload(
                robot_name=session.robot_name,
                stack_id=session.stack_id,
                goal_id=session.goal_id,
                state=state,
                detail=detail,
            ),
        )
        atomic_write_yaml(
            str(session.result_path),
            _result_payload(
                robot_name=session.robot_name,
                stack_id=session.stack_id,
                goal_id=session.goal_id,
                state=state,
                status_code=status_code,
                detail=detail,
                reported_pose=reported_pose,
            ),
        )
        self._sessions.pop(session.robot_name, None)

    def _cancel_session(self, session: GoalSession, *, reason: str):
        if session.goal_handle is not None:
            try:
                session.goal_handle.cancel_goal_async()
            except Exception:
                pass
        atomic_write_yaml(
            str(session.status_path),
            _status_payload(
                robot_name=session.robot_name,
                stack_id=session.stack_id,
                goal_id=session.goal_id,
                state="canceled",
                detail=reason,
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute external Nav2 goal requests")
    parser.add_argument(
        "--goal-request-root",
        default=os.environ.get("INTERNDATA_EXTERNAL_NAV2_GOAL_REQUEST_ROOT", DEFAULT_EXTERNAL_NAV2_GOAL_REQUEST_ROOT),
    )
    parser.add_argument(
        "--goal-status-root",
        default=os.environ.get("INTERNDATA_EXTERNAL_NAV2_GOAL_STATUS_ROOT", DEFAULT_EXTERNAL_NAV2_GOAL_STATUS_ROOT),
    )
    parser.add_argument(
        "--goal-result-root",
        default=os.environ.get("INTERNDATA_EXTERNAL_NAV2_GOAL_RESULT_ROOT", DEFAULT_EXTERNAL_NAV2_GOAL_RESULT_ROOT),
    )
    parser.add_argument(
        "--stack-status-root",
        default=os.environ.get("INTERNDATA_EXTERNAL_NAV2_STATUS_ROOT", DEFAULT_EXTERNAL_NAV2_STACK_STATUS_ROOT),
    )
    parser.add_argument("--poll-interval", type=float, default=0.1)
    args = parser.parse_args()

    executor = Nav2GoalExecutor(
        goal_request_root=_resolve_path(args.goal_request_root),
        goal_status_root=_resolve_path(args.goal_status_root),
        goal_result_root=_resolve_path(args.goal_result_root),
        stack_status_root=_resolve_path(args.stack_status_root),
        poll_interval=args.poll_interval,
    )

    def _request_stop(signum, _frame):
        del signum
        executor.stop()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    print(
        "[nav2-goal-executor] "
        f"goal_request_root={executor.goal_request_root} goal_status_root={executor.goal_status_root}",
        flush=True,
    )
    return executor.run()


if __name__ == "__main__":
    raise SystemExit(main())
