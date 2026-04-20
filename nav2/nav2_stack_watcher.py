#!/usr/bin/env python3
"""Watch external Nav2 stack requests and launch/refresh ROS-side Nav2 stacks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time

import rclpy
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nav2.protocol import (  # pylint: disable=wrong-import-position
    DEFAULT_EXTERNAL_NAV2_STACK_REQUEST_ROOT,
    DEFAULT_EXTERNAL_NAV2_STACK_STATUS_ROOT,
    atomic_write_yaml,
    read_yaml_file,
)


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _status_payload(*, robot_name: str, stack_id: str, state: str, detail: str = "", log_path: str = "", pid: int | None = None):
    payload = {
        "robot_name": robot_name,
        "stack_id": stack_id,
        "state": state,
        "detail": detail,
        "updated_at": datetime.now().isoformat(),
    }
    if log_path:
        payload["log_path"] = log_path
    if pid is not None:
        payload["pid"] = int(pid)
    return payload


def _stop_process(process: subprocess.Popen | None, log_file):
    if process is not None:
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
    if log_file is not None:
        log_file.close()


def _build_launch_command(params_path: Path) -> str:
    params_q = shlex.quote(str(params_path))
    return (
        "source /opt/ros/humble/setup.bash && "
        "trap 'kill 0 >/dev/null 2>&1 || true' EXIT INT TERM && "
        "ros2 run tf2_ros static_transform_publisher "
        "--x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 --frame-id map --child-frame-id odom "
        "> /dev/null 2>&1 & "
        f"ros2 run nav2_map_server map_server --ros-args --params-file {params_q} "
        "> /dev/null 2>&1 & "
        "ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args "
        "-p use_sim_time:=true -p autostart:=true -p node_names:=\"['map_server']\" "
        "> /dev/null 2>&1 & "
        "ros2 launch nav2_bringup navigation_launch.py "
        f"use_sim_time:=true autostart:=true use_composition:=False params_file:={params_q} & "
        "wait"
    )


class StackReadyProbe:
    def __init__(self):
        self._owns_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_context = True
        self.node = Node("interndata_nav2_stack_ready_probe")
        self.node.set_parameters([Parameter("use_sim_time", value=True)])
        self._clients: dict[str, ActionClient] = {}

    def is_ready(self, action_name: str) -> bool:
        action_name = str(action_name).strip() or "/navigate_to_pose"
        client = self._clients.get(action_name)
        if client is None:
            client = ActionClient(self.node, NavigateToPose, action_name)
            self._clients[action_name] = client
        ready = bool(client.server_is_ready())
        if not ready:
            client.wait_for_server(timeout_sec=0.01)
            ready = bool(client.server_is_ready())
        rclpy.spin_once(self.node, timeout_sec=0.0)
        return ready

    def destroy(self):
        self.node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()


@dataclass
class RuntimeStack:
    robot_name: str
    revision: str
    stack_id: str
    request_path: Path
    status_path: Path
    action_name: str
    process: subprocess.Popen | None = None
    log_file: object | None = None
    log_path: str = ""

    def stop(self):
        _stop_process(self.process, self.log_file)
        self.process = None
        self.log_file = None


class Nav2StackWatcher:
    def __init__(self, *, request_root: Path, status_root: Path, poll_interval: float):
        self.request_root = request_root
        self.status_root = status_root
        self.poll_interval = max(float(poll_interval), 0.1)
        self._running = True
        self._stacks: dict[str, RuntimeStack] = {}
        self._ready_probe = StackReadyProbe()

    def stop(self):
        self._running = False

    def run(self) -> int:
        self.request_root.mkdir(parents=True, exist_ok=True)
        self.status_root.mkdir(parents=True, exist_ok=True)
        try:
            while self._running:
                self._sync_once()
                time.sleep(self.poll_interval)
        finally:
            for stack in list(self._stacks.values()):
                stack.stop()
                atomic_write_yaml(
                    str(stack.status_path),
                    _status_payload(
                        robot_name=stack.robot_name,
                        stack_id=stack.stack_id,
                        state="stopped",
                        detail="watcher shutting down",
                        log_path=stack.log_path,
                    ),
                )
            self._ready_probe.destroy()
        return 0

    def _sync_once(self):
        active_robot_names: set[str] = set()
        request_paths = sorted(self.request_root.glob("*.yaml"))
        for request_path in request_paths:
            request = read_yaml_file(str(request_path))
            robot_name = str(request.get("robot_name", request_path.stem)).strip() or request_path.stem
            active_robot_names.add(robot_name)
            status_path = self.status_root / f"{robot_name}.yaml"
            enabled = bool(request.get("enabled", False))
            stack_id = str(request.get("stack_id", "")).strip()
            revision = str(request.get("request_revision", stack_id)).strip()
            params_path = _resolve_path(str(request.get("params_path", "")))
            map_yaml_path = _resolve_path(str(request.get("map_yaml_path", "")))
            stack_output_dir = _resolve_path(str(request.get("stack_output_dir", "")))
            action_name = str(request.get("action_name", "/navigate_to_pose")).strip() or "/navigate_to_pose"

            existing = self._stacks.get(robot_name)
            if not enabled:
                if existing is not None:
                    existing.stop()
                    atomic_write_yaml(
                        str(status_path),
                        _status_payload(
                            robot_name=robot_name,
                            stack_id=existing.stack_id,
                            state="stopped",
                            detail="request disabled",
                            log_path=existing.log_path,
                        ),
                    )
                    del self._stacks[robot_name]
                continue

            if not stack_id:
                atomic_write_yaml(
                    str(status_path),
                    _status_payload(robot_name=robot_name, stack_id="", state="invalid", detail="missing stack_id"),
                )
                continue

            if not params_path.exists() or not map_yaml_path.exists():
                atomic_write_yaml(
                    str(status_path),
                    _status_payload(
                        robot_name=robot_name,
                        stack_id=stack_id,
                        state="waiting_for_artifacts",
                        detail=f"params={params_path.exists()} map={map_yaml_path.exists()}",
                    ),
                )
                continue

            if existing is not None and existing.process is not None and existing.process.poll() is not None:
                existing.stop()
                del self._stacks[robot_name]
                existing = None

            if existing is not None and existing.revision == revision:
                self._write_live_status(existing)
                continue

            if existing is not None:
                existing.stop()
                del self._stacks[robot_name]

            stack_output_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(stack_output_dir / "external_nav2_stack.log")
            log_file = open(log_path, "w", encoding="utf-8")  # pylint: disable=consider-using-with
            env = dict(os.environ)
            env["ROS_DOMAIN_ID"] = str(request.get("ros_domain_id", env.get("ROS_DOMAIN_ID", "0")))
            env["RMW_IMPLEMENTATION"] = str(
                request.get("rmw_implementation", env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"))
            )
            command = _build_launch_command(params_path)
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                ["bash", "-lc", command],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            stack = RuntimeStack(
                robot_name=robot_name,
                revision=revision,
                stack_id=stack_id,
                request_path=request_path,
                status_path=status_path,
                action_name=action_name,
                process=process,
                log_file=log_file,
                log_path=log_path,
            )
            self._stacks[robot_name] = stack
            atomic_write_yaml(
                str(status_path),
                _status_payload(
                    robot_name=robot_name,
                    stack_id=stack_id,
                    state="launching",
                    detail="stack launched",
                    log_path=log_path,
                    pid=process.pid,
                ),
            )

        stale_robot_names = [name for name in self._stacks.keys() if name not in active_robot_names]
        for robot_name in stale_robot_names:
            stack = self._stacks.pop(robot_name)
            stack.stop()
            atomic_write_yaml(
                str(stack.status_path),
                _status_payload(
                    robot_name=robot_name,
                    stack_id=stack.stack_id,
                    state="stopped",
                    detail="request removed",
                    log_path=stack.log_path,
                ),
            )

    def _write_live_status(self, stack: RuntimeStack):
        if stack.process is not None and stack.process.poll() is not None:
            atomic_write_yaml(
                str(stack.status_path),
                _status_payload(
                    robot_name=stack.robot_name,
                    stack_id=stack.stack_id,
                    state="failed",
                    detail=f"stack exited rc={stack.process.returncode}",
                    log_path=stack.log_path,
                    pid=stack.process.pid,
                ),
            )
            return

        state = "ready" if self._ready_probe.is_ready(stack.action_name) else "launching"
        detail = "action server ready" if state == "ready" else "waiting for action server"
        atomic_write_yaml(
            str(stack.status_path),
            _status_payload(
                robot_name=stack.robot_name,
                stack_id=stack.stack_id,
                state=state,
                detail=detail,
                log_path=stack.log_path,
                pid=stack.process.pid if stack.process is not None else None,
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch Nav2 runtime requests and launch external ROS stacks")
    parser.add_argument(
        "--request-root",
        default=os.environ.get("INTERNDATA_EXTERNAL_NAV2_REQUEST_ROOT", DEFAULT_EXTERNAL_NAV2_STACK_REQUEST_ROOT),
    )
    parser.add_argument(
        "--status-root",
        default=os.environ.get("INTERNDATA_EXTERNAL_NAV2_STATUS_ROOT", DEFAULT_EXTERNAL_NAV2_STACK_STATUS_ROOT),
    )
    parser.add_argument("--poll-interval", type=float, default=0.5)
    args = parser.parse_args()

    watcher = Nav2StackWatcher(
        request_root=_resolve_path(args.request_root),
        status_root=_resolve_path(args.status_root),
        poll_interval=args.poll_interval,
    )

    def _request_stop(signum, _frame):
        del signum
        watcher.stop()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    print(
        f"[nav2-watcher] request_root={watcher.request_root} status_root={watcher.status_root}",
        flush=True,
    )
    return watcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
