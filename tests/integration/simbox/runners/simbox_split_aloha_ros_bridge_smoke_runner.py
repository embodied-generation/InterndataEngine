"""Smoke runner for SplitAloha ROS base bridge integration."""

import argparse
import json
import math
import os
import sys
import tempfile
import traceback

import numpy as np
import yaml
from isaacsim import SimulationApp

_runner_args = sys.argv[1:]
sys.argv = [sys.argv[0]]
simulation_app = SimulationApp({"headless": True})
sys.argv = [sys.argv[0], *_runner_args]

sys.path.append("./")
sys.path.append("./data_engine")
sys.path.append("workflows/simbox")

from omni.isaac.core import World  # pylint: disable=wrong-import-position

from nimbus.utils.utils import init_env  # pylint: disable=wrong-import-position
from workflows import import_extensions  # pylint: disable=wrong-import-position
from workflows.base import create_workflow  # pylint: disable=wrong-import-position


def _load_render_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_world(simulator_cfg: dict):
    return World(
        physics_dt=eval(str(simulator_cfg["physics_dt"])),
        rendering_dt=eval(str(simulator_cfg["rendering_dt"])),
        stage_units_in_meters=float(simulator_cfg["stage_units_in_meters"]),
    )


def _write_camera_free_task_cfg(task_cfg_path: str):
    with open(task_cfg_path, "r", encoding="utf-8") as file:
        task_cfg = yaml.safe_load(file)

    if isinstance(task_cfg, dict) and isinstance(task_cfg.get("tasks"), list):
        for task in task_cfg["tasks"]:
            if isinstance(task, dict):
                task["cameras"] = []

    tmp_dir = tempfile.mkdtemp(prefix="split_aloha_smoke_", dir="/tmp")
    patched_task_cfg_path = os.path.join(tmp_dir, os.path.basename(task_cfg_path))
    with open(patched_task_cfg_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(task_cfg, file, sort_keys=False)
    return patched_task_cfg_path


def _find_split_aloha(workflow):
    for robot in workflow.task.robots.values():
        if robot.__class__.__name__ == "SplitAloha":
            return robot
    raise RuntimeError("SplitAloha robot not found in workflow task")


def run_smoke(config_path: str, output_path: str, steps: int, wheel_velocity: float):
    bridge = None
    monitor_node = None
    report = {
        "config_path": config_path,
        "output_path": output_path,
        "steps": int(steps),
        "wheel_velocity": float(wheel_velocity),
        "status": "starting",
    }
    try:
        print("[smoke] importing ROS dependencies", flush=True)
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise RuntimeError(f"ROS message packages are required for this smoke test: {exc}") from exc

        from workflows.simbox.core.mobile.bridge import RangerMiniV3Bridge

        print("[smoke] initializing environment", flush=True)
        init_env()
        config = _load_render_config(config_path)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        task_cfg_path = _write_camera_free_task_cfg(scene_loader_cfg["cfg_path"])
        simulator_cfg = scene_loader_cfg["simulator"]
        report["workflow_type"] = workflow_type
        report["task_cfg_path"] = task_cfg_path

        import_extensions(workflow_type)
        print(f"[smoke] creating world for workflow={workflow_type}", flush=True)
        world = _build_world(simulator_cfg)
        workflow = create_workflow(workflow_type, world, task_cfg_path)
        print("[smoke] initializing task", flush=True)
        workflow.init_task(0)

        print("[smoke] locating SplitAloha robot", flush=True)
        robot = _find_split_aloha(workflow)
        base_interface = robot.get_base_interface()
        print("[smoke] creating direct /cmd_vel 4WIS bridge", flush=True)
        bridge = RangerMiniV3Bridge(
            robot,
            node_name="split_aloha_ros_bridge_smoke",
        )
        monitor_node = Node("split_aloha_ros_smoke_monitor")
        report["status"] = "running"

        counts = {"joint_states": 0, "odom": 0}
        odom_track = []
        max_abs_actual_wheel_velocity = 0.0
        max_abs_actual_steering_error = 0.0
        max_abs_requested_wheel_velocity = 0.0
        max_abs_requested_steering = 0.0

        def _on_joint_state(_msg):
            nonlocal max_abs_actual_wheel_velocity
            counts["joint_states"] += 1
            if not _msg.velocity:
                return
            wheel_velocity_slice = _msg.velocity[-4:]
            if wheel_velocity_slice:
                max_abs_actual_wheel_velocity = max(
                    max_abs_actual_wheel_velocity,
                    max(abs(float(v)) for v in wheel_velocity_slice),
                )

        def _on_odom_with_pose(msg):
            counts["odom"] += 1
            odom_track.append((float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)))

        monitor_node.create_subscription(JointState, bridge.ros_cfg["joint_state_topic"], _on_joint_state, 10)
        monitor_node.create_subscription(Odometry, bridge.ros_cfg["odom_topic"], _on_odom_with_pose, 10)

        command_pub = bridge.node.create_publisher(Twist, bridge.ros_cfg["cmd_vel_topic"], 10)
        wheel_radius = float(bridge.base_cfg["wheel_radius"])

        def _pump_monitor(iterations: int = 4):
            for _ in range(iterations):
                rclpy.spin_once(monitor_node, timeout_sec=0.0)

        # Let DDS discovery and topic matching settle before measuring traffic.
        for _ in range(40):
            bridge.step()
            max_abs_requested_wheel_velocity = max(
                max_abs_requested_wheel_velocity,
                float(np.max(np.abs(bridge._last_requested_wheel_velocities))),
            )
            max_abs_requested_steering = max(
                max_abs_requested_steering,
                float(np.max(np.abs(bridge._last_requested_steering))),
            )
            max_abs_actual_steering_error = max(
                max_abs_actual_steering_error,
                float(np.max(np.abs(bridge._last_requested_steering - bridge._last_applied_steering))),
            )
            _pump_monitor()
            world.step(render=False)

        print("[smoke] streaming cmd_vel sequence", flush=True)
        for step_idx in range(steps):
            steering_target = 0.25 * math.sin(step_idx * 0.05)
            linear_speed = (wheel_velocity * wheel_radius) if step_idx < int(steps * 0.8) else 0.0

            command_msg = Twist()
            command_msg.linear.x = linear_speed
            command_msg.linear.y = linear_speed * math.sin(steering_target)
            command_msg.angular.z = 0.6 * math.sin(step_idx * 0.03)
            command_pub.publish(command_msg)

            bridge.step()
            max_abs_requested_wheel_velocity = max(
                max_abs_requested_wheel_velocity,
                float(np.max(np.abs(bridge._last_requested_wheel_velocities))),
            )
            max_abs_requested_steering = max(
                max_abs_requested_steering,
                float(np.max(np.abs(bridge._last_requested_steering))),
            )
            max_abs_actual_steering_error = max(
                max_abs_actual_steering_error,
                float(np.max(np.abs(bridge._last_requested_steering - bridge._last_applied_steering))),
            )
            _pump_monitor()
            world.step(render=False)

        for _ in range(10):
            bridge.step()
            max_abs_requested_wheel_velocity = max(
                max_abs_requested_wheel_velocity,
                float(np.max(np.abs(bridge._last_requested_wheel_velocities))),
            )
            max_abs_requested_steering = max(
                max_abs_requested_steering,
                float(np.max(np.abs(bridge._last_requested_steering))),
            )
            max_abs_actual_steering_error = max(
                max_abs_actual_steering_error,
                float(np.max(np.abs(bridge._last_requested_steering - bridge._last_applied_steering))),
            )
            _pump_monitor()
            world.step(render=False)

        if len(odom_track) >= 2:
            first_xy = np.asarray(odom_track[0], dtype=np.float32)
            last_xy = np.asarray(odom_track[-1], dtype=np.float32)
            planar_displacement = float(np.linalg.norm(last_xy - first_xy))
        else:
            planar_displacement = 0.0

        report = {
            "task_cfg_path": task_cfg_path,
            "steps": int(steps),
            "wheel_velocity": float(wheel_velocity),
            "joint_state_count": int(counts["joint_states"]),
            "odom_count": int(counts["odom"]),
            "planar_displacement": planar_displacement,
            "bridge_received_cmd_vel_count": int(bridge._received_cmd_vel_count),
            "bridge_last_cmd_vel": dict(bridge._last_received_cmd_vel),
            "bridge_last_command": {
                "vx_body": float(bridge._command.vx_body),
                "vy_body": float(bridge._command.vy_body),
                "wz_body": float(bridge._command.wz_body),
            },
            "bridge_driver_command_message_count": int(bridge._driver_command_message_count),
            "bridge_motion_mode_message_count": int(bridge._motion_mode_message_count),
            "bridge_applied_driver_command_count": int(bridge._applied_driver_command_count),
            "bridge_pending_driver_command_count": int(bridge._pending_driver_command_count),
            "bridge_max_abs_requested_wheel_velocity": float(max_abs_requested_wheel_velocity),
            "bridge_max_abs_requested_steering": float(max_abs_requested_steering),
            "max_abs_actual_wheel_velocity": float(max_abs_actual_wheel_velocity),
            "max_abs_steering_tracking_error": float(max_abs_actual_steering_error),
            "status": "completed",
        }

        if counts["joint_states"] <= 0:
            raise RuntimeError("No /joint_states messages were observed during smoke test")
        if counts["odom"] <= 0:
            raise RuntimeError("No /odom messages were observed during smoke test")

        print("[smoke] writing success report", flush=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print(json.dumps(report, indent=2))
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        print(f"[smoke] failed: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if report.get("status") != "completed":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=2)
        if monitor_node is not None:
            monitor_node.destroy_node()
        if bridge is not None:
            bridge.destroy()
        simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="configs/simbox/de_render_template.yaml",
        help="Path to the SimBox render config used to load the task",
    )
    parser.add_argument(
        "--output-path",
        default="output/ros_bridge/split_aloha_ros_bridge_smoke.json",
        help="Where to save the smoke test report JSON",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="How many smoke loop steps to execute",
    )
    parser.add_argument(
        "--wheel-velocity",
        type=float,
        default=4.0,
        help="Wheel target velocity used in smoke command stream",
    )
    args = parser.parse_args()

    run_smoke(
        config_path=args.config_path,
        output_path=args.output_path,
        steps=args.steps,
        wheel_velocity=args.wheel_velocity,
    )
