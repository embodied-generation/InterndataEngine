"""Smoke runner for SplitAloha ROS base bridge integration."""

import argparse
import json
import math
import os
import sys

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


def _find_split_aloha(workflow):
    for robot in workflow.task.robots.values():
        if robot.__class__.__name__ == "SplitAloha":
            return robot
    raise RuntimeError("SplitAloha robot not found in workflow task")


def run_smoke(config_path: str, output_path: str, steps: int, wheel_velocity: float):
    bridge = None
    controller = None
    monitor_node = None
    try:
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise RuntimeError(f"ROS message packages are required for this smoke test: {exc}") from exc

        from workflows.simbox.core.mobile import RangerMiniV3Controller, SplitAlohaIsaacBaseBridge

        init_env()
        config = _load_render_config(config_path)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        task_cfg_path = scene_loader_cfg["cfg_path"]
        simulator_cfg = scene_loader_cfg["simulator"]

        import_extensions(workflow_type)
        world = _build_world(simulator_cfg)
        workflow = create_workflow(workflow_type, world, task_cfg_path)
        workflow.init_task(0)

        robot = _find_split_aloha(workflow)
        bridge = SplitAlohaIsaacBaseBridge(robot, node_name="split_aloha_ros_bridge_smoke")
        controller = RangerMiniV3Controller(bridge.base_cfg, node_name="split_aloha_ros_cmdvel_smoke")
        monitor_node = Node("split_aloha_ros_smoke_monitor")

        counts = {"joint_states": 0, "odom": 0}
        odom_track = []

        def _on_joint_state(_msg):
            counts["joint_states"] += 1

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
            controller.step()
            bridge.step()
            _pump_monitor()
            world.step(render=False)

        for step_idx in range(steps):
            steering_target = 0.25 * math.sin(step_idx * 0.05)
            linear_speed = (wheel_velocity * wheel_radius) if step_idx < int(steps * 0.8) else 0.0

            command_msg = Twist()
            command_msg.linear.x = linear_speed
            command_msg.angular.z = 2.0 * linear_speed * math.sin(steering_target) / max(float(bridge.base_cfg["wheel_base"]), 1e-6)
            command_pub.publish(command_msg)

            controller.step()
            bridge.step()
            _pump_monitor()
            world.step(render=False)

        for _ in range(10):
            controller.step()
            bridge.step()
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
            "bridge_has_motion_mode": bool(bridge._has_motion_mode),
            "bridge_last_linear_speed": float(bridge._command.linear_speed),
            "bridge_last_steering_angle": float(bridge._command.steering_angle),
        }

        if counts["joint_states"] <= 0:
            raise RuntimeError("No /joint_states messages were observed during smoke test")
        if counts["odom"] <= 0:
            raise RuntimeError("No /odom messages were observed during smoke test")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print(json.dumps(report, indent=2))
    finally:
        if monitor_node is not None:
            monitor_node.destroy_node()
        if controller is not None:
            controller.destroy()
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
