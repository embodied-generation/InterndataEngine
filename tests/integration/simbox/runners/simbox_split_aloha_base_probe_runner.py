"""Isaac Sim runner that probes the SplitAloha mobile base interface and geometry."""

import argparse
import json
import math
import os
import sys
import traceback

import numpy as np
import yaml
import cv2  # noqa: F401  # Preload OpenCV before Kit mutates dynamic library resolution.
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


def _select_nav_robot(task_cfg: dict) -> dict:
    robots = task_cfg.get("robots", [])
    if not robots:
        raise ValueError("Task config must contain at least one robot")

    for robot in robots:
        if str(robot.get("name", "")).lower() == "split_aloha":
            return dict(robot)
        if "split_aloha" in str(robot.get("path", "")).lower():
            return dict(robot)

    return dict(robots[0])


def _write_probe_arena_cfg(output_dir: str) -> str:
    arena_cfg = {
        "name": "base_probe_floor_arena",
        "fixtures": [
            {
                "name": "floor",
                "target_class": "PlaneObject",
                "size": [12.0, 12.0],
                "collision_enabled": True,
                "collision_thickness": 0.02,
                "translation": [0.0, 0.0, 0.0],
            },
            {
                "name": "boundary_left",
                "target_class": "PlaneObject",
                "size": [3.0, 12.0],
                "translation": [-6.0, 0.0, 1.5],
                "euler": [0.0, 90.0, 0.0],
            },
            {
                "name": "boundary_right",
                "target_class": "PlaneObject",
                "size": [3.0, 12.0],
                "translation": [6.0, 0.0, 1.5],
                "euler": [0.0, 90.0, 0.0],
            },
            {
                "name": "boundary_back",
                "target_class": "PlaneObject",
                "size": [12.0, 3.0],
                "translation": [0.0, -6.0, 1.5],
                "euler": [90.0, 0.0, 0.0],
            },
            {
                "name": "boundary_front",
                "target_class": "PlaneObject",
                "size": [12.0, 3.0],
                "translation": [0.0, 6.0, 1.5],
                "euler": [90.0, 0.0, 0.0],
            },
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    arena_cfg_path = os.path.join(output_dir, "split_aloha_base_probe_arena.yaml")
    with open(arena_cfg_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(arena_cfg, file, sort_keys=False)
    return arena_cfg_path


def _build_probe_floor_task(task_cfg: dict, arena_cfg_path: str) -> dict:
    nav_robot = _select_nav_robot(task_cfg)
    nav_robot_name = str(nav_robot["name"])

    nav_task = dict(task_cfg)
    nav_task["asset_root"] = str(task_cfg.get("asset_root", "workflows/simbox/assets"))
    nav_task["arena_file"] = arena_cfg_path
    nav_task["robots"] = [nav_robot]
    nav_task["objects"] = []
    nav_task["skills"] = []
    nav_task["cameras"] = []
    nav_task["render"] = False
    nav_task["regions"] = [
        {
            "object": nav_robot_name,
            "target": "floor",
            "random_type": "A_on_B_region_sampler",
            "random_config": {
                "pos_range": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "yaw_rotation": [0.0, 0.0],
            },
        }
    ]
    nav_task.pop("distractors", None)
    nav_task.pop("mem_distractors", None)
    nav_task.pop("nav2_obstacles", None)
    env_map_cfg = dict(nav_task.get("env_map", {}))
    if env_map_cfg:
        env_map_cfg["apply_randomization"] = False
        nav_task["env_map"] = env_map_cfg
    return nav_task


def _prepare_probe_task_cfg(task_cfg_path: str, output_dir: str) -> str:
    with open(task_cfg_path, "r", encoding="utf-8") as file:
        task_cfg = yaml.safe_load(file)

    tasks = task_cfg.get("tasks", [])
    if not isinstance(tasks, list):
        raise TypeError("Task config must contain a 'tasks' list")

    arena_cfg_path = _write_probe_arena_cfg(output_dir)
    prepared_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            raise TypeError("Each task entry must be a dict")
        prepared_tasks.append(_build_probe_floor_task(task, arena_cfg_path=arena_cfg_path))
    task_cfg["tasks"] = prepared_tasks

    os.makedirs(output_dir, exist_ok=True)
    patched_task_cfg_path = os.path.join(output_dir, "split_aloha_base_probe_task.yaml")
    with open(patched_task_cfg_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(task_cfg, file, sort_keys=False)
    return patched_task_cfg_path


def _yaw_from_wxyz(q_wxyz):
    w = float(q_wxyz[0])
    x = float(q_wxyz[1])
    y = float(q_wxyz[2])
    z = float(q_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(angle: float):
    return math.atan2(math.sin(angle), math.cos(angle))


def _world_delta_to_local(delta_xy: np.ndarray, yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.asarray(
        [
            cos_yaw * float(delta_xy[0]) + sin_yaw * float(delta_xy[1]),
            -sin_yaw * float(delta_xy[0]) + cos_yaw * float(delta_xy[1]),
        ],
        dtype=np.float64,
    )


def _steering_pattern(pattern: str, steer_angle: float) -> np.ndarray:
    steer = float(steer_angle)
    if pattern == "counter_phase":
        return np.asarray([steer, steer, -steer, -steer], dtype=np.float32)
    if pattern == "inverted_counter_phase":
        return np.asarray([-steer, -steer, steer, steer], dtype=np.float32)
    if pattern == "all_same":
        return np.asarray([steer, steer, steer, steer], dtype=np.float32)
    if pattern == "front_only":
        return np.asarray([steer, steer, 0.0, 0.0], dtype=np.float32)
    if pattern == "rear_only":
        return np.asarray([0.0, 0.0, -steer, -steer], dtype=np.float32)
    raise ValueError(f"Unsupported steering pattern: {pattern}")


def _resolve_wheel_velocities(base_interface: dict, steering_pattern: str, steer_angle: float, wheel_velocity: float, mode: str):
    if mode == "uniform":
        return np.full(4, float(wheel_velocity), dtype=np.float32)

    if mode != "ackermann":
        raise ValueError(f"Unsupported wheel speed mode: {mode}")

    base_cfg = dict(base_interface.get("base_cfg", {}))
    wheel_base = float(base_cfg["wheel_base"])
    track_width = float(base_cfg["track_width"])
    wheel_radius = float(base_cfg["wheel_radius"])
    linear_speed = float(wheel_velocity) * wheel_radius
    steer = float(steer_angle)

    if abs(steer) <= 1.0e-8 or abs(linear_speed) <= 1.0e-8:
        return np.full(4, float(wheel_velocity), dtype=np.float32)

    if steering_pattern == "front_only":
        axle_offsets = (
            ("fl", wheel_base, 0.5 * track_width),
            ("fr", wheel_base, -0.5 * track_width),
            ("rl", 0.0, 0.5 * track_width),
            ("rr", 0.0, -0.5 * track_width),
        )
        turn_radius = wheel_base / math.tan(abs(steer))
    elif steering_pattern == "rear_only":
        axle_offsets = (
            ("fl", 0.0, 0.5 * track_width),
            ("fr", 0.0, -0.5 * track_width),
            ("rl", wheel_base, 0.5 * track_width),
            ("rr", wheel_base, -0.5 * track_width),
        )
        turn_radius = wheel_base / math.tan(abs(steer))
    else:
        raise ValueError("ackermann wheel speed mode only supports front_only or rear_only steering patterns")

    speeds = []
    for _, wheel_x, wheel_y in axle_offsets:
        path_radius = math.hypot(wheel_x, turn_radius - wheel_y)
        wheel_linear_speed = abs(linear_speed) * path_radius / max(turn_radius, 1.0e-8)
        wheel_angular_speed = math.copysign(wheel_linear_speed / wheel_radius, wheel_velocity)
        speeds.append(float(wheel_angular_speed))
    return np.asarray(speeds, dtype=np.float32)


def run_probe(config_path: str, output_path: str):
    report = {"config_path": config_path, "status": "started"}
    try:
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
        dof_names = list(robot._articulation_view.dof_names)
        base_interface = robot.get_base_interface()

        report = {
            "robot_name": robot.name,
            "dof_names": dof_names,
            "base_interface": base_interface,
            "config_path": config_path,
            "status": "completed",
        }
    except Exception as exc:  # pylint: disable=broad-except
        report["status"] = "error"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(json.dumps(report, indent=2))
        simulation_app.close()


def run_steering_probe(
    config_path: str,
    output_path: str,
    steering_pattern: str,
    wheel_velocity: float,
    wheel_speed_mode: str,
    steer_angle: float,
    steps: int,
    settle_steps: int,
):
    report = {
        "config_path": config_path,
        "status": "started",
        "steering_pattern": steering_pattern,
        "steps": int(steps),
        "settle_steps": int(settle_steps),
    }
    try:
        init_env()
        config = _load_render_config(config_path)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        task_cfg_path = _prepare_probe_task_cfg(scene_loader_cfg["cfg_path"], os.path.dirname(output_path) or ".")
        simulator_cfg = scene_loader_cfg["simulator"]

        import_extensions(workflow_type)
        world = _build_world(simulator_cfg)
        workflow = create_workflow(workflow_type, world, task_cfg_path)
        workflow.init_task(0)
        workflow._destroy_nav2_navigators()
        workflow._destroy_ros_base_bridges()

        robot = _find_split_aloha(workflow)
        steering_positions = _steering_pattern(steering_pattern, steer_angle)
        base_interface = robot.get_base_interface()
        wheel_velocities = _resolve_wheel_velocities(
            base_interface=base_interface,
            steering_pattern=steering_pattern,
            steer_angle=steer_angle,
            wheel_velocity=wheel_velocity,
            mode=wheel_speed_mode,
        )

        start_translation, start_orientation = robot.get_mobile_base_pose()
        start_xy = np.asarray(start_translation[:2], dtype=np.float64)
        start_yaw = float(_yaw_from_wxyz(start_orientation))
        trajectory = []

        zero_steer = np.zeros(4, dtype=np.float32)
        zero_wheel = np.zeros(4, dtype=np.float32)
        for _ in range(max(int(settle_steps), 0)):
            robot.apply_base_command(zero_steer, zero_wheel)
            workflow._step_world(render=False)

        for step_idx in range(max(int(steps), 0)):
            robot.apply_base_command(steering_positions, wheel_velocities)
            workflow._step_world(render=False)
            pose, quat = robot.get_mobile_base_pose()
            xy = np.asarray(pose[:2], dtype=np.float64)
            yaw = float(_yaw_from_wxyz(quat))
            joint_state = robot.get_base_joint_state()
            trajectory.append(
                {
                    "step": int(step_idx),
                    "world_xy": [float(xy[0]), float(xy[1])],
                    "yaw": float(yaw),
                    "steering_positions": [float(v) for v in joint_state["steering_positions"]],
                    "wheel_velocities": [float(v) for v in joint_state["wheel_velocities"]],
                }
            )

        robot.apply_base_command(zero_steer, zero_wheel)
        for _ in range(6):
            workflow._step_world(render=False)

        final_translation, final_orientation = robot.get_mobile_base_pose()
        final_xy = np.asarray(final_translation[:2], dtype=np.float64)
        final_yaw = float(_yaw_from_wxyz(final_orientation))
        delta_world = final_xy - start_xy
        delta_local = _world_delta_to_local(delta_world, start_yaw)
        yaw_delta = float(_wrap_angle(final_yaw - start_yaw))
        planar_distance = float(np.linalg.norm(delta_world))
        signed_forward_distance = float(delta_local[0])
        effective_radius = float(abs(signed_forward_distance / yaw_delta)) if abs(yaw_delta) > 1.0e-4 else None
        max_abs_wheel_velocity = max(
            (abs(float(v)) for sample in trajectory for v in sample["wheel_velocities"]),
            default=0.0,
        )
        max_abs_steering_position = max(
            (abs(float(v)) for sample in trajectory for v in sample["steering_positions"]),
            default=0.0,
        )

        report = {
            "config_path": config_path,
            "prepared_task_cfg_path": task_cfg_path,
            "status": "completed",
            "steering_pattern": steering_pattern,
            "wheel_speed_mode": wheel_speed_mode,
            "commanded_steering_positions": [float(v) for v in steering_positions],
            "commanded_wheel_velocities": [float(v) for v in wheel_velocities],
            "steps": int(steps),
            "settle_steps": int(settle_steps),
            "start_world_xy": [float(start_xy[0]), float(start_xy[1])],
            "start_yaw": float(start_yaw),
            "final_world_xy": [float(final_xy[0]), float(final_xy[1])],
            "final_yaw": float(final_yaw),
            "delta_world_xy": [float(delta_world[0]), float(delta_world[1])],
            "delta_local_xy": [float(delta_local[0]), float(delta_local[1])],
            "yaw_delta": float(yaw_delta),
            "planar_distance": float(planar_distance),
            "effective_turn_radius_m": effective_radius,
            "max_abs_wheel_velocity": float(max_abs_wheel_velocity),
            "max_abs_steering_position": float(max_abs_steering_position),
            "trajectory": trajectory,
        }

    except Exception as exc:  # pylint: disable=broad-except
        report["status"] = "error"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(json.dumps(report, indent=2))
        simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="base-info",
        choices=["base-info", "steering-probe"],
        help="Which SplitAloha base probe to run.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/simbox/de_render_template.yaml",
        help="Path to the SimBox render config used to load the task",
    )
    parser.add_argument(
        "--output-path",
        default="output/ros_bridge/split_aloha_base_probe.json",
        help="Where to save the probe output JSON",
    )
    parser.add_argument(
        "--steering-pattern",
        default="counter_phase",
        choices=["counter_phase", "inverted_counter_phase", "all_same", "front_only", "rear_only"],
        help="Steering joint pattern to probe in steering-probe mode.",
    )
    parser.add_argument(
        "--wheel-velocity",
        type=float,
        default=4.0,
        help="Wheel angular velocity command used by steering-probe mode.",
    )
    parser.add_argument(
        "--wheel-speed-mode",
        default="uniform",
        choices=["uniform", "ackermann"],
        help="How steering-probe assigns per-wheel angular velocities.",
    )
    parser.add_argument(
        "--steer-angle",
        type=float,
        default=0.45,
        help="Steering angle magnitude in radians used by steering-probe mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=180,
        help="Number of sim steps for steering-probe mode.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=20,
        help="Initial zero-command settle steps before steering-probe mode starts.",
    )
    args = parser.parse_args()
    if args.mode == "steering-probe":
        run_steering_probe(
            config_path=args.config_path,
            output_path=args.output_path,
            steering_pattern=args.steering_pattern,
            wheel_velocity=args.wheel_velocity,
            wheel_speed_mode=args.wheel_speed_mode,
            steer_angle=args.steer_angle,
            steps=args.steps,
            settle_steps=args.settle_steps,
        )
    else:
        run_probe(config_path=args.config_path, output_path=args.output_path)
