"""Nav2 detour smoke runner for SplitAloha mobile base.

This runner initializes the normal SimBox workflow, sends one Nav2 goal that is
selected to have blocked line-of-sight but clear goal area, then steps the world
for a short horizon and saves a JSON report.
"""

import argparse
from datetime import datetime
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
import omni.physx as physx  # pylint: disable=wrong-import-position  # type: ignore[import-not-found]

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


def _yaw_from_wxyz(q_wxyz) -> float:
    w = float(q_wxyz[0])
    x = float(q_wxyz[1])
    y = float(q_wxyz[2])
    z = float(q_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _resolve_hit_distance(scene_query, origin, direction, max_range: float, ignore_prefixes: list[str]) -> float:
    hit = scene_query.raycast_closest(origin, direction, max_range, False)
    if not isinstance(hit, dict) or not bool(hit.get("hit", False)):
        return float(max_range)

    collision = str(hit.get("collision", ""))
    rigid_body = str(hit.get("rigidBody", ""))
    for prefix in ignore_prefixes:
        if collision.startswith(prefix) or rigid_body.startswith(prefix):
            return float(max_range)

    distance = float(hit.get("distance", max_range))
    if not math.isfinite(distance):
        return float(max_range)
    return float(min(max(distance, 0.0), max_range))


def _is_goal_area_clear(scene_query, goal_xy: np.ndarray, height: float, clearance: float, ignore_prefixes: list[str]) -> bool:
    origin = (float(goal_xy[0]), float(goal_xy[1]), float(height))
    for i in range(24):
        angle = 2.0 * math.pi * (float(i) / 24.0)
        direction = (math.cos(angle), math.sin(angle), 0.0)
        distance = _resolve_hit_distance(scene_query, origin, direction, clearance, ignore_prefixes)
        if distance < clearance * 0.95:
            return False
    return True


def _is_direct_path_blocked(
    scene_query,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    height: float,
    ignore_prefixes: list[str],
) -> bool:
    delta = goal_xy - start_xy
    distance = float(np.linalg.norm(delta))
    if distance < 1e-4:
        return False

    direction = (float(delta[0] / distance), float(delta[1] / distance), 0.0)
    origin = (float(start_xy[0]), float(start_xy[1]), float(height))
    hit_distance = _resolve_hit_distance(scene_query, origin, direction, distance, ignore_prefixes)
    return bool(hit_distance < distance * 0.90)


def _select_detour_goal(
    scene_query,
    start_xy: np.ndarray,
    start_yaw: float,
    height: float,
    clearance: float,
    ignore_prefixes: list[str],
) -> tuple[np.ndarray, dict]:
    candidates = [
        (1.6, 1.4),
        (1.8, 1.2),
        (1.4, 1.6),
        (1.6, -1.4),
        (1.8, -1.2),
        (1.2, 1.8),
        (1.2, -1.8),
        (2.0, 0.9),
        (2.0, -0.9),
    ]

    cos_yaw = math.cos(start_yaw)
    sin_yaw = math.sin(start_yaw)

    for dx_local, dy_local in candidates:
        dx_world = dx_local * cos_yaw - dy_local * sin_yaw
        dy_world = dx_local * sin_yaw + dy_local * cos_yaw
        candidate = np.asarray([start_xy[0] + dx_world, start_xy[1] + dy_world], dtype=np.float32)

        clear = _is_goal_area_clear(scene_query, candidate, height, clearance, ignore_prefixes)
        if not clear:
            continue

        blocked = _is_direct_path_blocked(scene_query, start_xy, candidate, height, ignore_prefixes)
        if not blocked:
            continue

        return candidate, {
            "candidate_local_offset": [float(dx_local), float(dy_local)],
            "line_of_sight_blocked": True,
            "goal_clearance_m": float(clearance),
        }

    fallback = np.asarray([start_xy[0] + 1.6, start_xy[1] + 1.2], dtype=np.float32)
    return fallback, {
        "candidate_local_offset": [1.6, 1.2],
        "line_of_sight_blocked": False,
        "goal_clearance_m": float(clearance),
        "fallback_used": True,
    }


def run_nav2_detour_smoke(config_path: str, output_path: str, steps: int, goal_clearance: float):
    report: dict = {
        "started_at": datetime.now().isoformat(),
        "status": "started",
        "config_path": config_path,
        "requested_steps": int(steps),
        "goal_clearance": float(goal_clearance),
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        report["status"] = "initializing_environment"
        init_env()
        config = _load_render_config(config_path)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        task_cfg_path = scene_loader_cfg["cfg_path"]
        simulator_cfg = scene_loader_cfg["simulator"]
        report["task_cfg_path"] = task_cfg_path

        import_extensions(workflow_type)
        world = _build_world(simulator_cfg)
        workflow = create_workflow(workflow_type, world, task_cfg_path)

        report["status"] = "initializing_task"
        workflow.init_task(0)

        robot = _find_split_aloha(workflow)
        navigator = workflow._nav2_navigators.get(robot.name)
        if navigator is None:
            raise RuntimeError("Nav2 navigator is not initialized, check split_aloha ros.nav2.enabled")
        report["nav2_enabled"] = True
        report["nav2_action_client_available"] = bool(getattr(navigator, "action_client_available", False))

        start_world_pose = robot.get_world_pose()
        start_xy = np.asarray(start_world_pose[0][:2], dtype=np.float32)
        start_yaw = _yaw_from_wxyz(start_world_pose[1])
        report["start_world_xy"] = [float(start_xy[0]), float(start_xy[1])]
        report["start_world_yaw"] = float(start_yaw)

        scene_query = physx.get_physx_scene_query_interface()
        ignore_prefixes = [str(getattr(robot, "robot_prim_path", ""))]
        ignore_prefixes = [prefix for prefix in ignore_prefixes if prefix]

        goal_xy, goal_meta = _select_detour_goal(
            scene_query=scene_query,
            start_xy=start_xy,
            start_yaw=start_yaw,
            height=0.25,
            clearance=max(float(goal_clearance), 0.10),
            ignore_prefixes=ignore_prefixes,
        )
        goal_yaw = math.atan2(float(goal_xy[1] - start_xy[1]), float(goal_xy[0] - start_xy[0]))
        report["goal_world_xy"] = [float(goal_xy[0]), float(goal_xy[1])]
        report["goal_world_yaw"] = float(goal_yaw)
        report["detour_goal_selection"] = goal_meta

        navigator.publish_initial_pose_from_robot()
        navigator.send_goal(float(goal_xy[0]), float(goal_xy[1]), float(goal_yaw))
        report["status"] = "goal_sent"

        path_xy = []
        nav2_status = None
        reached = False

        for step_idx in range(max(int(steps), 0)):
            workflow._step_world(render=False)

            pose = robot.get_world_pose()
            current_xy = np.asarray(pose[0][:2], dtype=np.float32)
            path_xy.append((float(current_xy[0]), float(current_xy[1])))

            nav2_status = navigator.latest_result_status
            if nav2_status == 4:
                reached = True
                break

        final_world_pose = robot.get_world_pose()
        final_xy = np.asarray(final_world_pose[0][:2], dtype=np.float32)
        final_dist_to_goal = float(np.linalg.norm(goal_xy - final_xy))
        world_planar_displacement = float(np.linalg.norm(final_xy - start_xy))

        if final_dist_to_goal <= 0.20:
            reached = True

        report["status"] = "completed"
        report["result"] = {
            "reached_target": bool(reached),
            "nav2_result_status": nav2_status,
            "final_distance_to_goal": final_dist_to_goal,
            "final_world_xy": [float(final_xy[0]), float(final_xy[1])],
            "world_planar_displacement": world_planar_displacement,
            "trajectory_point_count": len(path_xy),
        }
    except Exception as exc:  # pylint: disable=broad-except
        report["status"] = "error"
        report["error"] = str(exc)
    finally:
        report["finished_at"] = datetime.now().isoformat()
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(json.dumps(report, indent=2))
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
        default="output/ros_bridge/split_aloha_nav2_detour_smoke.json",
        help="Where to save the Nav2 detour smoke report JSON",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="How many world steps to execute after sending Nav2 goal",
    )
    parser.add_argument(
        "--goal-clearance",
        type=float,
        default=0.45,
        help="Required free-space radius around the detour goal (meters)",
    )
    args = parser.parse_args()

    run_nav2_detour_smoke(
        config_path=args.config_path,
        output_path=args.output_path,
        steps=args.steps,
        goal_clearance=args.goal_clearance,
    )
