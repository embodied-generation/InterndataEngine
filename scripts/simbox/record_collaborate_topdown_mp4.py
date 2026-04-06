#!/usr/bin/env python3
"""Record a top-down MP4 from collaborate arena while driving SplitAloha via ROS.

This script loads a real collaborate task, patches missing robot_config_file entries in-memory,
adds a debug camera at scene center (0,0,5m) facing downward, and records a moving MP4.
Motion profile: either local cmd_vel feedback control or Nav2 goal navigation.
"""

import argparse
from datetime import datetime
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from isaacsim import SimulationApp  # type: ignore[import-not-found]

_runner_args = sys.argv[1:]
sys.argv = [sys.argv[0]]
simulation_app = SimulationApp({"headless": True})
sys.argv = [sys.argv[0], *_runner_args]

sys.path.append("./")
sys.path.append("./data_engine")
sys.path.append("workflows/simbox")

from geometry_msgs.msg import Twist  # pylint: disable=wrong-import-position
from omni.isaac.core import World  # pylint: disable=wrong-import-position  # type: ignore[import-not-found]
import omni.physx as physx  # pylint: disable=wrong-import-position  # type: ignore[import-not-found]
from sensor_msgs.msg import LaserScan  # pylint: disable=wrong-import-position

from nimbus.utils.utils import init_env  # pylint: disable=wrong-import-position
from workflows import import_extensions  # pylint: disable=wrong-import-position
from workflows.base import create_workflow  # pylint: disable=wrong-import-position


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _dump_yaml(path: str, data):
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def _normalize_rgb(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.dtype == np.uint8:
        return image

    image = np.clip(image, 0.0, 255.0)
    if image.max() <= 1.0:
        image = image * 255.0
    return image.astype(np.uint8)


def _write_video(path: str, frames: list[np.ndarray], fps: int) -> str:
    try:
        import imageio.v2 as iio  # type: ignore[import-not-found]

        iio.mimwrite(path, frames, fps=fps, quality=8)
        return "imageio"
    except Exception:
        import cv2  # type: ignore[import-not-found]

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return "cv2"


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_wxyz(q_wxyz) -> float:
    w = float(q_wxyz[0])
    x = float(q_wxyz[1])
    y = float(q_wxyz[2])
    z = float(q_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _write_path_plot(path: str, trajectory_xy: list[tuple[float, float]], target_xy: tuple[float, float]) -> str:
    if not trajectory_xy:
        raise ValueError("trajectory_xy is empty")

    start_xy = trajectory_xy[0]
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        xs = [p[0] for p in trajectory_xy]
        ys = [p[1] for p in trajectory_xy]

        plt.figure(figsize=(6, 6))
        plt.plot(xs, ys, "-", linewidth=2.0, label="trajectory")
        plt.scatter([start_xy[0]], [start_xy[1]], c="green", s=70, label="start")
        plt.scatter([target_xy[0]], [target_xy[1]], c="red", s=70, marker="x", label="target")  # type: ignore[arg-type]
        plt.xlabel("virtual x (m)")
        plt.ylabel("virtual y (m)")
        plt.title("Point Navigation Path")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return "matplotlib"
    except Exception:
        import cv2  # type: ignore[import-not-found]

        xs = [p[0] for p in trajectory_xy] + [target_xy[0], start_xy[0]]
        ys = [p[1] for p in trajectory_xy] + [target_xy[1], start_xy[1]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        margin = 0.2
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin

        width, height = 800, 800
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)

        def _pix(pt):
            px = int((pt[0] - min_x) / span_x * (width - 1))
            py = int((1.0 - (pt[1] - min_y) / span_y) * (height - 1))
            return (max(0, min(width - 1, px)), max(0, min(height - 1, py)))

        poly = np.array([_pix(pt) for pt in trajectory_xy], dtype=np.int32)
        cv2.polylines(canvas, [poly], False, (40, 40, 200), 2)
        cv2.circle(canvas, _pix(start_xy), 6, (0, 170, 0), -1)
        cv2.drawMarker(canvas, _pix(target_xy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(canvas, "start", (_pix(start_xy)[0] + 8, _pix(start_xy)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)
        cv2.putText(canvas, "target", (_pix(target_xy)[0] + 8, _pix(target_xy)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1)
        cv2.imwrite(path, canvas)
        return "cv2"


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
    # Candidate offsets are ordered to prefer substantial lateral movement first.
    base_candidates = [
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

    for dx_local, dy_local in base_candidates:
        dx_world = dx_local * cos_yaw - dy_local * sin_yaw
        dy_world = dx_local * sin_yaw + dy_local * cos_yaw
        candidate = np.asarray([start_xy[0] + dx_world, start_xy[1] + dy_world], dtype=np.float32)

        clear = _is_goal_area_clear(scene_query, candidate, height, clearance, ignore_prefixes)
        if not clear:
            continue

        blocked = _is_direct_path_blocked(scene_query, start_xy, candidate, height, ignore_prefixes)
        if not blocked:
            continue

        meta = {
            "candidate_local_offset": [float(dx_local), float(dy_local)],
            "line_of_sight_blocked": True,
            "goal_clearance_m": float(clearance),
        }
        return candidate, meta

    # Fallback: choose a far clear point even if direct line is not blocked.
    fallback = np.asarray([start_xy[0] + 1.6, start_xy[1] + 1.2], dtype=np.float32)
    meta = {
        "candidate_local_offset": [1.6, 1.2],
        "line_of_sight_blocked": False,
        "goal_clearance_m": float(clearance),
        "fallback_used": True,
    }
    return fallback, meta


def _build_temp_task_cfg(task_cfg_source: str, split_aloha_usd_relpath: str) -> str:
    task_cfg = _load_yaml(task_cfg_source)

    robots = task_cfg["tasks"][0]["robots"]
    for robot_cfg in robots:
        if "robot_config_file" not in robot_cfg:
            robot_name = str(robot_cfg.get("name", "")).lower()
            if "split_aloha" in robot_name:
                robot_cfg["robot_config_file"] = "workflows/simbox/core/configs/robots/split_aloha.yaml"
            elif "lift2" in robot_name:
                robot_cfg["robot_config_file"] = "workflows/simbox/core/configs/robots/lift2.yaml"
            else:
                raise KeyError(f"Missing robot_config_file for unsupported robot: {robot_cfg.get('name')}")

        if str(robot_cfg.get("name", "")).lower() == "split_aloha" and split_aloha_usd_relpath:
            robot_cfg["path"] = split_aloha_usd_relpath

    temp_fd, temp_path = tempfile.mkstemp(prefix="collab_topdown_video_", suffix=".yaml")
    os.close(temp_fd)
    _dump_yaml(temp_path, task_cfg)
    return temp_path


def _find_split_aloha(workflow):
    for robot in workflow.task.robots.values():
        if robot.__class__.__name__ == "SplitAloha":
            return robot
    raise RuntimeError("SplitAloha robot not found in loaded scene")


class _ScanRecorder:
    def __init__(self, node, topic: str):
        self.topic = str(topic)
        self._records: list[dict] = []
        self._sub = node.create_subscription(LaserScan, self.topic, self._on_scan, 10)

    @property
    def records(self):
        return self._records

    def _on_scan(self, msg: LaserScan):
        stamp_sec = int(msg.header.stamp.sec)
        stamp_nanosec = int(msg.header.stamp.nanosec)
        ranges = [float(value) for value in msg.ranges]
        self._records.append(
            {
                "stamp_sec": stamp_sec,
                "stamp_nanosec": stamp_nanosec,
                "frame_id": str(msg.header.frame_id),
                "angle_min": float(msg.angle_min),
                "angle_max": float(msg.angle_max),
                "angle_increment": float(msg.angle_increment),
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
                "ranges": ranges,
            }
        )


def _save_scan_records(scan_records: list[dict], jsonl_path: Path, npz_path: Path):
    with open(jsonl_path, "w", encoding="utf-8") as file:
        for record in scan_records:
            file.write(json.dumps(record))
            file.write("\n")

    if not scan_records:
        np.savez_compressed(
            npz_path,
            stamp_sec=np.empty((0,), dtype=np.int64),
            stamp_nanosec=np.empty((0,), dtype=np.int64),
            ranges=np.empty((0, 0), dtype=np.float32),
        )
        return

    beam_count = min(len(record.get("ranges", [])) for record in scan_records)
    beam_count = max(beam_count, 0)

    stamp_sec = np.asarray([int(record["stamp_sec"]) for record in scan_records], dtype=np.int64)
    stamp_nanosec = np.asarray([int(record["stamp_nanosec"]) for record in scan_records], dtype=np.int64)

    if beam_count == 0:
        ranges = np.empty((len(scan_records), 0), dtype=np.float32)
    else:
        ranges = np.asarray(
            [[float(value) for value in record["ranges"][:beam_count]] for record in scan_records],
            dtype=np.float32,
        )

    np.savez_compressed(npz_path, stamp_sec=stamp_sec, stamp_nanosec=stamp_nanosec, ranges=ranges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="configs/simbox/de_render_template.yaml",
        help="SimBox render config path",
    )
    parser.add_argument(
        "--task-cfg-source",
        default="workflows/simbox/core/configs/tasks/long_horizon/split_aloha/collaborate_assemble_a_beef_sandwich_mix/collaborate_assemble_a_beef_sandwich_mix.yaml",
        help="Collaborate task config source path",
    )
    parser.add_argument(
        "--split-aloha-usd-relpath",
        default="../../../InternDataAssets/assets/split_aloha_mid_360/robot.usd",
        help="SplitAloha USD path relative to task asset_root",
    )
    parser.add_argument(
        "--output-path",
        default="output/camera_probe/collaborate_topdown_global.mp4",
        help="Output MP4 path",
    )
    parser.add_argument(
        "--output-root",
        default="output/camera_probe",
        help="Root directory for auto-created run folders",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Explicit run directory; if empty create <output-root>/<YYYYMMDD_HHMMSS>_<video-stem>",
    )
    parser.add_argument("--warmup-steps", type=int, default=40, help="Warmup steps before recording")
    parser.add_argument("--fps", type=int, default=20, help="Video fps")
    parser.add_argument("--camera-height", type=float, default=5.0, help="Top-down camera height (m)")
    parser.add_argument("--target-x", type=float, default=0.6, help="Target x in world frame (meters)")
    parser.add_argument("--target-y", type=float, default=0.0, help="Target y in world frame (meters)")
    parser.add_argument("--target-yaw", type=float, default=0.0, help="Target yaw in world frame (rad)")
    parser.add_argument(
        "--target-frame",
        choices=["relative", "absolute"],
        default="relative",
        help="Interpret target as offset from start world pose or absolute world coordinate",
    )
    parser.add_argument(
        "--require-yaw",
        action="store_true",
        help="Require target yaw alignment for success (off by default for pure point navigation)",
    )
    parser.add_argument("--max-steps", type=int, default=700, help="Max navigation steps")
    parser.add_argument("--linear-speed-limit", type=float, default=0.45, help="Linear speed limit (m/s)")
    parser.add_argument("--angular-speed-limit", type=float, default=1.2, help="Angular speed limit (rad/s)")
    parser.add_argument("--linear-kp", type=float, default=1.0, help="P gain for distance error")
    parser.add_argument("--angular-kp", type=float, default=2.0, help="P gain for heading/yaw error")
    parser.add_argument("--position-tolerance", type=float, default=0.03, help="Target position tolerance (m)")
    parser.add_argument("--yaw-tolerance", type=float, default=0.08, help="Target yaw tolerance (rad)")
    parser.add_argument("--arrival-hold-steps", type=int, default=30, help="Post-arrival hold steps for video tail")
    parser.add_argument(
        "--path-plot-path",
        default="",
        help="Output path figure path (default: <output-path stem>_path.png)",
    )
    parser.add_argument(
        "--use-nav2",
        action="store_true",
        help="Use Nav2 action navigation instead of local cmd_vel feedback control",
    )
    parser.add_argument(
        "--auto-detour-goal",
        action="store_true",
        help="Auto-pick a detour goal with blocked straight line and clear goal area",
    )
    parser.add_argument(
        "--detour-goal-clearance",
        type=float,
        default=0.45,
        help="Required free-space radius around auto-selected detour goal (m)",
    )
    parser.add_argument(
        "--nav2-success-status",
        type=int,
        default=4,
        help="Nav2 action status code treated as success (default: 4 = SUCCEEDED)",
    )
    args = parser.parse_args()

    temp_task_cfg = None
    bridge = None
    try:
        temp_task_cfg = _build_temp_task_cfg(args.task_cfg_source, args.split_aloha_usd_relpath)

        render_cfg = _load_yaml(args.config_path)
        scene_args = render_cfg["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_args["workflow_type"]
        simulator_cfg = scene_args["simulator"]
        physics_dt = float(eval(str(simulator_cfg["physics_dt"])))

        init_env()
        import_extensions(workflow_type)
        world = World(
            physics_dt=physics_dt,
            rendering_dt=eval(str(simulator_cfg["rendering_dt"])),
            stage_units_in_meters=float(simulator_cfg["stage_units_in_meters"]),
        )
        workflow = create_workflow(workflow_type, world, temp_task_cfg)
        workflow.init_task(0)

        split_aloha = _find_split_aloha(workflow)

        # Reuse workflow-managed ROS base modules to avoid duplicate nodes.
        bridge = workflow._ros_base_bridges.get(split_aloha.name)
        controller = workflow._ros_base_command_controllers.get(split_aloha.name)
        if bridge is None or controller is None:
            raise RuntimeError("Workflow ROS base bridge/controller is not initialized for SplitAloha")

        # Camera at scene center, 5m above floor, looking straight down in USD axes.
        debug_camera_cfg = {
            "name": "topdown_global_debug_camera",
            "translation": [0.0, 0.0, float(args.camera_height)],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "camera_axes": "usd",
            "camera_file": "workflows/simbox/core/configs/cameras/realsense_d455_v3.yaml",
            "parent": None,
            "apply_randomization": False,
        }
        workflow.task._load_camera(debug_camera_cfg)
        camera = workflow.task.cameras["topdown_global_debug_camera"]

        command_pub = bridge.node.create_publisher(Twist, bridge.ros_cfg["cmd_vel_topic"], 10)
        scan_topic = str(bridge.ros_cfg.get("scan_topic", "/scan"))
        scan_recorder = _ScanRecorder(bridge.node, scan_topic)

        navigator = None
        if bool(args.use_nav2):
            navigator = workflow._nav2_navigators.get(split_aloha.name)
            if navigator is None:
                raise RuntimeError(
                    "Nav2 is enabled by --use-nav2 but workflow has no Nav2 navigator for SplitAloha"
                )

        output_filename = Path(args.output_path).name
        output_root = Path(args.output_root)
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_root / f"{run_tag}_{Path(output_filename).stem}"
        run_dir.mkdir(parents=True, exist_ok=True)

        out_path = run_dir / output_filename
        if args.path_plot_path:
            path_plot_path = run_dir / Path(args.path_plot_path).name
        else:
            path_plot_path = run_dir / f"{out_path.stem}_path.png"
        scan_jsonl_path = run_dir / f"{out_path.stem}_scan.jsonl"
        scan_npz_path = run_dir / f"{out_path.stem}_scan.npz"

        # Warmup for stable rendering and ROS callback sync.
        for _ in range(max(int(args.warmup_steps), 0)):
            workflow._step_world(render=True)

        start_world_pos, _ = split_aloha.get_world_pose()
        start_world_xy = np.asarray(start_world_pos[:2], dtype=np.float32)
        start_world_pose = split_aloha.get_world_pose()
        start_world_yaw = _yaw_from_wxyz(start_world_pose[1])

        raw_target_xy = np.asarray([float(args.target_x), float(args.target_y)], dtype=np.float32)
        raw_target_yaw = float(args.target_yaw)
        target_frame = str(args.target_frame)

        if target_frame == "relative":
            target_world_xy = start_world_xy + raw_target_xy
            target_world_yaw = _wrap_angle(start_world_yaw + raw_target_yaw)
        else:
            target_world_xy = raw_target_xy.copy()
            target_world_yaw = _wrap_angle(raw_target_yaw)

        target_xy = target_world_xy
        target_yaw = target_world_yaw

        detour_goal_meta = None
        if bool(args.auto_detour_goal):
            scene_query = physx.get_physx_scene_query_interface()
            ignore_prefixes = [str(getattr(split_aloha, "robot_prim_path", ""))]
            detour_xy, detour_goal_meta = _select_detour_goal(
                scene_query=scene_query,
                start_xy=start_world_xy,
                start_yaw=start_world_yaw,
                height=0.25,
                clearance=max(float(args.detour_goal_clearance), 0.10),
                ignore_prefixes=[prefix for prefix in ignore_prefixes if prefix],
            )
            target_world_xy = detour_xy
            target_world_yaw = math.atan2(
                float(target_world_xy[1] - start_world_xy[1]),
                float(target_world_xy[0] - start_world_xy[0]),
            )
            target_xy = target_world_xy
            target_yaw = target_world_yaw

        require_yaw = bool(args.require_yaw)
        max_steps = max(int(args.max_steps), 1)
        linear_speed_limit = max(float(args.linear_speed_limit), 0.01)
        angular_speed_limit = max(float(args.angular_speed_limit), 0.01)
        linear_kp = max(float(args.linear_kp), 0.0)
        angular_kp = max(float(args.angular_kp), 0.0)
        position_tolerance = max(float(args.position_tolerance), 1e-4)
        yaw_tolerance = max(float(args.yaw_tolerance), 1e-4)
        arrival_hold_steps = max(int(args.arrival_hold_steps), 0)

        frames = []
        trajectory_xy: list[tuple[float, float]] = []
        reached_target = False
        reached_step = -1
        nav2_result_status = None

        if bool(args.use_nav2):
            navigator.publish_initial_pose_from_robot()
            navigator.send_goal(
                x=float(target_world_xy[0]),
                y=float(target_world_xy[1]),
                yaw=float(target_world_yaw),
            )

        for step_idx in range(max_steps):
            current_world_pose = split_aloha.get_world_pose()
            current_xy = np.asarray(current_world_pose[0][:2], dtype=np.float32)
            current_yaw = _yaw_from_wxyz(current_world_pose[1])
            trajectory_xy.append((float(current_xy[0]), float(current_xy[1])))

            dx = float(target_xy[0] - current_xy[0])
            dy = float(target_xy[1] - current_xy[1])
            dist_err = math.hypot(dx, dy)

            desired_heading = math.atan2(dy, dx) if dist_err > 1e-8 else current_yaw
            heading_err = _wrap_angle(desired_heading - current_yaw)
            yaw_err = _wrap_angle(target_yaw - current_yaw)

            if bool(args.use_nav2):
                nav2_result_status = navigator.latest_result_status
                if nav2_result_status == int(args.nav2_success_status):
                    reached_target = True
                    reached_step = step_idx
                    command_pub.publish(Twist())
                    workflow._step_world(render=True)
                    frame = _normalize_rgb(camera.get_observations()["color_image"])
                    frames.append(frame)
                    break

                if dist_err <= position_tolerance and (not require_yaw or abs(yaw_err) <= yaw_tolerance):
                    reached_target = True
                    reached_step = step_idx
                    command_pub.publish(Twist())
                    workflow._step_world(render=True)
                    frame = _normalize_rgb(camera.get_observations()["color_image"])
                    frames.append(frame)
                    break
            else:
                cmd = Twist()
                if dist_err > position_tolerance:
                    linear_cmd = min(linear_speed_limit, linear_kp * dist_err)
                    heading_scale = max(0.0, math.cos(heading_err))
                    linear_cmd *= heading_scale
                    angular_cmd = max(-angular_speed_limit, min(angular_speed_limit, angular_kp * heading_err))
                    cmd.linear.x = float(linear_cmd)
                    cmd.angular.z = float(angular_cmd)
                else:
                    if not require_yaw or abs(yaw_err) <= yaw_tolerance:
                        reached_target = True
                        reached_step = step_idx
                        command_pub.publish(Twist())
                        workflow._step_world(render=True)
                        frame = _normalize_rgb(camera.get_observations()["color_image"])
                        frames.append(frame)
                        break
                    cmd.linear.x = 0.0
                    cmd.angular.z = float(max(-angular_speed_limit, min(angular_speed_limit, angular_kp * yaw_err)))

                command_pub.publish(cmd)

            workflow._step_world(render=True)
            frame = _normalize_rgb(camera.get_observations()["color_image"])
            frames.append(frame)

        # Tail frames after reaching target for clearer video ending.
        if reached_target:
            for _ in range(arrival_hold_steps):
                command_pub.publish(Twist())
                workflow._step_world(render=True)
                frame = _normalize_rgb(camera.get_observations()["color_image"])
                frames.append(frame)

        final_world_pos, _ = split_aloha.get_world_pose()
        final_world_xy = np.asarray(final_world_pos[:2], dtype=np.float32)
        final_world_yaw = _yaw_from_wxyz(split_aloha.get_world_pose()[1])

        final_world_dist_err = float(
            math.hypot(float(target_world_xy[0] - final_world_xy[0]), float(target_world_xy[1] - final_world_xy[1]))
        )
        final_world_yaw_err = float(abs(_wrap_angle(target_world_yaw - final_world_yaw)))
        final_dist_err = final_world_dist_err
        final_yaw_err = final_world_yaw_err

        if trajectory_xy:
            trajectory_xy.append((float(final_world_xy[0]), float(final_world_xy[1])))

        world_planar_displacement = float(np.linalg.norm(final_world_xy - start_world_xy))

        writer = _write_video(str(out_path), frames, int(args.fps))

        path_plot_writer = _write_path_plot(str(path_plot_path), trajectory_xy, (float(target_world_xy[0]), float(target_world_xy[1])))
        world_path_plot_path = run_dir / f"{out_path.stem}_world_path.png"
        world_path_plot_writer = _write_path_plot(
            str(world_path_plot_path), trajectory_xy, (float(target_world_xy[0]), float(target_world_xy[1]))
        )
        _save_scan_records(scan_recorder.records, scan_jsonl_path, scan_npz_path)

        report = {
            "task_cfg_source": args.task_cfg_source,
            "task_cfg_used": temp_task_cfg,
            "arena_file": _load_yaml(temp_task_cfg)["tasks"][0]["arena_file"],
            "camera_name": "topdown_global_debug_camera",
            "camera_pose": {
                "translation": debug_camera_cfg["translation"],
                "orientation": debug_camera_cfg["orientation"],
                "camera_axes": debug_camera_cfg["camera_axes"],
            },
            "frame_count": len(frames),
            "resolution": [int(frames[0].shape[0]), int(frames[0].shape[1])],
            "fps": int(args.fps),
            "navigation_profile": {
                "target_frame": target_frame,
                "raw_target_x": float(raw_target_xy[0]),
                "raw_target_y": float(raw_target_xy[1]),
                "raw_target_yaw": raw_target_yaw,
                "target_x": float(target_xy[0]),
                "target_y": float(target_xy[1]),
                "target_yaw": target_yaw,
                "target_world_x": float(target_world_xy[0]),
                "target_world_y": float(target_world_xy[1]),
                "target_world_yaw": target_world_yaw,
                "navigation_mode": "nav2" if bool(args.use_nav2) else "local_cmd_vel",
                "require_yaw": require_yaw,
                "max_steps": max_steps,
                "linear_speed_limit": linear_speed_limit,
                "angular_speed_limit": angular_speed_limit,
                "linear_kp": linear_kp,
                "angular_kp": angular_kp,
                "position_tolerance": position_tolerance,
                "yaw_tolerance": yaw_tolerance,
                "arrival_hold_steps": arrival_hold_steps,
                "physics_dt": physics_dt,
            },
            "result": {
                "reached_target": reached_target,
                "reached_step": reached_step,
                "final_distance_error": final_dist_err,
                "final_yaw_error": final_yaw_err,
                "final_world_distance_error": final_world_dist_err,
                "final_world_yaw_error": final_world_yaw_err,
                "start_world_xy": [float(start_world_xy[0]), float(start_world_xy[1])],
                "final_world_xy": [float(final_world_xy[0]), float(final_world_xy[1])],
                "world_planar_displacement": world_planar_displacement,
                "nav2_result_status": nav2_result_status,
                "trajectory_point_count": len(trajectory_xy),
            },
            "detour_goal_selection": detour_goal_meta,
            "scan_capture": {
                "topic": scan_topic,
                "message_count": len(scan_recorder.records),
                "jsonl_path": str(scan_jsonl_path),
                "npz_path": str(scan_npz_path),
            },
            "run_dir": str(run_dir),
            "path_plot_path": str(path_plot_path),
            "path_plot_writer": path_plot_writer,
            "world_path_plot_path": str(world_path_plot_path),
            "world_path_plot_writer": world_path_plot_writer,
            "video_path": str(out_path),
            "writer": writer,
        }

        report_path = out_path.with_suffix(".json")
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print(json.dumps(report, indent=2))

    finally:
        if temp_task_cfg and os.path.exists(temp_task_cfg):
            os.remove(temp_task_cfg)
        simulation_app.close()


if __name__ == "__main__":
    main()
