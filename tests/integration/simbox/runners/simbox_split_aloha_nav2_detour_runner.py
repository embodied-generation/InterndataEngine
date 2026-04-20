"""Nav2 smoke runner for the SplitAloha mobile base.

This runner initializes the normal SimBox workflow, sends one explicit Nav2
goal, steps the world for a short horizon, and saves a JSON report.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from typing import Iterable

import numpy as np
import yaml
import cv2  # noqa: F401  # Preload OpenCV before Kit mutates dynamic library resolution.
from isaacsim import SimulationApp
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock

_runner_args = sys.argv[1:]
sys.argv = [sys.argv[0]]
simulation_app = SimulationApp({"headless": True})
sys.argv = [sys.argv[0], *_runner_args]

sys.path.append("./")
sys.path.append("./data_engine")
sys.path.append("workflows/simbox")

from omni.isaac.core import World  # pylint: disable=wrong-import-position
from omni.isaac.core.utils.prims import get_prim_at_path  # pylint: disable=wrong-import-position
from omni.isaac.core.utils.transformations import (  # pylint: disable=wrong-import-position
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
import omni.physx as physx  # pylint: disable=wrong-import-position  # type: ignore[import-not-found]

from nimbus.utils.utils import init_env  # pylint: disable=wrong-import-position
from workflows import import_extensions  # pylint: disable=wrong-import-position
from workflows.base import create_workflow  # pylint: disable=wrong-import-position
from nav2.localization_stack import IsaacStaticMapExporter  # pylint: disable=wrong-import-position

FLOOR_ARENA_CFG = "workflows/simbox/core/configs/arenas/nav2_floor_arena.yaml"
NAV2_SMOKE_MAX_ACKERMANN_STEER_RAD = 0.55
NAV2_OBSTACLE_HEIGHT_M = 0.10
NAV2_OBSTACLE_CENTER_Z_M = NAV2_OBSTACLE_HEIGHT_M * 0.5
NAV2_CAMERA_RENDER_STRIDE = 2
NAV2_CAMERA_WARMUP_RENDER_STEPS = 2
NAV2_SUCCESS_POSITION_TOLERANCE_M = 0.10
NAV2_SUCCESS_YAW_TOLERANCE_RAD = 0.10
NAV2_DEFAULT_GOAL_X_M = -1.09915
NAV2_DEFAULT_GOAL_Y_M = 4.09205
NAV2_DEFAULT_GOAL_YAW_RAD = 1.83280
SPLIT_ALOHA_NAV2_FOOTPRINT_POINTS = [
    [0.36, 0.24],
    [0.32, 0.29],
    [-0.32, 0.29],
    [-0.36, 0.24],
    [-0.36, -0.24],
    [-0.32, -0.29],
    [0.32, -0.29],
    [0.36, -0.24],
]
SPLIT_ALOHA_NAV2_INFLATION_RADIUS_M = 0.18
SPLIT_ALOHA_NAV2_MIN_TURN_RADIUS_M = 0.95
LOCALIZATION_SCENE_NAME = "nav2_floor_arena"


def _strip_env_paths(value: str, disallowed_fragments: Iterable[str]) -> str:
    fragments = tuple(fragment for fragment in disallowed_fragments if fragment)
    parts = []
    for part in value.split(":"):
        if not part:
            continue
        if any(fragment in part for fragment in fragments):
            continue
        parts.append(part)
    return ":".join(parts)


def _build_clean_ros2_launch_env() -> dict[str, str]:
    env = dict(os.environ)
    isaac_fragments = (
        "/workspace/isaac-sim",
        "/isaac-sim",
        "omni.isaac.ros2_bridge/humble",
    )
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
        current_value = str(env.get(key, ""))
        cleaned = _strip_env_paths(current_value, isaac_fragments)
        if cleaned:
            env[key] = cleaned
        else:
            env.pop(key, None)

    env.pop("ROS_DISTRO", None)
    env.pop("ROS_VERSION", None)
    env.pop("ROS_PYTHON_VERSION", None)
    return env


def _load_render_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _format_nav2_footprint(points: list[list[float]]) -> str:
    return "[" + ", ".join(f"[{float(x):.3f}, {float(y):.3f}]" for x, y in points) + "]"


def _build_nav2_obstacle_layout():
    obstacle_specs = [
        {
            "name": "nav_wall_primary",
            "translation": [0.05, 1.95, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [1.55, 0.42, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.77, 0.29, 0.20],
        },
        {
            "name": "nav_right_gate_lower",
            "translation": [1.25, 1.60, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.60, 0.82, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.86, 0.76, 0.27],
        },
        {
            "name": "nav_right_gate_upper",
            "translation": [1.35, 2.77, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.82, 1.10, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.24, 0.67, 0.44],
        },
        {
            "name": "nav_left_mid_block",
            "translation": [-2.25, 2.47, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.62, 0.88, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.20, 0.49, 0.76],
        },
        {
            "name": "nav_left_far_block",
            "translation": [-2.15, 3.57, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.52, 0.64, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.28, 0.70, 0.74],
        },
        {
            "name": "nav_center_far_block",
            "translation": [0.05, 3.47, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [1.02, 0.46, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.35, 0.58, 0.88],
        },
        {
            "name": "nav_right_far_pillar",
            "translation": [1.18, 3.67, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.52, 0.58, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.89, 0.55, 0.22],
        },
        {
            "name": "nav_right_far_wall",
            "translation": [1.85, 3.50, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.60, 0.90, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.62, 0.36, 0.79],
        },
        {
            "name": "nav_right_mid_pillar",
            "translation": [0.78, 2.28, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.46, 0.60, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.76, 0.48, 0.18],
        },
        {
            "name": "nav_upper_mid_pillar",
            "translation": [-0.18, 3.96, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.38, 0.52, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.51, 0.42, 0.82],
        },
        {
            "name": "nav_upper_right_bar",
            "translation": [0.88, 4.18, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.96, 0.34, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.30, 0.63, 0.79],
        },
        {
            "name": "nav_far_left_bumper",
            "translation": [-2.95, 3.18, NAV2_OBSTACLE_CENTER_Z_M],
            "scale": [0.44, 0.58, NAV2_OBSTACLE_HEIGHT_M],
            "color": [0.70, 0.38, 0.33],
        },
    ]

    objects = []
    for spec in obstacle_specs:
        objects.append(
            {
                "name": spec["name"],
                "target_class": "BoxObject",
                "translation": [float(v) for v in spec["translation"]],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "scale": [float(v) for v in spec["scale"]],
                "visible": True,
                "collision_enabled": True,
                "color": [float(v) for v in spec["color"]],
            }
        )

    return obstacle_specs, objects


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


def _select_nav_cameras(task_cfg: dict, robot_name: str) -> list[dict]:
    cameras = task_cfg.get("cameras", [])
    if not cameras:
        return []

    filtered_cameras = []
    robot_name_lower = robot_name.lower()
    for camera in cameras:
        name = str(camera.get("name", "")).lower()
        parent = str(camera.get("parent", "")).lower()
        if robot_name_lower in name or robot_name_lower in parent:
            filtered_cameras.append(dict(camera))

    if filtered_cameras:
        return filtered_cameras
    return [dict(cameras[0])]


def _build_nav2_floor_task(
    task_cfg: dict,
    keep_task_cameras: bool,
    topdown_camera: bool,
    robot_config_file_override: str = "",
) -> dict:
    nav_robot = _select_nav_robot(task_cfg)
    if robot_config_file_override:
        nav_robot["robot_config_file"] = robot_config_file_override
    nav_robot_name = str(nav_robot["name"])
    obstacle_specs, obstacle_objects = _build_nav2_obstacle_layout()

    nav_task = dict(task_cfg)
    nav_task["arena_file"] = FLOOR_ARENA_CFG
    nav_task["robots"] = [nav_robot]
    nav_task["objects"] = obstacle_objects
    nav_task["skills"] = []
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
    nav_task["nav2_obstacles"] = obstacle_specs

    if keep_task_cameras:
        nav_task["render"] = True
        source_cameras = _select_nav_cameras(task_cfg, nav_robot_name)
        if topdown_camera:
            if not source_cameras:
                raise ValueError("Cannot synthesize topdown camera because task has no source camera config")
            source_camera = dict(source_cameras[0])
            nav_task["cameras"] = [
                {
                    "name": "nav2_topdown",
                    "translation": [0.0, 0.0, 5.0],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                    "camera_axes": source_camera.get("camera_axes", "usd"),
                    "camera_file": source_camera["camera_file"],
                    "parent": "",
                }
            ]
        else:
            nav_task["cameras"] = source_cameras
    else:
        nav_task["cameras"] = []
        nav_task["render"] = False

    return nav_task


def _prepare_headless_nav_task_cfg(
    task_cfg_path: str,
    output_dir: str,
    keep_task_cameras: bool = False,
    topdown_camera: bool = False,
    ackermann_split_steering: bool | None = None,
    ackermann_split_wheel_speeds: bool | None = None,
) -> str:
    with open(task_cfg_path, "r", encoding="utf-8") as file:
        task_cfg = yaml.safe_load(file)

    tasks = task_cfg.get("tasks", [])
    if not isinstance(tasks, list):
        raise TypeError("Task config must contain a 'tasks' list")

    robot_config_file_override = ""
    if tasks:
        first_robot = _select_nav_robot(tasks[0])
        robot_config_file = str(first_robot.get("robot_config_file", "")).strip()
        if robot_config_file:
            robot_config_file_override = _prepare_nav_robot_config(
                robot_config_file,
                output_dir,
                ackermann_split_steering=ackermann_split_steering,
                ackermann_split_wheel_speeds=ackermann_split_wheel_speeds,
            )

    nav_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            raise TypeError("Each task entry must be a dict")
        nav_tasks.append(
            _build_nav2_floor_task(
                task,
                keep_task_cameras=keep_task_cameras,
                topdown_camera=topdown_camera,
                robot_config_file_override=robot_config_file_override,
            )
        )

    task_cfg["tasks"] = nav_tasks

    os.makedirs(output_dir, exist_ok=True)
    if keep_task_cameras:
        patched_task_cfg_path = os.path.join(output_dir, "split_aloha_nav2_camera_task.yaml")
    else:
        patched_task_cfg_path = os.path.join(output_dir, "split_aloha_nav2_headless_task.yaml")
    with open(patched_task_cfg_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(task_cfg, file, sort_keys=False)
    return patched_task_cfg_path


def _prepare_nav_robot_config(
    robot_config_path: str,
    output_dir: str,
    ackermann_split_steering: bool | None = None,
    ackermann_split_wheel_speeds: bool | None = None,
) -> str:
    with open(robot_config_path, "r", encoding="utf-8") as file:
        robot_cfg = yaml.safe_load(file)

    base_cfg = robot_cfg.setdefault("base", {})
    ros_cfg = base_cfg.setdefault("ros", {})
    current_limit = ros_cfg.get("max_steer_angle_ackermann", NAV2_SMOKE_MAX_ACKERMANN_STEER_RAD)
    ros_cfg["max_steer_angle_ackermann"] = float(
        min(float(current_limit), float(NAV2_SMOKE_MAX_ACKERMANN_STEER_RAD))
    )
    ros_cfg["allow_spinning"] = True
    ros_cfg["spinning_requires_zero_linear"] = True
    localization_cfg = ros_cfg.setdefault("localization", {})
    localization_cfg.setdefault("enabled", True)
    localization_cfg["mode"] = "static_map_truth_pose"
    localization_cfg.setdefault("map_resolution", 0.05)
    localization_cfg.setdefault("map_output_dir", "output/nav2_maps")
    localization_cfg.setdefault("map_z_min", 0.0)
    localization_cfg.setdefault("map_z_max", 0.35)
    localization_cfg.setdefault("map_frame", "map")
    localization_cfg.setdefault("odom_frame", ros_cfg.get("odom_frame", "odom"))
    localization_cfg.setdefault("base_frame", ros_cfg.get("base_frame", "base_link"))
    ros_cfg.setdefault("nav2", {})
    ros_cfg["nav2"]["global_frame"] = str(localization_cfg.get("map_frame", "map"))
    if ackermann_split_steering is not None:
        base_cfg["ackermann_split_steering"] = bool(ackermann_split_steering)
    if ackermann_split_wheel_speeds is not None:
        base_cfg["ackermann_split_wheel_speeds"] = bool(ackermann_split_wheel_speeds)

    os.makedirs(output_dir, exist_ok=True)
    patched_robot_cfg_path = os.path.join(output_dir, "split_aloha_nav2_robot.yaml")
    with open(patched_robot_cfg_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(robot_cfg, file, sort_keys=False)
    return patched_robot_cfg_path


def _enable_viewportless_syntheticdata_fallback():
    import omni.usd
    import omni.syntheticdata.scripts.SyntheticData as syntheticdata_module
    from pxr import Sdf, Usd

    syntheticdata_cls = syntheticdata_module.SyntheticData
    if getattr(syntheticdata_cls, "_viewportless_patch_applied", False):
        return

    def _edit_render_var(render_product_path: str, render_var: str, usd_stage=None, remove: bool = False) -> None:
        stage = usd_stage
        if stage is None:
            stage = omni.usd.get_context().get_stage()
            if not stage:
                raise RuntimeError("No stage provided or active in the default UsdContext")

        with Usd.EditContext(stage, stage.GetSessionLayer()):
            render_product_prim = stage.GetPrimAtPath(render_product_path)
            if not render_product_prim:
                raise RuntimeError(f"invalid renderProduct {render_product_path}")

            render_var_prim_path = f"/Render/Vars/{render_var}"
            render_product_render_var_rel = render_product_prim.GetRelationship("orderedVars")

            if remove:
                if render_product_render_var_rel:
                    render_product_render_var_rel.RemoveTarget(render_var_prim_path)
                return

            render_var_prim = stage.GetPrimAtPath(render_var_prim_path)
            if not render_var_prim:
                render_var_prim = stage.DefinePrim(render_var_prim_path)
            if not render_var_prim:
                raise RuntimeError(f"cannot create renderVar {render_var_prim_path}")

            render_var_prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(render_var)
            render_var_prim.SetMetadata("hide_in_stage_window", True)
            render_var_prim.SetMetadata("no_delete", True)

            if not render_product_render_var_rel:
                render_product_render_var_rel = render_product_prim.CreateRelationship("orderedVars")
            if not render_product_render_var_rel:
                raise RuntimeError(f"cannot set orderedVars relationship for renderProduct {render_product_path}")
            render_product_render_var_rel.AddTarget(render_var_prim_path)

    original_add_rendervar = syntheticdata_cls._add_rendervar
    original_remove_rendervar = syntheticdata_cls._remove_rendervar

    def _patched_add_rendervar(render_product_path: str, render_var: str, usd_stage=None) -> None:
        try:
            original_add_rendervar(render_product_path, render_var, usd_stage)
        except RuntimeError as exc:
            if "omni::kit::IViewport" not in str(exc):
                raise
            _edit_render_var(render_product_path, render_var, usd_stage=usd_stage, remove=False)

    def _patched_remove_rendervar(render_product_path: str, render_var: str, usd_stage=None) -> None:
        try:
            original_remove_rendervar(render_product_path, render_var, usd_stage)
        except RuntimeError as exc:
            if "omni::kit::IViewport" not in str(exc):
                raise
            _edit_render_var(render_product_path, render_var, usd_stage=usd_stage, remove=True)

    syntheticdata_cls._add_rendervar = staticmethod(_patched_add_rendervar)
    syntheticdata_cls._remove_rendervar = staticmethod(_patched_remove_rendervar)
    syntheticdata_cls._viewportless_patch_applied = True


def _build_world(simulator_cfg: dict):
    return World(
        physics_dt=eval(str(simulator_cfg["physics_dt"])),
        rendering_dt=eval(str(simulator_cfg["rendering_dt"])),
        stage_units_in_meters=float(simulator_cfg["stage_units_in_meters"]),
    )


class _SimClockPublisher:
    def __init__(self, world):
        self.world = world
        self.node = Node("simbox_nav2_clock_publisher")
        self._clock_pub = self.node.create_publisher(Clock, "/clock", 10)

    def publish(self):
        sim_time = float(getattr(self.world, "current_time", 0.0))
        secs = int(math.floor(sim_time))
        nanosecs = int(round((sim_time - secs) * 1.0e9))
        if nanosecs >= 1_000_000_000:
            secs += 1
            nanosecs -= 1_000_000_000

        msg = Clock()
        msg.clock.sec = secs
        msg.clock.nanosec = nanosecs
        self._clock_pub.publish(msg)
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def destroy(self):
        self.node.destroy_node()


def _enable_node_use_sim_time(node: Node):
    node.set_parameters([Parameter("use_sim_time", value=True)])


def _write_nav2_bt_files(output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    nav_to_pose_bt_path = os.path.join(output_dir, "navigate_to_pose_w_replanning_no_motion_recovery.xml")
    nav_through_poses_bt_path = os.path.join(
        output_dir,
        "navigate_through_poses_w_replanning_no_motion_recovery.xml",
    )

    nav_to_pose_bt = """<!-- Ackermann-oriented navigation with explicit recovery actions. -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="4" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">
        <RateController hz="0.25">
          <RecoveryNode number_of_retries="1" name="ComputePathToPose">
            <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
            <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
          </RecoveryNode>
        </RateController>
        <RecoveryNode number_of_retries="1" name="FollowPath">
          <FollowPath path="{path}" controller_id="FollowPath"/>
          <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
        </RecoveryNode>
      </PipelineSequence>
      <ReactiveFallback name="RecoveryFallback">
        <GoalUpdated/>
        <RoundRobin name="RecoveryActions">
          <Sequence name="ClearingActions">
            <ClearEntireCostmap name="ClearLocalCostmap-Subtree" service_name="local_costmap/clear_entirely_local_costmap"/>
            <ClearEntireCostmap name="ClearGlobalCostmap-Subtree" service_name="global_costmap/clear_entirely_global_costmap"/>
          </Sequence>
          <Spin spin_dist="1.57"/>
          <BackUp backup_dist="0.30" backup_speed="0.05"/>
          <DriveOnHeading dist_to_travel="0.20" speed="0.05" time_allowance="10.0"/>
          <Wait wait_duration="2.0"/>
        </RoundRobin>
      </ReactiveFallback>
    </RecoveryNode>
  </BehaviorTree>
</root>
"""

    nav_through_poses_bt = """<!-- Ackermann-oriented navigation through poses with explicit recovery actions. -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="4" name="NavigateThroughPosesRecovery">
      <PipelineSequence name="NavigateThroughPosesWithReplanning">
        <RateController hz="0.25">
          <RecoveryNode number_of_retries="1" name="ComputePathThroughPoses">
            <ReactiveSequence>
              <RemovePassedGoals input_goals="{goals}" output_goals="{goals}" radius="0.7"/>
              <ComputePathThroughPoses goals="{goals}" path="{path}" planner_id="GridBased"/>
            </ReactiveSequence>
            <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
          </RecoveryNode>
        </RateController>
        <RecoveryNode number_of_retries="1" name="FollowPath">
          <FollowPath path="{path}" controller_id="FollowPath"/>
          <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
        </RecoveryNode>
      </PipelineSequence>
      <ReactiveFallback name="RecoveryFallback">
        <GoalUpdated/>
        <RoundRobin name="RecoveryActions">
          <Sequence name="ClearingActions">
            <ClearEntireCostmap name="ClearLocalCostmap-Subtree" service_name="local_costmap/clear_entirely_local_costmap"/>
            <ClearEntireCostmap name="ClearGlobalCostmap-Subtree" service_name="global_costmap/clear_entirely_global_costmap"/>
          </Sequence>
          <Spin spin_dist="1.57"/>
          <BackUp backup_dist="0.30" backup_speed="0.05"/>
          <DriveOnHeading dist_to_travel="0.20" speed="0.05" time_allowance="10.0"/>
          <Wait wait_duration="2.0"/>
        </RoundRobin>
      </ReactiveFallback>
    </RecoveryNode>
  </BehaviorTree>
</root>
"""

    with open(nav_to_pose_bt_path, "w", encoding="utf-8") as file:
        file.write(nav_to_pose_bt)
    with open(nav_through_poses_bt_path, "w", encoding="utf-8") as file:
        file.write(nav_through_poses_bt)

    return nav_to_pose_bt_path, nav_through_poses_bt_path


def _build_real_nav2_params(nav_to_pose_bt: str, nav_through_poses_bt: str, base_cfg: dict):
    ros_cfg = dict(base_cfg.get("ros", {}))
    nav2_cfg = dict(ros_cfg.get("nav2", {}))
    localization_cfg = dict(ros_cfg.get("localization", {}))
    map_frame = str(localization_cfg.get("map_frame", nav2_cfg.get("global_frame", "map")))
    odom_frame = str(localization_cfg.get("odom_frame", ros_cfg.get("odom_frame", "odom")))
    base_frame = str(localization_cfg.get("base_frame", nav2_cfg.get("robot_base_frame", ros_cfg.get("base_frame", "base_link"))))
    footprint = _format_nav2_footprint(SPLIT_ALOHA_NAV2_FOOTPRINT_POINTS)
    bt_plugins = [
        "nav2_compute_path_to_pose_action_bt_node",
        "nav2_compute_path_through_poses_action_bt_node",
        "nav2_smooth_path_action_bt_node",
        "nav2_follow_path_action_bt_node",
        "nav2_spin_action_bt_node",
        "nav2_wait_action_bt_node",
        "nav2_assisted_teleop_action_bt_node",
        "nav2_back_up_action_bt_node",
        "nav2_drive_on_heading_bt_node",
        "nav2_clear_costmap_service_bt_node",
        "nav2_is_stuck_condition_bt_node",
        "nav2_goal_reached_condition_bt_node",
        "nav2_goal_updated_condition_bt_node",
        "nav2_globally_updated_goal_condition_bt_node",
        "nav2_is_path_valid_condition_bt_node",
        "nav2_rate_controller_bt_node",
        "nav2_distance_controller_bt_node",
        "nav2_speed_controller_bt_node",
        "nav2_truncate_path_action_bt_node",
        "nav2_truncate_path_local_action_bt_node",
        "nav2_goal_updater_node_bt_node",
        "nav2_recovery_node_bt_node",
        "nav2_pipeline_sequence_bt_node",
        "nav2_round_robin_node_bt_node",
        "nav2_transform_available_condition_bt_node",
        "nav2_time_expired_condition_bt_node",
        "nav2_path_expiring_timer_condition",
        "nav2_distance_traveled_condition_bt_node",
        "nav2_single_trigger_bt_node",
        "nav2_goal_updated_controller_bt_node",
        "nav2_is_battery_low_condition_bt_node",
        "nav2_navigate_through_poses_action_bt_node",
        "nav2_navigate_to_pose_action_bt_node",
        "nav2_remove_passed_goals_action_bt_node",
        "nav2_planner_selector_bt_node",
        "nav2_controller_selector_bt_node",
        "nav2_goal_checker_selector_bt_node",
        "nav2_controller_cancel_bt_node",
        "nav2_path_longer_on_approach_bt_node",
        "nav2_wait_cancel_bt_node",
        "nav2_spin_cancel_bt_node",
        "nav2_back_up_cancel_bt_node",
        "nav2_assisted_teleop_cancel_bt_node",
        "nav2_drive_on_heading_cancel_bt_node",
    ]

    return {
        "bt_navigator": {
            "ros__parameters": {
                "use_sim_time": True,
                "global_frame": map_frame,
                "robot_base_frame": base_frame,
                "odom_topic": str(ros_cfg.get("odom_topic", "/odom")),
                "bt_loop_duration": 10,
                "default_server_timeout": 20,
                "wait_for_service_timeout": 1000,
                "default_bt_xml_filename": nav_to_pose_bt,
                "default_nav_to_pose_bt_xml": nav_to_pose_bt,
                "default_nav_through_poses_bt_xml": nav_through_poses_bt,
                "plugin_lib_names": bt_plugins,
            }
        },
        "bt_navigator_navigate_to_pose_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "controller_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "controller_frequency": 20.0,
                "min_x_velocity_threshold": 0.001,
                "min_y_velocity_threshold": 0.001,
                "min_theta_velocity_threshold": 0.001,
                "failure_tolerance": 1.20,
                "progress_checker_plugin": "progress_checker",
                "goal_checker_plugins": ["general_goal_checker"],
                "controller_plugins": ["FollowPath"],
                "progress_checker": {
                    "plugin": "nav2_controller::SimpleProgressChecker",
                    "required_movement_radius": 0.05,
                    "movement_time_allowance": 90.0,
                },
                "general_goal_checker": {
                    "stateful": False,
                    "plugin": "nav2_controller::SimpleGoalChecker",
                    "xy_goal_tolerance": float(NAV2_SUCCESS_POSITION_TOLERANCE_M),
                    "yaw_goal_tolerance": float(NAV2_SUCCESS_YAW_TOLERANCE_RAD),
                },
                "FollowPath": {
                    "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
                    "desired_linear_vel": 0.4,
                    "lookahead_dist": 0.70,
                    "min_lookahead_dist": 0.45,
                    "max_lookahead_dist": 1.00,
                    "lookahead_time": 1.0,
                    "use_velocity_scaled_lookahead_dist": True,
                    "transform_tolerance": 0.4,
                    "min_approach_linear_velocity": 0.26,
                    "approach_velocity_scaling_dist": 1.20,
                    "use_collision_detection": False,
                    "max_allowed_time_to_collision_up_to_carrot": 0.6,
                    "use_regulated_linear_velocity_scaling": True,
                    "use_cost_regulated_linear_velocity_scaling": True,
                    "cost_scaling_dist": 0.25,
                    "cost_scaling_gain": 0.65,
                    "inflation_cost_scaling_factor": 3.0,
                    "regulated_linear_scaling_min_radius": SPLIT_ALOHA_NAV2_MIN_TURN_RADIUS_M,
                    "regulated_linear_scaling_min_speed": 0.26,
                    "use_rotate_to_heading": True,
                    "rotate_to_heading_angular_vel": 0.35,
                    "rotate_to_heading_min_angle": 0.10,
                    "max_angular_accel": 0.9,
                    "allow_reversing": False,
                    "use_interpolation": True,
                },
            }
        },
        "local_costmap": {
            "local_costmap": {
                "ros__parameters": {
                    "use_sim_time": True,
                    "update_frequency": 10.0,
                    "publish_frequency": 4.0,
                    "global_frame": map_frame,
                    "robot_base_frame": base_frame,
                    "rolling_window": False,
                    "resolution": 0.05,
                    "footprint": footprint,
                    "footprint_padding": 0.0,
                    "plugins": ["static_layer", "inflation_layer"],
                    "static_layer": {
                        "plugin": "nav2_costmap_2d::StaticLayer",
                        "map_subscribe_transient_local": True,
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        "cost_scaling_factor": 3.0,
                        "inflation_radius": SPLIT_ALOHA_NAV2_INFLATION_RADIUS_M,
                    },
                    "always_send_full_costmap": True,
                }
            }
        },
        "global_costmap": {
            "global_costmap": {
                "ros__parameters": {
                    "use_sim_time": True,
                    "update_frequency": 4.0,
                    "publish_frequency": 2.0,
                    "global_frame": map_frame,
                    "robot_base_frame": base_frame,
                    "rolling_window": False,
                    "resolution": 0.05,
                    "track_unknown_space": False,
                    "footprint": footprint,
                    "footprint_padding": 0.0,
                    "plugins": ["static_layer", "inflation_layer"],
                    "static_layer": {
                        "plugin": "nav2_costmap_2d::StaticLayer",
                        "map_subscribe_transient_local": True,
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        "cost_scaling_factor": 3.0,
                        "inflation_radius": SPLIT_ALOHA_NAV2_INFLATION_RADIUS_M,
                    },
                    "always_send_full_costmap": True,
                }
            }
        },
        "planner_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "expected_planner_frequency": 10.0,
                "planner_plugins": ["GridBased"],
                "GridBased": {
                    "plugin": "nav2_smac_planner/SmacPlannerHybrid",
                    "tolerance": 0.15,
                    "allow_unknown": False,
                    "downsample_costmap": False,
                    "max_iterations": 1000000,
                    "max_on_approach_iterations": 1000,
                    "max_planning_time": 2.0,
                    "motion_model_for_search": "REEDS_SHEPP",
                    "angle_quantization_bins": 72,
                    "analytic_expansion_ratio": 3.5,
                    "analytic_expansion_max_length": 3.0,
                    "minimum_turning_radius": SPLIT_ALOHA_NAV2_MIN_TURN_RADIUS_M,
                    "reverse_penalty": 8.0,
                    "change_penalty": 1.2,
                    "non_straight_penalty": 1.15,
                    "cost_penalty": 2.5,
                    "retrospective_penalty": 0.015,
                    "lookup_table_size": 20.0,
                    "cache_obstacle_heuristic": False,
                    "smooth_path": True,
                },
            }
        },
        "smoother_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "smoother_plugins": ["simple_smoother"],
                "simple_smoother": {
                    "plugin": "nav2_smoother::SimpleSmoother",
                    "tolerance": 1.0e-10,
                    "max_its": 1000,
                    "do_refinement": True,
                },
            }
        },
        "behavior_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "costmap_topic": "local_costmap/costmap_raw",
                "footprint_topic": "local_costmap/published_footprint",
                "cycle_frequency": 10.0,
                "behavior_plugins": ["spin", "backup", "drive_on_heading", "wait"],
                "spin": {"plugin": "nav2_behaviors/Spin"},
                "backup": {"plugin": "nav2_behaviors/BackUp"},
                "drive_on_heading": {"plugin": "nav2_behaviors/DriveOnHeading"},
                "wait": {"plugin": "nav2_behaviors/Wait"},
                "global_frame": map_frame,
                "robot_base_frame": base_frame,
                "transform_tolerance": 0.2,
                "simulate_ahead_time": 2.0,
                "max_rotational_vel": 0.35,
                "min_rotational_vel": 0.1,
                "rotational_acc_lim": 1.0,
            }
        },
        "waypoint_follower": {
            "ros__parameters": {
                "use_sim_time": True,
                "loop_rate": 20,
                "stop_on_failure": False,
                "waypoint_task_executor_plugin": "wait_at_waypoint",
                "wait_at_waypoint": {
                    "plugin": "nav2_waypoint_follower::WaitAtWaypoint",
                    "enabled": True,
                    "waypoint_pause_duration": 0,
                },
            }
        },
        "velocity_smoother": {
            "ros__parameters": {
                "use_sim_time": True,
                "smoothing_frequency": 20.0,
                "scale_velocities": False,
                "feedback": "OPEN_LOOP",
                "max_velocity": [0.4, 0.0, 0.30],
                "min_velocity": [-0.4, 0.0, -0.30],
                "max_accel": [0.35, 0.0, 0.7],
                "max_decel": [-0.35, 0.0, -0.7],
                "odom_topic": str(ros_cfg.get("odom_topic", "/odom")),
                "odom_duration": 0.1,
                "deadband_velocity": [0.0, 0.0, 0.0],
                "velocity_timeout": 1.0,
            }
        },
        "map_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "yaml_filename": str(localization_cfg.get("map_yaml_path", "")),
            }
        },
    }


def _launch_real_nav2(output_dir: str, map_yaml_path: str, base_cfg: dict):
    os.makedirs(output_dir, exist_ok=True)
    _cleanup_stale_nav2_processes()
    nav_to_pose_bt_path, nav_through_poses_bt_path = _write_nav2_bt_files(output_dir)
    params_path = os.path.join(output_dir, "split_aloha_nav2_real_params.yaml")
    log_path = os.path.join(output_dir, "split_aloha_nav2_real.log")

    with open(params_path, "w", encoding="utf-8") as file:
        params = _build_real_nav2_params(
            nav_to_pose_bt=nav_to_pose_bt_path,
            nav_through_poses_bt=nav_through_poses_bt_path,
            base_cfg=base_cfg,
        )
        params["map_server"]["ros__parameters"]["yaml_filename"] = str(map_yaml_path)
        yaml.safe_dump(params, file, sort_keys=False)

    log_file = open(log_path, "w", encoding="utf-8")
    command = (
        "source /opt/ros/humble/setup.bash && "
        "trap 'kill 0 >/dev/null 2>&1 || true' EXIT INT TERM && "
        "ros2 run tf2_ros static_transform_publisher "
        "--x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 --frame-id map --child-frame-id odom "
        "> /dev/null 2>&1 & "
        f"ros2 run nav2_map_server map_server --ros-args --params-file {params_path} "
        "> /dev/null 2>&1 & "
        "ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args "
        "-p use_sim_time:=true -p autostart:=true -p node_names:=\"['map_server']\" "
        "> /dev/null 2>&1 & "
        "ros2 launch nav2_bringup navigation_launch.py "
        f"use_sim_time:=true autostart:=true use_composition:=False params_file:={params_path} & "
        "wait"
    )
    launch_env = _build_clean_ros2_launch_env()
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        ["bash", "-lc", command],
        cwd="/workspace/InterndataEngine",
        env=launch_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return process, log_file, params_path, log_path


def _launch_external_base_driver(output_dir: str, base_cfg: dict):
    os.makedirs(output_dir, exist_ok=True)
    driver_cfg_path = os.path.join(output_dir, "split_aloha_external_driver_base.yaml")
    log_path = os.path.join(output_dir, "split_aloha_external_driver.log")

    driver_base_cfg = deepcopy(base_cfg)
    driver_ros_cfg = driver_base_cfg.setdefault("ros", {})
    driver_ros_cfg["publish_driver_topics"] = True
    driver_ros_cfg["internal_cmdvel_controller_enabled"] = False

    with open(driver_cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(driver_base_cfg, handle, sort_keys=False)

    log_file = open(log_path, "w", encoding="utf-8")
    command = (
        "source /opt/ros/humble/setup.bash && "
        f"exec python3 -m nav2.ranger_driver_node --base-config {driver_cfg_path}"
    )
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        ["bash", "-lc", command],
        cwd="/workspace/InterndataEngine",
        env=_build_clean_ros2_launch_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return process, log_file, driver_cfg_path, log_path


def _cleanup_stale_nav2_processes():
    patterns = [
        "/opt/ros/humble/bin/ros2 launch nav2_bringup navigation_launch.py",
        "/opt/ros/humble/lib/nav2_controller/controller_server",
        "/opt/ros/humble/lib/nav2_smoother/smoother_server",
        "/opt/ros/humble/lib/nav2_planner/planner_server",
        "/opt/ros/humble/lib/nav2_behaviors/behavior_server",
        "/opt/ros/humble/lib/nav2_bt_navigator/bt_navigator",
        "/opt/ros/humble/lib/nav2_waypoint_follower/waypoint_follower",
        "/opt/ros/humble/lib/nav2_velocity_smoother/velocity_smoother",
        "/opt/ros/humble/lib/nav2_map_server/map_server",
        "/opt/ros/humble/lib/nav2_lifecycle_manager/lifecycle_manager",
        "/opt/ros/humble/lib/tf2_ros/static_transform_publisher",
    ]
    for signal_name in ("-INT", "-TERM", "-KILL"):
        for pattern in patterns:
            subprocess.run(
                ["pkill", signal_name, "-f", pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        time.sleep(0.5)


def _stop_subprocess(process, log_file):
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
    _cleanup_stale_nav2_processes()
    if log_file is not None:
        log_file.close()


def _read_log_tail(log_path: str, max_lines: int = 60) -> str:
    if not log_path or not os.path.exists(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
    return "".join(lines[-max_lines:])


def _run_camera_replay_from_report(
    config_path: str,
    replay_report_path: str,
    output_path: str,
    video_path: str = "",
    camera_name: str = "",
    camera_image_path: str = "",
    camera_video_path: str = "",
    camera_view: str = "",
    camera_height_m: float = 5.0,
):
    replay_report: dict = {
        "started_at": datetime.now().isoformat(),
        "status": "started",
        "mode": "camera_replay",
        "config_path": config_path,
        "replay_report_path": replay_report_path,
        "camera_view": camera_view,
        "camera_height_m": float(camera_height_m),
    }
    if camera_name:
        replay_report["camera_name_requested"] = camera_name
    if video_path:
        replay_report["video_path"] = video_path
    if camera_image_path:
        replay_report["camera_image_path"] = camera_image_path
    if camera_video_path:
        replay_report["camera_video_path"] = camera_video_path

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    camera_writer = None
    last_camera_image_path = ""
    sim_clock_publisher = None

    try:
        with open(replay_report_path, "r", encoding="utf-8") as file:
            source_report = json.load(file)

        source_result = dict(source_report.get("result", {}))
        trajectory_xy = list(source_result.get("trajectory_xy", []))
        trajectory_yaw = list(source_result.get("trajectory_yaw", []))
        if not trajectory_xy or len(trajectory_xy) != len(trajectory_yaw):
            raise ValueError("Replay source report does not contain a valid trajectory_xy/trajectory_yaw sequence")
        if video_path and not any([camera_name, camera_image_path, camera_video_path, camera_view]):
            _write_topdown_mp4(
                video_path=video_path,
                path_xy=trajectory_xy,
                path_yaw=trajectory_yaw,
                start_xy=trajectory_xy[0],
                goal_xy=source_report.get("goal_world_xy", trajectory_xy[-1]),
                goal_yaw=float(source_report.get("goal_world_yaw", trajectory_yaw[-1])),
                reached_target=bool(source_result.get("reached_target", False)),
                obstacle_specs=source_report.get("nav2_obstacles", []),
            )
            replay_report["status"] = "completed"
            replay_report["result"] = {
                "debug_video_path": video_path,
                "rendered_trajectory_point_count": len(trajectory_xy),
                "reverse_motion_events": _build_reverse_motion_events(trajectory_xy, trajectory_yaw),
                "motion_segments": _infer_motion_segments(trajectory_xy, trajectory_yaw),
            }
            return

        replay_report["status"] = "initializing_environment"
        init_env()
        config = _load_render_config(config_path)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        task_cfg_path = scene_loader_cfg["cfg_path"]
        simulator_cfg = scene_loader_cfg["simulator"]
        replay_report["original_task_cfg_path"] = task_cfg_path
        topdown_camera = camera_view == "topdown"
        task_cfg_path = _prepare_headless_nav_task_cfg(
            task_cfg_path,
            output_dir or ".",
            keep_task_cameras=True,
            topdown_camera=topdown_camera,
        )
        replay_report["task_cfg_path"] = task_cfg_path
        replay_report["task_cameras_enabled"] = True
        replay_report["topdown_camera_enabled"] = bool(topdown_camera)
        replay_report["camera_render_stride"] = 1
        _enable_viewportless_syntheticdata_fallback()

        import_extensions(workflow_type)
        world = _build_world(simulator_cfg)
        workflow = create_workflow(workflow_type, world, task_cfg_path)

        replay_report["status"] = "initializing_task"
        workflow.init_task(0)
        robot = _find_split_aloha(workflow)
        sim_clock_publisher = _SimClockPublisher(world)
        sim_clock_publisher.publish()

        root_start_pose = robot.get_world_pose()
        base_start_pose = _get_robot_base_pose(robot)
        root_from_world_start = tf_matrix_from_pose(
            np.asarray(root_start_pose[0], dtype=np.float32),
            np.asarray(root_start_pose[1], dtype=np.float32),
        )
        base_from_world_start = tf_matrix_from_pose(
            np.asarray(base_start_pose[0], dtype=np.float32),
            np.asarray(base_start_pose[1], dtype=np.float32),
        )
        base_to_root = np.asarray(np.linalg.inv(base_from_world_start) @ root_from_world_start, dtype=np.float32)

        base_height = float(base_start_pose[0][2])
        replay_report["replay_base_height_m"] = base_height
        replay_report["source_goal_world_xy"] = list(source_report.get("goal_world_xy", []))
        replay_report["source_nav2_obstacles"] = list(source_report.get("nav2_obstacles", []))

        selected_camera_name = camera_name or ("nav2_topdown" if topdown_camera else "")
        if topdown_camera:
            start_xy = np.asarray(trajectory_xy[0], dtype=np.float32)
            goal_xy = np.asarray(source_report.get("goal_world_xy", trajectory_xy[-1]), dtype=np.float32)
            topdown_center_xy, topdown_height_m = _compute_topdown_camera_view(
                workflow=workflow,
                camera_name=selected_camera_name,
                start_xy=start_xy,
                goal_xy=goal_xy,
                obstacle_specs=source_report.get("nav2_obstacles", []),
                min_height_m=camera_height_m,
            )
            replay_report["topdown_camera_center_xy"] = [float(topdown_center_xy[0]), float(topdown_center_xy[1])]
            replay_report["topdown_camera_height_m"] = float(topdown_height_m)
        else:
            topdown_center_xy = None
            topdown_height_m = float(camera_height_m)

        for _ in range(NAV2_CAMERA_WARMUP_RENDER_STEPS):
            if topdown_camera:
                _update_topdown_camera_pose(
                    workflow,
                    robot,
                    camera_name=selected_camera_name,
                    height_m=topdown_height_m,
                    center_xy=topdown_center_xy,
                )
            sim_clock_publisher.publish()
            workflow._step_world(render=True)
            sim_clock_publisher.publish()

        if topdown_camera:
            _update_topdown_camera_pose(
                workflow,
                robot,
                camera_name=selected_camera_name,
                height_m=topdown_height_m,
                center_xy=topdown_center_xy,
            )

        selected_camera_name, warmup_frame = _capture_task_camera_frame(workflow, camera_name=selected_camera_name)
        replay_report["camera_name"] = selected_camera_name
        replay_report["camera_resolution"] = [int(warmup_frame.shape[1]), int(warmup_frame.shape[0])]

        camera_frame_count = 0
        if camera_image_path:
            _write_rgb_png(camera_image_path, warmup_frame)
            last_camera_image_path = camera_image_path
        if camera_video_path:
            camera_writer = _open_rgb_video_writer(camera_video_path, warmup_frame)
            camera_writer.write(cv2.cvtColor(warmup_frame, cv2.COLOR_RGB2BGR))
            camera_frame_count += 1

        replay_report["status"] = "replaying"
        rendered_world_xy = []
        rendered_world_yaw = []

        for frame_idx, (xy, yaw) in enumerate(zip(trajectory_xy, trajectory_yaw)):
            world_from_base = tf_matrix_from_pose(
                np.asarray([float(xy[0]), float(xy[1]), base_height], dtype=np.float32),
                _wxyz_from_yaw(float(yaw)),
            )
            world_from_root = np.asarray(world_from_base @ base_to_root, dtype=np.float32)
            root_translation, root_orientation = pose_from_tf_matrix(
                np.asarray(world_from_root, dtype=np.float64)
            )
            robot.set_world_pose(
                position=np.asarray(root_translation, dtype=np.float32),
                orientation=np.asarray(root_orientation, dtype=np.float32),
            )
            if topdown_camera:
                _update_topdown_camera_pose(
                    workflow,
                    robot,
                    camera_name=selected_camera_name,
                    height_m=topdown_height_m,
                    center_xy=topdown_center_xy,
                )
            sim_clock_publisher.publish()
            workflow._step_world(render=True)
            sim_clock_publisher.publish()
            _, rgb_frame = _capture_task_camera_frame(workflow, camera_name=selected_camera_name)
            if camera_image_path and frame_idx == len(trajectory_xy) - 1:
                _write_rgb_png(camera_image_path, rgb_frame)
                last_camera_image_path = camera_image_path
            if camera_writer is not None:
                camera_writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                camera_frame_count += 1
            rendered_world_xy.append([float(xy[0]), float(xy[1])])
            rendered_world_yaw.append(float(yaw))

        replay_report["status"] = "completed"
        replay_report["result"] = {
            "camera_name": selected_camera_name,
            "camera_frame_count": int(camera_frame_count),
            "camera_view": camera_view or "task",
            "rendered_trajectory_point_count": len(rendered_world_xy),
            "rendered_trajectory_xy": rendered_world_xy,
            "rendered_trajectory_yaw": rendered_world_yaw,
        }
        if camera_image_path:
            replay_report["result"]["camera_image_path"] = last_camera_image_path or camera_image_path
        if camera_video_path:
            replay_report["result"]["camera_video_path"] = camera_video_path
        if video_path:
            _write_topdown_mp4(
                video_path=video_path,
                path_xy=trajectory_xy,
                path_yaw=trajectory_yaw,
                start_xy=trajectory_xy[0],
                goal_xy=source_report.get("goal_world_xy", trajectory_xy[-1]),
                goal_yaw=float(source_report.get("goal_world_yaw", trajectory_yaw[-1])),
                reached_target=bool(source_result.get("reached_target", False)),
                obstacle_specs=source_report.get("nav2_obstacles", []),
            )
            replay_report["result"]["debug_video_path"] = video_path
    except Exception as exc:  # pylint: disable=broad-except
        replay_report["status"] = "error"
        replay_report["error"] = str(exc)
        replay_report["traceback"] = traceback.format_exc()
    finally:
        if camera_writer is not None:
            camera_writer.release()
        if sim_clock_publisher is not None:
            sim_clock_publisher.destroy()
        replay_report["finished_at"] = datetime.now().isoformat()
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(replay_report, file, indent=2)
        print(json.dumps(replay_report, indent=2))
        simulation_app.close()


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


def _wxyz_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * float(yaw)
    return np.asarray([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float32)


def _get_robot_base_pose(robot):
    getter = getattr(robot, "get_mobile_base_pose", None)
    if callable(getter):
        return getter()
    return robot.get_world_pose()


def _compute_topdown_bounds(start_xy, goal_xy, path_xy, obstacle_specs=None, margin: float = 0.35):
    points = [np.asarray(start_xy, dtype=np.float32), np.asarray(goal_xy, dtype=np.float32)]
    points.extend(np.asarray(point, dtype=np.float32) for point in path_xy)
    for obstacle_spec in obstacle_specs or []:
        translation = obstacle_spec.get("translation", [0.0, 0.0, 0.0])
        scale = obstacle_spec.get("scale", [0.0, 0.0, 0.0])
        half_extent = 0.5 * np.asarray(scale[:2], dtype=np.float32)
        center_xy = np.asarray(translation[:2], dtype=np.float32)
        points.extend(
            [
                center_xy + np.asarray([half_extent[0], half_extent[1]], dtype=np.float32),
                center_xy + np.asarray([half_extent[0], -half_extent[1]], dtype=np.float32),
                center_xy + np.asarray([-half_extent[0], half_extent[1]], dtype=np.float32),
                center_xy + np.asarray([-half_extent[0], -half_extent[1]], dtype=np.float32),
            ]
        )
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]

    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    span = max(span_x, span_y)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    half = 0.5 * span
    return {
        "min_x": center_x - half,
        "max_x": center_x + half,
        "min_y": center_y - half,
        "max_y": center_y + half,
    }


def _world_to_pixel(xy, bounds, canvas_size: int, border_px: int):
    usable = float(canvas_size - 2 * border_px)
    span_x = max(float(bounds["max_x"] - bounds["min_x"]), 1e-6)
    span_y = max(float(bounds["max_y"] - bounds["min_y"]), 1e-6)

    px = border_px + (float(xy[0]) - float(bounds["min_x"])) / span_x * usable
    py = border_px + (1.0 - (float(xy[1]) - float(bounds["min_y"])) / span_y) * usable
    px = int(round(max(0.0, min(float(canvas_size - 1), px))))
    py = int(round(max(0.0, min(float(canvas_size - 1), py))))
    return (px, py)


def _angle_diff_rad(target: float, current: float) -> float:
    return float(math.atan2(math.sin(float(target) - float(current)), math.cos(float(target) - float(current))))


def _draw_heading_arrow(frame, center_px, yaw: float, length_px: int, color):
    end_x = int(round(center_px[0] + math.cos(float(yaw)) * float(length_px)))
    end_y = int(round(center_px[1] - math.sin(float(yaw)) * float(length_px)))
    cv2.arrowedLine(
        frame,
        center_px,
        (end_x, end_y),
        color=color,
        thickness=2,
        tipLength=0.3,
        line_type=cv2.LINE_AA,
    )


def _draw_obstacle_rect(frame, obstacle_spec: dict, bounds, canvas_size: int, border_px: int):
    translation = obstacle_spec.get("translation", [0.0, 0.0, 0.0])
    scale = obstacle_spec.get("scale", [0.0, 0.0, 0.0])
    half_x = 0.5 * float(scale[0])
    half_y = 0.5 * float(scale[1])
    min_xy = (float(translation[0]) - half_x, float(translation[1]) - half_y)
    max_xy = (float(translation[0]) + half_x, float(translation[1]) + half_y)
    p0 = _world_to_pixel((min_xy[0], max_xy[1]), bounds, canvas_size, border_px)
    p1 = _world_to_pixel((max_xy[0], min_xy[1]), bounds, canvas_size, border_px)
    left = min(p0[0], p1[0])
    right = max(p0[0], p1[0])
    top = min(p0[1], p1[1])
    bottom = max(p0[1], p1[1])
    fill_color = tuple(int(round(255.0 * float(channel))) for channel in obstacle_spec.get("color", [0.8, 0.3, 0.2]))
    edge_color = (36, 36, 36)
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), fill_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0.0, dst=frame)
    cv2.rectangle(frame, (left, top), (right, bottom), edge_color, 2, cv2.LINE_AA)
    label = str(obstacle_spec.get("name", "obs"))
    cv2.putText(
        frame,
        label,
        (left + 4, max(18, top + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        edge_color,
        1,
        cv2.LINE_AA,
    )


def _draw_topdown_grid(frame, bounds, canvas_size: int, border_px: int, spacing_m: float = 0.25):
    color = (232, 236, 240)
    min_x = float(bounds["min_x"])
    max_x = float(bounds["max_x"])
    min_y = float(bounds["min_y"])
    max_y = float(bounds["max_y"])

    start_x = math.floor(min_x / spacing_m) * spacing_m
    start_y = math.floor(min_y / spacing_m) * spacing_m

    x = start_x
    while x <= max_x + 1e-6:
        p0 = _world_to_pixel((x, min_y), bounds, canvas_size, border_px)
        p1 = _world_to_pixel((x, max_y), bounds, canvas_size, border_px)
        cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)
        x += spacing_m

    y = start_y
    while y <= max_y + 1e-6:
        p0 = _world_to_pixel((min_x, y), bounds, canvas_size, border_px)
        p1 = _world_to_pixel((max_x, y), bounds, canvas_size, border_px)
        cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)
        y += spacing_m


def _infer_motion_segments(path_xy, path_yaw, distance_epsilon: float = 1.0e-4):
    segments = []
    if not path_xy or len(path_xy) != len(path_yaw):
        return segments

    last_state = "stationary"
    segment_start = 0
    for idx in range(1, len(path_xy)):
        prev_xy = np.asarray(path_xy[idx - 1], dtype=np.float32)
        curr_xy = np.asarray(path_xy[idx], dtype=np.float32)
        delta = curr_xy - prev_xy
        distance = float(np.linalg.norm(delta))
        if distance <= distance_epsilon:
            state = "stationary"
            signed_progress = 0.0
        else:
            heading = np.asarray([math.cos(float(path_yaw[idx])), math.sin(float(path_yaw[idx]))], dtype=np.float32)
            signed_progress = float(np.dot(delta, heading))
            state = "forward" if signed_progress >= 0.0 else "reverse"

        if idx == 1:
            last_state = state
            segment_start = 0
            continue
        if state != last_state:
            segments.append(
                {
                    "start_index": int(segment_start),
                    "end_index": int(idx - 1),
                    "motion_state": str(last_state),
                }
            )
            segment_start = idx - 1
            last_state = state

    segments.append(
        {
            "start_index": int(segment_start),
            "end_index": int(len(path_xy) - 1),
            "motion_state": str(last_state),
        }
    )
    return segments


def _build_reverse_motion_events(path_xy, path_yaw, min_reverse_steps: int = 3):
    reverse_events = []
    for segment in _infer_motion_segments(path_xy, path_yaw):
        if segment["motion_state"] != "reverse":
            continue
        step_count = int(segment["end_index"] - segment["start_index"] + 1)
        if step_count < int(min_reverse_steps):
            continue
        start_xy = np.asarray(path_xy[segment["start_index"]], dtype=np.float32)
        end_xy = np.asarray(path_xy[segment["end_index"]], dtype=np.float32)
        reverse_events.append(
            {
                "start_index": int(segment["start_index"]),
                "end_index": int(segment["end_index"]),
                "step_count": int(step_count),
                "start_xy": [float(start_xy[0]), float(start_xy[1])],
                "end_xy": [float(end_xy[0]), float(end_xy[1])],
                "path_length_m": float(np.linalg.norm(end_xy - start_xy)),
            }
        )
    return reverse_events


def _normalize_rgb_frame(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected camera frame shaped like HxWx3(+), got {rgb.shape}")

    rgb = rgb[..., :3]
    if rgb.dtype == np.uint8:
        return rgb

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    scale = 255.0 if float(np.max(rgb)) <= 1.0 else 1.0
    return np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)


def _write_rgb_png(png_path: str, rgb_frame: np.ndarray):
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(png_path, bgr_frame):
        raise RuntimeError(f"Failed to write camera PNG to {png_path}")


def _open_rgb_video_writer(video_path: str, rgb_frame: np.ndarray, fps: int = 15):
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    height, width = rgb_frame.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open camera video writer for {video_path}")
    return writer


def _capture_task_camera_frame(workflow, camera_name: str = ""):
    cameras = getattr(workflow.task, "cameras", {})
    if not cameras:
        return None, None

    if camera_name:
        camera = cameras.get(camera_name)
        if camera is None:
            available = ", ".join(sorted(cameras.keys()))
            raise KeyError(f"Camera '{camera_name}' not found. Available cameras: {available}")
        selected_name = camera_name
    else:
        selected_name = next(iter(cameras.keys()))
        camera = cameras[selected_name]

    camera_obs = camera.get_observations()
    rgb_frame = _normalize_rgb_frame(camera_obs["color_image"])
    return selected_name, rgb_frame


def _capture_task_camera_observation(workflow, camera_name: str = ""):
    cameras = getattr(workflow.task, "cameras", {})
    if not cameras:
        return None, None, None

    if camera_name:
        camera = cameras.get(camera_name)
        if camera is None:
            available = ", ".join(sorted(cameras.keys()))
            raise KeyError(f"Camera '{camera_name}' not found. Available cameras: {available}")
        selected_name = camera_name
    else:
        selected_name = next(iter(cameras.keys()))
        camera = cameras[selected_name]

    camera_obs = camera.get_observations()
    rgb_frame = _normalize_rgb_frame(camera_obs["color_image"])
    return selected_name, camera_obs, rgb_frame


def _get_prim_transform_to_world(prim_path: str) -> np.ndarray:
    if not prim_path:
        return np.eye(4, dtype=np.float32)
    prim = get_prim_at_path(prim_path)
    world_prim = get_prim_at_path("/World")
    if prim is None or world_prim is None:
        return np.eye(4, dtype=np.float32)
    return np.asarray(get_relative_transform(prim, world_prim), dtype=np.float32)


def _get_camera_world_transform(camera, camera_obs: dict | None = None) -> np.ndarray:
    if camera_obs is not None and "camera2env_pose" in camera_obs:
        env_from_camera = np.asarray(camera_obs["camera2env_pose"], dtype=np.float32)
    else:
        env_root = get_prim_at_path(getattr(camera, "root_prim_path", ""))
        camera_prim = get_prim_at_path(getattr(camera, "prim_path", ""))
        if env_root is None or camera_prim is None:
            return np.eye(4, dtype=np.float32)
        env_from_camera = np.asarray(get_relative_transform(camera_prim, env_root), dtype=np.float32)

    world_from_env = _get_prim_transform_to_world(getattr(camera, "root_prim_path", ""))
    return np.asarray(world_from_env @ env_from_camera, dtype=np.float32)


def _set_camera_world_pose(camera, world_translation, world_orientation):
    world_from_camera = tf_matrix_from_pose(
        np.asarray(world_translation, dtype=np.float32),
        np.asarray(world_orientation, dtype=np.float32),
    )
    world_from_env = _get_prim_transform_to_world(getattr(camera, "root_prim_path", ""))
    env_from_world = np.linalg.inv(world_from_env)
    env_from_camera = np.asarray(env_from_world @ world_from_camera, dtype=np.float64)
    local_translation, local_orientation = pose_from_tf_matrix(env_from_camera)
    camera.set_local_pose(
        translation=np.asarray(local_translation, dtype=np.float32),
        orientation=np.asarray(local_orientation, dtype=np.float32),
        camera_axes="usd",
    )


def _compute_topdown_camera_view(workflow, camera_name: str, start_xy, goal_xy, obstacle_specs, min_height_m: float):
    cameras = getattr(workflow.task, "cameras", {})
    camera = cameras.get(camera_name)
    if camera is None:
        available = ", ".join(sorted(cameras.keys()))
        raise KeyError(f"Camera '{camera_name}' not found for topdown view compute. Available cameras: {available}")

    xy_points = [np.asarray(start_xy, dtype=np.float32), np.asarray(goal_xy, dtype=np.float32)]
    for obstacle in obstacle_specs or []:
        translation = obstacle.get("translation", [0.0, 0.0, 0.0])
        scale = obstacle.get("scale", [0.0, 0.0, 0.0])
        center_xy = np.asarray(translation[:2], dtype=np.float32)
        half_extent = 0.5 * np.asarray(scale[:2], dtype=np.float32)
        xy_points.extend(
            [
                center_xy + np.asarray([half_extent[0], half_extent[1]], dtype=np.float32),
                center_xy + np.asarray([half_extent[0], -half_extent[1]], dtype=np.float32),
                center_xy + np.asarray([-half_extent[0], half_extent[1]], dtype=np.float32),
                center_xy + np.asarray([-half_extent[0], -half_extent[1]], dtype=np.float32),
            ]
        )

    xy_stack = np.asarray(xy_points, dtype=np.float32)
    min_xy = np.min(xy_stack, axis=0)
    max_xy = np.max(xy_stack, axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    half_span = 0.5 * (max_xy - min_xy) + np.asarray([0.75, 0.75], dtype=np.float32)

    camera_matrix = np.asarray(getattr(camera, "is_camera_matrix", np.eye(3)), dtype=np.float32)
    resolution = getattr(camera, "get_resolution", None)
    if callable(resolution):
        width, height = camera.get_resolution()
    else:
        width, height = int(camera_matrix[0, 2] * 2.0), int(camera_matrix[1, 2] * 2.0)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    width = max(int(width), 1)
    height = max(int(height), 1)

    height_from_x = float(half_span[0]) * fx / max(width * 0.5, 1.0)
    height_from_y = float(half_span[1]) * fy / max(height * 0.5, 1.0)
    camera_height_m = max(float(min_height_m), height_from_x, height_from_y)
    return center_xy.astype(np.float32), float(camera_height_m)


def _update_topdown_camera_pose(
    workflow,
    robot,
    camera_name: str,
    height_m: float,
    center_xy: np.ndarray | None = None,
):
    cameras = getattr(workflow.task, "cameras", {})
    camera = cameras.get(camera_name)
    if camera is None:
        available = ", ".join(sorted(cameras.keys()))
        raise KeyError(f"Camera '{camera_name}' not found for topdown update. Available cameras: {available}")

    robot_pose = _get_robot_base_pose(robot)
    robot_position = np.asarray(robot_pose[0], dtype=np.float32)
    if center_xy is None:
        target_xy = robot_position[:2]
    else:
        target_xy = np.asarray(center_xy, dtype=np.float32)
    topdown_translation = np.asarray(
        [float(target_xy[0]), float(target_xy[1]), float(robot_position[2]) + float(height_m)],
        dtype=np.float32,
    )
    topdown_orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _set_camera_world_pose(camera, topdown_translation, topdown_orientation)


def _project_world_point_to_image(camera, camera_obs: dict, world_xyz):
    camera_matrix = np.asarray(
        camera_obs.get("camera_params", getattr(camera, "is_camera_matrix", np.eye(3))),
        dtype=np.float32,
    )
    world_from_camera = _get_camera_world_transform(camera, camera_obs=camera_obs)
    camera_from_world = np.linalg.inv(world_from_camera)
    world_point = np.asarray([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0], dtype=np.float32)
    camera_point = np.asarray(camera_from_world @ world_point, dtype=np.float32)

    depth = float(-camera_point[2])
    if depth <= 1.0e-5:
        return None

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    pixel_x = cx + fx * float(camera_point[0]) / depth
    pixel_y = cy - fy * float(camera_point[1]) / depth
    return int(round(pixel_x)), int(round(pixel_y))


def _collect_obstacle_world_corners(workflow, obstacle_spec: dict):
    name = str(obstacle_spec.get("name", ""))
    obj = getattr(workflow.task, "objects", {}).get(name)

    if obj is not None:
        center_world = np.asarray(obj.get_world_pose()[0], dtype=np.float32)
    else:
        center_world = np.asarray(obstacle_spec.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)

    scale = np.asarray(obstacle_spec.get("scale", [0.0, 0.0, 0.0]), dtype=np.float32)
    half_extent = 0.5 * scale
    corners = []
    for dx in (-half_extent[0], half_extent[0]):
        for dy in (-half_extent[1], half_extent[1]):
            for dz in (-half_extent[2], half_extent[2]):
                corners.append(
                    np.asarray(
                        [float(center_world[0] + dx), float(center_world[1] + dy), float(center_world[2] + dz)],
                        dtype=np.float32,
                    )
                )
    return corners


def _annotate_topdown_camera_frame(
    workflow,
    camera,
    camera_obs: dict,
    rgb_frame: np.ndarray,
    start_xy,
    goal_xy,
    current_xy,
    obstacle_specs,
    start_z: float,
    goal_z: float,
    current_z: float,
    path_xy=None,
):
    annotated = rgb_frame.copy()
    overlay = annotated.copy()
    image_h, image_w = annotated.shape[:2]

    def _inside_image(pixel_xy) -> bool:
        return pixel_xy is not None and 0 <= pixel_xy[0] < image_w and 0 <= pixel_xy[1] < image_h

    for obstacle in obstacle_specs or []:
        projected = []
        for world_corner in _collect_obstacle_world_corners(workflow, obstacle):
            pixel_xy = _project_world_point_to_image(camera, camera_obs, world_corner)
            if _inside_image(pixel_xy):
                projected.append(pixel_xy)
        if len(projected) < 3:
            continue

        polygon = cv2.convexHull(np.asarray(projected, dtype=np.int32))
        color = tuple(int(round(255.0 * float(channel))) for channel in obstacle.get("color", [0.8, 0.3, 0.2]))
        cv2.fillConvexPoly(overlay, polygon, color, lineType=cv2.LINE_AA)
        cv2.polylines(annotated, [polygon], True, (20, 20, 20), 2, cv2.LINE_AA)
        polygon_points = polygon.reshape(-1, 2)
        label_px = tuple(int(v) for v in np.mean(polygon_points, axis=0))
        cv2.putText(
            annotated,
            str(obstacle.get("name", "obs")),
            label_px,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0.0, dst=annotated)

    if path_xy:
        path_pixels = []
        for path_point in path_xy:
            pixel_xy = _project_world_point_to_image(
                camera,
                camera_obs,
                [float(path_point[0]), float(path_point[1]), float(current_z)],
            )
            if _inside_image(pixel_xy):
                path_pixels.append(pixel_xy)
        if len(path_pixels) >= 2:
            cv2.polylines(
                annotated,
                [np.asarray(path_pixels, dtype=np.int32)],
                False,
                (255, 215, 0),
                2,
                cv2.LINE_AA,
            )

    start_px = _project_world_point_to_image(camera, camera_obs, [float(start_xy[0]), float(start_xy[1]), float(start_z)])
    goal_px = _project_world_point_to_image(camera, camera_obs, [float(goal_xy[0]), float(goal_xy[1]), float(goal_z)])
    current_px = _project_world_point_to_image(
        camera,
        camera_obs,
        [float(current_xy[0]), float(current_xy[1]), float(current_z)],
    )

    if _inside_image(start_px):
        cv2.circle(annotated, start_px, 9, (40, 180, 80), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            "start",
            (start_px[0] + 10, start_px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 100, 40),
            2,
            cv2.LINE_AA,
        )
    if _inside_image(goal_px):
        cv2.drawMarker(annotated, goal_px, (40, 40, 220), cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            "goal",
            (goal_px[0] + 10, goal_px[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 180),
            2,
            cv2.LINE_AA,
        )
    if _inside_image(current_px):
        cv2.circle(annotated, current_px, 8, (255, 140, 0), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            "robot",
            (current_px[0] + 10, current_px[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 70, 10),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _write_topdown_mp4(
    video_path: str,
    path_xy,
    path_yaw,
    start_xy,
    goal_xy,
    goal_yaw: float,
    reached_target: bool,
    obstacle_specs=None,
    fps: int = 15,
):
    if not path_xy:
        raise ValueError("path_xy is empty, cannot export topdown MP4")

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    canvas_size = 720
    border_px = 54
    bounds = _compute_topdown_bounds(
        start_xy=start_xy,
        goal_xy=goal_xy,
        path_xy=path_xy,
        obstacle_specs=obstacle_specs,
    )
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (canvas_size, canvas_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    try:
        final_hold_frames = max(int(fps), 12)
        motion_segments = _infer_motion_segments(path_xy, path_yaw)
        reverse_events = _build_reverse_motion_events(path_xy, path_yaw)
        final_frame = None
        motion_colors = {
            "forward": (56, 104, 255),
            "reverse": (46, 180, 255),
            "stationary": (150, 150, 150),
        }
        for idx, xy in enumerate(path_xy):
            frame = np.full((canvas_size, canvas_size, 3), 248, dtype=np.uint8)
            _draw_topdown_grid(frame, bounds, canvas_size, border_px)
            for obstacle_spec in obstacle_specs or []:
                _draw_obstacle_rect(frame, obstacle_spec, bounds, canvas_size, border_px)

            start_px = _world_to_pixel(start_xy, bounds, canvas_size, border_px)
            goal_px = _world_to_pixel(goal_xy, bounds, canvas_size, border_px)
            current_px = _world_to_pixel(xy, bounds, canvas_size, border_px)

            for segment in motion_segments:
                seg_start = int(segment["start_index"])
                seg_end = min(int(segment["end_index"]), idx)
                if seg_end <= seg_start:
                    continue
                segment_points = [
                    _world_to_pixel(path_xy[point_idx], bounds, canvas_size, border_px)
                    for point_idx in range(seg_start, seg_end + 1)
                ]
                if len(segment_points) >= 2:
                    cv2.polylines(
                        frame,
                        [np.asarray(segment_points, dtype=np.int32)],
                        False,
                        color=motion_colors.get(str(segment["motion_state"]), (90, 90, 90)),
                        thickness=4,
                        lineType=cv2.LINE_AA,
                    )

            for reverse_event in reverse_events:
                if int(reverse_event["start_index"]) > idx:
                    continue
                marker_xy = reverse_event["start_xy"]
                marker_px = _world_to_pixel(marker_xy, bounds, canvas_size, border_px)
                cv2.drawMarker(frame, marker_px, (0, 120, 220), cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)

            cv2.circle(frame, start_px, 8, (40, 180, 80), -1, cv2.LINE_AA)
            cv2.circle(frame, goal_px, 10, (40, 40, 220), 2, cv2.LINE_AA)
            cv2.drawMarker(frame, goal_px, (40, 40, 220), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
            _draw_heading_arrow(frame, goal_px, float(goal_yaw), length_px=26, color=(40, 40, 220))
            cv2.circle(frame, current_px, 10, (255, 140, 0), -1, cv2.LINE_AA)
            _draw_heading_arrow(frame, current_px, float(path_yaw[idx]), length_px=26, color=(30, 30, 30))

            remaining = float(np.linalg.norm(np.asarray(goal_xy, dtype=np.float32) - np.asarray(xy, dtype=np.float32)))
            goal_yaw_error = abs(_angle_diff_rad(float(goal_yaw), float(path_yaw[idx])))
            current_motion = "stationary"
            if idx > 0:
                prev_xy = np.asarray(path_xy[idx - 1], dtype=np.float32)
                curr_xy = np.asarray(xy, dtype=np.float32)
                delta = curr_xy - prev_xy
                if float(np.linalg.norm(delta)) > 1.0e-4:
                    heading = np.asarray([math.cos(float(path_yaw[idx])), math.sin(float(path_yaw[idx]))], dtype=np.float32)
                    current_motion = "forward" if float(np.dot(delta, heading)) >= 0.0 else "reverse"
            status_text = "goal reached" if reached_target and idx == len(path_xy) - 1 else "navigating"
            cv2.putText(
                frame,
                f"Nav2 debug replay  step {idx + 1}/{len(path_xy)}",
                (24, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (32, 32, 32),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"remaining {remaining:.3f} m   {status_text}",
                (24, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (70, 70, 70),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"pose ({float(xy[0]):.2f}, {float(xy[1]):.2f})   yaw {math.degrees(float(path_yaw[idx])):.1f} deg   motion {current_motion}",
                (24, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                motion_colors.get(current_motion, (80, 80, 80)),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"goal yaw {math.degrees(float(goal_yaw)):.1f} deg   yaw err {math.degrees(goal_yaw_error):.1f} deg",
                (24, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (40, 40, 180),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            final_frame = frame

        if final_frame is None:
            raise RuntimeError("Failed to render topdown replay frames")

        outcome = "SUCCEEDED" if reached_target else "FAILED"
        cv2.putText(
            final_frame,
            f"Result: {outcome}   reverse_events: {len(reverse_events)}",
            (24, canvas_size - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 120, 20) if reached_target else (30, 30, 180),
            2,
            cv2.LINE_AA,
        )
        for _ in range(final_hold_frames):
            writer.write(final_frame)
    finally:
        writer.release()

def run_nav2_smoke(
    config_path: str,
    output_path: str,
    steps: int,
    goal_x: float,
    goal_y: float,
    goal_yaw: float,
    video_path: str = "",
    camera_name: str = "",
    camera_image_path: str = "",
    camera_video_path: str = "",
    camera_view: str = "",
    camera_height_m: float = 5.0,
    ackermann_split_steering: bool | None = None,
    ackermann_split_wheel_speeds: bool | None = None,
):
    report: dict = {
        "started_at": datetime.now().isoformat(),
        "status": "started",
        "config_path": config_path,
        "requested_steps": int(steps),
        "goal_world_xy_requested": [float(goal_x), float(goal_y)],
        "goal_world_yaw_requested": float(goal_yaw),
    }
    if video_path:
        report["video_path"] = video_path
    if camera_name:
        report["camera_name_requested"] = camera_name
    if camera_image_path:
        report["camera_image_path"] = camera_image_path
    if camera_video_path:
        report["camera_video_path"] = camera_video_path
    if camera_view:
        report["camera_view"] = camera_view
    report["camera_height_m"] = float(camera_height_m)
    report["ackermann_split_steering_override"] = ackermann_split_steering
    report["ackermann_split_wheel_speeds_override"] = ackermann_split_wheel_speeds
    report["camera_render_stride"] = int(NAV2_CAMERA_RENDER_STRIDE)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    camera_writer = None
    selected_camera_name = None
    camera_frame_count = 0
    last_camera_image_path = ""
    nav2_process = None
    nav2_log_file = None
    nav2_params_path = ""
    nav2_log_path = ""
    driver_process = None
    driver_log_file = None
    driver_cfg_path = ""
    driver_log_path = ""
    sim_clock_publisher = None
    topdown_center_xy = None
    topdown_height_m = float(camera_height_m)

    try:
        report["status"] = "initializing_environment"
        init_env()
        config = _load_render_config(config_path)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        task_cfg_path = scene_loader_cfg["cfg_path"]
        simulator_cfg = scene_loader_cfg["simulator"]
        report["original_task_cfg_path"] = task_cfg_path
        keep_task_cameras = bool(camera_image_path or camera_video_path or camera_name or camera_view)
        topdown_camera = keep_task_cameras and camera_view == "topdown"
        task_cfg_path = _prepare_headless_nav_task_cfg(
            task_cfg_path,
            output_dir or ".",
            keep_task_cameras=keep_task_cameras,
            topdown_camera=topdown_camera,
            ackermann_split_steering=ackermann_split_steering,
            ackermann_split_wheel_speeds=ackermann_split_wheel_speeds,
        )
        report["task_cfg_path"] = task_cfg_path
        report["task_cameras_enabled"] = keep_task_cameras
        report["topdown_camera_enabled"] = bool(topdown_camera)
        if keep_task_cameras:
            _enable_viewportless_syntheticdata_fallback()

        import_extensions(workflow_type)
        world = _build_world(simulator_cfg)
        workflow = create_workflow(workflow_type, world, task_cfg_path)

        report["status"] = "initializing_task"
        workflow.init_task(0)

        robot = _find_split_aloha(workflow)
        base_interface = robot.get_base_interface()
        base_cfg = dict(base_interface.get("base_cfg", {}))
        ros_cfg = dict(base_cfg.get("ros", {}))
        localization_cfg = dict(ros_cfg.get("localization", {}))
        navigator = workflow._nav2_navigators.get(robot.name)
        if navigator is None:
            raise RuntimeError("Nav2 navigator is not initialized, check split_aloha ros.nav2.enabled")
        bridge = workflow._ros_base_bridges.get(robot.name)
        controller = workflow._ros_base_command_controllers.get(robot.name)
        if bridge is not None:
            _enable_node_use_sim_time(bridge.node)
        if controller is not None:
            _enable_node_use_sim_time(controller.node)
        if navigator is not None:
            _enable_node_use_sim_time(navigator.node)
        sim_clock_publisher = _SimClockPublisher(world)
        sim_clock_publisher.publish()
        nav_step_dt = float(getattr(world, "physics_dt", 1.0 / 30.0))

        start_world_pose = _get_robot_base_pose(robot)
        start_xy = np.asarray(start_world_pose[0][:2], dtype=np.float32)
        start_yaw = _yaw_from_wxyz(start_world_pose[1])
        report["start_world_xy"] = [float(start_xy[0]), float(start_xy[1])]
        report["start_world_yaw"] = float(start_yaw)

        map_output_root = str(localization_cfg.get("map_output_dir", "output/nav2_maps")).strip() or "output/nav2_maps"
        if not os.path.isabs(map_output_root):
            map_output_root = os.path.join("/workspace/InterndataEngine", map_output_root)
        map_exporter = IsaacStaticMapExporter(
            workflow=workflow,
            robot=robot,
            base_cfg=base_cfg,
            scene_name=LOCALIZATION_SCENE_NAME,
        )
        map_artifact = map_exporter.export_map(output_dir=map_output_root, clear_center_xy=start_xy)
        report["localization_enabled"] = bool(localization_cfg.get("enabled", True))
        report["localization_mode"] = str(localization_cfg.get("mode", "static_map_truth_pose"))
        report["localization_map_yaml_path"] = str(map_artifact["yaml_path"])
        report["localization_map_pgm_path"] = str(map_artifact["pgm_path"])
        report["localization_map_resolution"] = float(map_artifact["resolution"])
        report["localization_map_origin"] = list(map_artifact["origin"])
        report["localization_map_bounds_xy"] = dict(map_artifact["bounds_xy"])
        report["localization_map_z_bounds"] = dict(map_artifact["z_bounds"])
        report["localization_robot_clear_radius_m"] = float(map_artifact["robot_clear_radius_m"])
        report["map_to_odom_xy_yaw"] = [0.0, 0.0, 0.0]

        driver_process, driver_log_file, driver_cfg_path, driver_log_path = _launch_external_base_driver(
            output_dir or ".",
            base_cfg=base_cfg,
        )
        report["external_driver_config_path"] = driver_cfg_path
        report["external_driver_log_path"] = driver_log_path
        report["external_driver_pid"] = int(driver_process.pid)

        nav2_process, nav2_log_file, nav2_params_path, nav2_log_path = _launch_real_nav2(
            output_dir or ".",
            map_yaml_path=str(map_artifact["yaml_path"]),
            base_cfg=base_cfg,
        )
        report["nav2_obstacles"] = list(workflow.task.cfg.get("nav2_obstacles", []))
        report["nav2_enabled"] = True
        report["nav2_params_path"] = nav2_params_path
        report["nav2_log_path"] = nav2_log_path
        report["nav2_launch_pid"] = int(nav2_process.pid)
        report["nav2_action_client_available"] = bool(getattr(navigator, "action_client_available", False))
        report["nav2_action_server_ready_at_init"] = bool(getattr(navigator, "action_server_ready", False))
        report["ros_base_bridge_enabled"] = bridge is not None
        report["ros_base_controller_enabled"] = controller is not None
        report["ros_external_driver_enabled"] = True
        if bridge is not None:
            report["effective_ackermann_split_steering"] = bool(getattr(bridge, "_ackermann_split_steering", False))
            report["effective_ackermann_split_wheel_speeds"] = bool(
                getattr(bridge, "_ackermann_split_wheel_speeds", False)
            )

        report["status"] = "waiting_for_nav2_action_server"
        wait_deadline = time.monotonic() + 60.0
        wait_camera_name = camera_name or ("nav2_topdown" if topdown_camera else "")
        while time.monotonic() < wait_deadline:
            if topdown_camera:
                _update_topdown_camera_pose(
                    workflow,
                    robot,
                    camera_name=wait_camera_name,
                    height_m=camera_height_m,
                )
            sim_clock_publisher.publish()
            workflow._step_world(render=False)
            sim_clock_publisher.publish()
            navigator.step(step_dt=nav_step_dt)
            if nav2_process.poll() is not None:
                raise RuntimeError(
                    f"Real Nav2 process exited before action server became ready. See log: {nav2_log_path}"
                )
            if navigator.action_server_ready:
                break
            time.sleep(0.05)

        report["nav2_action_server_ready_after_wait"] = bool(getattr(navigator, "action_server_ready", False))
        if not navigator.action_server_ready:
            raise RuntimeError(f"Timed out waiting for real Nav2 action server. See log: {nav2_log_path}")

        report["status"] = "settling_nav2_stack"
        settle_deadline = time.monotonic() + 3.0
        while time.monotonic() < settle_deadline:
            if topdown_camera:
                _update_topdown_camera_pose(
                    workflow,
                    robot,
                    camera_name=wait_camera_name,
                    height_m=camera_height_m,
                )
            sim_clock_publisher.publish()
            workflow._step_world(render=False)
            sim_clock_publisher.publish()
            navigator.step(step_dt=nav_step_dt)
            if nav2_process.poll() is not None:
                raise RuntimeError(f"Real Nav2 process exited during stack settle. See log: {nav2_log_path}")
            time.sleep(0.05)

        pre_navigation_pose = _get_robot_base_pose(robot)
        pre_navigation_xy = np.asarray(pre_navigation_pose[0][:2], dtype=np.float32)
        pre_navigation_yaw = _yaw_from_wxyz(pre_navigation_pose[1])
        report["pre_navigation_world_xy"] = [float(pre_navigation_xy[0]), float(pre_navigation_xy[1])]
        report["pre_navigation_world_yaw"] = float(pre_navigation_yaw)
        report["pre_navigation_world_delta_xy"] = [
            float(pre_navigation_xy[0] - start_xy[0]),
            float(pre_navigation_xy[1] - start_xy[1]),
        ]
        goal_xy = np.asarray([float(goal_x), float(goal_y)], dtype=np.float32)
        goal_yaw = float(goal_yaw)
        report["goal_world_xy"] = [float(goal_xy[0]), float(goal_xy[1])]
        report["goal_world_yaw"] = float(goal_yaw)
        report["goal_source"] = "explicit_final_goal"
        if topdown_camera:
            topdown_center_xy, topdown_height_m = _compute_topdown_camera_view(
                workflow=workflow,
                camera_name=wait_camera_name,
                start_xy=pre_navigation_xy,
                goal_xy=goal_xy,
                obstacle_specs=report.get("nav2_obstacles", []),
                min_height_m=camera_height_m,
            )
            report["topdown_camera_center_xy"] = [float(topdown_center_xy[0]), float(topdown_center_xy[1])]
            report["topdown_camera_height_m"] = float(topdown_height_m)

        for _ in range(6):
            if topdown_camera:
                _update_topdown_camera_pose(
                    workflow,
                    robot,
                    camera_name=wait_camera_name,
                    height_m=topdown_height_m,
                    center_xy=topdown_center_xy,
                )
            sim_clock_publisher.publish()
            workflow._step_world(render=False)
            sim_clock_publisher.publish()
        report["localization_startup_status"] = {
            "localization_converged": True,
            "localization_pose_xy_yaw": None,
            "map_to_odom_xy_yaw": [0.0, 0.0, 0.0],
            "gt_vs_localization_error": {"position_error_m": 0.0, "yaw_error_rad": 0.0},
            "localization_covariance_trace": None,
        }

        if keep_task_cameras:
            if topdown_camera and not camera_name:
                camera_name = "nav2_topdown"
            for _ in range(NAV2_CAMERA_WARMUP_RENDER_STEPS):
                if topdown_camera:
                    _update_topdown_camera_pose(
                        workflow,
                        robot,
                        camera_name=camera_name,
                        height_m=topdown_height_m,
                        center_xy=topdown_center_xy,
                    )
                sim_clock_publisher.publish()
                workflow._step_world(render=True)
                sim_clock_publisher.publish()

            if topdown_camera:
                _update_topdown_camera_pose(
                    workflow,
                    robot,
                    camera_name=camera_name,
                    height_m=topdown_height_m,
                    center_xy=topdown_center_xy,
                )
            selected_camera_name, warmup_frame = _capture_task_camera_frame(workflow, camera_name=camera_name)
            report["camera_name"] = selected_camera_name
            report["camera_resolution"] = [int(warmup_frame.shape[1]), int(warmup_frame.shape[0])]

            if camera_image_path:
                _write_rgb_png(camera_image_path, warmup_frame)
                last_camera_image_path = camera_image_path
            if camera_video_path:
                camera_writer = _open_rgb_video_writer(camera_video_path, warmup_frame)
                camera_writer.write(cv2.cvtColor(warmup_frame, cv2.COLOR_RGB2BGR))
                camera_frame_count += 1

        navigator.send_goal(float(goal_xy[0]), float(goal_xy[1]), float(goal_yaw))
        report["status"] = "goal_sent"
        report["active_goal_name"] = "final_goal"

        path_xy = []
        path_yaw = []
        nav2_status = None
        reached = False
        final_nav_xy = start_xy.copy()
        final_nav_yaw = start_yaw
        final_world_yaw = start_yaw
        camera_render_count = 0
        base_telemetry = {
            "samples": 0,
            "mean_abs_steering_sum": 0.0,
            "mean_abs_wheel_velocity_sum": 0.0,
            "max_abs_steering": 0.0,
            "max_abs_wheel_velocity": 0.0,
            "max_abs_steering_tracking_error": 0.0,
            "max_abs_wheel_velocity_tracking_error": 0.0,
            "final_tracking_state": {},
            "final_state": {},
        }

        for step_idx in range(max(int(steps), 0)):
            capture_camera_this_step = bool(keep_task_cameras and step_idx % NAV2_CAMERA_RENDER_STRIDE == 0)
            if capture_camera_this_step and topdown_camera:
                _update_topdown_camera_pose(
                    workflow,
                    robot,
                    camera_name=selected_camera_name or camera_name,
                    height_m=topdown_height_m,
                    center_xy=topdown_center_xy,
                )
            sim_clock_publisher.publish()
            workflow._step_world(render=capture_camera_this_step)
            sim_clock_publisher.publish()
            navigator.step(step_dt=nav_step_dt)
            if capture_camera_this_step:
                camera_render_count += 1
            if nav2_process.poll() is not None:
                raise RuntimeError(f"Real Nav2 process exited during navigation. See log: {nav2_log_path}")

            pose = _get_robot_base_pose(robot)
            current_xy = np.asarray(pose[0][:2], dtype=np.float32)
            current_yaw = _yaw_from_wxyz(pose[1])
            path_xy.append((float(current_xy[0]), float(current_xy[1])))
            path_yaw.append(float(current_yaw))

            base_joint_state = robot.get_base_joint_state()
            mean_abs_steering = float(np.mean(np.abs(base_joint_state["steering_positions"])))
            mean_abs_wheel_velocity = float(np.mean(np.abs(base_joint_state["wheel_velocities"])))
            base_telemetry["samples"] += 1
            base_telemetry["mean_abs_steering_sum"] += mean_abs_steering
            base_telemetry["mean_abs_wheel_velocity_sum"] += mean_abs_wheel_velocity
            base_telemetry["max_abs_steering"] = max(base_telemetry["max_abs_steering"], mean_abs_steering)
            base_telemetry["max_abs_wheel_velocity"] = max(
                base_telemetry["max_abs_wheel_velocity"],
                float(np.max(np.abs(base_joint_state["wheel_velocities"]))),
            )
            base_telemetry["final_state"] = {
                "steering_positions": [float(v) for v in base_joint_state["steering_positions"]],
                "wheel_positions": [float(v) for v in base_joint_state["wheel_positions"]],
                "steering_velocities": [float(v) for v in base_joint_state["steering_velocities"]],
                "wheel_velocities": [float(v) for v in base_joint_state["wheel_velocities"]],
            }
            if bridge is not None:
                requested_steering = np.asarray(getattr(bridge, "_last_requested_steering", []), dtype=np.float32)
                applied_steering = np.asarray(getattr(bridge, "_last_applied_steering", []), dtype=np.float32)
                requested_wheels = np.asarray(getattr(bridge, "_last_requested_wheel_velocities", []), dtype=np.float32)
                steering_tracking_error = (
                    np.asarray(base_joint_state["steering_positions"], dtype=np.float32) - requested_steering
                    if requested_steering.size == base_joint_state["steering_positions"].size
                    else np.zeros_like(base_joint_state["steering_positions"], dtype=np.float32)
                )
                wheel_velocity_tracking_error = (
                    np.asarray(base_joint_state["wheel_velocities"], dtype=np.float32) - requested_wheels
                    if requested_wheels.size == base_joint_state["wheel_velocities"].size
                    else np.zeros_like(base_joint_state["wheel_velocities"], dtype=np.float32)
                )
                base_telemetry["max_abs_steering_tracking_error"] = max(
                    base_telemetry["max_abs_steering_tracking_error"],
                    float(np.max(np.abs(steering_tracking_error))),
                )
                base_telemetry["max_abs_wheel_velocity_tracking_error"] = max(
                    base_telemetry["max_abs_wheel_velocity_tracking_error"],
                    float(np.max(np.abs(wheel_velocity_tracking_error))),
                )
                base_telemetry["final_tracking_state"] = {
                    "requested_steering_positions": [float(v) for v in requested_steering],
                    "applied_steering_positions": [float(v) for v in applied_steering],
                    "requested_wheel_velocities": [float(v) for v in requested_wheels],
                    "steering_tracking_error": [float(v) for v in steering_tracking_error],
                    "wheel_velocity_tracking_error": [float(v) for v in wheel_velocity_tracking_error],
                }

            if capture_camera_this_step:
                _, rgb_frame = _capture_task_camera_frame(workflow, camera_name=selected_camera_name or camera_name)
                if camera_image_path:
                    _write_rgb_png(camera_image_path, rgb_frame)
                    last_camera_image_path = camera_image_path
                if camera_writer is not None:
                    camera_writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                    camera_frame_count += 1

            nav_x, nav_y, nav_yaw = navigator.get_current_pose_xy_yaw()
            final_nav_xy = np.asarray([float(nav_x), float(nav_y)], dtype=np.float32)
            final_nav_yaw = float(nav_yaw)
            nav2_status = navigator.latest_result_status
            active_goal_dist = float(np.linalg.norm(goal_xy - current_xy))
            active_goal_yaw_error = abs(_angle_diff_rad(goal_yaw, float(current_yaw)))
            if (
                active_goal_dist <= NAV2_SUCCESS_POSITION_TOLERANCE_M
                and active_goal_yaw_error <= NAV2_SUCCESS_YAW_TOLERANCE_RAD
            ):
                reached = True
                break
            if nav2_status is not None:
                break

        final_world_pose = _get_robot_base_pose(robot)
        final_xy = np.asarray(final_world_pose[0][:2], dtype=np.float32)
        final_world_yaw = _yaw_from_wxyz(final_world_pose[1])
        final_dist_to_goal = float(np.linalg.norm(goal_xy - final_xy))
        world_planar_displacement = float(np.linalg.norm(final_xy - start_xy))
        nav_planar_displacement = float(np.linalg.norm(final_nav_xy - start_xy))
        nav_dist_to_goal = float(np.linalg.norm(goal_xy - final_nav_xy))
        final_yaw_error = abs(_angle_diff_rad(goal_yaw, final_world_yaw))
        reverse_motion_events = _build_reverse_motion_events(path_xy, path_yaw)
        motion_segments = _infer_motion_segments(path_xy, path_yaw)

        if (
            final_dist_to_goal <= NAV2_SUCCESS_POSITION_TOLERANCE_M
            and nav_dist_to_goal <= NAV2_SUCCESS_POSITION_TOLERANCE_M
            and final_yaw_error <= NAV2_SUCCESS_YAW_TOLERANCE_RAD
        ):
            reached = True

        report["status"] = "completed"
        report["result"] = {
            "reached_target": bool(reached),
            "nav2_result_status": nav2_status,
            "nav2_action_server_ready": bool(getattr(navigator, "action_server_ready", False)),
            "success_position_tolerance_m": float(NAV2_SUCCESS_POSITION_TOLERANCE_M),
            "success_yaw_tolerance_rad": float(NAV2_SUCCESS_YAW_TOLERANCE_RAD),
            "final_distance_to_goal": final_dist_to_goal,
            "final_yaw_error_rad": final_yaw_error,
            "final_world_xy": [float(final_xy[0]), float(final_xy[1])],
            "final_world_yaw": float(final_world_yaw),
            "world_planar_displacement": world_planar_displacement,
            "final_nav_xy": [float(final_nav_xy[0]), float(final_nav_xy[1])],
            "final_nav_yaw": final_nav_yaw,
            "nav_distance_to_goal": nav_dist_to_goal,
            "nav_planar_displacement": nav_planar_displacement,
            "trajectory_point_count": len(path_xy),
            "navigator_pending_goal": bool(navigator._pending_goal_pose is not None),
            "nav2_latest_plan_point_count": len(getattr(navigator, "latest_plan_xy", [])),
            "nav2_latest_plan_xy": [[float(x), float(y)] for x, y in getattr(navigator, "latest_plan_xy", [])],
            "nav2_latest_plan_source_topic": str(getattr(navigator, "latest_plan_source_topic", "")),
            "trajectory_xy": [[float(x), float(y)] for x, y in path_xy],
            "trajectory_yaw": [float(yaw) for yaw in path_yaw],
            "motion_segments": motion_segments,
            "reverse_motion_events": reverse_motion_events,
            "localization_pose_xy_yaw": None,
            "map_to_odom_xy_yaw": [0.0, 0.0, 0.0],
            "gt_vs_localization_error": {"position_error_m": 0.0, "yaw_error_rad": 0.0},
            "localization_covariance_trace": None,
            "localization_converged": True,
            "localization_trace": [],
        }
        if bridge is not None:
            report["result"]["bridge_has_motion_mode"] = bool(getattr(bridge, "_has_motion_mode", False))
            report["result"]["bridge_motion_mode_message_count"] = int(getattr(bridge, "_motion_mode_message_count", 0))
            report["result"]["bridge_driver_command_message_count"] = int(
                getattr(bridge, "_driver_command_message_count", 0)
            )
            report["result"]["bridge_pending_driver_command_count"] = int(
                getattr(bridge, "_pending_driver_command_count", 0)
            )
            report["result"]["bridge_applied_driver_command_count"] = int(
                getattr(bridge, "_applied_driver_command_count", 0)
            )
            report["result"]["bridge_last_linear_speed"] = float(getattr(bridge._command, "linear_speed", 0.0))
            report["result"]["bridge_last_lateral_speed"] = float(getattr(bridge._command, "lateral_speed", 0.0))
            report["result"]["bridge_last_steering_angle"] = float(getattr(bridge._command, "steering_angle", 0.0))
            report["result"]["bridge_last_angular_speed"] = float(getattr(bridge._command, "angular_speed", 0.0))
            report["result"]["bridge_last_requested_steering"] = [
                float(v) for v in getattr(bridge, "_last_requested_steering", [])
            ]
            report["result"]["bridge_last_requested_wheel_velocities"] = [
                float(v) for v in getattr(bridge, "_last_requested_wheel_velocities", [])
            ]
            report["result"]["bridge_last_applied_steering"] = [
                float(v) for v in getattr(bridge, "_last_applied_steering", [])
            ]
            report["result"]["bridge_virtual_xy"] = [float(getattr(bridge, "_virtual_x", 0.0)), float(getattr(bridge, "_virtual_y", 0.0))]
            report["result"]["bridge_virtual_yaw"] = float(getattr(bridge, "_virtual_yaw", 0.0))
        if controller is not None:
            report["result"]["controller_received_cmd_vel_count"] = int(
                getattr(controller, "_received_cmd_vel_count", 0)
            )
            report["result"]["controller_published_command_count"] = int(
                getattr(controller, "_published_command_count", 0)
            )
            report["result"]["controller_last_received_cmd_vel"] = dict(
                getattr(controller, "_last_received_cmd_vel", {})
            )
            report["result"]["controller_last_published_motion_mode"] = getattr(
                controller, "_last_published_motion_mode", None
            )
        if base_telemetry["samples"] > 0:
            report["result"]["base_joint_state_final"] = dict(base_telemetry["final_state"])
            report["result"]["base_joint_state_metrics"] = {
                "sample_count": int(base_telemetry["samples"]),
                "mean_abs_steering_position": float(
                    base_telemetry["mean_abs_steering_sum"] / base_telemetry["samples"]
                ),
                "mean_abs_wheel_velocity": float(
                    base_telemetry["mean_abs_wheel_velocity_sum"] / base_telemetry["samples"]
                ),
                "max_abs_steering_position": float(base_telemetry["max_abs_steering"]),
                "max_abs_wheel_velocity": float(base_telemetry["max_abs_wheel_velocity"]),
                "max_abs_steering_tracking_error": float(base_telemetry["max_abs_steering_tracking_error"]),
                "max_abs_wheel_velocity_tracking_error": float(base_telemetry["max_abs_wheel_velocity_tracking_error"]),
            }
            if base_telemetry["final_tracking_state"]:
                report["result"]["base_joint_tracking_final"] = dict(base_telemetry["final_tracking_state"])
        if video_path:
            _write_topdown_mp4(
                video_path=video_path,
                path_xy=path_xy,
                path_yaw=path_yaw,
                start_xy=start_xy,
                goal_xy=goal_xy,
                goal_yaw=float(goal_yaw),
                reached_target=reached,
                obstacle_specs=report.get("nav2_obstacles", []),
            )
        if keep_task_cameras:
            report["result"]["camera_name"] = selected_camera_name
            report["result"]["camera_frame_count"] = int(camera_frame_count)
            report["result"]["camera_render_count"] = int(camera_render_count)
            report["result"]["camera_view"] = camera_view or "task"
            if camera_image_path:
                report["result"]["camera_image_path"] = last_camera_image_path or camera_image_path
            if camera_video_path:
                report["result"]["camera_video_path"] = camera_video_path
        if not reached:
            raise RuntimeError(
                "Nav2 smoke did not reach target: "
                f"nav_status={nav2_status}, nav_distance_to_goal={nav_dist_to_goal:.3f}, "
                f"world_distance_to_goal={final_dist_to_goal:.3f}, "
                f"world_displacement={world_planar_displacement:.3f}, nav_displacement={nav_planar_displacement:.3f}"
            )
    except Exception as exc:  # pylint: disable=broad-except
        report["status"] = "error"
        report["error"] = str(exc)
        if nav2_process is not None:
            report["nav2_process_returncode"] = nav2_process.poll()
        if driver_process is not None:
            report["external_driver_returncode"] = driver_process.poll()
        if nav2_log_path:
            report["nav2_log_tail"] = _read_log_tail(nav2_log_path)
        if driver_log_path:
            report["external_driver_log_tail"] = _read_log_tail(driver_log_path)
        report["traceback"] = traceback.format_exc()
    finally:
        _stop_subprocess(nav2_process, nav2_log_file)
        _stop_subprocess(driver_process, driver_log_file)
        if camera_writer is not None:
            camera_writer.release()
        if sim_clock_publisher is not None:
            sim_clock_publisher.destroy()
        report["finished_at"] = datetime.now().isoformat()
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(json.dumps(report, indent=2))
        simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="configs/simbox/de_plan_with_render_template.yaml",
        help="Path to the SimBox render config used to load the task",
    )
    parser.add_argument(
        "--output-path",
        default="output/ros_bridge/split_aloha_nav2_smoke.json",
        help="Where to save the Nav2 smoke report JSON",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1200,
        help="How many world steps to execute after sending Nav2 goal",
    )
    parser.add_argument(
        "--goal-x",
        type=float,
        default=NAV2_DEFAULT_GOAL_X_M,
        help="World-frame x coordinate of the explicit final navigation goal.",
    )
    parser.add_argument(
        "--goal-y",
        type=float,
        default=NAV2_DEFAULT_GOAL_Y_M,
        help="World-frame y coordinate of the explicit final navigation goal.",
    )
    parser.add_argument(
        "--goal-yaw",
        type=float,
        default=NAV2_DEFAULT_GOAL_YAW_RAD,
        help="World-frame yaw of the explicit final navigation goal in radians.",
    )
    parser.add_argument(
        "--video-path",
        default="",
        help="Optional path to export a topdown MP4 replay of the executed trajectory",
    )
    parser.add_argument(
        "--camera-name",
        default="",
        help="Optional task camera name to capture. Defaults to the first task camera when camera export is enabled.",
    )
    parser.add_argument(
        "--camera-image-path",
        default="",
        help="Optional path to export the latest Isaac task camera frame as a PNG.",
    )
    parser.add_argument(
        "--camera-video-path",
        default="",
        help="Optional path to export an MP4 built from Isaac task camera frames.",
    )
    parser.add_argument(
        "--camera-view",
        default="",
        choices=["", "task", "topdown"],
        help="Optional camera layout. Use 'topdown' for a nadir camera above the robot.",
    )
    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=5.0,
        help="Height of the topdown camera above the robot base when --camera-view=topdown.",
    )
    parser.add_argument(
        "--replay-from-report",
        default="",
        help="Optional path to a previously saved Nav2 report JSON. When set, skip live Nav2 and render Isaac camera replay from the recorded trajectory.",
    )
    parser.add_argument(
        "--ackermann-split-steering",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to split inner/outer steering angles in dual-Ackermann mode.",
    )
    parser.add_argument(
        "--ackermann-split-wheel-speeds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether to split inner/outer wheel speeds in dual-Ackermann mode.",
    )
    args = parser.parse_args()

    if args.replay_from_report:
        _run_camera_replay_from_report(
            config_path=args.config_path,
            replay_report_path=args.replay_from_report,
            output_path=args.output_path,
            video_path=args.video_path,
            camera_name=args.camera_name,
            camera_image_path=args.camera_image_path,
            camera_video_path=args.camera_video_path,
            camera_view=args.camera_view,
            camera_height_m=args.camera_height_m,
        )
    else:
        run_nav2_smoke(
            config_path=args.config_path,
            output_path=args.output_path,
            steps=args.steps,
            goal_x=args.goal_x,
            goal_y=args.goal_y,
            goal_yaw=args.goal_yaw,
            video_path=args.video_path,
            camera_name=args.camera_name,
            camera_image_path=args.camera_image_path,
            camera_video_path=args.camera_video_path,
            camera_view=args.camera_view,
            camera_height_m=args.camera_height_m,
            ackermann_split_steering=args.ackermann_split_steering,
            ackermann_split_wheel_speeds=args.ackermann_split_wheel_speeds,
        )
