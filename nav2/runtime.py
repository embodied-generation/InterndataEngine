"""Workflow-side external Nav2 session manager for split Isaac/ROS deployments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
import math
import os
import shutil
from typing import Optional

import yaml

from .protocol import safe_name

NAV2_DEFAULT_MAX_ACKERMANN_STEER_RAD = 0.6981
NAV2_DEFAULT_POSITION_TOLERANCE_M = 0.10
NAV2_DEFAULT_YAW_TOLERANCE_RAD = 0.10

LOGGER = logging.getLogger("simbox.nav2_skill")
DEFAULT_NAV2_SKILL_FOOTPRINT_POINTS = [
    [0.36, 0.24],
    [0.32, 0.29],
    [-0.32, 0.29],
    [-0.36, 0.24],
    [-0.36, -0.24],
    [-0.32, -0.29],
    [0.32, -0.29],
    [0.36, -0.24],
]
DEFAULT_NAV2_SKILL_INFLATION_RADIUS_M = 0.34
DEFAULT_NAV2_SKILL_MIN_TURN_RADIUS_M = 0.47644


def _angle_diff_rad(target: float, current: float) -> float:
    return math.atan2(math.sin(float(target) - float(current)), math.cos(float(target) - float(current)))


def _yaw_from_wxyz(q_wxyz) -> float:
    w = float(q_wxyz[0])
    x = float(q_wxyz[1])
    y = float(q_wxyz[2])
    z = float(q_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = float(bx) - float(ax)
    aby = float(by) - float(ay)
    apx = float(px) - float(ax)
    apy = float(py) - float(ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1.0e-12:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    closest_x = float(ax) + t * abx
    closest_y = float(ay) + t * aby
    return math.hypot(float(px) - closest_x, float(py) - closest_y)


def _footprint_inscribed_radius(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    radius = float("inf")
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        radius = min(
            radius,
            _distance_point_to_segment(
                0.0,
                0.0,
                float(point[0]),
                float(point[1]),
                float(next_point[0]),
                float(next_point[1]),
            ),
        )
    return 0.0 if not math.isfinite(radius) else float(radius)


def _runtime_control_debug_snapshot(robot) -> dict:
    snapshot = {}

    bridge = getattr(robot, "_simbox_ros_base_bridge", None)
    if bridge is not None:
        requested_steering = getattr(bridge, "_last_requested_steering", None)
        requested_wheel_velocities = getattr(bridge, "_last_requested_wheel_velocities", None)
        bridge_info = {
            "received_cmd_vel_count": int(getattr(bridge, "_received_cmd_vel_count", 0)),
            "driver_command_message_count": int(getattr(bridge, "_driver_command_message_count", 0)),
            "pending_driver_command_count": int(getattr(bridge, "_pending_driver_command_count", 0)),
            "applied_driver_command_count": int(getattr(bridge, "_applied_driver_command_count", 0)),
            "last_received_cmd_vel": dict(getattr(bridge, "_last_received_cmd_vel", {}) or {}),
            "recent_cmd_vel_history": list(getattr(bridge, "_debug_cmd_vel_history", []))[-20:],
            "steering_command_sign": float(getattr(bridge, "_steering_command_sign", 1.0)),
            "virtual_odom_enabled": False,
            "last_requested_steering": [float(value) for value in list(requested_steering)]
            if requested_steering is not None
            else [],
            "last_requested_wheel_velocities": [float(value) for value in list(requested_wheel_velocities)]
            if requested_wheel_velocities is not None
            else [],
            "last_published_pose": dict(getattr(bridge, "_last_published_pose_debug", {}) or {}),
            "recent_command_history": list(getattr(bridge, "_debug_command_history", []))[-20:],
        }
        active_command = getattr(bridge, "_command", None)
        if active_command is not None:
            bridge_info["active_command"] = {
                "vx_body": float(getattr(active_command, "vx_body", 0.0)),
                "vy_body": float(getattr(active_command, "vy_body", 0.0)),
                "wz_body": float(getattr(active_command, "wz_body", 0.0)),
                "received_time_sec": float(getattr(active_command, "received_time_sec", 0.0)),
            }
        snapshot["bridge"] = bridge_info

    return snapshot


def _format_nav2_footprint(points: list[list[float]]) -> str:
    return "[" + ", ".join(f"[{float(x):.3f}, {float(y):.3f}]" for x, y in points) + "]"


def _nav2_skill_cfg(base_cfg: dict) -> dict:
    return dict(base_cfg.get("nav2_skill", {}))


def configure_base_cfg_for_nav2_skill(
    base_cfg: dict,
    *,
    map_output_dir: str = "output/nav2_maps",
    map_resolution: float = 0.05,
    map_z_min: float = 0.0,
    map_z_max: float = 0.35,
    position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
    yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
):
    """Normalize mobile-base config for external compose-managed Nav2 sessions."""

    base_cfg = deepcopy(base_cfg)
    ros_cfg = base_cfg.setdefault("ros", {})

    ros_cfg["cmd_vel_topic"] = str(ros_cfg.get("cmd_vel_topic", "/cmd_vel"))
    ros_cfg.pop("command_topic", None)
    ros_cfg.pop("motion_mode_topic", None)
    ros_cfg.pop("command_type", None)
    ros_cfg.pop("internal_cmdvel_controller_enabled", None)
    current_limit = ros_cfg.get("max_steer_angle_ackermann", NAV2_DEFAULT_MAX_ACKERMANN_STEER_RAD)
    ros_cfg["max_steer_angle_ackermann"] = float(
        min(float(current_limit), float(NAV2_DEFAULT_MAX_ACKERMANN_STEER_RAD))
    )
    virtual_odom_cfg = ros_cfg.setdefault("virtual_odom", {})
    virtual_odom_cfg["enabled"] = False
    virtual_odom_cfg["publish_twist"] = True
    virtual_odom_cfg["use_world_z"] = True
    virtual_odom_cfg["default_z"] = float(virtual_odom_cfg.get("default_z", 0.0))

    localization_cfg = ros_cfg.setdefault("localization", {})
    localization_cfg["enabled"] = True
    localization_cfg["mode"] = "static_map_truth_pose"
    localization_cfg["map_resolution"] = float(map_resolution)
    localization_cfg["map_output_dir"] = str(map_output_dir)
    localization_cfg["map_z_min"] = float(map_z_min)
    localization_cfg["map_z_max"] = float(map_z_max)
    localization_cfg["map_frame"] = str(localization_cfg.get("map_frame", "map"))
    localization_cfg["odom_frame"] = str(localization_cfg.get("odom_frame", ros_cfg.get("odom_frame", "odom")))
    localization_cfg["base_frame"] = str(localization_cfg.get("base_frame", ros_cfg.get("base_frame", "base_link")))

    nav2_cfg = ros_cfg.setdefault("nav2", {})
    nav2_cfg["enabled"] = True
    nav2_cfg["global_frame"] = str(localization_cfg.get("map_frame", "map"))
    nav2_cfg["robot_base_frame"] = str(localization_cfg.get("base_frame", ros_cfg.get("base_frame", "base_link")))
    nav2_cfg["skill_managed"] = True
    nav2_cfg["runtime_mode"] = "external_compose"
    nav2_cfg["stack_request_root"] = str(nav2_cfg.get("stack_request_root", "output/ros_bridge/runtime_requests"))
    nav2_cfg["stack_status_root"] = str(nav2_cfg.get("stack_status_root", "output/ros_bridge/runtime_status"))
    nav2_cfg["goal_request_root"] = str(nav2_cfg.get("goal_request_root", "output/ros_bridge/goal_requests"))
    nav2_cfg["goal_status_root"] = str(nav2_cfg.get("goal_status_root", "output/ros_bridge/goal_status"))
    nav2_cfg["goal_result_root"] = str(nav2_cfg.get("goal_result_root", "output/ros_bridge/goal_result"))
    nav2_cfg["stack_reuse"] = bool(nav2_cfg.get("stack_reuse", True))
    nav2_cfg["goal_transport"] = str(nav2_cfg.get("goal_transport", "ros_topic_bridge"))
    nav2_cfg["load_map_service"] = str(nav2_cfg.get("load_map_service", "/map_server/load_map"))
    nav2_cfg["clear_global_costmap_service"] = str(
        nav2_cfg.get("clear_global_costmap_service", "/global_costmap/clear_entirely_global_costmap")
    )
    nav2_cfg["clear_local_costmap_service"] = str(
        nav2_cfg.get("clear_local_costmap_service", "/local_costmap/clear_entirely_local_costmap")
    )
    nav2_cfg["bridge_map_update_topic"] = str(nav2_cfg.get("bridge_map_update_topic", "/simbox/nav_bridge/map_update"))
    nav2_cfg["bridge_goal_topic"] = str(nav2_cfg.get("bridge_goal_topic", "/simbox/nav_bridge/goal"))
    nav2_cfg["bridge_cancel_topic"] = str(nav2_cfg.get("bridge_cancel_topic", "/simbox/nav_bridge/cancel"))
    nav2_cfg["bridge_status_topic"] = str(nav2_cfg.get("bridge_status_topic", "/simbox/nav_bridge/status"))
    nav2_cfg["bridge_result_topic"] = str(nav2_cfg.get("bridge_result_topic", "/simbox/nav_bridge/result"))
    nav2_cfg["bridge_alive_timeout_sec"] = float(nav2_cfg.get("bridge_alive_timeout_sec", 3.0))

    base_cfg["ackermann_split_steering"] = True
    base_cfg["ackermann_split_wheel_speeds"] = True
    base_cfg.setdefault("nav2_skill", {})
    base_cfg["nav2_skill"]["position_tolerance_m"] = float(position_tolerance_m)
    base_cfg["nav2_skill"]["yaw_tolerance_rad"] = float(yaw_tolerance_rad)
    return deepcopy(base_cfg)


def configure_robot_for_nav2_skill(
    robot,
    *,
    map_output_dir: str = "output/nav2_maps",
    map_resolution: float = 0.05,
    map_z_min: float = 0.0,
    map_z_max: float = 0.35,
    position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
    yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
):
    base_cfg = configure_base_cfg_for_nav2_skill(
        getattr(robot, "base_cfg", {}),
        map_output_dir=map_output_dir,
        map_resolution=map_resolution,
        map_z_min=map_z_min,
        map_z_max=map_z_max,
        position_tolerance_m=position_tolerance_m,
        yaw_tolerance_rad=yaw_tolerance_rad,
    )
    robot.base_cfg = base_cfg
    return deepcopy(base_cfg)


def generate_nav2_bringup_artifacts(
    output_dir: str,
    *,
    base_cfg: dict,
    map_yaml_path: str,
    position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
    yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
    params_filename: str = "split_aloha_nav2_skill_params.yaml",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    nav_to_pose_bt, nav_through_poses_bt = _write_nav2_bt_files(output_dir, base_cfg)
    params = _build_nav2_params(
        nav_to_pose_bt=nav_to_pose_bt,
        nav_through_poses_bt=nav_through_poses_bt,
        base_cfg=base_cfg,
        position_tolerance_m=position_tolerance_m,
        yaw_tolerance_rad=yaw_tolerance_rad,
    )
    params["map_server"]["ros__parameters"]["yaml_filename"] = str(map_yaml_path)
    params_path = os.path.join(output_dir, params_filename)
    with open(params_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(params, handle, sort_keys=False)
    return {
        "params_path": params_path,
        "params": params,
        "nav_to_pose_bt": nav_to_pose_bt,
        "nav_through_poses_bt": nav_through_poses_bt,
    }


def _write_nav2_bt_files(output_dir: str, base_cfg: dict) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    nav_to_pose_bt_path = os.path.join(output_dir, "navigate_to_pose_w_replanning_no_motion_recovery.xml")
    nav_through_poses_bt_path = os.path.join(
        output_dir,
        "navigate_through_poses_w_replanning_no_motion_recovery.xml",
    )
    nav2_skill_cfg = _nav2_skill_cfg(base_cfg)
    bt_cfg = dict(nav2_skill_cfg.get("bt_navigator", {}))
    replanning_hz = float(bt_cfg.get("replanning_hz", 0.25))
    navigate_recovery_retries = int(bt_cfg.get("navigate_recovery_retries", 4))
    compute_path_retries = int(bt_cfg.get("compute_path_retries", 1))
    follow_path_retries = int(bt_cfg.get("follow_path_retries", 1))
    remove_passed_goals_radius = float(bt_cfg.get("remove_passed_goals_radius", 0.7))
    wait_duration = float(bt_cfg.get("wait_duration", 2.0))

    nav_to_pose_bt = f"""<!-- Holonomic Nav2 navigation without recovery/fallback behaviors. -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateWithReplanning">
      <RateController hz="{replanning_hz}">
        <ComputePathToPose goal="{{goal}}" path="{{raw_path}}" planner_id="GridBased"/>
      </RateController>
      <SmoothPath unsmoothed_path="{{raw_path}}" smoothed_path="{{path}}" smoother_id="simple_smoother"/>
      <FollowPath path="{{path}}" controller_id="FollowPath"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
"""

    nav_through_poses_bt = f"""<!-- Holonomic Nav2 navigation through poses without recovery/fallback behaviors. -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateThroughPosesWithReplanning">
      <RateController hz="{replanning_hz}">
        <Sequence>
          <RemovePassedGoals input_goals="{{goals}}" output_goals="{{goals}}" radius="{remove_passed_goals_radius}"/>
          <ComputePathThroughPoses goals="{{goals}}" path="{{raw_path}}" planner_id="GridBased"/>
        </Sequence>
      </RateController>
      <SmoothPath unsmoothed_path="{{raw_path}}" smoothed_path="{{path}}" smoother_id="simple_smoother"/>
      <FollowPath path="{{path}}" controller_id="FollowPath"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
"""

    with open(nav_to_pose_bt_path, "w", encoding="utf-8") as file:
        file.write(nav_to_pose_bt)
    with open(nav_through_poses_bt_path, "w", encoding="utf-8") as file:
        file.write(nav_through_poses_bt)
    return nav_to_pose_bt_path, nav_through_poses_bt_path


def _build_nav2_params(
    nav_to_pose_bt: str,
    nav_through_poses_bt: str,
    base_cfg: dict,
    *,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
):
    ros_cfg = dict(base_cfg.get("ros", {}))
    nav2_cfg = dict(ros_cfg.get("nav2", {}))
    localization_cfg = dict(ros_cfg.get("localization", {}))
    map_frame = str(localization_cfg.get("map_frame", nav2_cfg.get("global_frame", "map")))
    odom_frame = str(localization_cfg.get("odom_frame", ros_cfg.get("odom_frame", "odom")))
    base_frame = str(
        localization_cfg.get("base_frame", nav2_cfg.get("robot_base_frame", ros_cfg.get("base_frame", "base_link")))
    )
    nav2_skill_cfg = _nav2_skill_cfg(base_cfg)
    footprint_points = nav2_skill_cfg.get("footprint_points", DEFAULT_NAV2_SKILL_FOOTPRINT_POINTS)
    footprint = _format_nav2_footprint(footprint_points)
    inflation_radius = float(nav2_skill_cfg.get("inflation_radius_m", DEFAULT_NAV2_SKILL_INFLATION_RADIUS_M))
    inscribed_radius = _footprint_inscribed_radius(footprint_points)
    inflation_radius = max(inflation_radius, inscribed_radius)
    minimum_turning_radius = float(nav2_skill_cfg.get("minimum_turning_radius_m", DEFAULT_NAV2_SKILL_MIN_TURN_RADIUS_M))
    bt_cfg = dict(nav2_skill_cfg.get("bt_navigator", {}))
    bt_plugins = list(
        bt_cfg.get(
            "plugin_lib_names",
            [
                "nav2_compute_path_to_pose_action_bt_node",
                "nav2_compute_path_through_poses_action_bt_node",
                "nav2_smooth_path_action_bt_node",
                "nav2_follow_path_action_bt_node",
                "nav2_clear_costmap_service_bt_node",
                "nav2_goal_updated_condition_bt_node",
                "nav2_remove_passed_goals_action_bt_node",
                "nav2_rate_controller_bt_node",
                "nav2_pipeline_sequence_bt_node",
                "nav2_navigate_to_pose_action_bt_node",
                "nav2_navigate_through_poses_action_bt_node",
            ],
        )
    )
    controller_cfg = dict(nav2_skill_cfg.get("controller_server", {}))
    progress_checker_cfg = dict(controller_cfg.get("progress_checker", {}))
    goal_checker_cfg = dict(controller_cfg.get("goal_checker", {}))
    follow_path_cfg = dict(controller_cfg.get("follow_path", {}))
    local_costmap_cfg = dict(nav2_skill_cfg.get("local_costmap", {}))
    global_costmap_cfg = dict(nav2_skill_cfg.get("global_costmap", {}))
    planner_cfg = dict(nav2_skill_cfg.get("planner_server", {}))
    smoother_cfg = dict(nav2_skill_cfg.get("smoother_server", {}))
    behavior_cfg = dict(nav2_skill_cfg.get("behavior_server", {}))
    waypoint_cfg = dict(nav2_skill_cfg.get("waypoint_follower", {}))
    velocity_smoother_cfg = dict(nav2_skill_cfg.get("velocity_smoother", {}))

    follow_path_plugin = str(follow_path_cfg.get("plugin", "nav2_mppi_controller::MPPIController"))
    if follow_path_plugin == "dwb_core::DWBLocalPlanner":
        follow_path_params = {
            "plugin": follow_path_plugin,
            "debug_trajectory_details": bool(follow_path_cfg.get("debug_trajectory_details", False)),
            "short_circuit_trajectory_evaluation": bool(
                follow_path_cfg.get("short_circuit_trajectory_evaluation", True)
            ),
            "stateful": bool(follow_path_cfg.get("stateful", True)),
            "min_vel_x": float(follow_path_cfg.get("min_vel_x", -0.35)),
            "max_vel_x": float(follow_path_cfg.get("max_vel_x", 0.35)),
            "min_vel_y": float(follow_path_cfg.get("min_vel_y", -0.25)),
            "max_vel_y": float(follow_path_cfg.get("max_vel_y", 0.25)),
            "max_vel_theta": float(follow_path_cfg.get("max_vel_theta", 0.60)),
            "min_speed_xy": float(follow_path_cfg.get("min_speed_xy", 0.0)),
            "max_speed_xy": float(follow_path_cfg.get("max_speed_xy", 0.40)),
            "min_speed_theta": float(follow_path_cfg.get("min_speed_theta", 0.0)),
            "acc_lim_x": float(follow_path_cfg.get("acc_lim_x", 0.35)),
            "acc_lim_y": float(follow_path_cfg.get("acc_lim_y", 0.35)),
            "acc_lim_theta": float(follow_path_cfg.get("acc_lim_theta", 0.70)),
            "decel_lim_x": float(follow_path_cfg.get("decel_lim_x", -0.35)),
            "decel_lim_y": float(follow_path_cfg.get("decel_lim_y", -0.35)),
            "decel_lim_theta": float(follow_path_cfg.get("decel_lim_theta", -0.70)),
            "vx_samples": int(follow_path_cfg.get("vx_samples", 15)),
            "vy_samples": int(follow_path_cfg.get("vy_samples", 15)),
            "vtheta_samples": int(follow_path_cfg.get("vtheta_samples", 20)),
            "sim_time": float(follow_path_cfg.get("sim_time", 1.2)),
            "linear_granularity": float(follow_path_cfg.get("linear_granularity", 0.05)),
            "angular_granularity": float(follow_path_cfg.get("angular_granularity", 0.05)),
            "transform_tolerance": float(follow_path_cfg.get("transform_tolerance", 0.4)),
            "critics": list(
                follow_path_cfg.get(
                    "critics",
                    [
                        "BaseObstacle",
                        "GoalAlign",
                        "PathAlign",
                        "PathDist",
                        "GoalDist",
                        "Oscillation",
                        "RotateToGoal",
                    ],
                )
            ),
            "BaseObstacle.scale": float(follow_path_cfg.get("BaseObstacle.scale", 0.06)),
            "PathAlign.scale": float(follow_path_cfg.get("PathAlign.scale", 20.0)),
            "PathAlign.forward_point_distance": float(
                follow_path_cfg.get("PathAlign.forward_point_distance", 0.12)
            ),
            "GoalAlign.scale": float(follow_path_cfg.get("GoalAlign.scale", 16.0)),
            "GoalAlign.forward_point_distance": float(
                follow_path_cfg.get("GoalAlign.forward_point_distance", 0.12)
            ),
            "PathDist.scale": float(follow_path_cfg.get("PathDist.scale", 24.0)),
            "GoalDist.scale": float(follow_path_cfg.get("GoalDist.scale", 20.0)),
            "RotateToGoal.scale": float(follow_path_cfg.get("RotateToGoal.scale", 18.0)),
            "RotateToGoal.slowing_factor": float(follow_path_cfg.get("RotateToGoal.slowing_factor", 4.0)),
            "RotateToGoal.lookahead_time": float(follow_path_cfg.get("RotateToGoal.lookahead_time", -1.0)),
        }
    elif follow_path_plugin == "nav2_mppi_controller::MPPIController":
        follow_path_params = {
            "plugin": follow_path_plugin,
            "time_steps": int(follow_path_cfg.get("time_steps", 40)),
            "model_dt": float(follow_path_cfg.get("model_dt", 0.05)),
            "batch_size": int(follow_path_cfg.get("batch_size", 1200)),
            "iteration_count": int(follow_path_cfg.get("iteration_count", 1)),
            "prune_distance": float(follow_path_cfg.get("prune_distance", 1.8)),
            "transform_tolerance": float(follow_path_cfg.get("transform_tolerance", 0.3)),
            "temperature": float(follow_path_cfg.get("temperature", 0.3)),
            "gamma": float(follow_path_cfg.get("gamma", 0.015)),
            "motion_model": str(follow_path_cfg.get("motion_model", "Omni")),
            "open_loop": bool(follow_path_cfg.get("open_loop", False)),
            "visualize": bool(follow_path_cfg.get("visualize", False)),
            "regenerate_noises": bool(follow_path_cfg.get("regenerate_noises", False)),
            "reset_period": float(follow_path_cfg.get("reset_period", 1.0)),
            "retry_attempt_limit": int(follow_path_cfg.get("retry_attempt_limit", 1)),
            "vx_max": float(follow_path_cfg.get("vx_max", 0.35)),
            "vx_min": float(follow_path_cfg.get("vx_min", -0.35)),
            "vy_max": float(follow_path_cfg.get("vy_max", 0.25)),
            "vy_min": float(follow_path_cfg.get("vy_min", -0.25)),
            "wz_max": float(follow_path_cfg.get("wz_max", 0.60)),
            "ax_max": float(follow_path_cfg.get("ax_max", 0.35)),
            "ax_min": float(follow_path_cfg.get("ax_min", -0.35)),
            "ay_max": float(follow_path_cfg.get("ay_max", 0.35)),
            "ay_min": float(follow_path_cfg.get("ay_min", -0.35)),
            "az_max": float(follow_path_cfg.get("az_max", 0.70)),
            "vx_std": float(follow_path_cfg.get("vx_std", 0.12)),
            "vy_std": float(follow_path_cfg.get("vy_std", 0.14)),
            "wz_std": float(follow_path_cfg.get("wz_std", 0.25)),
            "TrajectoryVisualizer": dict(
                follow_path_cfg.get(
                    "TrajectoryVisualizer",
                    {
                        "trajectory_step": 5,
                        "time_step": 3,
                    },
                )
            ),
            "TrajectoryValidator": dict(
                follow_path_cfg.get(
                    "TrajectoryValidator",
                    {
                        "plugin": "mppi::DefaultOptimalTrajectoryValidator",
                        "collision_lookahead_time": 2.0,
                        "consider_footprint": True,
                    },
                )
            ),
            "critics": list(
                follow_path_cfg.get(
                    "critics",
                    [
                        "ConstraintCritic",
                        "CostCritic",
                        "GoalCritic",
                        "GoalAngleCritic",
                        "PathAlignCritic",
                        "PathFollowCritic",
                        "PathAngleCritic",
                        "TwirlingCritic",
                    ],
                )
            ),
            "ConstraintCritic": dict(
                follow_path_cfg.get(
                    "ConstraintCritic",
                    {
                        "enabled": True,
                        "cost_power": 1,
                        "cost_weight": 4.0,
                    },
                )
            ),
            "CostCritic": dict(
                follow_path_cfg.get(
                    "CostCritic",
                    {
                        "enabled": True,
                        "cost_power": 1,
                        "cost_weight": 3.8,
                        "critical_cost": 300.0,
                        "consider_footprint": True,
                        "collision_cost": 1000000.0,
                        "near_goal_distance": 0.4,
                        "trajectory_point_step": 2,
                    },
                )
            ),
            "GoalCritic": dict(
                follow_path_cfg.get(
                    "GoalCritic",
                    {
                        "enabled": True,
                        "cost_power": 1,
                        "cost_weight": 5.0,
                        "threshold_to_consider": 1.4,
                    },
                )
            ),
            "GoalAngleCritic": dict(
                follow_path_cfg.get(
                    "GoalAngleCritic",
                    {
                        "enabled": True,
                        "cost_power": 1,
                        "cost_weight": 3.0,
                        "threshold_to_consider": 0.4,
                    },
                )
            ),
            "PathAlignCritic": dict(
                follow_path_cfg.get(
                    "PathAlignCritic",
                    {
                        "enabled": True,
                        "cost_power": 1,
                        "cost_weight": 10.0,
                        "threshold_to_consider": 0.8,
                        "offset_from_furthest": 10,
                        "max_path_occupancy_ratio": 0.2,
                        "use_path_orientations": True,
                    },
                )
            ),
            "PathFollowCritic": dict(
                follow_path_cfg.get(
                    "PathFollowCritic",
                    {
                        "enabled": True,
                        "cost_power": 1,
                        "cost_weight": 8.0,
                        "threshold_to_consider": 1.4,
                        "offset_from_furthest": 6,
                    },
                )
            ),
            "PathAngleCritic": dict(
                follow_path_cfg.get(
                    "PathAngleCritic",
                    {
                        "enabled": True,
                        "cost_power": 1,
                        "cost_weight": 3.0,
                        "threshold_to_consider": 0.8,
                        "offset_from_furthest": 10,
                        "max_angle_to_furthest": 0.78539816339,
                        "mode": 2,
                    },
                )
            ),
            "TwirlingCritic": dict(
                follow_path_cfg.get(
                    "TwirlingCritic",
                    {
                        "enabled": True,
                        "cost_power": 1,
                        "cost_weight": 8.0,
                    },
                )
            ),
        }
    else:
        follow_path_params = {"plugin": follow_path_plugin}
    for key, value in follow_path_cfg.items():
        if key != "plugin":
            follow_path_params[key] = value

    local_costmap_frame = str(
        local_costmap_cfg.get(
            "global_frame",
            odom_frame if bool(local_costmap_cfg.get("rolling_window", True)) else map_frame,
        )
    )
    local_costmap_plugins = list(local_costmap_cfg.get("plugins", ["static_layer", "inflation_layer"]))
    global_costmap_frame = str(global_costmap_cfg.get("global_frame", map_frame))
    global_costmap_plugins = list(global_costmap_cfg.get("plugins", ["static_layer", "inflation_layer"]))

    planner_plugin = str(planner_cfg.get("plugin", "nav2_smac_planner/SmacPlannerLattice"))
    planner_params = {
        "plugin": planner_plugin,
        "tolerance": float(planner_cfg.get("tolerance", 0.10)),
        "allow_unknown": bool(planner_cfg.get("allow_unknown", False)),
    }
    if planner_plugin == "nav2_navfn_planner/NavfnPlanner":
        planner_params["use_astar"] = bool(planner_cfg.get("use_astar", True))
    elif planner_plugin == "nav2_smac_planner/SmacPlanner2D":
        planner_params.update(
            {
                "downsample_costmap": bool(planner_cfg.get("downsample_costmap", False)),
                "downsampling_factor": int(planner_cfg.get("downsampling_factor", 1)),
                "max_iterations": int(planner_cfg.get("max_iterations", 1000000)),
                "max_on_approach_iterations": int(planner_cfg.get("max_on_approach_iterations", 1000)),
                "max_planning_time": float(planner_cfg.get("max_planning_time", 2.0)),
                "cost_travel_multiplier": float(planner_cfg.get("cost_travel_multiplier", 2.0)),
            }
        )
    elif planner_plugin == "nav2_smac_planner/SmacPlannerLattice":
        planner_params.update(
            {
                "downsample_costmap": bool(planner_cfg.get("downsample_costmap", False)),
                "downsampling_factor": int(planner_cfg.get("downsampling_factor", 1)),
                "max_iterations": int(planner_cfg.get("max_iterations", 1000000)),
                "max_on_approach_iterations": int(planner_cfg.get("max_on_approach_iterations", 2000)),
                "max_planning_time": float(planner_cfg.get("max_planning_time", 3.0)),
                "smooth_path": bool(planner_cfg.get("smooth_path", True)),
                "minimum_turning_radius": float(planner_cfg.get("minimum_turning_radius", minimum_turning_radius)),
                "reverse_penalty": float(planner_cfg.get("reverse_penalty", 1.0)),
                "change_penalty": float(planner_cfg.get("change_penalty", 0.0)),
                "non_straight_penalty": float(planner_cfg.get("non_straight_penalty", 1.05)),
                "cost_penalty": float(planner_cfg.get("cost_penalty", 2.0)),
                "rotation_penalty": float(planner_cfg.get("rotation_penalty", 3.0)),
                "retrospective_penalty": float(planner_cfg.get("retrospective_penalty", 0.015)),
                "analytic_expansion_ratio": float(planner_cfg.get("analytic_expansion_ratio", 3.5)),
                "analytic_expansion_max_length": float(planner_cfg.get("analytic_expansion_max_length", 2.5)),
                "analytic_expansion_max_cost": float(planner_cfg.get("analytic_expansion_max_cost", 200.0)),
                "analytic_expansion_max_cost_override": bool(
                    planner_cfg.get("analytic_expansion_max_cost_override", False)
                ),
                "cache_obstacle_heuristic": bool(planner_cfg.get("cache_obstacle_heuristic", True)),
                "allow_reverse_expansion": bool(planner_cfg.get("allow_reverse_expansion", False)),
                "lattice_filepath": str(
                    planner_cfg.get(
                        "lattice_filepath",
                        "/opt/ros/humble/share/nav2_smac_planner/sample_primitives/5cm_resolution/0.5m_turning_radius/omni/output.json",
                    )
                ),
                "smoother": dict(
                    planner_cfg.get(
                        "smoother",
                        {
                            "tolerance": 1.0e-10,
                            "max_iterations": 1000,
                            "w_data": 0.2,
                            "w_smooth": 0.3,
                            "do_refinement": True,
                        },
                    )
                ),
            }
        )
    for key, value in planner_cfg.items():
        if key != "plugin":
            planner_params[key] = value

    return {
        "bt_navigator": {
            "ros__parameters": {
                "use_sim_time": True,
                "global_frame": map_frame,
                "robot_base_frame": base_frame,
                "odom_topic": str(ros_cfg.get("odom_topic", "/odom")),
                "bt_loop_duration": int(bt_cfg.get("bt_loop_duration", 10)),
                "default_server_timeout": int(bt_cfg.get("default_server_timeout", 20)),
                "wait_for_service_timeout": int(bt_cfg.get("wait_for_service_timeout", 1000)),
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
                "controller_frequency": float(controller_cfg.get("controller_frequency", 20.0)),
                "min_x_velocity_threshold": float(controller_cfg.get("min_x_velocity_threshold", 0.001)),
                "min_y_velocity_threshold": float(controller_cfg.get("min_y_velocity_threshold", 0.001)),
                "min_theta_velocity_threshold": float(controller_cfg.get("min_theta_velocity_threshold", 0.001)),
                "failure_tolerance": float(controller_cfg.get("failure_tolerance", 1.20)),
                "progress_checker_plugin": "progress_checker",
                "goal_checker_plugins": ["general_goal_checker"],
                "controller_plugins": ["FollowPath"],
                "progress_checker": {
                    "plugin": str(progress_checker_cfg.get("plugin", "nav2_controller::SimpleProgressChecker")),
                    "required_movement_radius": float(progress_checker_cfg.get("required_movement_radius", 0.05)),
                    "movement_time_allowance": float(progress_checker_cfg.get("movement_time_allowance", 90.0)),
                },
                "general_goal_checker": {
                    "stateful": bool(goal_checker_cfg.get("stateful", False)),
                    "plugin": str(goal_checker_cfg.get("plugin", "nav2_controller::SimpleGoalChecker")),
                    "xy_goal_tolerance": float(position_tolerance_m),
                    "yaw_goal_tolerance": float(yaw_tolerance_rad),
                },
                "FollowPath": follow_path_params,
            }
        },
        "local_costmap": {
            "local_costmap": {
                "ros__parameters": {
                    "use_sim_time": True,
                    "update_frequency": float(local_costmap_cfg.get("update_frequency", 10.0)),
                    "publish_frequency": float(local_costmap_cfg.get("publish_frequency", 4.0)),
                    "global_frame": local_costmap_frame,
                    "robot_base_frame": base_frame,
                    "rolling_window": bool(local_costmap_cfg.get("rolling_window", True)),
                    "width": int(local_costmap_cfg.get("width", 6)),
                    "height": int(local_costmap_cfg.get("height", 6)),
                    "resolution": float(local_costmap_cfg.get("resolution", 0.05)),
                    "footprint": footprint,
                    "footprint_padding": float(local_costmap_cfg.get("footprint_padding", 0.0)),
                    "plugins": local_costmap_plugins,
                    "static_layer": {
                        "plugin": "nav2_costmap_2d::StaticLayer",
                        "map_subscribe_transient_local": True,
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        "cost_scaling_factor": float(local_costmap_cfg.get("cost_scaling_factor", 3.0)),
                        "inflation_radius": inflation_radius,
                    },
                    "always_send_full_costmap": bool(local_costmap_cfg.get("always_send_full_costmap", True)),
                }
            }
        },
        "global_costmap": {
            "global_costmap": {
                "ros__parameters": {
                    "use_sim_time": True,
                    "update_frequency": float(global_costmap_cfg.get("update_frequency", 4.0)),
                    "publish_frequency": float(global_costmap_cfg.get("publish_frequency", 2.0)),
                    "global_frame": global_costmap_frame,
                    "robot_base_frame": base_frame,
                    "rolling_window": bool(global_costmap_cfg.get("rolling_window", False)),
                    "resolution": float(global_costmap_cfg.get("resolution", 0.05)),
                    "track_unknown_space": bool(global_costmap_cfg.get("track_unknown_space", False)),
                    "footprint": footprint,
                    "footprint_padding": float(global_costmap_cfg.get("footprint_padding", 0.0)),
                    "plugins": global_costmap_plugins,
                    "static_layer": {
                        "plugin": "nav2_costmap_2d::StaticLayer",
                        "map_subscribe_transient_local": True,
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        "cost_scaling_factor": float(global_costmap_cfg.get("cost_scaling_factor", 3.0)),
                        "inflation_radius": inflation_radius,
                    },
                    "always_send_full_costmap": bool(global_costmap_cfg.get("always_send_full_costmap", True)),
                }
            }
        },
        "planner_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "expected_planner_frequency": float(planner_cfg.get("expected_planner_frequency", 10.0)),
                "planner_plugins": ["GridBased"],
                "GridBased": planner_params,
            }
        },
        "smoother_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "smoother_plugins": ["simple_smoother"],
                "simple_smoother": {
                    "plugin": str(smoother_cfg.get("plugin", "nav2_smoother::SimpleSmoother")),
                    "tolerance": float(smoother_cfg.get("tolerance", 1.0e-10)),
                    "max_its": int(smoother_cfg.get("max_its", 1000)),
                    "do_refinement": bool(smoother_cfg.get("do_refinement", True)),
                },
            }
        },
        "behavior_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "costmap_topic": "local_costmap/costmap_raw",
                "footprint_topic": "local_costmap/published_footprint",
                "cycle_frequency": float(behavior_cfg.get("cycle_frequency", 10.0)),
                "behavior_plugins": list(
                    behavior_cfg.get("behavior_plugins", ["wait"])
                ),
                "spin": dict(behavior_cfg.get("spin", {"plugin": "nav2_behaviors/Spin"})),
                "backup": dict(behavior_cfg.get("backup", {"plugin": "nav2_behaviors/BackUp"})),
                "drive_on_heading": dict(behavior_cfg.get("drive_on_heading", {"plugin": "nav2_behaviors/DriveOnHeading"})),
                "wait": dict(behavior_cfg.get("wait", {"plugin": "nav2_behaviors/Wait"})),
                "global_frame": map_frame,
                "robot_base_frame": base_frame,
                "transform_tolerance": float(behavior_cfg.get("transform_tolerance", 0.2)),
                "simulate_ahead_time": float(behavior_cfg.get("simulate_ahead_time", 2.0)),
                "max_rotational_vel": float(behavior_cfg.get("max_rotational_vel", 0.35)),
                "min_rotational_vel": float(behavior_cfg.get("min_rotational_vel", 0.1)),
                "rotational_acc_lim": float(behavior_cfg.get("rotational_acc_lim", 1.0)),
            }
        },
        "waypoint_follower": {
            "ros__parameters": {
                "use_sim_time": True,
                "loop_rate": int(waypoint_cfg.get("loop_rate", 20)),
                "stop_on_failure": bool(waypoint_cfg.get("stop_on_failure", False)),
                "waypoint_task_executor_plugin": str(waypoint_cfg.get("waypoint_task_executor_plugin", "wait_at_waypoint")),
                "wait_at_waypoint": dict(
                    waypoint_cfg.get(
                        "wait_at_waypoint",
                        {
                            "plugin": "nav2_waypoint_follower::WaitAtWaypoint",
                            "enabled": True,
                            "waypoint_pause_duration": 0,
                        },
                    )
                ),
            }
        },
        "velocity_smoother": {
            "ros__parameters": {
                "use_sim_time": True,
                "smoothing_frequency": float(velocity_smoother_cfg.get("smoothing_frequency", 20.0)),
                "scale_velocities": bool(velocity_smoother_cfg.get("scale_velocities", False)),
                "feedback": str(velocity_smoother_cfg.get("feedback", "OPEN_LOOP")),
                "max_velocity": list(velocity_smoother_cfg.get("max_velocity", [0.35, 0.25, 0.60])),
                "min_velocity": list(velocity_smoother_cfg.get("min_velocity", [-0.35, -0.25, -0.60])),
                "max_accel": list(velocity_smoother_cfg.get("max_accel", [0.35, 0.35, 0.70])),
                "max_decel": list(velocity_smoother_cfg.get("max_decel", [-0.35, -0.35, -0.70])),
                "odom_topic": str(ros_cfg.get("odom_topic", "/odom")),
                "odom_duration": float(velocity_smoother_cfg.get("odom_duration", 0.1)),
                "deadband_velocity": list(velocity_smoother_cfg.get("deadband_velocity", [0.0, 0.0, 0.0])),
                "velocity_timeout": float(velocity_smoother_cfg.get("velocity_timeout", 1.0)),
            }
        },
        "map_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "yaml_filename": str(localization_cfg.get("map_yaml_path", "")),
            }
        },
        "global_costmap_client": {"ros__parameters": {"use_sim_time": True}},
        "local_costmap_client": {"ros__parameters": {"use_sim_time": True}},
        "planner_server_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "controller_server_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "behavior_server_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "waypoint_follower_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "amcl": {"ros__parameters": {"enabled": False}},
        "amcl_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
    }


class _TaskShim:
    def __init__(self, task):
        self.task = task

@dataclass
class Nav2SkillResult:
    done: bool = False
    success: bool = False
    failure_reason: str = ""
    error_message: str = ""
    final_world_xy: tuple[float, float] = (0.0, 0.0)
    final_world_yaw: float = 0.0
    final_nav_xy: tuple[float, float] = (0.0, 0.0)
    final_nav_yaw: float = 0.0
    final_distance_to_goal: float = float("inf")
    final_nav_distance_to_goal: float = float("inf")
    final_yaw_error_rad: float = float("inf")


class PersistentNav2RuntimeManager:
    """Workflow-side manager that drives Nav2 through a standard-message ROS bridge."""

    STATE_IDLE = "idle"
    STATE_WAITING_FOR_STACK_READY = "waiting_for_stack_ready"
    STATE_WAITING_FOR_MAP_READY = "waiting_for_map_ready"
    STATE_WAITING_FOR_GOAL_ACCEPTED = "waiting_for_goal_accepted"
    STATE_RUNNING = "running"
    STATE_POST_SUCCESS_SETTLING = "post_success_settling"
    STATE_SUCCEEDED = "succeeded"
    STATE_FAILED = "failed"

    def __init__(
        self,
        *,
        world,
        task,
        robot,
        output_root: str = "output/ros_bridge/skills",
        scene_name: str = "nav2_skill_scene",
    ):
        self.world = world
        self.task = task
        self.robot = robot
        self.output_root = str(output_root)
        self.scene_name = str(scene_name)
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.position_tolerance_m = NAV2_DEFAULT_POSITION_TOLERANCE_M
        self.yaw_tolerance_rad = NAV2_DEFAULT_YAW_TOLERANCE_RAD
        self.startup_timeout_sec = 60.0
        self.runtime_timeout_sec = 240.0

        self.state = self.STATE_IDLE
        self.result = Nav2SkillResult()
        self._base_cfg = deepcopy(getattr(robot, "base_cfg", {}))
        self._map_info = None
        self._params_path = ""
        self._stack_output_dir = ""
        self._goal_output_dir = ""
        self._stack_id = ""
        self._request_id = ""
        self._startup_deadline = None
        self._goal_accept_deadline = None
        self._runtime_deadline = None
        self._post_success_deadline = None
        self._bridge_client = None
        self._bridge_robot_name = ""
        self._goal_output_tag = ""
        self._goal_debug_map_info = None
        self._goal_params_path = ""
        self._cleaned_up = False

    @property
    def done(self) -> bool:
        return bool(self.result.done)

    @property
    def success(self) -> bool:
        return bool(self.result.success)

    def bind(self, *, world, task, robot, scene_name: Optional[str] = None):
        self.world = world
        self.task = task
        self.robot = robot
        self._base_cfg = deepcopy(getattr(robot, "base_cfg", {}))
        if scene_name is not None:
            self.scene_name = str(scene_name)
        current_robot_name = safe_name(getattr(self.robot, "name", "robot"))
        if self._bridge_client is not None and current_robot_name != self._bridge_robot_name:
            try:
                self._bridge_client.destroy()
            finally:
                self._bridge_client = None
                self._bridge_robot_name = ""

    def _ensure_bridge_client(self):
        from .bridge_client import Nav2BridgeClient

        robot_name = safe_name(getattr(self.robot, "name", "robot"))
        if self._bridge_client is not None and robot_name == self._bridge_robot_name:
            return self._bridge_client

        if self._bridge_client is not None:
            try:
                self._bridge_client.destroy()
            finally:
                self._bridge_client = None

        self._bridge_client = Nav2BridgeClient(
            self.robot,
            self._base_cfg,
            node_name=f"nav2_bridge_client_{robot_name}",
        )
        self._bridge_robot_name = robot_name
        return self._bridge_client

    def _stack_signature(self, params: dict) -> str:
        payload = {
            "scene_name": str(self.scene_name),
            "params": params,
            "map_info": self._map_info,
            "nav2_cfg": dict(self._base_cfg.get("ros", {}).get("nav2", {})),
            "nav2_skill_cfg": dict(self._base_cfg.get("nav2_skill", {})),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:12]

    def _prepare_stack_artifacts(self):
        from .localization_stack import IsaacStaticMapExporter

        robot_tag = safe_name(getattr(self.robot, "name", "robot"))
        scene_tag = safe_name(self.scene_name)
        stack_tag = f"{robot_tag}_{scene_tag}"
        self._stack_output_dir = os.path.join(self.output_root, stack_tag, "stack")
        os.makedirs(self._stack_output_dir, exist_ok=True)

        exporter = IsaacStaticMapExporter(
            workflow=_TaskShim(self.task),
            robot=self.robot,
            base_cfg=self._base_cfg,
            scene_name=self.scene_name,
        )
        translation, orientation = self.robot.get_mobile_base_pose()
        self._map_info = exporter.export_map(
            output_dir=self._base_cfg["ros"]["localization"]["map_output_dir"],
            clear_center_xy=(
                float(translation[0]),
                float(translation[1]),
                float(_yaw_from_wxyz(orientation)),
            ),
        )
        self._base_cfg.setdefault("ros", {}).setdefault("localization", {})["map_yaml_path"] = self._map_info["yaml_path"]
        artifacts = generate_nav2_bringup_artifacts(
            self._stack_output_dir,
            base_cfg=self._base_cfg,
            map_yaml_path=str(self._map_info["yaml_path"]),
            position_tolerance_m=self.position_tolerance_m,
            yaw_tolerance_rad=self.yaw_tolerance_rad,
        )
        self._params_path = str(artifacts["params_path"])
        self._stack_id = f"{robot_tag}::{scene_tag}::{self._stack_signature(artifacts['params'])}"

    def _new_goal_output_dir(self) -> str:
        robot_tag = safe_name(getattr(self.robot, "name", "robot"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_root, f"{robot_tag}_nav2_goal_{stamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _freeze_goal_debug_artifacts(self):
        self._goal_debug_map_info = dict(self._map_info or {})
        self._goal_params_path = str(self._params_path)
        if not self._goal_output_dir:
            return

        debug_dir = os.path.join(self._goal_output_dir, "debug_inputs")
        os.makedirs(debug_dir, exist_ok=True)

        map_info = dict(self._goal_debug_map_info or {})
        yaml_path = str(map_info.get("yaml_path", "")).strip()
        pgm_path = str(map_info.get("pgm_path", "")).strip()
        if yaml_path:
            yaml_src = yaml_path if os.path.isabs(yaml_path) else os.path.abspath(yaml_path)
            yaml_dst = os.path.join(debug_dir, "map.yaml")
            shutil.copy2(yaml_src, yaml_dst)
            map_info["yaml_path"] = yaml_dst
        if pgm_path:
            pgm_src = pgm_path if os.path.isabs(pgm_path) else os.path.abspath(pgm_path)
            pgm_dst = os.path.join(debug_dir, "map.pgm")
            shutil.copy2(pgm_src, pgm_dst)
            map_info["pgm_path"] = pgm_dst
        self._goal_debug_map_info = map_info

        if self._params_path:
            params_src = self._params_path if os.path.isabs(self._params_path) else os.path.abspath(self._params_path)
            params_dst = os.path.join(debug_dir, os.path.basename(params_src) or "nav2_skill_params.yaml")
            shutil.copy2(params_src, params_dst)
            self._goal_params_path = params_dst

    def begin_goal(
        self,
        *,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
        yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
        startup_timeout_sec: float = 60.0,
        runtime_timeout_sec: float = 240.0,
    ):
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_yaw = float(goal_yaw)
        self.position_tolerance_m = float(position_tolerance_m)
        self.yaw_tolerance_rad = float(yaw_tolerance_rad)
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.runtime_timeout_sec = float(runtime_timeout_sec)
        previous_request_id = str(self._request_id)
        self.result = Nav2SkillResult()
        self.state = self.STATE_IDLE
        self._startup_deadline = None
        self._goal_accept_deadline = None
        self._runtime_deadline = None
        self._post_success_deadline = None
        self._goal_output_dir = self._new_goal_output_dir()
        self._goal_output_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._request_id = self._goal_output_tag
        self._goal_debug_map_info = None
        self._goal_params_path = ""

        bridge_client = self._ensure_bridge_client()
        bridge_client.reset_debug_trace()
        if previous_request_id:
            bridge_client.cancel_request(previous_request_id)
        self._prepare_stack_artifacts()
        self._freeze_goal_debug_artifacts()
        self.state = self.STATE_WAITING_FOR_STACK_READY
        self._startup_deadline = time_monotonic() + self.startup_timeout_sec

    def prepare_for_reset(self):
        if self._bridge_client is not None:
            try:
                self._bridge_client.cancel_request(self._request_id)
            except Exception:
                LOGGER.exception("failed to cancel nav2 goal during reset")
        self.result = Nav2SkillResult()
        self.state = self.STATE_IDLE
        self._startup_deadline = None
        self._goal_accept_deadline = None
        self._runtime_deadline = None
        self._post_success_deadline = None
        self._request_id = ""

    def step(self):
        if self.done or self.state == self.STATE_IDLE:
            return

        bridge_client = self._ensure_bridge_client()
        bridge_client.step(step_dt=self._step_dt())

        if self.state == self.STATE_WAITING_FOR_STACK_READY:
            if bridge_client.bridge_online and bridge_client.nav_stack_ready:
                bridge_client.publish_map_update(
                    request_id=self._request_id,
                    stack_id=self._stack_id,
                    map_yaml_path=str(self._map_info["yaml_path"]),
                    scene_name=self.scene_name,
                )
                self.state = self.STATE_WAITING_FOR_MAP_READY
                return
            if time_monotonic() >= float(self._startup_deadline):
                self._fail("stack_not_ready", "Timed out waiting for Nav2 bridge heartbeat/action readiness.")
            return

        status = bridge_client.request_status(self._request_id)
        result = bridge_client.request_result(self._request_id)
        bridge_state = self._bridge_state_name(status=status, result=result)

        if self.state == self.STATE_WAITING_FOR_MAP_READY:
            if bridge_state == "ready":
                bridge_client.publish_goal(
                    request_id=self._request_id,
                    goal_x=self.goal_x,
                    goal_y=self.goal_y,
                    goal_yaw=self.goal_yaw,
                )
                self.state = self.STATE_WAITING_FOR_GOAL_ACCEPTED
                self._goal_accept_deadline = time_monotonic() + min(self.startup_timeout_sec, 30.0)
                return
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail(
                    "bridge_" + bridge_state,
                    self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}",
                )
                return
            if time_monotonic() >= float(self._startup_deadline):
                self._fail("map_update_timeout", "Timed out waiting for bridge adapter to load the map.")
            return

        if self.state == self.STATE_WAITING_FOR_GOAL_ACCEPTED:
            if bridge_state in {"accepted", "running"}:
                self.state = self.STATE_RUNNING
                self._runtime_deadline = time_monotonic() + self.runtime_timeout_sec
                return
            if bridge_state == "succeeded":
                self.state = self.STATE_POST_SUCCESS_SETTLING
                self._post_success_deadline = time_monotonic() + 2.0
                return
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail(
                    "bridge_" + bridge_state,
                    self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}",
                )
                return
            if time_monotonic() >= float(self._goal_accept_deadline):
                bridge_client.cancel_request(self._request_id)
                self._fail("goal_not_accepted", "Timed out waiting for bridge adapter to accept the goal.")
            return

        if self.state == self.STATE_RUNNING:
            self._update_pose_result_fields()
            if bridge_state == "succeeded":
                self.state = self.STATE_POST_SUCCESS_SETTLING
                self._post_success_deadline = time_monotonic() + 2.0
                return
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail(
                    "bridge_" + bridge_state,
                    self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}",
                )
                return
            if time_monotonic() >= float(self._runtime_deadline):
                bridge_client.cancel_request(self._request_id)
                self._fail("runtime_timeout", "Timed out while waiting for the navigation goal to finish.")
            return

        if self.state == self.STATE_POST_SUCCESS_SETTLING:
            self._update_pose_result_fields()
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail(
                    "bridge_" + bridge_state,
                    self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}",
                )
                return
            if self._goal_within_tolerance():
                self.result.done = True
                self.result.success = True
                self.state = self.STATE_SUCCEEDED
                self._write_debug_snapshot(
                    "success_snapshot.json",
                    "goal_succeeded",
                    "Nav2 goal reached within skill tolerances after post-success settling.",
                )
                return
            if time_monotonic() >= float(self._post_success_deadline):
                self._fail(
                    "goal_tolerance_not_met",
                    "Nav2 reported success but skill tolerances were not met after post-success settling.",
                )

    def shutdown(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self._bridge_client is not None and not self.result.done:
            try:
                self._bridge_client.cancel_request(self._request_id)
            except Exception:
                LOGGER.exception("failed to cancel nav2 goal during shutdown")
            try:
                self._write_debug_snapshot(
                    "shutdown_snapshot.json",
                    "runtime_shutdown",
                    "Nav2 runtime manager was shut down before the goal finished",
                )
            except Exception:
                LOGGER.exception("failed to write nav2 shutdown snapshot")
        if self._bridge_client is not None:
            try:
                self._bridge_client.destroy()
            except Exception:
                LOGGER.exception("failed to destroy nav2 bridge client")
            finally:
                self._bridge_client = None
                self._bridge_robot_name = ""

    def _bridge_state_name(self, *, status: dict, result: dict) -> str:
        payload = result or status
        return str(payload.get("state", "")).strip().lower()

    @staticmethod
    def _bridge_detail(*, status: dict, result: dict) -> str:
        payload = result or status
        return str(payload.get("detail", "")).strip()

    def _goal_within_tolerance(self) -> bool:
        return (
            self.result.final_distance_to_goal <= self.position_tolerance_m
            and self.result.final_nav_distance_to_goal <= self.position_tolerance_m
            and self.result.final_yaw_error_rad <= self.yaw_tolerance_rad
        )

    def _step_dt(self) -> float:
        get_physics_dt = getattr(self.world, "get_physics_dt", None)
        if callable(get_physics_dt):
            return float(get_physics_dt())
        return float(getattr(self.world, "physics_dt", 1.0 / 60.0))

    def _write_debug_snapshot(self, filename: str, reason: str, message: str):
        self._update_pose_result_fields()
        control_snapshot = _runtime_control_debug_snapshot(self.robot)
        bridge_client = self._bridge_client
        planning_payload = dict((bridge_client.latest_result if bridge_client is not None else {}).get("planning", {}))
        trajectory_payload = bridge_client.odom_trace if bridge_client is not None else []
        artifacts = self._write_navigation_artifacts(
            planning_payload=planning_payload,
            trajectory_payload=trajectory_payload,
            control_snapshot=control_snapshot,
        )
        debug_snapshot = {
            "robot": getattr(self.robot, "name", "robot"),
            "state": str(self.state),
            "reason": str(reason),
            "message": str(message),
            "goal": {"x": float(self.goal_x), "y": float(self.goal_y), "yaw": float(self.goal_yaw)},
            "world_xy": list(self.result.final_world_xy),
            "world_yaw": float(self.result.final_world_yaw),
            "nav_xy": list(self.result.final_nav_xy),
            "nav_yaw": float(self.result.final_nav_yaw),
            "world_dist": float(self.result.final_distance_to_goal),
            "nav_dist": float(self.result.final_nav_distance_to_goal),
            "yaw_err": float(self.result.final_yaw_error_rad),
            "control": control_snapshot,
            "map_info": dict(self._goal_debug_map_info or self._map_info or {}),
            "params_path": str(self._goal_params_path or self._params_path),
            "stack_id": str(self._stack_id),
            "nav2_runtime": {
                "bridge_online": bool(bridge_client.bridge_online) if bridge_client is not None else False,
                "nav_stack_ready": bool(bridge_client.nav_stack_ready) if bridge_client is not None else False,
                "latest_status": bridge_client.latest_status if bridge_client is not None else {},
                "latest_result": bridge_client.latest_result if bridge_client is not None else {},
            },
            "artifacts": artifacts,
        }
        if self._goal_output_dir:
            snapshot_path = os.path.join(self._goal_output_dir, filename)
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(debug_snapshot, handle, indent=2, ensure_ascii=False)
        return debug_snapshot

    def _write_navigation_artifacts(
        self,
        *,
        planning_payload: dict,
        trajectory_payload: list[dict],
        control_snapshot: dict,
    ) -> dict:
        artifacts = {}
        if not self._goal_output_dir:
            return artifacts

        planning_summary = {}
        if planning_payload:
            planning_summary = {
                "state": str(planning_payload.get("state", "")),
                "source": str(planning_payload.get("source", "")),
                "status_code": planning_payload.get("status_code"),
                "planning_time_sec": planning_payload.get("planning_time_sec"),
            }
            path_payload = dict(planning_payload.get("path", {}))
            if path_payload:
                planning_summary["frame_id"] = str(path_payload.get("frame_id", ""))
                planning_summary["num_poses"] = int(path_payload.get("num_poses", 0))
                planning_summary["path_length_m"] = float(path_payload.get("path_length_m", 0.0))
            planned_path_path = os.path.join(self._goal_output_dir, "planned_path.json")
            with open(planned_path_path, "w", encoding="utf-8") as handle:
                json.dump(planning_payload, handle, indent=2, ensure_ascii=False)
            artifacts["planned_path"] = planned_path_path
        if planning_summary:
            artifacts["planned_path_summary"] = planning_summary

        if trajectory_payload:
            actual_trajectory_path = os.path.join(self._goal_output_dir, "actual_trajectory.json")
            with open(actual_trajectory_path, "w", encoding="utf-8") as handle:
                json.dump(trajectory_payload, handle, indent=2, ensure_ascii=False)
            artifacts["actual_trajectory"] = actual_trajectory_path
            artifacts["actual_trajectory_summary"] = {
                "num_samples": len(trajectory_payload),
                "start_xy": [float(trajectory_payload[0]["x"]), float(trajectory_payload[0]["y"])],
                "end_xy": [float(trajectory_payload[-1]["x"]), float(trajectory_payload[-1]["y"])],
            }

        bridge_snapshot = dict(control_snapshot.get("bridge", {}) or {})
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        bridge_cmd_vel_history = list(getattr(bridge, "_debug_cmd_vel_history", [])) if bridge is not None else []
        if bridge_cmd_vel_history:
            cmd_vel_history_path = os.path.join(self._goal_output_dir, "cmd_vel_history.json")
            with open(cmd_vel_history_path, "w", encoding="utf-8") as handle:
                json.dump(bridge_cmd_vel_history, handle, indent=2, ensure_ascii=False)
            artifacts["cmd_vel_history"] = cmd_vel_history_path
            artifacts["cmd_vel_history_summary"] = {
                "num_samples": len(bridge_cmd_vel_history),
                "last_received_cmd_vel": bridge_snapshot.get("last_received_cmd_vel"),
            }

        bridge_command_history = list(getattr(bridge, "_debug_command_history", [])) if bridge is not None else []
        if bridge_command_history:
            bridge_command_history_path = os.path.join(self._goal_output_dir, "bridge_command_history.json")
            with open(bridge_command_history_path, "w", encoding="utf-8") as handle:
                json.dump(bridge_command_history, handle, indent=2, ensure_ascii=False)
            artifacts["bridge_command_history"] = bridge_command_history_path
            artifacts["bridge_command_history_summary"] = {
                "num_samples": len(bridge_command_history),
                "steering_command_sign": bridge_snapshot.get("steering_command_sign"),
            }
        return artifacts

    def _fail(self, reason: str, message: str):
        self._update_pose_result_fields()
        self.result.done = True
        self.result.success = False
        self.result.failure_reason = str(reason)
        self.result.error_message = str(message)
        control_snapshot = _runtime_control_debug_snapshot(self.robot)
        failure_snapshot = self._write_debug_snapshot("failure_snapshot.json", reason, message)
        LOGGER.error(
            "nav2 skill failed: robot=%s reason=%s message=%s world_xy=%s nav_xy=%s world_dist=%.3f nav_dist=%.3f yaw_err=%.3f control=%s",
            failure_snapshot["robot"],
            self.result.failure_reason,
            self.result.error_message,
            self.result.final_world_xy,
            self.result.final_nav_xy,
            self.result.final_distance_to_goal,
            self.result.final_nav_distance_to_goal,
            self.result.final_yaw_error_rad,
            control_snapshot,
        )
        self.state = self.STATE_FAILED

    def _update_pose_result_fields(self):
        world_translation, world_orientation = self.robot.get_mobile_base_pose()
        world_xy = (float(world_translation[0]), float(world_translation[1]))
        world_yaw = float(_yaw_from_wxyz(world_orientation))

        bridge_client = self._bridge_client
        result_payload = bridge_client.request_result(self._request_id) if bridge_client is not None else {}
        status_payload = bridge_client.request_status(self._request_id) if bridge_client is not None else {}
        reported_pose = dict((result_payload or status_payload).get("reported_pose", {}))

        if bridge_client is not None:
            nav_x, nav_y, nav_yaw = bridge_client.get_current_pose_xy_yaw()
            nav_xy = (float(nav_x), float(nav_y))
            nav_yaw = float(nav_yaw)
        elif {"x", "y", "yaw"} <= set(reported_pose.keys()):
            nav_xy = (float(reported_pose["x"]), float(reported_pose["y"]))
            nav_yaw = float(reported_pose["yaw"])
        else:
            nav_xy = world_xy
            nav_yaw = world_yaw

        self.result.final_world_xy = world_xy
        self.result.final_world_yaw = world_yaw
        self.result.final_nav_xy = nav_xy
        self.result.final_nav_yaw = nav_yaw
        self.result.final_distance_to_goal = math.hypot(self.goal_x - world_xy[0], self.goal_y - world_xy[1])
        self.result.final_nav_distance_to_goal = math.hypot(self.goal_x - nav_xy[0], self.goal_y - nav_xy[1])
        self.result.final_yaw_error_rad = abs(_angle_diff_rad(self.goal_yaw, world_yaw))

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


def time_monotonic() -> float:
    import time

    return time.monotonic()


SkillManagedNav2Session = PersistentNav2RuntimeManager
