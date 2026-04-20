"""Unit tests for ROS navigation helper logic."""

import math
from types import SimpleNamespace
from typing import Any, cast

from geometry_msgs.msg import Twist

from workflows.simbox.core.mobile.controllers.ranger_mini_v3_controller import RangerDriverCommand, RangerROS2Driver
from workflows.simbox.core.mobile.bridge.isaac_base_bridge import (
    SplitAlohaIsaacBaseBridge,
    _BaseCommand,
)
from nav2.nav2_navigator import Nav2Navigator


def _make_goal(frame_id: str, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
        ),
    )


def test_nav2_local_goal_echo_filter_window():
    navigator = Nav2Navigator.__new__(Nav2Navigator)
    navigator._local_goal_echo_filter_sec = 0.5
    navigator._last_local_goal_signature = None
    navigator._last_local_goal_time_sec = -1e9
    navigator._now_sec = lambda: 10.0

    goal = _make_goal("map", 1.0, 2.0, 0.0, 0.0, 0.0, 0.1, 0.995)
    navigator._remember_local_goal(cast(Any, goal))

    assert navigator._should_ignore_local_goal_echo(cast(Any, goal))

    navigator._now_sec = lambda: 10.8
    assert not navigator._should_ignore_local_goal_echo(cast(Any, goal))


def test_bridge_dual_ackermann_twist_mapping():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._steering_limit = 2.1
    bridge._wheel_base = 0.5
    bridge._steering_command_sign = 1.0
    bridge._ackermann_rear_steering_mode = "counter_phase"

    command = _BaseCommand(
        motion_mode=SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN,
        linear_speed=1.0,
        lateral_speed=0.0,
        steering_angle=0.2,
        angular_speed=0.0,
        received_time_sec=0.0,
    )

    vx, vy, wz = bridge._command_to_body_twist(command)

    assert math.isclose(vx, 1.0, rel_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-9)
    assert math.isclose(wz, 2.0 * math.tan(0.2) / 0.5, rel_tol=1e-6)


def test_bridge_dual_ackermann_geometry_splits_inner_outer():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._wheel_base = 0.5
    bridge._track_width = 0.3
    bridge._ackermann_rear_steering_mode = "counter_phase"

    left_steering = bridge._compute_dual_ackermann_steering_positions(0.35)
    right_steering = bridge._compute_dual_ackermann_steering_positions(-0.35)

    assert left_steering[0] > left_steering[1] > 0.0
    assert left_steering[2] < left_steering[3] < 0.0
    assert right_steering[0] > right_steering[1] and right_steering[1] < 0.0
    assert right_steering[2] < right_steering[3] and right_steering[2] > 0.0


def test_bridge_dual_ackermann_wheel_speeds_stay_forward_when_turning_right():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._wheel_base = 0.5
    bridge._track_width = 0.3
    bridge._ackermann_rear_steering_mode = "counter_phase"

    wheel_linear = bridge._compute_dual_ackermann_wheel_linear_speeds(1.0, -0.35)

    assert all(speed > 0.0 for speed in wheel_linear)
    assert wheel_linear[0] > wheel_linear[1]
    assert wheel_linear[2] > wheel_linear[3]


def test_bridge_map_ranger_command_can_disable_ackermann_steering_split():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._steering_limit = 2.1
    bridge._wheel_velocity_limit = 20.0
    bridge._wheel_radius = 0.1
    bridge._wheel_base = 0.5
    bridge._track_width = 0.3
    bridge._steering_command_sign = 1.0
    bridge._ackermann_split_steering = False
    bridge._ackermann_split_wheel_speeds = True
    bridge._ackermann_rear_steering_mode = "counter_phase"

    command = _BaseCommand(
        motion_mode=SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN,
        linear_speed=1.0,
        lateral_speed=0.0,
        steering_angle=0.35,
        angular_speed=0.0,
        received_time_sec=0.0,
    )

    steering_positions, _ = bridge._map_ranger_command(command)

    expected = [0.35, 0.35, -0.35, -0.35]
    assert all(math.isclose(float(actual), target, rel_tol=1e-6, abs_tol=1e-6) for actual, target in zip(steering_positions, expected))


def test_bridge_map_ranger_command_can_disable_ackermann_wheel_speed_split():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._steering_limit = 2.1
    bridge._wheel_velocity_limit = 20.0
    bridge._wheel_radius = 0.1
    bridge._wheel_base = 0.5
    bridge._track_width = 0.3
    bridge._steering_command_sign = 1.0
    bridge._ackermann_split_steering = True
    bridge._ackermann_split_wheel_speeds = False
    bridge._ackermann_rear_steering_mode = "counter_phase"

    command = _BaseCommand(
        motion_mode=SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN,
        linear_speed=1.0,
        lateral_speed=0.0,
        steering_angle=-0.35,
        angular_speed=0.0,
        received_time_sec=0.0,
    )

    _, wheel_velocities = bridge._map_ranger_command(command)

    assert all(math.isclose(float(speed), 10.0, rel_tol=1e-6) for speed in wheel_velocities)


def test_bridge_world_linear_velocity_to_body_frame():
    velocity_world = [0.0, 1.0, 0.0]
    yaw = math.pi / 2.0
    orientation = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    velocity_body = SplitAlohaIsaacBaseBridge._world_linear_velocity_to_body(velocity_world, orientation)

    assert math.isclose(float(velocity_body[0]), 1.0, rel_tol=1e-6)
    assert math.isclose(float(velocity_body[1]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(velocity_body[2]), 0.0, abs_tol=1e-6)


def test_bridge_virtual_odom_parallel_update():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._steering_limit = 2.1
    bridge._wheel_base = 0.5
    bridge._steering_command_sign = 1.0
    bridge._ackermann_rear_steering_mode = "counter_phase"
    bridge._virtual_odom_enabled = True
    bridge._virtual_odom_use_world_z = False
    bridge._virtual_x = 0.0
    bridge._virtual_y = 0.0
    bridge._virtual_yaw = 0.0
    bridge._virtual_z = 0.0

    command = _BaseCommand(
        motion_mode=SplitAlohaIsaacBaseBridge.MOTION_MODE_PARALLEL,
        linear_speed=1.0,
        lateral_speed=0.0,
        steering_angle=math.pi / 2.0,
        angular_speed=0.0,
        received_time_sec=0.0,
    )

    bridge._update_virtual_odometry(command, dt=1.0)

    assert math.isclose(bridge._virtual_x, 0.0, abs_tol=1e-6)
    assert math.isclose(bridge._virtual_y, 1.0, abs_tol=1e-6)
    assert math.isclose(bridge._virtual_yaw, 0.0, abs_tol=1e-6)


def test_bridge_steering_command_sign_flips_ackermann_direction():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._steering_limit = 2.1
    bridge._wheel_base = 0.5
    bridge._steering_command_sign = -1.0
    bridge._ackermann_rear_steering_mode = "counter_phase"

    command = _BaseCommand(
        motion_mode=SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN,
        linear_speed=1.0,
        lateral_speed=0.0,
        steering_angle=0.2,
        angular_speed=0.0,
        received_time_sec=0.0,
    )

    _, _, wz = bridge._command_to_body_twist(command)

    assert math.isclose(wz, 2.0 * math.tan(-0.2) / 0.5, rel_tol=1e-6)


def test_bridge_front_only_ackermann_keeps_rear_steering_zero():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._steering_limit = 2.1
    bridge._wheel_velocity_limit = 20.0
    bridge._wheel_radius = 0.1
    bridge._wheel_base = 0.5
    bridge._track_width = 0.3
    bridge._steering_command_sign = 1.0
    bridge._ackermann_split_steering = False
    bridge._ackermann_split_wheel_speeds = False
    bridge._ackermann_rear_steering_mode = "fixed_zero"

    command = _BaseCommand(
        motion_mode=SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN,
        linear_speed=1.0,
        lateral_speed=0.0,
        steering_angle=0.35,
        angular_speed=0.0,
        received_time_sec=0.0,
    )

    steering_positions, _ = bridge._map_ranger_command(command)

    expected = [0.35, 0.35, 0.0, 0.0]
    assert all(math.isclose(float(actual), target, rel_tol=1e-6, abs_tol=1e-6) for actual, target in zip(steering_positions, expected))


def test_bridge_front_only_ackermann_twist_mapping():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge._steering_limit = 2.1
    bridge._wheel_base = 0.5
    bridge._steering_command_sign = 1.0
    bridge._ackermann_rear_steering_mode = "fixed_zero"

    command = _BaseCommand(
        motion_mode=SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN,
        linear_speed=1.0,
        lateral_speed=0.0,
        steering_angle=0.2,
        angular_speed=0.0,
        received_time_sec=0.0,
    )

    _, _, wz = bridge._command_to_body_twist(command)

    assert math.isclose(wz, math.tan(0.2) / 0.5, rel_tol=1e-6)


def test_ranger_ros2_driver_dual_ackermann_matches_official_semantics():
    driver = RangerROS2Driver.__new__(RangerROS2Driver)
    driver.MOTION_MODE_DUAL_ACKERMAN = 0
    driver.MOTION_MODE_PARALLEL = 1
    driver.MOTION_MODE_SPINNING = 2
    driver.MOTION_MODE_SIDE_SLIP = 3
    driver._robot_model = "ranger_mini_v3"
    driver._robot_params = RangerROS2Driver._MODEL_PARAMS["ranger_mini_v3"]
    driver._parking_mode = False
    driver._last_nonzero_x = 1.0
    driver._now_sec = lambda: 12.0

    msg = Twist()
    msg.linear.x = 0.5
    msg.angular.z = 0.2

    command = driver._resolve_command(msg)

    assert command.motion_mode == RangerROS2Driver.MOTION_MODE_DUAL_ACKERMAN
    assert math.isclose(command.linear_speed, 0.5, rel_tol=1e-6)
    assert math.isclose(command.angular_speed, 0.0, abs_tol=1e-6)
    assert command.steering_angle > 0.0


def test_ranger_ros2_driver_switches_to_parallel_for_lateral_motion():
    driver = RangerROS2Driver.__new__(RangerROS2Driver)
    driver.MOTION_MODE_DUAL_ACKERMAN = 0
    driver.MOTION_MODE_PARALLEL = 1
    driver.MOTION_MODE_SPINNING = 2
    driver.MOTION_MODE_SIDE_SLIP = 3
    driver._robot_model = "ranger_mini_v3"
    driver._robot_params = RangerROS2Driver._MODEL_PARAMS["ranger_mini_v3"]
    driver._parking_mode = False
    driver._last_nonzero_x = 1.0
    driver._now_sec = lambda: 13.0

    msg = Twist()
    msg.linear.x = 0.4
    msg.linear.y = 0.2

    command = driver._resolve_command(msg)

    assert command.motion_mode == RangerROS2Driver.MOTION_MODE_PARALLEL
    assert math.isclose(command.linear_speed, math.sqrt(0.4 * 0.4 + 0.2 * 0.2), rel_tol=1e-6)
    assert command.steering_angle > 0.0


def test_ranger_ros2_driver_parallel_mode_handles_zero_forward_speed():
    driver = RangerROS2Driver.__new__(RangerROS2Driver)
    driver.MOTION_MODE_DUAL_ACKERMAN = 0
    driver.MOTION_MODE_PARALLEL = 1
    driver.MOTION_MODE_SPINNING = 2
    driver.MOTION_MODE_SIDE_SLIP = 3
    driver._robot_model = "ranger_mini_v3"
    driver._robot_params = RangerROS2Driver._MODEL_PARAMS["ranger_mini_v3"]
    driver._parking_mode = False
    driver._last_nonzero_x = 1.0
    driver._now_sec = lambda: 14.0

    msg = Twist()
    msg.linear.x = 0.0
    msg.linear.y = 0.2

    command = driver._resolve_command(msg)

    assert command.motion_mode == RangerROS2Driver.MOTION_MODE_PARALLEL
    assert math.isclose(command.linear_speed, 0.2, rel_tol=1e-6)
    assert command.steering_angle > 0.0


def test_bridge_can_pull_command_from_internal_driver():
    bridge = SplitAlohaIsaacBaseBridge.__new__(SplitAlohaIsaacBaseBridge)
    bridge.driver = SimpleNamespace(
        get_active_command=lambda: RangerDriverCommand(
                motion_mode=SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN,
                linear_speed=0.6,
                lateral_speed=0.0,
                steering_angle=0.15,
                angular_speed=0.0,
                received_time_sec=9.0,
        )
    )
    bridge._has_motion_mode = False
    bridge._latest_motion_mode = SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN
    bridge._motion_mode_message_count = 0
    bridge._driver_command_message_count = 0
    bridge._applied_driver_command_count = 0
    bridge._last_internal_driver_received_time_sec = None

    bridge._refresh_command_from_internal_driver()

    assert bridge._has_motion_mode
    assert bridge._command.motion_mode == SplitAlohaIsaacBaseBridge.MOTION_MODE_DUAL_ACKERMAN
    assert math.isclose(bridge._command.linear_speed, 0.6, rel_tol=1e-6)
    assert math.isclose(bridge._command.steering_angle, 0.15, rel_tol=1e-6)
