"""SplitAloha robot implementation - Dual-arm manipulator."""
from copy import deepcopy

import numpy as np
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot


# pylint: disable=line-too-long,unused-argument
@register_robot
class SplitAloha(TemplateRobot):
    """SplitAloha dual-arm robot with 6-DOF arms."""

    def __init__(self, *args, **kwargs):
        self.base_cfg = {}
        self.base_steering_joint_names = []
        self.base_wheel_joint_names = []
        self.base_steering_joint_indices = []
        self.base_wheel_joint_indices = []
        super().__init__(*args, **kwargs)
        self.base_cfg = deepcopy(self.cfg.get("base", {}))
        self.base_steering_joint_names = list(self.base_cfg.get("steering_joint_names", []))
        self.base_wheel_joint_names = list(self.base_cfg.get("wheel_joint_names", []))

    def _setup_joint_indices(self):
        self.left_joint_indices = self.cfg["left_joint_indices"]
        self.right_joint_indices = self.cfg["right_joint_indices"]
        self.left_gripper_indices = self.cfg["left_gripper_indices"]
        self.right_gripper_indices = self.cfg["right_gripper_indices"]
        self.body_indices = []
        self.head_indices = []
        self.lift_indices = []

    def _setup_paths(self):
        fl_ee_path = self.cfg["fl_ee_path"]
        fr_ee_path = self.cfg["fr_ee_path"]
        self.fl_ee_path = f"{self.robot_prim_path}/{fl_ee_path}"
        self.fr_ee_path = f"{self.robot_prim_path}/{fr_ee_path}"
        self.fl_base_path = f"{self.robot_prim_path}/{self.cfg['fl_base_path']}"
        self.fr_base_path = f"{self.robot_prim_path}/{self.cfg['fr_base_path']}"
        self.fl_hand_path = self.fl_ee_path
        self.fr_hand_path = self.fr_ee_path

    def _setup_gripper_keypoints(self):
        self.fl_gripper_keypoints = self.cfg["fl_gripper_keypoints"]
        self.fr_gripper_keypoints = self.cfg["fr_gripper_keypoints"]

    def _setup_collision_paths(self):
        self.fl_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_filter_paths"]]
        self.fr_filter_paths_expr = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_filter_paths"]]
        self.fl_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fl_forbid_collision_paths"]]
        self.fr_forbid_collision_paths = [f"{self.robot_prim_path}/{p}" for p in self.cfg["fr_forbid_collision_paths"]]

    def _get_gripper_state(self, gripper_home):
        return 1.0 if gripper_home and gripper_home[0] >= 0.05 else -1.0

    def _setup_joint_velocities(self):
        # SplitAloha has 12 joints for velocity control
        all_joint_indices = self.left_joint_indices + self.right_joint_indices
        if all_joint_indices:
            self._articulation_view.set_max_joint_velocities(
                [500.0] * 12,
                joint_indices=all_joint_indices,
            )

    def _set_initial_positions(self):
        positions = self.left_joint_home + self.right_joint_home + self.left_gripper_home + self.right_gripper_home
        indices = (
            self.left_joint_indices + self.right_joint_indices + self.left_gripper_indices + self.right_gripper_indices
        )
        if positions and indices:
            self._articulation_view.set_joint_positions(positions, joint_indices=indices)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._setup_base_joint_indices()

    def _setup_base_joint_indices(self):
        dof_names = list(self._articulation_view.dof_names)
        self.base_steering_joint_indices = [dof_names.index(name) for name in self.base_steering_joint_names]
        self.base_wheel_joint_indices = [dof_names.index(name) for name in self.base_wheel_joint_names]

    def get_base_interface(self):
        return {
            "steering_joint_names": list(self.base_steering_joint_names),
            "wheel_joint_names": list(self.base_wheel_joint_names),
            "steering_joint_indices": list(self.base_steering_joint_indices),
            "wheel_joint_indices": list(self.base_wheel_joint_indices),
            "base_cfg": deepcopy(self.base_cfg),
        }

    def apply_base_command(self, steering_positions, wheel_velocities):
        steering_positions = np.asarray(steering_positions, dtype=np.float32)
        wheel_velocities = np.asarray(wheel_velocities, dtype=np.float32)
        if steering_positions.shape[0] != len(self.base_steering_joint_indices):
            raise ValueError("steering_positions size does not match steering joints")
        if wheel_velocities.shape[0] != len(self.base_wheel_joint_indices):
            raise ValueError("wheel_velocities size does not match wheel joints")

        self._articulation_view.set_joint_position_targets(
            steering_positions.reshape(1, -1),
            joint_indices=np.array(self.base_steering_joint_indices, dtype=np.int32),
        )
        self._articulation_view.set_joint_velocity_targets(
            wheel_velocities.reshape(1, -1),
            joint_indices=np.array(self.base_wheel_joint_indices, dtype=np.int32),
        )

    def get_base_joint_state(self):
        joint_positions = self._articulation_view.get_joint_positions()[0]
        joint_velocities = self._articulation_view.get_joint_velocities()[0]
        return {
            "steering_positions": joint_positions[self.base_steering_joint_indices].copy(),
            "wheel_positions": joint_positions[self.base_wheel_joint_indices].copy(),
            "steering_velocities": joint_velocities[self.base_steering_joint_indices].copy(),
            "wheel_velocities": joint_velocities[self.base_wheel_joint_indices].copy(),
        }
