"""SplitAloha robot implementation - Dual-arm manipulator."""
from copy import deepcopy
import math
import os

import numpy as np
from core.robots.base_robot import register_robot
from core.robots.template_robot import TemplateRobot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.xforms import get_world_pose


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
        self.mobile_base_prim_path = None
        self._mobile_support_joint_paths = []
        self._mobile_support_body_paths = []
        self._wheel_collision_paths = []
        self._disabled_collision_paths = []
        self._wheel_physics_material_path = None
        self._wheel_joint_paths = []
        self._steering_joint_paths = []
        super().__init__(*args, **kwargs)
        self.base_cfg = deepcopy(self.cfg.get("base", {}))
        self.base_steering_joint_names = list(self.base_cfg.get("steering_joint_names", []))
        self.base_wheel_joint_names = list(self.base_cfg.get("wheel_joint_names", []))
        self._setup_mobile_base_interface()
        self._configure_mobile_support_joints_for_physical_drive()

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

    def _setup_mobile_base_interface(self):
        mobile_root = os.path.dirname(os.path.dirname(self.fl_base_path))
        base_frame = str(self.base_cfg.get("ros", {}).get("base_frame", "base_link"))
        candidate_path = f"{mobile_root}/{base_frame}"
        if get_prim_at_path(candidate_path).IsValid():
            self.mobile_base_prim_path = candidate_path
        else:
            self.mobile_base_prim_path = None

        self._mobile_support_joint_paths = [
            (f"{mobile_root}/dummy_base_x/mobile_translate_x", "linear"),
            (f"{mobile_root}/dummy_base_y/mobile_translate_y", "linear"),
            (f"{mobile_root}/dummy_base_rotate/mobile_rotate", "angular"),
        ]
        self._mobile_support_body_paths = [
            f"{mobile_root}/dummy_base_x",
            f"{mobile_root}/dummy_base_y",
            f"{mobile_root}/dummy_base_rotate",
        ]
        # Only keep the real wheel collision geometry in the physical drive path.
        # Steering support / fork geometry can easily catch the floor or low obstacles
        # and destabilize the base when we are trying to validate wheel-ground contact.
        self._wheel_collision_paths = [
            f"{mobile_root}/fl_wheel_link/collisions",
            f"{mobile_root}/fr_wheel_link/collisions",
            f"{mobile_root}/rl_wheel_link/collisions",
            f"{mobile_root}/rr_wheel_link/collisions",
        ]
        self._disabled_collision_paths = [
            f"{mobile_root}/fl_steering_wheel_link/collisions",
            f"{mobile_root}/fr_steering_wheel_link/collisions",
            f"{mobile_root}/rl_steering_wheel_link/collisions",
            f"{mobile_root}/rr_steering_wheel_link/collisions",
        ]
        self._wheel_physics_material_path = f"{mobile_root}/Looks/wheel_physics_material"
        self._wheel_joint_paths = [
            f"{mobile_root}/fl_steering_wheel_link/fl_wheel",
            f"{mobile_root}/fr_steering_wheel_link/fr_wheel",
            f"{mobile_root}/rl_steering_wheel_link/rl_wheel",
            f"{mobile_root}/rr_steering_wheel_link/rr_wheel",
        ]
        self._steering_joint_paths = [f"{mobile_root}/{base_frame}/{joint_name}" for joint_name in self.base_steering_joint_names]

    def _configure_mobile_support_joints_for_physical_drive(self):
        try:
            from pxr import Gf, UsdPhysics  # pylint: disable=import-outside-toplevel
            from omni.physx.scripts import physicsUtils, utils  # pylint: disable=import-outside-toplevel
        except ImportError:
            return

        static_friction = float(self.base_cfg.get("wheel_static_friction", 1.5))
        dynamic_friction = float(self.base_cfg.get("wheel_dynamic_friction", static_friction))
        restitution = float(self.base_cfg.get("wheel_restitution", 0.0))
        wheel_drive_stiffness = float(self.base_cfg.get("wheel_drive_stiffness", 0.0))
        wheel_drive_damping = float(self.base_cfg.get("wheel_drive_damping", 150.0))
        wheel_drive_max_force = float(self.base_cfg.get("wheel_drive_max_force", 300.0))
        steering_drive_stiffness = float(self.base_cfg.get("steering_drive_stiffness", 1.0e7))
        steering_drive_damping = float(self.base_cfg.get("steering_drive_damping", 1.0e5))
        steering_drive_max_force = float(self.base_cfg.get("steering_drive_max_force", 1.0e6))
        stage = get_prim_at_path(self.robot_prim_path).GetStage()
        if self._wheel_physics_material_path and not get_prim_at_path(self._wheel_physics_material_path).IsValid():
            utils.addRigidBodyMaterial(
                stage,
                self._wheel_physics_material_path,
                staticFriction=static_friction,
                dynamicFriction=dynamic_friction,
                restitution=restitution,
            )

        for collision_path in self._wheel_collision_paths:
            prim = get_prim_at_path(collision_path)
            if not prim.IsValid():
                continue
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr().Set(True)
            if self._wheel_physics_material_path:
                physicsUtils.add_physics_material_to_prim(stage, prim, self._wheel_physics_material_path)

        for collision_path in self._disabled_collision_paths:
            prim = get_prim_at_path(collision_path)
            if not prim.IsValid():
                continue
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr().Set(False)

        for body_path in self._mobile_support_body_paths:
            prim = get_prim_at_path(body_path)
            if not prim.IsValid():
                continue
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_attr = mass_api.GetMassAttr()
            mass_value = mass_attr.Get() if mass_attr.HasAuthoredValue() else None
            if mass_value is None or not self._is_finite_scalar(mass_value) or float(mass_value) <= 0.0:
                self._set_or_create_attr(mass_attr, 0.5, mass_api.CreateMassAttr)
            inertia_attr = mass_api.GetDiagonalInertiaAttr()
            inertia_value = inertia_attr.Get() if inertia_attr.HasAuthoredValue() else None
            if not self._is_finite_vec3(inertia_value) or any(float(component) <= 0.0 for component in inertia_value):
                self._set_or_create_attr(
                    inertia_attr,
                    Gf.Vec3f(0.01, 0.01, 0.01),
                    mass_api.CreateDiagonalInertiaAttr,
                )
            com_attr = mass_api.GetCenterOfMassAttr()
            com_value = com_attr.Get() if com_attr.HasAuthoredValue() else None
            if not self._is_finite_vec3(com_value):
                self._set_or_create_attr(
                    com_attr,
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    mass_api.CreateCenterOfMassAttr,
                )
            axes_attr = mass_api.GetPrincipalAxesAttr()
            axes_value = axes_attr.Get() if axes_attr.HasAuthoredValue() else None
            if not self._is_finite_quat(axes_value):
                self._set_or_create_attr(
                    axes_attr,
                    Gf.Quatf(1.0, 0.0, 0.0, 0.0),
                    mass_api.CreatePrincipalAxesAttr,
                )

        for joint_path, drive_type in self._mobile_support_joint_paths:
            prim = get_prim_at_path(joint_path)
            if not prim.IsValid():
                continue
            drive_api = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if not drive_api:
                continue
            if drive_api.GetStiffnessAttr().HasAuthoredValue():
                drive_api.GetStiffnessAttr().Set(float(drive_api.GetStiffnessAttr().Get()))
            else:
                drive_api.GetStiffnessAttr().Set(0.0)
            if drive_api.GetDampingAttr().HasAuthoredValue():
                drive_api.GetDampingAttr().Set(float(drive_api.GetDampingAttr().Get()))
            else:
                drive_api.GetDampingAttr().Set(0.0)
            if drive_api.GetMaxForceAttr().HasAuthoredValue():
                drive_api.GetMaxForceAttr().Set(float(drive_api.GetMaxForceAttr().Get()))
            else:
                drive_api.GetMaxForceAttr().Set(0.0)
            if not drive_api.GetTargetVelocityAttr().HasAuthoredValue():
                drive_api.GetTargetVelocityAttr().Set(0.0)

        for joint_path in self._steering_joint_paths:
            prim = get_prim_at_path(joint_path)
            if not prim.IsValid():
                continue
            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive_api:
                continue
            drive_api.GetStiffnessAttr().Set(steering_drive_stiffness)
            drive_api.GetDampingAttr().Set(steering_drive_damping)
            drive_api.GetMaxForceAttr().Set(steering_drive_max_force)

        for joint_path in self._wheel_joint_paths:
            prim = get_prim_at_path(joint_path)
            if not prim.IsValid():
                continue
            drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive_api:
                continue
            drive_api.GetStiffnessAttr().Set(wheel_drive_stiffness)
            drive_api.GetDampingAttr().Set(wheel_drive_damping)
            drive_api.GetMaxForceAttr().Set(wheel_drive_max_force)

    @staticmethod
    def _is_finite_scalar(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _is_finite_vec3(cls, value):
        try:
            return value is not None and all(cls._is_finite_scalar(component) for component in value)
        except TypeError:
            return False

    @classmethod
    def _is_finite_quat(cls, value):
        if value is None:
            return False
        try:
            return all(
                cls._is_finite_scalar(component)
                for component in (value.GetReal(), *value.GetImaginary())
            )
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def _set_or_create_attr(attr, value, create_fn):
        authored_attr = attr if attr.IsValid() else create_fn()
        authored_attr.Set(value)

    def get_base_interface(self):
        return {
            "steering_joint_names": list(self.base_steering_joint_names),
            "wheel_joint_names": list(self.base_wheel_joint_names),
            "steering_joint_indices": list(self.base_steering_joint_indices),
            "wheel_joint_indices": list(self.base_wheel_joint_indices),
            "mobile_base_prim_path": self.mobile_base_prim_path,
            "base_cfg": deepcopy(self.base_cfg),
        }

    def get_mobile_base_pose(self):
        if self.mobile_base_prim_path:
            return get_world_pose(self.mobile_base_prim_path)
        return self.get_world_pose()

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
