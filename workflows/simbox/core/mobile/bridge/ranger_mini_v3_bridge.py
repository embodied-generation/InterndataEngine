"""SplitAloha/Ranger 风格底盘在 Isaac Sim 中的 4WIS 直接桥接实现。"""

from __future__ import annotations

import math

import numpy as np

from .base_bridge import BaseBridge
from .types import BaseCommand


class RangerMiniV3Bridge(BaseBridge):
    """将标准车体 twist 直接映射为 4WIS 底盘关节控制。"""

    def __init__(self, robot, node_name: str = "ranger_mini_v3_bridge", driver=None):
        self._module_positions = np.zeros((4, 2), dtype=np.float32)
        self._body_twist_deadband = 1.0e-4
        self._module_speed_deadband = 1.0e-5
        super().__init__(robot=robot, node_name=node_name, driver=driver)

    def _validate_bridge_configuration(self, *, steering_count: int, wheel_count: int):
        if steering_count != 4 or wheel_count != 4:
            raise ValueError(
                f"RangerMiniV3Bridge expects 4 steering and 4 wheel joints, got {steering_count}/{wheel_count}"
            )

        half_wheel_base = 0.5 * self._wheel_base
        half_track_width = 0.5 * self._track_width
        self._module_positions = np.asarray(
            [
                [half_wheel_base, half_track_width],
                [half_wheel_base, -half_track_width],
                [-half_wheel_base, half_track_width],
                [-half_wheel_base, -half_track_width],
            ],
            dtype=np.float32,
        )
        self._body_twist_deadband = float(self.base_cfg.get("body_twist_deadband", 1.0e-4))
        self._module_speed_deadband = float(self.base_cfg.get("module_speed_deadband", 1.0e-5))

    def _map_command(self, command: BaseCommand) -> tuple[np.ndarray, np.ndarray]:
        current_steering = self._last_applied_steering.astype(np.float32).copy()
        requested_steering = current_steering.copy()
        wheel_linear_speeds = np.zeros(4, dtype=np.float32)

        if (
            abs(command.vx_body) <= self._body_twist_deadband
            and abs(command.vy_body) <= self._body_twist_deadband
            and abs(command.wz_body) <= self._body_twist_deadband
        ):
            return requested_steering, np.zeros(4, dtype=np.float32)

        for module_index, (module_x, module_y) in enumerate(self._module_positions):
            module_vx = float(command.vx_body) - float(command.wz_body) * float(module_y)
            module_vy = float(command.vy_body) + float(command.wz_body) * float(module_x)
            module_speed = math.hypot(module_vx, module_vy)
            if module_speed <= self._module_speed_deadband:
                requested_steering[module_index] = float(current_steering[module_index])
                wheel_linear_speeds[module_index] = 0.0
                continue

            desired_joint_heading = self._steering_command_sign * math.atan2(module_vy, module_vx)
            desired_joint_heading, signed_module_speed = self._minimize_steering_rotation(
                desired_joint_heading=desired_joint_heading,
                current_joint_heading=float(current_steering[module_index]),
                module_speed=module_speed,
            )
            requested_steering[module_index] = float(
                np.clip(desired_joint_heading, -self._steering_limit, self._steering_limit)
            )
            wheel_linear_speeds[module_index] = float(signed_module_speed)

        wheel_velocities = wheel_linear_speeds / self._wheel_radius
        peak_wheel_velocity = float(np.max(np.abs(wheel_velocities))) if wheel_velocities.size else 0.0
        if peak_wheel_velocity > self._wheel_velocity_limit > 0.0:
            wheel_velocities *= float(self._wheel_velocity_limit / peak_wheel_velocity)
        return requested_steering.astype(np.float32), wheel_velocities.astype(np.float32)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _minimize_steering_rotation(
        self,
        *,
        desired_joint_heading: float,
        current_joint_heading: float,
        module_speed: float,
    ) -> tuple[float, float]:
        desired_joint_heading = self._wrap_to_pi(desired_joint_heading)
        current_joint_heading = self._wrap_to_pi(current_joint_heading)
        heading_error = self._wrap_to_pi(desired_joint_heading - current_joint_heading)
        if abs(heading_error) > 0.5 * math.pi:
            desired_joint_heading = self._wrap_to_pi(desired_joint_heading + math.pi)
            module_speed = -float(module_speed)
        return desired_joint_heading, float(module_speed)
