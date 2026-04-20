"""桥接层通用数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseCommand:
    """桥接层内部统一命令格式，直接表达车体坐标系 twist。"""

    vx_body: float
    vy_body: float
    wz_body: float
    received_time_sec: float

    @classmethod
    def from_twist_message(cls, msg, *, received_time_sec: float) -> "BaseCommand":
        return cls(
            vx_body=float(msg.linear.x),
            vy_body=float(msg.linear.y),
            wz_body=float(msg.angular.z),
            received_time_sec=float(received_time_sec),
        )

    @classmethod
    def zero(cls, *, received_time_sec: float) -> "BaseCommand":
        return cls(
            vx_body=0.0,
            vy_body=0.0,
            wz_body=0.0,
            received_time_sec=float(received_time_sec),
        )

    # Compatibility-only accessors for older debug/reporting code.
    @property
    def motion_mode(self) -> int:
        return 0

    @property
    def linear_speed(self) -> float:
        return float(self.vx_body)

    @property
    def lateral_speed(self) -> float:
        return float(self.vy_body)

    @property
    def steering_angle(self) -> float:
        return 0.0

    @property
    def angular_speed(self) -> float:
        return float(self.wz_body)
