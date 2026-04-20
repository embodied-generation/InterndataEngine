"""Isaac Sim ROS 2 bridge helpers for publishing /clock without system ROS installs."""

from __future__ import annotations

import importlib
import math
import time


_BRIDGE_EXTENSION = "omni.isaac.ros2_bridge"
_ROS2_IMPORTS = None


def _app_instance():
    import omni.kit.app

    return omni.kit.app.get_app()


def _load_ros2_imports():
    rclpy = importlib.import_module("rclpy")
    node_module = importlib.import_module("rclpy.node")
    parameter_module = importlib.import_module("rclpy.parameter")
    clock_module = importlib.import_module("rosgraph_msgs.msg")
    return {
        "rclpy": rclpy,
        "Node": node_module.Node,
        "Parameter": parameter_module.Parameter,
        "Clock": clock_module.Clock,
    }


def ensure_isaac_ros2_bridge_ready(simulation_app=None, *, max_wait_sec: float = 90.0):
    """Enable Isaac's ROS 2 bridge and wait for its bundled Python ROS modules."""

    global _ROS2_IMPORTS
    if _ROS2_IMPORTS is not None:
        return _ROS2_IMPORTS

    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension(_BRIDGE_EXTENSION)

    deadline = time.monotonic() + max(float(max_wait_sec), 1.0)
    last_error = None
    while time.monotonic() < deadline:
        app = _app_instance()
        if simulation_app is not None:
            simulation_app.update()
        else:
            app.update()
        try:
            _ROS2_IMPORTS = _load_ros2_imports()
            return _ROS2_IMPORTS
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc

    raise RuntimeError(
        f"Isaac ROS 2 bridge '{_BRIDGE_EXTENSION}' did not expose internal ROS Python modules in time"
    ) from last_error


class SimClockPublisher:
    def __init__(self, world, *, simulation_app=None):
        self.world = world
        self.simulation_app = simulation_app
        ros2_imports = ensure_isaac_ros2_bridge_ready(simulation_app=simulation_app)
        self._rclpy = ros2_imports["rclpy"]
        self._Clock = ros2_imports["Clock"]
        self._owns_context = False
        if not self._rclpy.ok():
            self._rclpy.init(args=None)
            self._owns_context = True
        self.node = ros2_imports["Node"]("simbox_nav2_skill_clock_publisher")
        self.node.set_parameters([ros2_imports["Parameter"]("use_sim_time", value=False)])
        self._clock_pub = self.node.create_publisher(self._Clock, "/clock", 10)

    def publish(self):
        sim_time = float(getattr(self.world, "current_time", 0.0))
        secs = int(math.floor(sim_time))
        nanosecs = int(round((sim_time - secs) * 1.0e9))
        if nanosecs >= 1_000_000_000:
            secs += 1
            nanosecs -= 1_000_000_000
        msg = self._Clock()
        msg.clock.sec = secs
        msg.clock.nanosec = nanosecs
        self._clock_pub.publish(msg)
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def destroy(self):
        self.node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
