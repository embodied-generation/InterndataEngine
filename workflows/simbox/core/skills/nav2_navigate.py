"""Main-flow Nav2 navigation skill."""

from __future__ import annotations

from copy import deepcopy

from core.skills.base_skill import BaseSkill, SKILL_DICT, register_skill
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask

from nav2.runtime import configure_robot_for_nav2_skill


@register_skill
class Nav2_Navigate(BaseSkill):
    """Block until SplitAloha reaches an explicit world-frame navigation goal."""

    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.world = kwargs["world"]
        self.workflow = kwargs.get("workflow")
        self.skill_cfg = cfg

        try:
            self.goal_x = float(cfg["goal_x"])
            self.goal_y = float(cfg["goal_y"])
            self.goal_yaw = float(cfg["goal_yaw"])
        except KeyError as exc:
            raise KeyError("nav2__navigate requires goal_x, goal_y, and goal_yaw") from exc

        self.position_tolerance_m = float(cfg.get("xy_goal_tolerance", 0.10))
        self.yaw_tolerance_rad = float(cfg.get("yaw_goal_tolerance", 0.10))
        self.startup_timeout_sec = float(cfg.get("startup_timeout_sec", 60.0))
        self.runtime_timeout_sec = float(cfg.get("runtime_timeout_sec", 240.0))
        self.output_root = str(cfg.get("output_root", "output/ros_bridge/skills"))
        self.scene_name = str(cfg.get("scene_name", getattr(task, "name", "nav2_skill_scene")))

        self._configured_base_cfg = configure_robot_for_nav2_skill(
            self.robot,
            map_output_dir=str(cfg.get("map_output_dir", "output/nav2_maps")),
            map_resolution=float(cfg.get("map_resolution", 0.05)),
            map_z_min=float(cfg.get("map_z_min", 0.0)),
            map_z_max=float(cfg.get("map_z_max", 0.35)),
            position_tolerance_m=self.position_tolerance_m,
            yaw_tolerance_rad=self.yaw_tolerance_rad,
        )
        self._manager = None
        self._goal_started = False
        self._local_done = False
        self._local_success = False
        self._hold_command = None
        self.manip_list = []
        self.failure_reason = ""
        self.error_message = ""

    def simple_generate_manip_cmds(self):
        # Navigation skill should not inject manipulator hold commands into the
        # main action loop. The test runner leaves the arm untouched while Nav2
        # drives the base, and the formal mainflow should match that behavior.
        self._hold_command = None
        self.manip_list = []

    def is_ready(self):
        # This skill advances through the runtime manager rather than by
        # producing arm joint actions for the main action loop.
        return bool(self.manip_list)

    def update(self):
        if self._local_done:
            return

        if self._manager is None:
            if self.workflow is not None and hasattr(self.workflow, "get_navigation_session_manager"):
                self._manager = self.workflow.get_navigation_session_manager(getattr(self.robot, "name", ""))
            if self._manager is None:
                self.failure_reason = "manager_unavailable"
                self.error_message = "Workflow did not initialize a navigation session manager for this robot"
                self._local_done = True
                self._local_success = False
                return

        self._manager.bind(
            world=self.world,
            task=self.task,
            robot=self.robot,
            scene_name=self.scene_name,
        )
        if not self._goal_started:
            self._manager.begin_goal(
                goal_x=self.goal_x,
                goal_y=self.goal_y,
                goal_yaw=self.goal_yaw,
                position_tolerance_m=self.position_tolerance_m,
                yaw_tolerance_rad=self.yaw_tolerance_rad,
                startup_timeout_sec=self.startup_timeout_sec,
                runtime_timeout_sec=self.runtime_timeout_sec,
            )
            self._goal_started = True

        if self._manager.done:
            self.failure_reason = str(self._manager.result.failure_reason)
            self.error_message = str(self._manager.result.error_message)
            self._local_done = True
            self._local_success = bool(self._manager.success)

    def is_done(self):
        return bool(self._local_done)

    def is_success(self):
        return bool(self._local_success)

    def is_feasible(self):
        if self._manager is None:
            return True
        return not (self._local_done and not self._local_success)


SKILL_DICT["nav2__navigate"] = Nav2_Navigate
