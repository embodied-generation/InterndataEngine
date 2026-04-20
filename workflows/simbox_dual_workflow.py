import glob
import json
import os
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import numpy as np
import yaml
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from omni.physx import acquire_physx_interface
from tqdm import tqdm
from yaml import Loader

from deps.world_toolkit.world_recorder import WorldRecorder
from workflows.simbox.utils.task_config_parser import TaskConfigParser

from .base import NimbusWorkFlow
from .simbox.core.controllers import get_controller_cls
from .simbox.core.loggers.lmdb_logger import LmdbLogger
from .simbox.core.loggers.utils import log_dual_obs
from .simbox.core.skills import get_skill_cls
from .simbox.core.tasks import get_task_cls
from .simbox.core.utils.collision_utils import filter_collisions
from .simbox.core.utils.utils import set_random_seed


class _PassiveSkillController:
    """Placeholder controller for skills that do not emit manipulator actions."""

    def __init__(self, *, robot_name: str, controller_name: str):
        self.name = robot_name
        self.robot_file = f"{controller_name}_passive_skill_controller"
        self._gripper_state = 1.0

    def reset(self):
        return None

    def forward(self, _command):
        return {
            "joint_positions": np.array([], dtype=np.float32),
            "joint_indices": np.array([], dtype=np.int64),
        }


# pylint: disable=unused-argument
@NimbusWorkFlow.register("SimBoxDualWorkFlow")
class SimBoxDualWorkFlow(NimbusWorkFlow):
    def __init__(
        self,
        world,
        task_cfg_path: str,
        scene_info: str = "dining_room_scene_info",
        random_seed: int = None,
    ):
        self.scene_info = scene_info
        self.step_replay = False
        self.random_seed = random_seed
        self._ros_base_command_controllers = {}
        self._ros_base_bridges = {}
        self._nav2_navigators = {}
        self._navigation_session_managers = {}
        self._nav2_clock_publisher = None
        self._ros_bridge_unavailable_reported = False
        super().__init__(world, task_cfg_path)

    @staticmethod
    def _task_uses_nav2_skill(task_cfg: dict, robot_name: str) -> bool:
        skills = task_cfg.get("skills", [])
        if not isinstance(skills, list):
            return False
        for cfg_skill_dict in skills:
            if not isinstance(cfg_skill_dict, dict):
                continue
            robot_skill_list = cfg_skill_dict.get(robot_name, [])
            if not isinstance(robot_skill_list, list):
                continue
            for lr_skill_dict in robot_skill_list:
                if not isinstance(lr_skill_dict, dict):
                    continue
                for lr_skill_list in lr_skill_dict.values():
                    if not isinstance(lr_skill_list, list):
                        continue
                    for skill_cfg in lr_skill_list:
                        if isinstance(skill_cfg, dict) and str(skill_cfg.get("name", "")).strip() == "nav2__navigate":
                            return True
        return False

    def _normalize_skill_managed_mobile_configs(self, task_cfg: dict):
        try:
            from nav2.runtime import configure_base_cfg_for_nav2_skill
        except Exception:
            return

        robots = task_cfg.get("robots", [])
        if not isinstance(robots, list):
            return

        for robot in robots:
            if not isinstance(robot, dict):
                continue
            robot_name = str(robot.get("name", "")).strip()
            if not robot_name or not self._task_uses_nav2_skill(task_cfg, robot_name):
                continue
            base_cfg = robot.get("base", {})
            if not isinstance(base_cfg, dict):
                continue
            robot["base"] = configure_base_cfg_for_nav2_skill(base_cfg)

    @staticmethod
    def _skill_requires_controller(skill_cfg: dict) -> bool:
        if not isinstance(skill_cfg, dict):
            return True
        return str(skill_cfg.get("name", "")).strip() != "nav2__navigate"

    def _skill_controller_names(self, task_cfg: dict, robot_name: str) -> set[str]:
        controller_names = set()
        skills = task_cfg.get("skills", [])
        if not isinstance(skills, list):
            return controller_names
        for cfg_skill_dict in skills:
            if not isinstance(cfg_skill_dict, dict):
                continue
            robot_skill_list = cfg_skill_dict.get(robot_name, [])
            if not isinstance(robot_skill_list, list):
                continue
            for lr_skill_dict in robot_skill_list:
                if not isinstance(lr_skill_dict, dict):
                    continue
                for lr_name in lr_skill_dict.keys():
                    controller_names.add(str(lr_name))
        return controller_names

    def _required_controller_names(self, task_cfg: dict, robot_name: str) -> set[str]:
        required = set()
        skills = task_cfg.get("skills", [])
        if not isinstance(skills, list):
            return required
        for cfg_skill_dict in skills:
            if not isinstance(cfg_skill_dict, dict):
                continue
            robot_skill_list = cfg_skill_dict.get(robot_name, [])
            if not isinstance(robot_skill_list, list):
                continue
            for lr_skill_dict in robot_skill_list:
                if not isinstance(lr_skill_dict, dict):
                    continue
                for lr_name, lr_skill_list in lr_skill_dict.items():
                    if not isinstance(lr_skill_list, list):
                        continue
                    if any(self._skill_requires_controller(skill_cfg) for skill_cfg in lr_skill_list):
                        required.add(str(lr_name))
        return required

    def parse_task_cfgs(self, task_cfg_path: str) -> list:
        task_cfgs = TaskConfigParser(task_cfg_path).parse_tasks()
        # Merge robot configs for each task
        for task_cfg in task_cfgs:
            self._merge_robot_configs(task_cfg)
        return task_cfgs

    def _merge_robot_configs(self, task_cfg: dict):
        """Merge robot configs from robot_config_file into task_cfg['robots']."""
        robots = task_cfg.get("robots", [])

        for robot in robots:
            robot_config_file = robot.get("robot_config_file")
            if robot_config_file:
                with open(robot_config_file, "r", encoding="utf-8") as f:
                    robot_base_cfg = yaml.load(f, Loader=Loader)

                # Merge: robot_base_cfg as base, task_cfg['robots'][i] overrides
                merged_cfg = deepcopy(robot_base_cfg)
                merged_cfg.update(robot)
                base_cfg = merged_cfg.get("base")
                if isinstance(base_cfg, dict):
                    self._merge_base_configs(base_cfg)
                robot.clear()
                robot.update(merged_cfg)

        self._normalize_skill_managed_mobile_configs(task_cfg)

    def _merge_base_configs(self, base_cfg: dict):
        """Merge mobile base/nav config references into base_cfg in-place."""
        override_cfg = deepcopy(base_cfg)
        merged_base_cfg = {}
        base_config_file = override_cfg.get("base_config_file")
        nav_config_file = override_cfg.get("nav_config_file")

        if base_config_file:
            with open(base_config_file, "r", encoding="utf-8") as f:
                loaded_base_cfg = yaml.load(f, Loader=Loader)
            if isinstance(loaded_base_cfg, dict):
                merged_base_cfg = deepcopy(loaded_base_cfg)

        if nav_config_file:
            with open(nav_config_file, "r", encoding="utf-8") as f:
                loaded_nav_cfg = yaml.load(f, Loader=Loader)
            if isinstance(loaded_nav_cfg, dict):
                self._deep_update_dict(merged_base_cfg, loaded_nav_cfg)

        self._deep_update_dict(merged_base_cfg, override_cfg)
        base_cfg.clear()
        base_cfg.update(merged_base_cfg)

    def _deep_update_dict(self, base: dict, override: dict):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update_dict(base[key], value)
            else:
                base[key] = value

    def reset(self, need_preload: bool = True):
        self._prepare_navigation_session_managers_for_reset()
        self._destroy_navigation_session_managers()
        self._destroy_nav2_clock_publisher()
        self._destroy_ros_base_bridges()

        # A previous task can remain registered if scene setup fails during world.reset().
        # Clear the world before constructing the next task so retries do not trip the
        # duplicate-name guard in omni.isaac.core.world.World.add_task().
        if self.world.get_current_tasks() or self.world.is_tasks_scene_built():
            self.world.clear()

        # source code noted this as debug, so it could be removed later
        from omni.isaac.core.utils.viewports import set_camera_view

        set_camera_view(eye=[1.3, 0.7, 2.7], target=[0.0, 0, 1.5], camera_prim_path="/OmniverseKit_Persp")
        # Modify config
        arena_file_path = self.task_cfg.get("arena_file", None)
        with open(arena_file_path, "r", encoding="utf-8") as arena_file:
            arena = yaml.load(arena_file, Loader=Loader)

        # if "involved_scenes" in arena:
        #     arena["involved_scenes"] = self.scene_info

        self.task_cfg["arena"] = arena

        for obj_cfg in self.task_cfg["objects"]:
            if obj_cfg["target_class"] == "ArticulatedObject":
                if obj_cfg.get("apply_randomization", False):
                    asset_root = self.task_cfg["asset_root"]
                    art_paths = glob.glob(os.path.join(asset_root, obj_cfg["art_cat"], "*"))
                    art_paths.sort()
                    path = random.choice(art_paths)
                    info_name = obj_cfg["info_name"]
                    info_path = f"{path}/Kps/{info_name}/info.json"
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    scale = info["object_scale"][:3]

                    obj_cfg["path"] = path.replace(f"{asset_root}/", "", 1) + "/instance.usd"
                    obj_cfg["category"] = path.split("/")[-2]
                    obj_cfg["obj_info_path"] = info_path.replace(f"{asset_root}/", "", 1)
                    obj_cfg["scale"] = scale
                    self.task_cfg["data"]["collect_info"] = obj_cfg["category"]

        self.task_cfg.pop("arena_file", None)
        self.task_cfg.pop("camera_file", None)
        self.task_cfg.pop("logger_file", None)
        # Modify config done
        if self.task_cfg.get("fluid", None):
            # for fluid manipulation, only gpu mode is supportive
            physx_interface = acquire_physx_interface()
            physx_interface.overwrite_gpu_setting(1)

        self.task = get_task_cls(self.task_cfg["task"])(self.task_cfg)
        self.stage = self.world.stage
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        self.world.add_task(self.task)

        # # Add hidden ground plane for physics simulation
        # from omni.isaac.core.objects import GroundPlane
        # plane = GroundPlane(
        #     prim_path="/World/GroundPlane",
        #     z_position=0.0,
        #     visible=False,
        # )

        prim_paths = []  # do not collide with each other
        global_collision_paths = []  # collide with everything

        self.robots_prim_paths = []
        for robot in self.task_cfg["robots"]:
            robot_prim_path = self.task.root_prim_path + "/" + robot["name"]
            prim_paths.append(robot_prim_path)
            self.robots_prim_paths.append(robot_prim_path)
        neglect_collision_names = self.task_cfg.get("neglect_collision_names", [])
        candidates = self.task_cfg["objects"] + self.task_cfg["arena"]["fixtures"]
        for candidate in candidates:
            candidate_prim_path = self.task.root_prim_path + "/" + candidate["name"]
            global_collision_paths.append(candidate_prim_path)
            for neglect_collision_name in neglect_collision_names:
                if neglect_collision_name in candidate["name"]:
                    prim_paths.append(candidate_prim_path)
                    global_collision_paths.remove(candidate_prim_path)

        collision_root_path = "/World/collisions"
        filter_collisions(
            self.stage,
            self.world.get_physics_context().prim_path,
            collision_root_path,
            prim_paths,
            global_collision_paths,
        )
        self.world.reset()
        self._step_world(render=True)
        self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)
        self._initialize_ros_base_bridges()
        self._initialize_navigation_session_managers()

        for _ in range(50):
            self._init_static_objects(self.task)
            self._step_world(render=False)

        self.logger = LmdbLogger(
            task_dir=self.task_cfg["data"]["task_dir"],
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
            collect_info=self.task_cfg["data"]["collect_info"],
            version=self.task_cfg["data"].get("version", "v1.0"),
        )
        # Motion vectors are large dense tensors; keep LMDB logging opt-in.
        self.log_motion_vectors = bool(self.task_cfg["data"].get("log_motion_vectors", False))

        if self.random_seed is not None:
            seed = self.random_seed
        else:
            seed = time.time_ns() % (2**32)
        set_random_seed(seed)

        # while True:
        #     self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

    def _initialize_skills(self, task, task_cfg, controllers, world):
        draw_points = False
        if draw_points:
            from omni.isaac.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
        else:
            draw = None

        # Initialize skills for each robot.
        skills = []
        for cfg_skill_dict in task_cfg["skills"]:
            skill_dict = defaultdict(list)
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                robot = task.robots[robot_name]
                controller = controllers[robot_name]

                for lr_skill_dict in robot_skill_list:
                    skill_sequence = [
                        [
                            get_skill_cls(skill_cfg["name"])(
                                robot,
                                controller[lr_name],
                                task,
                                skill_cfg,
                                world=world,
                                workflow=self,
                                draw=draw,
                            )
                            for skill_cfg in lr_skill_list
                        ]
                        for lr_name, lr_skill_list in lr_skill_dict.items()
                    ]
                    skill_dict[robot_name].append(skill_sequence)
            skills.append(skill_dict)
        return skills

    def _initialize_controllers(self, task, task_cfg, world):
        """Initialize controllers for each robot."""
        controllers = {}
        for robot in task_cfg["robots"]:
            robot_name = robot["name"]
            controllers[robot_name] = {}
            required_controller_names = self._required_controller_names(task_cfg, robot_name)
            declared_controller_names = self._skill_controller_names(task_cfg, robot_name)

            robot_files = robot.get("robot_file", [])
            if isinstance(robot_files, str):
                robot_files = [robot_files]
            robot_files_by_name = {}
            for robot_file in robot_files:
                controller_name = "left" if "left" in robot_file else "right"
                robot_files_by_name[controller_name] = robot_file

            for controller_name in required_controller_names:
                robot_file = robot_files_by_name.get(controller_name)
                if robot_file is None:
                    raise KeyError(
                        f"Robot '{robot_name}' is missing robot_file for controller '{controller_name}'"
                    )
                controllers[robot_name][controller_name] = get_controller_cls(robot["target_class"])(
                    name=robot_name,
                    robot_file=robot_file,
                    constrain_grasp_approach=robot.get("constrain_grasp_approach", False),
                    collision_activation_distance=robot.get("collision_activation_distance", 0.03),
                    task=task,
                    world=world,
                    ignore_substring=robot.get("ignore_substring", ["material", "Plane", "conveyor", "scene", "table"]),
                    use_batch=robot.get("use_batch", False),
                )
                controllers[robot_name][controller_name].reset()

            passive_controller_names = (
                declared_controller_names | set(robot_files_by_name.keys())
            ) - required_controller_names
            for controller_name in passive_controller_names:
                controllers[robot_name][controller_name] = _PassiveSkillController(
                    robot_name=robot_name,
                    controller_name=controller_name,
                )
        return controllers

    def _initialize_ros_base_bridges(self):
        self._destroy_ros_base_bridges()
        self._ros_base_command_controllers = {}
        self._ros_base_bridges = {}
        try:
            from .simbox.core.mobile.bridge import RangerMiniV3Bridge
        except Exception as exc:
            if not self._ros_bridge_unavailable_reported:
                print(f"[ros-base-bridge] ROS bridge unavailable, skip initialization: {exc}")
                self._ros_bridge_unavailable_reported = True
            return

        for robot_name, robot in self.task.robots.items():
            if not hasattr(robot, "get_base_interface") or not hasattr(robot, "apply_base_command"):
                continue

            try:
                base_interface = robot.get_base_interface()
            except Exception:
                continue

            base_cfg = base_interface.get("base_cfg", {}) if isinstance(base_interface, dict) else {}
            ros_cfg = base_cfg.get("ros", {}) if isinstance(base_cfg, dict) else {}
            if not isinstance(ros_cfg, dict) or not ros_cfg:
                continue
            if not bool(ros_cfg.get("enabled", True)):
                continue

            bridge_node_name = f"{robot_name}_ranger_mini_v3_bridge".replace("-", "_")
            bridge = None
            try:
                bridge = RangerMiniV3Bridge(robot, node_name=bridge_node_name)
            except Exception as exc:
                print(f"[ros-base-bridge] Failed to initialize bridge for '{robot_name}': {exc}")
                if bridge is not None:
                    try:
                        bridge.destroy()
                    except Exception:
                        pass
                continue
            self._ros_base_bridges[robot_name] = bridge
            setattr(robot, "_simbox_ros_base_command_controller", None)
            setattr(robot, "_simbox_ros_base_bridge", bridge)
            print(f"[ros-base-bridge] '{robot_name}' using DIRECT /cmd_vel 4WIS bridge")

        if self._ros_base_bridges:
            robot_names = sorted(self._ros_base_bridges.keys())
            print(f"[ros-base-bridge] Initialized {len(robot_names)} bridge(s): {robot_names}")

    def _step_ros_base_bridges(self):
        if not self._ros_base_bridges:
            return

        get_physics_dt = getattr(self.world, "get_physics_dt", None)
        if callable(get_physics_dt):
            step_dt = float(get_physics_dt())
        else:
            step_dt = float(getattr(self.world, "physics_dt", 1.0 / 60.0))

        broken_robot_names = []
        for robot_name, controller in list(self._ros_base_command_controllers.items()):
            try:
                controller.step()
            except Exception as exc:
                print(f"[ros-base-bridge] Command controller step failed for '{robot_name}': {exc}")
                broken_robot_names.append(robot_name)

        for robot_name, bridge in list(self._ros_base_bridges.items()):
            try:
                bridge.step(step_dt=step_dt)
            except Exception as exc:
                print(f"[ros-base-bridge] Bridge step failed for '{robot_name}': {exc}")
                broken_robot_names.append(robot_name)

        for robot_name in set(broken_robot_names):
            controller = self._ros_base_command_controllers.pop(robot_name, None)
            if controller is not None:
                try:
                    controller.destroy()
                except Exception as exc:
                    print(f"[ros-base-bridge] Command controller destroy failed for '{robot_name}': {exc}")
                robot = self.task.robots.get(robot_name)
                if robot is not None and hasattr(robot, "_simbox_ros_base_command_controller"):
                    setattr(robot, "_simbox_ros_base_command_controller", None)

            bridge = self._ros_base_bridges.pop(robot_name, None)
            if bridge is None:
                continue
            try:
                bridge.destroy()
            except Exception as exc:
                print(f"[ros-base-bridge] Bridge destroy failed for '{robot_name}': {exc}")
            robot = self.task.robots.get(robot_name)
            if robot is not None and hasattr(robot, "_simbox_ros_base_bridge"):
                setattr(robot, "_simbox_ros_base_bridge", None)

    def _robot_nav2_enabled(self, robot) -> bool:
        if not hasattr(robot, "get_base_interface"):
            return False
        try:
            base_interface = robot.get_base_interface()
        except Exception:
            return False
        base_cfg = base_interface.get("base_cfg", {}) if isinstance(base_interface, dict) else {}
        ros_cfg = base_cfg.get("ros", {}) if isinstance(base_cfg, dict) else {}
        nav2_cfg = ros_cfg.get("nav2", {}) if isinstance(ros_cfg, dict) else {}
        return isinstance(nav2_cfg, dict) and bool(nav2_cfg.get("enabled", False))

    def _initialize_navigation_session_managers(self):
        try:
            from nav2.isaac_ros_clock import SimClockPublisher
            from nav2.runtime import PersistentNav2RuntimeManager
        except Exception as exc:
            print(f"[ros-nav2-runtime] Runtime manager unavailable, skip initialization: {exc}")
            return

        live_robot_names = set()
        for robot_name, robot in self.task.robots.items():
            if not self._robot_nav2_enabled(robot):
                continue
            robot = self.task.robots.get(robot_name)
            if robot is None:
                continue
            live_robot_names.add(robot_name)
            manager = self._navigation_session_managers.get(robot_name)
            if manager is None:
                manager = PersistentNav2RuntimeManager(
                    world=self.world,
                    task=self.task,
                    robot=robot,
                    output_root="output/ros_bridge/skills",
                    scene_name=str(getattr(self.task, "name", "nav2_skill_scene")),
                )
                self._navigation_session_managers[robot_name] = manager
            manager.bind(
                world=self.world,
                task=self.task,
                robot=robot,
                scene_name=str(getattr(self.task, "name", "nav2_skill_scene")),
            )
        if live_robot_names and self._nav2_clock_publisher is None:
            self._nav2_clock_publisher = SimClockPublisher(
                self.world,
                simulation_app=getattr(self, "simulation_app", None),
            )

        stale_robot_names = [name for name in self._navigation_session_managers.keys() if name not in live_robot_names]
        for robot_name in stale_robot_names:
            manager = self._navigation_session_managers.pop(robot_name, None)
            if manager is None:
                continue
            try:
                manager.shutdown()
            except Exception as exc:
                print(f"[ros-nav2-runtime] Runtime manager shutdown failed for '{robot_name}': {exc}")

        if self._navigation_session_managers:
            robot_names = sorted(self._navigation_session_managers.keys())
            print(f"[ros-nav2-runtime] Initialized {len(robot_names)} session manager(s): {robot_names}")

    def _prepare_navigation_session_managers_for_reset(self):
        for robot_name, manager in list(self._navigation_session_managers.items()):
            try:
                manager.prepare_for_reset()
            except Exception as exc:
                print(f"[ros-nav2-runtime] Runtime manager prepare_for_reset failed for '{robot_name}': {exc}")

    def _step_navigation_session_managers(self):
        if not self._navigation_session_managers:
            return

        broken_robot_names = []
        for robot_name, manager in list(self._navigation_session_managers.items()):
            try:
                manager.step()
            except Exception as exc:
                print(f"[ros-nav2-runtime] Runtime manager step failed for '{robot_name}': {exc}")
                broken_robot_names.append(robot_name)

        for robot_name in broken_robot_names:
            manager = self._navigation_session_managers.pop(robot_name, None)
            if manager is None:
                continue
            try:
                manager.shutdown()
            except Exception as exc:
                print(f"[ros-nav2-runtime] Runtime manager shutdown failed for '{robot_name}': {exc}")

    def _destroy_navigation_session_managers(self):
        for robot_name, manager in list(self._navigation_session_managers.items()):
            try:
                manager.shutdown()
            except Exception as exc:
                print(f"[ros-nav2-runtime] Runtime manager shutdown failed for '{robot_name}': {exc}")
        self._navigation_session_managers = {}

    def get_navigation_session_manager(self, robot_name: str):
        return self._navigation_session_managers.get(robot_name)

    def _publish_nav2_clock(self):
        if self._nav2_clock_publisher is None:
            return
        try:
            self._nav2_clock_publisher.world = self.world
            self._nav2_clock_publisher.publish()
        except Exception as exc:
            print(f"[ros-nav2-runtime] Clock publish failed: {exc}")
            self._destroy_nav2_clock_publisher()

    def _destroy_nav2_clock_publisher(self):
        if self._nav2_clock_publisher is None:
            return
        try:
            self._nav2_clock_publisher.destroy()
        except Exception as exc:
            print(f"[ros-nav2-runtime] Clock publisher destroy failed: {exc}")
        self._nav2_clock_publisher = None

    def _destroy_nav2_navigators(self):
        self._nav2_navigators = {}

    def _destroy_ros_base_bridges(self):
        for robot_name, controller in list(self._ros_base_command_controllers.items()):
            try:
                controller.destroy()
            except Exception as exc:
                print(f"[ros-base-bridge] Command controller destroy failed for '{robot_name}': {exc}")
            robot = self.task.robots.get(robot_name)
            if robot is not None and hasattr(robot, "_simbox_ros_base_command_controller"):
                setattr(robot, "_simbox_ros_base_command_controller", None)
        self._ros_base_command_controllers = {}

        for robot_name, bridge in list(self._ros_base_bridges.items()):
            try:
                bridge.destroy()
            except Exception as exc:
                print(f"[ros-base-bridge] Bridge destroy failed for '{robot_name}': {exc}")
            robot = self.task.robots.get(robot_name)
            if robot is not None and hasattr(robot, "_simbox_ros_base_bridge"):
                setattr(robot, "_simbox_ros_base_bridge", None)
        self._ros_base_bridges = {}

    def _step_world(self, render: bool = False):
        # Match the successful test runner more closely:
        # 1. pump ROS/nav before physics
        # 2. step physics
        # 3. pump ROS/nav once more after physics so odom/cmd_vel callbacks are
        #    processed against the updated simulation state.
        self._publish_nav2_clock()
        self._step_navigation_session_managers()
        self._step_ros_base_bridges()
        self.world.step(render=render)
        self._publish_nav2_clock()
        self._step_navigation_session_managers()

    def __del__(self):
        try:
            self._destroy_navigation_session_managers()
            self._destroy_nav2_clock_publisher()
        except Exception:
            pass

    def _initialize_world_recorder(self):
        """
        Initialize WorldRecorder with appropriate mode based on configuration.

        Supports two modes:
        - step_replay=False: Records prim poses for fast geometric replay (compatible with old workflow)
        - step_replay=True: Uses preprocessed joint position data for physics-accurate replay (new default)
        """
        self.world_recorder = WorldRecorder(
            self.world,
            self.task.robots,
            self.task.objects | self.task.distractors | self.task.visuals,
            step_replay=self.step_replay,
        )
        self.world_recorder.reset()

    def _reset_controllers(self, controllers):
        """Reset all controllers."""
        for _, controller in controllers.items():
            for _, ctrl in controller.items():
                ctrl.reset()

    def _init_static_objects(self, task):
        for _, obj in task.objects.items():
            try:
                init_translation = obj.init_translation
                init_orientation = obj.init_orientation
                init_parent = obj.init_parent
                if init_translation and init_orientation and init_parent:
                    parent_world_pose = get_relative_transform(
                        get_prim_at_path(task.root_prim_path + "/" + init_parent), get_prim_at_path(task.root_prim_path)
                    )
                    parent_translation, _ = pose_from_tf_matrix(parent_world_pose)
                    obj.set_local_pose(
                        translation=(parent_translation + init_translation), orientation=init_orientation
                    )
                    obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            except Exception:
                pass

    def _randomization_layout_mem(self):
        # Reset world
        self.world.reset()

        # Individual initialize
        self.task.individual_randomize_from_mem()
        self.task.post_reset()

        self._step_world(render=False)

        # Reset controllers
        self._reset_controllers(self.controllers)

        # Reset skills
        del self.skills
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        # Warmup
        for _ in range(20):
            self.world.get_observations()
            self._init_static_objects(self.task)
            self._step_world(render=False)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def _randomization_layout(self):
        # Reset world
        self.world.reset()

        # Individual initialize
        self.task.individual_randomize()
        self.task.post_reset()

        self._step_world(render=False)

        # Reset controllers
        if self.task_cfg.get("fluid", None):
            # Fluid, Bug, Why !!!!!!
            # For fluid manipulation, only delete controllers and reinitialize controllers can plan successfully
            if hasattr(self, "controllers"):
                del self.controllers
            self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)

        # del self.controllers
        # self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)
        self._reset_controllers(self.controllers)

        # Reset skills
        if hasattr(self, "skills"):
            del self.skills

        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        # Warmup
        for _ in range(20):
            self.world.get_observations()
            self._init_static_objects(self.task)
            self._step_world(render=False)

        if self.task_cfg.get("fluid", None):
            self.task._set_fluid()
            # Fluid need additional warmup
            for _ in range(150):
                self._step_world(render=False)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def randomization(self, layout_path=None) -> bool:
        try:
            if layout_path is None:
                # Individual Reset
                self.task.individual_reset()
                self._randomization_layout()
            else:
                with open(layout_path, "rb") as f:
                    data = pickle.load(f)
                self.data = data
                self.randomization_from_mem(data)
            return True
        except Exception as e:
            raise e

    def update_skill_states(self, skills, episode_success, should_continue):
        """Update and manage skill states."""
        current_skills = skills[0]

        # Check if any skills remain
        if not any(current_skills.values()):
            skills.pop(0)
            if skills:
                should_continue = self.plan_first_skill(skills, should_continue)
            return episode_success, should_continue

        # Update each robot's skills
        for _, skill_sequences in current_skills.items():
            if not skill_sequences:
                continue

            # Update all skills first
            for lr_skill_list in skill_sequences[0]:
                if lr_skill_list:
                    start_lr_skill = lr_skill_list[0]
                    start_lr_skill.update()  # Must update regardless of completion
                    if start_lr_skill.is_done():
                        if not start_lr_skill.is_success():
                            episode_success = False
                            should_continue = False
                        lr_skill_list.remove(start_lr_skill)

                        if lr_skill_list:
                            next_skill = lr_skill_list[0]
                            next_skill.simple_generate_manip_cmds()
                            if hasattr(next_skill, "visualize_target"):
                                next_skill.visualize_target(self.world)
                            if len(next_skill.manip_list) == 0:
                                should_continue = not next_skill.is_ready()
                    if hasattr(start_lr_skill, "visualize_target"):
                        start_lr_skill.visualize_target(self.world)

            # Remove empty skill sequences
            completed_skills = []
            for lr_skill_list in skill_sequences[0]:
                if not lr_skill_list:
                    completed_skills.append(lr_skill_list)
            for completed_skill in completed_skills:
                skill_sequences[0].remove(completed_skill)

            # Move to next sequence if current is empty
            if not skill_sequences[0]:
                skill_sequences.pop(0)
                if skill_sequences:
                    for skill in skill_sequences[0]:
                        skill[0].simple_generate_manip_cmds()
                        if len(skill[0].manip_list) == 0:
                            should_continue = not skill[0].is_ready()
        return episode_success, should_continue

    def plan_first_skill(self, skills, should_continue):
        for _, robot_skill_list in skills[0].items():
            for lr_skill_list in robot_skill_list[0]:
                lr_skill_list[0].simple_generate_manip_cmds()
                if hasattr(lr_skill_list[0], "visualize_target"):
                    lr_skill_list[0].visualize_target(self.world)
                if len(lr_skill_list[0].manip_list) == 0:
                    should_continue = not lr_skill_list[0].is_ready()
        return should_continue

    def _dump_nav2_runtime_debug_snapshots(self, tag: str):
        for robot_name, manager in getattr(self, "_navigation_session_managers", {}).items():
            if manager is None:
                continue
            try:
                manager._write_debug_snapshot(f"workflow_{tag}_snapshot.json", f"workflow_{tag}", f"workflow terminated while nav2 skill was still active for {robot_name}")
            except Exception as exc:
                print(f"[ros-nav2-runtime] Failed to dump debug snapshot for '{robot_name}': {exc}")

    def generate_seq(self) -> list:
        end = False

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        step_id = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        should_continue = self.plan_first_skill(self.skills, should_continue)

        # Warmup
        for _ in range(10):
            obs = self.world.get_observations()
            # self._init_static_objects(self.task)
            self._step_world(render=False)

        while not (step_id >= max_episode_length or (not self.skills and not episode_success) or (not should_continue)):
            obs = self.world.get_observations()
            action_dict = {}
            record_flag = True
            if self.skills and should_continue:
                # Process current skills
                current_skills = self.skills[0]
                for robot_name, skill_sequences in current_skills.items():
                    if skill_sequences and skill_sequences[0]:
                        action = [
                            skill[0].controller.forward(skill[0].manip_list[0])
                            for skill in skill_sequences[0]
                            if skill[0] and skill[0].is_ready()
                        ]

                        feasible_labels = [skill[0].is_feasible() for skill in skill_sequences[0] if skill[0]]
                        record_labels = [skill[0].is_record() for skill in skill_sequences[0] if skill[0]]

                        if False in feasible_labels:
                            should_continue = False
                        if False in record_labels:
                            record_flag = False

                        if action:
                            action_dict[robot_name] = {
                                "joint_positions": np.concatenate([a["joint_positions"] for a in action]),
                                "joint_indices": np.concatenate([a["joint_indices"] for a in action]),
                                "raw_action": action,
                            }
            elif not self.skills and episode_success:
                print("Task is successful")
                end = True
                for j_idx in range(1, 7):
                    self._step_world(render=False)
                    obs = self.world.get_observations()
                    log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id + j_idx)
                    self.world_recorder.record()

                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id)
                self.world_recorder.record()
            self.task.apply_action(action_dict)
            self._step_world(render=False)

            step_id += 1
            if self.skills:
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        if end:
            if self.step_replay:
                return [None] * step_id
            else:
                # Prim poses mode: return recorded poses for compatibility
                return self.world_recorder.prim_poses
        else:
            if step_id >= max_episode_length:
                self._dump_nav2_runtime_debug_snapshots("step_limit")
            elif not should_continue:
                self._dump_nav2_runtime_debug_snapshots("skill_stop")
            return []

    def recover_seq(self, seq_path):
        data = self.data
        return self.recover_seq_from_mem(data)

    def _record_rgb_depth(self, step_idx: int):
        for key, value in self.task.cameras.items():
            for robot_name, _ in self.task.robots.items():
                if robot_name in key:
                    camera_obs = value.get_observations()
                    rgb_img = camera_obs["color_image"]
                    # Special processing if enabled
                    camera2env_pose = camera_obs["camera2env_pose"]
                    save_camera_name = key.replace(f"{robot_name}_", "")
                    self.logger.add_color_image(
                        robot_name, "images.rgb." + save_camera_name, rgb_img, step_idx=step_idx
                    )
                    if "depth_image" in camera_obs:
                        depth_image = camera_obs["depth_image"]
                        depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
                        self.logger.add_depth_image(
                            robot_name, "images.depth." + save_camera_name, depth_img, step_idx=step_idx
                        )
                    if "semantic_mask" in camera_obs:
                        self.logger.add_seg_image(
                            robot_name, "images.seg." + save_camera_name, seg_mask, step_idx=step_idx
                        )
                        if "semantic_mask_id2labels" in camera_obs:
                            self.logger.add_scalar_data(
                                robot_name,
                                "labels.seg." + save_camera_name,
                                camera_obs["semantic_mask_id2labels"],
                            )
                    if "bbox2d_tight" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox2d_tight." + save_camera_name, camera_obs["bbox2d_tight"]
                        )
                    if "bbox2d_tight_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox2d_tight_id2labels." + save_camera_name,
                            camera_obs["bbox2d_tight_id2labels"],
                        )
                    if "bbox2d_loose" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox2d_loose." + save_camera_name, camera_obs["bbox2d_loose"]
                        )
                    if "bbox2d_loose_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox2d_loose_id2labels." + save_camera_name,
                            camera_obs["bbox2d_loose_id2labels"],
                        )
                    if "bbox3d" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox3d." + save_camera_name, camera_obs["bbox3d"]
                        )
                    if "bbox3d_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox3d_id2labels." + save_camera_name,
                            camera_obs["bbox3d_id2labels"],
                        )
                    if self.log_motion_vectors and "motion_vectors" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.motion_vectors." + save_camera_name, camera_obs["motion_vectors"]
                        )
                    self.logger.add_scalar_data(
                        robot_name, "camera2env_pose." + save_camera_name, camera2env_pose
                    )
                    if step_idx == 0:
                        save_camera_name = key.replace(f"{robot_name}_", "")
                        self.logger.add_json_data(
                            robot_name, f"{save_camera_name}_camera_params", camera_obs["camera_params"]
                        )

                    # depth_img = get_src(value, "depth")
                    # depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)

                    # # Initialize lists for new camera keys
                    # if key not in self.rgb:
                    #     self.rgb[key] = []
                    # if key not in self.depth:
                    #     self.depth[key] = []

                    # # Append current frame to the corresponding camera's list
                    # self.rgb[key].append(rgb_img)
                    # self.depth[key].append(depth_img)

    def seq_replay(self, sequence: list) -> int:
        """
        Replay recorded sequence with mode-specific data preparation.

        Returns:
            int: Number of steps replayed
        """
        if not self.step_replay:
            self.world_recorder.prim_poses = sequence

        # warmup before replay formally
        self.world_recorder.warmup()

        # Get total steps from WorldRecorder
        total_steps = self.world_recorder.get_total_steps()
        step_idx = 0

        # Unified replay loop - WorldRecorder handles rendering internally
        with tqdm(total=total_steps, desc="Replay Progress") as pbar:
            while not self.world_recorder.replay():
                # Record RGB/depth at current step
                self._record_rgb_depth(step_idx)
                step_idx += 1
                pbar.update(1)

        self.length = total_steps
        print("Replay finished.")
        return total_steps

    def get_task_name(self):
        return self.task_cfg["task"]

    def save_seq(self, save_path: str) -> int:
        ser_bytes = self.dump_plan_info()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        save_path = os.path.join(save_path, "plan")
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, f"{timestamp}.pkl")
        with open(path, "wb") as f:
            f.write(ser_bytes)
        return self.world_recorder.get_total_steps()

    def save(self, save_path: str) -> int:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        self.logger.save(save_path, timestamp, save_img=True)

        return self.length

    def plan_with_render(self):
        end = False

        step_id = 0
        length = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        should_continue = self.plan_first_skill(self.skills, should_continue)

        # Warmup
        for _ in range(10):
            obs = self.world.get_observations()
            # self._init_static_objects(self.task)
            self._step_world(render=True)

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        while not (step_id >= max_episode_length or (not self.skills and not episode_success) or (not should_continue)):
            obs = self.world.get_observations()
            action_dict = {}
            record_flag = True
            if self.skills and should_continue:
                # Process current skills
                current_skills = self.skills[0]
                for robot_name, skill_sequences in current_skills.items():
                    if skill_sequences and skill_sequences[0]:
                        action = [
                            skill[0].controller.forward(skill[0].manip_list[0])
                            for skill in skill_sequences[0]
                            if skill[0] and skill[0].is_ready()
                        ]

                        feasible_labels = [skill[0].is_feasible() for skill in skill_sequences[0] if skill[0]]
                        record_labels = [skill[0].is_record() for skill in skill_sequences[0] if skill[0]]

                        if False in feasible_labels:
                            should_continue = False
                        if False in record_labels:
                            record_flag = False

                        if action:
                            action_dict[robot_name] = {
                                "joint_positions": np.concatenate([a["joint_positions"] for a in action]),
                                "joint_indices": np.concatenate([a["joint_indices"] for a in action]),
                                "raw_action": action,
                            }
            elif not self.skills and episode_success:
                print("Task is successful")
                end = True
                for j_idx in range(1, 7):
                    self._step_world(render=True)
                    obs = self.world.get_observations()
                    log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id + j_idx)
                    self._record_rgb_depth(step_id + j_idx)
                    self.world_recorder.record()
                length = step_id + 6
                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id)
                self._record_rgb_depth(step_id)
            self.task.apply_action(action_dict)
            self._step_world(render=True)

            step_id += 1
            if self.skills:
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        self.length = length
        if end:
            return length
        else:
            self.length = step_id
            return 0

    def _dump_task_cfg(self, task_cfg):
        task_cfg_copy = deepcopy(task_cfg)
        return pickle.dumps(task_cfg_copy)

    def dump_plan_info(self) -> bytes:
        logger_ser = self.logger.dump()
        cfg_ser = self._dump_task_cfg(self.task_cfg)
        ser = pickle.dumps((cfg_ser, self.world_recorder.dumps(), logger_ser))
        return ser

    def dedump_plan_info(self, ser_obj: bytes) -> object:
        res = pickle.loads(ser_obj)
        return res

    def randomization_from_mem(self, data) -> bool:
        try:
            cfg_ser, _, _ = data
            task_cfg = pickle.loads(cfg_ser)
            self.task_cfg = task_cfg
            self.task.cfg = task_cfg

            # Individual Reset
            self.task.individual_reset_from_mem()
            self._randomization_layout_mem()
            return True
        except Exception as e:
            raise e

    def recover_seq_from_mem(self, data) -> list:
        """
        Recover sequence from memory based on WorldRecorder mode.

        Returns:
            - step_replay=False: Returns prim_poses list
            - step_replay=True: Returns placeholder list (replay data is in WorldRecorder)
        """
        try:
            _, wr_ser, logger_ser = data
            self.logger.dedump(logger_ser)

            if wr_ser:
                self.world_recorder.loads(wr_ser)

            if self.step_replay:
                return [None] * self.world_recorder.num_steps
            else:
                return self.world_recorder.prim_poses

        except Exception as e:
            raise e
