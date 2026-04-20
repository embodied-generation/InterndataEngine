import os
import time
from fractions import Fraction
from pathlib import Path

from nimbus.components.data.iterator import Iterator
from nimbus.components.data.package import Package
from nimbus.components.data.scene import Scene
from nimbus.components.load import SceneLoader
from nimbus.daemon import ComponentStatus, StatusReporter
from nimbus.daemon.decorators import status_monitor
from nimbus.utils.flags import get_random_seed
from workflows.base import create_workflow


def _resolve_headless_experience() -> str:
    candidate_paths = []
    for env_key in ("ISAAC_SIM_ROOT", "ISAAC_SIM_PATH"):
        value = os.environ.get(env_key, "").strip()
        if value:
            root = Path(value)
            candidate_paths.extend(
                [
                    root / "apps/omni.isaac.sim.python.gym.headless.kit",
                    root / "apps/omni.isaac.sim.headless.native.kit",
                ]
            )
    for root in (Path("/isaac-sim"), Path("/workspace/isaac-sim")):
        candidate_paths.extend(
            [
                root / "apps/omni.isaac.sim.python.gym.headless.kit",
                root / "apps/omni.isaac.sim.headless.native.kit",
            ]
        )

    for experience in candidate_paths:
        if experience.is_file():
            return str(experience)

    return "/isaac-sim/apps/omni.isaac.sim.python.gym.headless.kit"


def _resolve_experience(configured_path: str, *, headless: bool) -> str:
    configured_path = str(configured_path or "").strip()
    if configured_path:
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return str(candidate.resolve())
    if headless:
        return _resolve_headless_experience()
    return ""


class EnvLoader(SceneLoader):
    """
    Environment loader that initializes Isaac Sim and loads scenes based on workflow configurations.

    This loader integrates with the workflow system to manage scene loading and task execution.
    It supports two operating modes:
    - Standalone mode (pack_iter=None): Loads tasks directly from workflow configuration
    - Pipeline mode (pack_iter provided): Loads tasks from a package iterator

    It also supports task repetition for data augmentation across different random seeds.

    Args:
        pack_iter (Iterator[Package]): An iterator from the previous component. None for standalone.
        cfg_path (str): Path to the workflow configuration file.
        workflow_type (str): Type of workflow to create (e.g., 'SimBoxDualWorkFlow').
        simulator (dict): Simulator configuration including physics_dt, rendering_dt, headless, etc.
        task_repeat (int): How many times to repeat each task before advancing (-1 means single execution).
        need_preload (bool): Whether to preload assets on scene initialization.
        scene_info (str): Configuration key for scene information in the workflow config.
    """

    def __init__(
        self,
        pack_iter: Iterator[Package],
        cfg_path: str,
        workflow_type: str,
        simulator: dict,
        task_repeat: int = -1,
        need_preload: bool = False,
        scene_info: str = "dining_room_scene_info",
    ):
        init_start_time = time.time()
        super().__init__(pack_iter)

        self.status_reporter = StatusReporter(self.__class__.__name__)
        self.status_reporter.update_status(ComponentStatus.IDLE)
        self.need_preload = need_preload
        self.task_repeat_cnt = task_repeat
        self.task_repeat_idx = 0
        self.workflow_type = workflow_type

        # Parse simulator config
        physics_dt = simulator.get("physics_dt", "1/30")
        rendering_dt = simulator.get("rendering_dt", "1/30")
        if isinstance(physics_dt, str):
            physics_dt = float(Fraction(physics_dt))
        if isinstance(rendering_dt, str):
            rendering_dt = float(Fraction(rendering_dt))

        from isaacsim import SimulationApp

        launch_config = {
            "headless": simulator.get("headless", True),
            "anti_aliasing": simulator.get("anti_aliasing", 3),
            "multi_gpu": simulator.get("multi_gpu", True),
            "renderer": simulator.get("renderer", "RayTracedLighting"),
        }
        experience = _resolve_experience(simulator.get("experience", ""), headless=bool(launch_config["headless"]))
        if launch_config["headless"]:
            if "disable_viewport_updates" in simulator:
                launch_config["disable_viewport_updates"] = bool(simulator.get("disable_viewport_updates"))
            elif not simulator.get("experience", ""):
                launch_config["disable_viewport_updates"] = True
            self.simulation_app = SimulationApp(
                launch_config,
                experience=experience,
            )
        else:
            if experience:
                self.simulation_app = SimulationApp(launch_config, experience=experience)
            else:
                self.simulation_app = SimulationApp(launch_config)

        if workflow_type == "SimBoxDualWorkFlow":
            from nav2.isaac_ros_clock import ensure_isaac_ros2_bridge_ready

            ensure_isaac_ros2_bridge_ready(simulation_app=self.simulation_app)

        self.logger.info(f"simulator params: physics dt={physics_dt}, rendering dt={rendering_dt}")
        from omni.isaac.core import World

        world = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=simulator.get("stage_units_in_meters", 1.0),
        )

        # Import workflow extensions and create workflow
        from workflows import import_extensions

        import_extensions(workflow_type)
        self.workflow = create_workflow(
            workflow_type,
            world,
            cfg_path,
            scene_info=scene_info,
            random_seed=get_random_seed(),
        )
        self.workflow.simulation_app = self.simulation_app

        self.scene = None
        self.task_finish = False
        self.cur_index = 0
        self.record_init_time(time.time() - init_start_time)

        self.status_reporter.update_status(ComponentStatus.READY)

    @status_monitor()
    def _init_next_task(self):
        """
        Internal helper method to initialize and return the next task as a Scene object.

        Handles task repetition logic and advances the task index when all repetitions are complete.

        Returns:
            Scene: Initialized scene object for the next task.

        Raises:
            StopIteration: When all tasks have been exhausted.
        """
        if self.scene is not None and self.task_repeat_cnt > 0 and self.task_repeat_idx < self.task_repeat_cnt:
            self.logger.info(f"Task execute times {self.task_repeat_idx + 1}/{self.task_repeat_cnt}")
            self.workflow.init_task(self.cur_index - 1, self.need_preload)
            self.task_repeat_idx += 1
            scene = Scene(
                name=self.workflow.get_task_name(),
                wf=self.workflow,
                task_id=self.cur_index - 1,
                task_exec_num=self.task_repeat_idx,
                simulation_app=self.simulation_app,
            )
            return scene
        if self.cur_index >= len(self.workflow.task_cfgs):
            self.logger.info("No more tasks to load, stopping iteration.")
            raise StopIteration
        self.logger.info(f"Loading task {self.cur_index + 1}/{len(self.workflow.task_cfgs)}")
        self.workflow.init_task(self.cur_index, self.need_preload)
        self.task_repeat_idx = 1
        scene = Scene(
            name=self.workflow.get_task_name(),
            wf=self.workflow,
            task_id=self.cur_index,
            task_exec_num=self.task_repeat_idx,
            simulation_app=self.simulation_app,
        )
        self.cur_index += 1
        return scene

    def load_asset(self) -> Scene:
        """
        Load and initialize the next scene from workflow.

        Supports two modes:
        - Standalone: Iterates through workflow tasks directly
        - Pipeline: Synchronizes with incoming packages and applies plan info to scene

        Returns:
            Scene: The loaded and initialized Scene object.

        Raises:
            StopIteration: When no more scenes are available.
        """
        try:
            # Standalone mode: load tasks directly from workflow
            if self.pack_iter is None:
                self.scene = self._init_next_task()
            # Pipeline mode: load tasks from package iterator
            else:
                package = next(self.pack_iter)
                self.cur_index = package.task_id

                # Initialize scene if this is the first package or a new task
                if self.scene is None:
                    self.scene = self._init_next_task()
                elif self.cur_index > self.scene.task_id:
                    self.scene = self._init_next_task()

                # Apply plan information from package to scene
                package.data = self.scene.wf.dedump_plan_info(package.data)
                self.scene.add_plan_info(package.data)

            return self.scene
        except StopIteration:
            raise StopIteration
        except Exception as e:
            raise e
