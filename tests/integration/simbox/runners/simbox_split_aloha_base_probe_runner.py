"""Isaac Sim runner that probes the SplitAloha real base joint interface."""

import argparse
import json
import os
import sys

import yaml
from isaacsim import SimulationApp

_runner_args = sys.argv[1:]
sys.argv = [sys.argv[0]]
simulation_app = SimulationApp({"headless": True})
sys.argv = [sys.argv[0], *_runner_args]

sys.path.append("./")
sys.path.append("./data_engine")
sys.path.append("workflows/simbox")

from omni.isaac.core import World  # pylint: disable=wrong-import-position

from nimbus.utils.utils import init_env  # pylint: disable=wrong-import-position
from workflows import import_extensions  # pylint: disable=wrong-import-position
from workflows.base import create_workflow  # pylint: disable=wrong-import-position


def _load_render_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_world(simulator_cfg: dict):
    return World(
        physics_dt=eval(str(simulator_cfg["physics_dt"])),
        rendering_dt=eval(str(simulator_cfg["rendering_dt"])),
        stage_units_in_meters=float(simulator_cfg["stage_units_in_meters"]),
    )


def _find_split_aloha(workflow):
    for robot in workflow.task.robots.values():
        if robot.__class__.__name__ == "SplitAloha":
            return robot
    raise RuntimeError("SplitAloha robot not found in workflow task")


def run_probe(config_path: str, output_path: str):
    try:
        init_env()
        config = _load_render_config(config_path)
        scene_loader_cfg = config["load_stage"]["scene_loader"]["args"]
        workflow_type = scene_loader_cfg["workflow_type"]
        task_cfg_path = scene_loader_cfg["cfg_path"]
        simulator_cfg = scene_loader_cfg["simulator"]

        import_extensions(workflow_type)
        world = _build_world(simulator_cfg)
        workflow = create_workflow(workflow_type, world, task_cfg_path)
        workflow.init_task(0)

        robot = _find_split_aloha(workflow)
        dof_names = list(robot._articulation_view.dof_names)
        base_interface = robot.get_base_interface()

        report = {
            "robot_name": robot.name,
            "dof_names": dof_names,
            "base_interface": base_interface,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print(json.dumps(report, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="configs/simbox/de_render_template.yaml",
        help="Path to the SimBox render config used to load the task",
    )
    parser.add_argument(
        "--output-path",
        default="output/ros_bridge/split_aloha_base_probe.json",
        help="Where to save the probe output JSON",
    )
    args = parser.parse_args()
    run_probe(config_path=args.config_path, output_path=args.output_path)
