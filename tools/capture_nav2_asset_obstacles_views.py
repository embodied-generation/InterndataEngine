from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
import yaml
from isaacsim import SimulationApp


TASK_CFG_PATH = "workflows/simbox/core/configs/tasks/navigation/split_aloha/nav2_asset_obstacles.yaml"
OUTPUT_ROOT = "output/ros_bridge/nav2_asset_obstacles_capture"


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_rgb(path: str, image: np.ndarray):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def main():
    simulation_app = SimulationApp(
        {
            "headless": True,
            "anti_aliasing": 0,
            "multi_gpu": True,
            "renderer": "RayTracedLighting",
        }
    )

    try:
        from omni.isaac.core import World
        from nimbus.utils.utils import init_env
        from workflows import import_extensions
        from workflows.base import create_workflow

        init_env()
        import_extensions("SimBoxDualWorkFlow")

        with open(TASK_CFG_PATH, "r", encoding="utf-8") as f:
            task_cfg = yaml.safe_load(f)["tasks"][0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_ROOT, timestamp)
        _ensure_dir(output_dir)

        world = World(physics_dt=1 / 30, rendering_dt=1 / 30, stage_units_in_meters=1.0)
        workflow = create_workflow("SimBoxDualWorkFlow", world, TASK_CFG_PATH, scene_info="dining_room_scene_info")
        workflow.init_task(0, need_preload=False)

        for _ in range(20):
            workflow._step_world(render=True)  # pylint: disable=protected-access

        obs = workflow.task.get_observations()
        for camera_name in ("nav2_global", "split_aloha_head", "split_aloha_hand_left", "split_aloha_hand_right"):
            camera_obs = obs["cameras"][camera_name]
            _save_rgb(os.path.join(output_dir, f"{camera_name}.png"), np.asarray(camera_obs["color_image"]))

        global_camera = workflow.task.cameras["nav2_global"]
        for obj_cfg in task_cfg["objects"]:
            name = obj_cfg["name"]
            tx, ty, _ = obj_cfg["translation"]
            global_camera.set_local_pose(
                translation=np.asarray([float(tx), float(ty), 5.0], dtype=np.float32),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                camera_axes="usd",
            )
            for _ in range(4):
                workflow._step_world(render=True)  # pylint: disable=protected-access
            obstacle_obs = workflow.task.get_observations()["cameras"]["nav2_global"]
            _save_rgb(os.path.join(output_dir, f"{name}.png"), np.asarray(obstacle_obs["color_image"]))

        summary = {
            "task_cfg_path": TASK_CFG_PATH,
            "output_dir": output_dir,
            "cameras": ["nav2_global", "split_aloha_head", "split_aloha_hand_left", "split_aloha_hand_right"],
            "obstacles": [obj["name"] for obj in task_cfg["objects"]],
        }
        with open(os.path.join(output_dir, "summary.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

        print(output_dir)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
