#!/usr/bin/env python3
"""ROS-side Ranger driver that converts /cmd_vel into bridge driver topics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import signal
import sys
import time

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.simbox.core.mobile.controllers import RangerMiniV3Controller  # pylint: disable=wrong-import-position


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_base_cfg(base_config_path: str) -> dict:
    resolved_path = _resolve_path(base_config_path)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle) or {}
    if not isinstance(base_cfg, dict):
        raise TypeError(f"base config at {resolved_path} must be a YAML mapping")
    return deepcopy(base_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish Ranger driver topics from /cmd_vel")
    parser.add_argument(
        "--base-config",
        default="workflows/simbox/core/configs/bases/ranger_mini_v3.yaml",
        help="Path to the merged or base-only mobile config YAML",
    )
    parser.add_argument("--node-name", default="interndata_ranger_driver")
    parser.add_argument("--loop-interval", type=float, default=0.01)
    args = parser.parse_args()

    base_cfg = _load_base_cfg(args.base_config)
    ros_cfg = base_cfg.setdefault("ros", {})
    ros_cfg["publish_driver_topics"] = True
    ros_cfg["internal_cmdvel_controller_enabled"] = False

    controller = RangerMiniV3Controller(base_cfg, node_name=args.node_name)
    running = True

    def _request_stop(signum, _frame):
        del signum
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print(
        "[ranger-driver] listening on "
        f"{ros_cfg.get('cmd_vel_topic', '/cmd_vel')} -> "
        f"{ros_cfg.get('motion_mode_topic', '/ranger/sim/motion_mode')}, "
        f"{ros_cfg.get('command_topic', '/ranger/sim/base_command')}",
        flush=True,
    )
    try:
        while running:
            controller.step()
            time.sleep(max(float(args.loop_interval), 0.001))
    finally:
        controller.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
