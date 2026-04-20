#!/usr/bin/env python3
"""Generate a resident Nav2 bringup config for the split Isaac/Nav2 deployment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from nav2.runtime import (
    NAV2_DEFAULT_POSITION_TOLERANCE_M,
    NAV2_DEFAULT_YAW_TOLERANCE_RAD,
    configure_base_cfg_for_nav2_skill,
    generate_nav2_bringup_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a dict: {path}")
    return data


def _deep_update_dict(base: dict, override: dict):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update_dict(base[key], value)
        else:
            base[key] = value


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_robot_base_cfg(robot_config_path: Path) -> dict:
    robot_cfg = _load_yaml(robot_config_path)
    base_cfg = deepcopy(robot_cfg.get("base", {}))
    if not isinstance(base_cfg, dict):
        raise TypeError(f"robot config base must be a dict: {robot_config_path}")

    merged_base_cfg: dict = {}
    base_config_file = str(base_cfg.get("base_config_file", "")).strip()
    nav_config_file = str(base_cfg.get("nav_config_file", "")).strip()

    if base_config_file:
        _deep_update_dict(merged_base_cfg, _load_yaml(_resolve_repo_path(base_config_file)))
    if nav_config_file:
        _deep_update_dict(merged_base_cfg, _load_yaml(_resolve_repo_path(nav_config_file)))

    _deep_update_dict(merged_base_cfg, base_cfg)
    return merged_base_cfg


def _write_bootstrap_map(map_dir: Path) -> Path:
    map_dir.mkdir(parents=True, exist_ok=True)
    pgm_path = map_dir / "map.pgm"
    yaml_path = map_dir / "map.yaml"

    if not pgm_path.exists():
        width = 64
        height = 64
        free_value = 254
        with open(pgm_path, "wb") as handle:
            handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
            handle.write(bytes([free_value]) * width * height)

    yaml_payload = "\n".join(
        [
            f"image: {pgm_path.name}",
            "mode: trinary",
            "resolution: 0.050000",
            "origin: [-1.600000, -1.600000, 0.0]",
            "negate: 0",
            "occupied_thresh: 0.65",
            "free_thresh: 0.25",
            "",
        ]
    )
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(yaml_payload)
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-config",
        default="workflows/simbox/core/configs/robots/split_aloha.yaml",
        help="Robot config that contains base_config_file/nav_config_file",
    )
    parser.add_argument(
        "--output-dir",
        default="output/nav2_runtime/bootstrap",
        help="Directory to write resident Nav2 params and BT files into",
    )
    parser.add_argument(
        "--map-dir",
        default="output/nav2_runtime/bootstrap_map",
        help="Directory to place the bootstrap map used before first load_map",
    )
    parser.add_argument(
        "--position-tolerance-m",
        type=float,
        default=NAV2_DEFAULT_POSITION_TOLERANCE_M,
    )
    parser.add_argument(
        "--yaw-tolerance-rad",
        type=float,
        default=NAV2_DEFAULT_YAW_TOLERANCE_RAD,
    )
    args = parser.parse_args()

    robot_config_path = _resolve_repo_path(args.robot_config)
    output_dir = _resolve_repo_path(args.output_dir)
    map_dir = _resolve_repo_path(args.map_dir)

    base_cfg = configure_base_cfg_for_nav2_skill(_load_robot_base_cfg(robot_config_path))
    bootstrap_map_yaml = _write_bootstrap_map(map_dir)
    base_cfg.setdefault("ros", {}).setdefault("localization", {})["map_yaml_path"] = str(bootstrap_map_yaml)

    artifacts = generate_nav2_bringup_artifacts(
        str(output_dir),
        base_cfg=base_cfg,
        map_yaml_path=str(bootstrap_map_yaml),
        position_tolerance_m=float(args.position_tolerance_m),
        yaw_tolerance_rad=float(args.yaw_tolerance_rad),
        params_filename="nav2_params.yaml",
    )

    print(artifacts["params_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
