#!/usr/bin/env python3
"""Visualize a Nav2 navigation sample with a correctly aligned static map.

This script overlays:
- exported Nav2 static map
- planned path
- actual trajectory
- final pose footprint

It fixes a common pitfall in previous ad-hoc visualizations:
`map.pgm` row 0 corresponds to world max-y, so the image must be flipped
vertically before plotting with `origin="lower"`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
import yaml


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_snapshot_path(sample_dir: Path) -> Path:
    for filename in ("failure_snapshot.json", "success_snapshot.json", "shutdown_snapshot.json"):
        path = sample_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No supported snapshot file found in {sample_dir}. "
        "Expected one of failure_snapshot.json, success_snapshot.json, shutdown_snapshot.json."
    )


def _resolve_map_yaml_path(snapshot: dict, repo_root: Path) -> Path:
    map_info = dict(snapshot.get("map_info", {}) or {})
    yaml_path = str(map_info.get("yaml_path", "")).strip()
    if not yaml_path:
        raise FileNotFoundError("map_info.yaml_path is missing in failure_snapshot.json")
    path = Path(yaml_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_footprint_points(snapshot: dict, repo_root: Path) -> np.ndarray:
    params_path = str(snapshot.get("params_path", "")).strip()
    if not params_path:
        raise FileNotFoundError("params_path is missing in failure_snapshot.json")
    path = Path(params_path)
    if not path.is_absolute():
        path = repo_root / path
    params = _load_yaml(path)
    footprint_raw = (
        params["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"]
    )
    return np.asarray(json.loads(footprint_raw), dtype=float)


def _rotated_footprint(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    return np.stack(
        [
            x + cos_yaw * points[:, 0] - sin_yaw * points[:, 1],
            y + sin_yaw * points[:, 0] + cos_yaw * points[:, 1],
        ],
        axis=1,
    )


def _nearest_obstacle_center(
    map_img: np.ndarray,
    *,
    resolution: float,
    origin_x: float,
    origin_y: float,
    pose_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
    height, _ = map_img.shape
    occupied = np.argwhere(map_img == 0)
    if occupied.size == 0:
        return np.asarray([math.nan, math.nan], dtype=float), math.inf
    xs = origin_x + (occupied[:, 1] + 0.5) * resolution
    ys = origin_y + ((height - 1 - occupied[:, 0]) + 0.5) * resolution
    centers = np.stack([xs, ys], axis=1)
    distances = np.hypot(centers[:, 0] - pose_xy[0], centers[:, 1] - pose_xy[1])
    index = int(np.argmin(distances))
    return centers[index], float(distances[index])


def _path_track_error(planned_xy: np.ndarray, pose_xy: np.ndarray) -> tuple[np.ndarray, float]:
    distances = np.hypot(planned_xy[:, 0] - pose_xy[0], planned_xy[:, 1] - pose_xy[1])
    index = int(np.argmin(distances))
    return planned_xy[index], float(distances[index])


def _default_output_path(sample_dir: Path, repo_root: Path) -> Path:
    return repo_root / "output" / "analysis" / f"{sample_dir.name}_failure_diagnostic.png"


def visualize_failure(sample_dir: Path, output_path: Path):
    repo_root = Path.cwd()
    snapshot_path = _resolve_snapshot_path(sample_dir)
    failure_snapshot = _load_json(snapshot_path)
    planned_path = _load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load_json(sample_dir / "actual_trajectory.json")

    map_yaml_path = _resolve_map_yaml_path(failure_snapshot, repo_root)
    map_yaml = _load_yaml(map_yaml_path)
    map_img = np.asarray(Image.open(map_yaml_path.parent / map_yaml["image"]))
    map_img_plot = np.flipud(map_img)

    resolution = float(map_yaml["resolution"])
    origin_x = float(map_yaml["origin"][0])
    origin_y = float(map_yaml["origin"][1])
    height, width = map_img.shape
    extent = [origin_x, origin_x + width * resolution, origin_y, origin_y + height * resolution]

    planned_xy = np.asarray(
        [(float(p["x"]), float(p["y"])) for p in planned_path["path"]["poses"]],
        dtype=float,
    )
    actual_xy = np.asarray(
        [(float(p["x"]), float(p["y"])) for p in actual_trajectory],
        dtype=float,
    )

    start_xy = actual_xy[0]
    goal_xy = np.asarray(
        [
            float(failure_snapshot["goal"]["x"]),
            float(failure_snapshot["goal"]["y"]),
        ],
        dtype=float,
    )
    failure_xy = actual_xy[-1]
    failure_yaw = float(actual_trajectory[-1]["yaw"])

    footprint_points = _load_footprint_points(failure_snapshot, repo_root)
    failure_footprint = _rotated_footprint(
        footprint_points,
        x=float(failure_xy[0]),
        y=float(failure_xy[1]),
        yaw=failure_yaw,
    )

    nearest_plan_xy, track_error = _path_track_error(planned_xy, failure_xy)
    nearest_obstacle_xy, obstacle_distance = _nearest_obstacle_center(
        map_img,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        pose_xy=failure_xy,
    )
    remaining_distance = float(
        math.hypot(goal_xy[0] - failure_xy[0], goal_xy[1] - failure_xy[1])
    )
    yaw_error = float(failure_snapshot.get("yaw_err", math.nan))

    figure, (ax_global, ax_zoom) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    free_space = np.where(map_img_plot == 0, 0.0, 1.0)

    for axis in (ax_global, ax_zoom):
        axis.imshow(
            free_space,
            cmap="gray",
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        axis.plot(planned_xy[:, 0], planned_xy[:, 1], color="#1565C0", linewidth=2.0, label="planned path")
        axis.plot(actual_xy[:, 0], actual_xy[:, 1], color="#D32F2F", linewidth=2.0, label="actual trajectory")
        axis.scatter([start_xy[0]], [start_xy[1]], c="#2E7D32", s=45, marker="o", label="start")
        axis.scatter([goal_xy[0]], [goal_xy[1]], c="#6A1B9A", s=60, marker="*", label="goal")
        axis.scatter([failure_xy[0]], [failure_xy[1]], c="#EF6C00", s=55, marker="x", label="failure pose")
        axis.add_patch(
            Polygon(failure_footprint, closed=True, fill=False, edgecolor="#EF6C00", linewidth=2.0)
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)

    ax_global.set_title("Global Overlay")
    ax_global.legend(loc="upper left", fontsize=8)

    zoom_padding_x = 2.2
    zoom_padding_y = 1.4
    ax_zoom.set_xlim(float(failure_xy[0] - zoom_padding_x), float(failure_xy[0] + zoom_padding_x))
    ax_zoom.set_ylim(float(failure_xy[1] - zoom_padding_y), float(failure_xy[1] + zoom_padding_y))
    snapshot_reason = str(failure_snapshot.get("reason", "")).strip() or snapshot_path.stem
    ax_zoom.set_title("Final Pose Zoom")

    ax_zoom.scatter([nearest_plan_xy[0]], [nearest_plan_xy[1]], c="#1565C0", s=32, marker="o")
    if math.isfinite(obstacle_distance):
        ax_zoom.scatter([nearest_obstacle_xy[0]], [nearest_obstacle_xy[1]], c="black", s=28, marker="s")
        ax_zoom.annotate(
            f"nearest obstacle center = {obstacle_distance:.2f} m",
            xy=(nearest_obstacle_xy[0], nearest_obstacle_xy[1]),
            xytext=(failure_xy[0] + 0.55, failure_xy[1] - 0.15),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
            fontsize=9,
        )
    ax_zoom.annotate(
        f"track error = {track_error:.2f} m",
        xy=(failure_xy[0], failure_xy[1]),
        xytext=(failure_xy[0] + 0.45, failure_xy[1] + 0.55),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9,
    )

    figure.suptitle(
        "Nav2 sample diagnostic"
        f"\nsample = {sample_dir.name}, snapshot = {snapshot_reason}, "
        f"remaining_dist = {remaining_distance:.2f} m, yaw_err = {yaw_error:.2f} rad",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Visualize one Nav2 failure sample.")
    parser.add_argument(
        "sample_dir",
        type=Path,
        help="Path to a failure sample directory containing failure_snapshot.json, planned_path.json and actual_trajectory.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to output/analysis/<sample_name>_failure_diagnostic.png",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    if not sample_dir.is_dir():
        raise NotADirectoryError(f"Sample directory does not exist: {sample_dir}")

    output_path = args.output.resolve() if args.output is not None else _default_output_path(sample_dir, Path.cwd())
    visualize_failure(sample_dir=sample_dir, output_path=output_path)
    print(output_path)


if __name__ == "__main__":
    main()
