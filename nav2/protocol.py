"""Shared file-based protocol helpers for external Nav2 coordination."""

from __future__ import annotations

import os
from typing import Any

import yaml

DEFAULT_EXTERNAL_NAV2_STACK_REQUEST_ROOT = "output/ros_bridge/runtime_requests"
DEFAULT_EXTERNAL_NAV2_STACK_STATUS_ROOT = "output/ros_bridge/runtime_status"
DEFAULT_EXTERNAL_NAV2_GOAL_REQUEST_ROOT = "output/ros_bridge/goal_requests"
DEFAULT_EXTERNAL_NAV2_GOAL_STATUS_ROOT = "output/ros_bridge/goal_status"
DEFAULT_EXTERNAL_NAV2_GOAL_RESULT_ROOT = "output/ros_bridge/goal_result"


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value).strip())
    return cleaned or "robot"


def atomic_write_yaml(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    os.replace(tmp_path, path)


def read_yaml_file(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def remove_file_if_exists(path: str):
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def nav2_protocol_roots(base_cfg: dict | None = None) -> dict[str, str]:
    base_cfg = base_cfg or {}
    ros_cfg = dict(base_cfg.get("ros", {}))
    nav2_cfg = dict(ros_cfg.get("nav2", {}))

    def _value(cfg_key: str, env_key: str, default: str) -> str:
        configured = str(nav2_cfg.get(cfg_key, "")).strip()
        if configured:
            return configured
        return str(os.environ.get(env_key, default)).strip()

    return {
        "stack_request_root": _value(
            "stack_request_root",
            "INTERNDATA_EXTERNAL_NAV2_REQUEST_ROOT",
            DEFAULT_EXTERNAL_NAV2_STACK_REQUEST_ROOT,
        ),
        "stack_status_root": _value(
            "stack_status_root",
            "INTERNDATA_EXTERNAL_NAV2_STATUS_ROOT",
            DEFAULT_EXTERNAL_NAV2_STACK_STATUS_ROOT,
        ),
        "goal_request_root": _value(
            "goal_request_root",
            "INTERNDATA_EXTERNAL_NAV2_GOAL_REQUEST_ROOT",
            DEFAULT_EXTERNAL_NAV2_GOAL_REQUEST_ROOT,
        ),
        "goal_status_root": _value(
            "goal_status_root",
            "INTERNDATA_EXTERNAL_NAV2_GOAL_STATUS_ROOT",
            DEFAULT_EXTERNAL_NAV2_GOAL_STATUS_ROOT,
        ),
        "goal_result_root": _value(
            "goal_result_root",
            "INTERNDATA_EXTERNAL_NAV2_GOAL_RESULT_ROOT",
            DEFAULT_EXTERNAL_NAV2_GOAL_RESULT_ROOT,
        ),
    }


def status_summary(path: str) -> str:
    status = read_yaml_file(path)
    if not status:
        return ""
    state = str(status.get("state", "")).strip()
    detail = str(status.get("detail", "")).strip()
    stack_id = str(status.get("stack_id", "")).strip()
    goal_id = str(status.get("goal_id", "")).strip()
    updated_at = str(status.get("updated_at", "")).strip()
    parts = [
        part
        for part in (
            f"state={state}" if state else "",
            f"stack_id={stack_id}" if stack_id else "",
            f"goal_id={goal_id}" if goal_id else "",
            f"updated_at={updated_at}" if updated_at else "",
        )
        if part
    ]
    if detail:
        parts.append(f"detail={detail}")
    return ", ".join(parts)
