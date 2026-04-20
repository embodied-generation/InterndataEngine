"""Unit tests for file-based external Nav2 protocol helpers."""

from nav2.protocol import (
    DEFAULT_EXTERNAL_NAV2_GOAL_REQUEST_ROOT,
    DEFAULT_EXTERNAL_NAV2_GOAL_RESULT_ROOT,
    DEFAULT_EXTERNAL_NAV2_GOAL_STATUS_ROOT,
    DEFAULT_EXTERNAL_NAV2_STACK_REQUEST_ROOT,
    DEFAULT_EXTERNAL_NAV2_STACK_STATUS_ROOT,
    atomic_write_yaml,
    nav2_protocol_roots,
    safe_name,
    status_summary,
)


def test_nav2_protocol_roots_defaults():
    roots = nav2_protocol_roots({})

    assert roots["stack_request_root"] == DEFAULT_EXTERNAL_NAV2_STACK_REQUEST_ROOT
    assert roots["stack_status_root"] == DEFAULT_EXTERNAL_NAV2_STACK_STATUS_ROOT
    assert roots["goal_request_root"] == DEFAULT_EXTERNAL_NAV2_GOAL_REQUEST_ROOT
    assert roots["goal_status_root"] == DEFAULT_EXTERNAL_NAV2_GOAL_STATUS_ROOT
    assert roots["goal_result_root"] == DEFAULT_EXTERNAL_NAV2_GOAL_RESULT_ROOT


def test_nav2_protocol_roots_honor_config_override():
    roots = nav2_protocol_roots(
        {
            "ros": {
                "nav2": {
                    "stack_request_root": "stack/req",
                    "stack_status_root": "stack/status",
                    "goal_request_root": "goal/req",
                    "goal_status_root": "goal/status",
                    "goal_result_root": "goal/result",
                }
            }
        }
    )

    assert roots == {
        "stack_request_root": "stack/req",
        "stack_status_root": "stack/status",
        "goal_request_root": "goal/req",
        "goal_status_root": "goal/status",
        "goal_result_root": "goal/result",
    }


def test_safe_name_and_status_summary(tmp_path):
    status_path = tmp_path / "status.yaml"
    atomic_write_yaml(
        str(status_path),
        {
            "state": "ready",
            "stack_id": "robot::scene::abc",
            "goal_id": "goal-1",
            "detail": "action server ready",
            "updated_at": "2026-04-18T12:00:00",
        },
    )

    assert safe_name("robot/name 1") == "robot_name_1"
    summary = status_summary(str(status_path))
    assert "state=ready" in summary
    assert "stack_id=robot::scene::abc" in summary
    assert "goal_id=goal-1" in summary
