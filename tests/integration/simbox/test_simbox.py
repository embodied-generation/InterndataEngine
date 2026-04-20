"""
simbox tests that run in Isaac Sim environment using subprocess wrappers.
Migrated from original test files to use the new IntegrationTestHarness framework.
"""

import os
import sys

import pytest

from tests.integration.base.test_harness import IntegrationTestHarness

# Add path for proper imports
sys.path.append("./")
sys.path.append("./data_engine")
sys.path.append("./tests/integration")

ISAAC_TEST_PYTHON = os.environ.get("ISAAC_SIM_PYTHON", sys.executable)


def test_simbox_pipeline():
    """
    Test simbox pipeline by running it in Isaac Sim subprocess.
    This test uses a subprocess wrapper to handle Isaac Sim process separation.
    """
    harness = IntegrationTestHarness(config_path="configs/simbox/de_pipe_template.yaml", seed=42, random_num=1)

    # Run in subprocess using Isaac Sim Python interpreter
    result = harness.run_data_engine_subprocess(
        runner_script="tests/integration/simbox/runners/simbox_pipeline_runner.py",
        interpreter=ISAAC_TEST_PYTHON,
        timeout=1800,  # 30 minutes
    )

    # Verify subprocess completed successfully
    assert result.returncode == 0, f"simbox pipeline test failed with return code: {result.returncode}"

    # Validate that output was generated
    # harness.validate_output_generated(min_files=6)

    print("✓ simbox pipeline test completed successfully")


def test_simbox_plan():
    """
    Test simbox plan generation by running it in Isaac Sim subprocess.
    """
    harness = IntegrationTestHarness(
        config_path="configs/simbox/de_plan_template.yaml",
        seed=42,
        random_num=1,
    )

    # Run in subprocess using Isaac Sim Python interpreter
    result = harness.run_data_engine_subprocess(
        runner_script="tests/integration/simbox/runners/simbox_plan_runner.py",
        interpreter=ISAAC_TEST_PYTHON,
        timeout=1800,  # 30 minutes
    )

    # Verify subprocess completed successfully
    assert result.returncode == 0, f"simbox plan test failed with return code: {result.returncode}"

    # Validate that output was generated
    # harness.validate_output_generated(min_files=1)

    print("✓ simbox plan test completed successfully")


def test_simbox_render():
    """
    Test simbox render by running it in Isaac Sim subprocess.
    """
    harness = IntegrationTestHarness(
        config_path="configs/simbox/de_render_template.yaml",
        # config_path="tests/integration/simbox/configs/simbox_test_render_configs.yaml",
        seed=42,
    )

    reference_dir = "/shared/smartbot_new/zhangyuchang/CI/manip/simbox/simbox_render_ci"
    # Run in subprocess using Isaac Sim Python interpreter
    result = harness.run_data_engine_subprocess(
        runner_script="tests/integration/simbox/runners/simbox_render_runner.py",
        interpreter=ISAAC_TEST_PYTHON,
        timeout=1800,  # 30 minutes
        compare_output=True,
        reference_dir=reference_dir,
        comparator="simbox",
        comparator_args={
            # Use defaults; override here if needed
            # "tolerance": 1e-6,
            # "image_psnr": 37.0,
        },
    )

    # Verify subprocess completed successfully
    assert result.returncode == 0, f"simbox render test failed with return code: {result.returncode}"

    # Validate that output was generated
    # harness.validate_output_generated(min_files=1)

    print("✓ simbox render test completed successfully")


@pytest.mark.skipif(
    os.environ.get("RUN_ROS_BASE_BRIDGE_SMOKE_TEST") != "1",
    reason="Set RUN_ROS_BASE_BRIDGE_SMOKE_TEST=1 to enable ROS base bridge smoke test",
)
def test_simbox_split_aloha_ros_bridge_smoke():
    """Smoke test for SplitAloha ROS base bridge in Isaac Sim subprocess."""
    harness = IntegrationTestHarness(
        config_path="configs/simbox/de_render_template.yaml",
        seed=42,
    )

    result = harness.run_data_engine_subprocess(
        runner_script="tests/integration/simbox/runners/simbox_split_aloha_ros_bridge_smoke_runner.py",
        interpreter=ISAAC_TEST_PYTHON,
        timeout=1800,
    )

    assert result.returncode == 0, f"simbox split_aloha ros bridge smoke test failed: {result.returncode}"

    print("✓ simbox split_aloha ros bridge smoke test completed successfully")


@pytest.mark.skipif(
    os.environ.get("RUN_ROS_NAV2_SMOKE_TEST") != "1",
    reason="Set RUN_ROS_NAV2_SMOKE_TEST=1 to enable ROS Nav2 detour smoke test",
)
def test_simbox_split_aloha_nav2_detour_smoke():
    """Smoke test for SplitAloha ROS Nav2 detour flow in Isaac Sim subprocess."""
    harness = IntegrationTestHarness(
        config_path="configs/simbox/de_render_template.yaml",
        seed=42,
    )

    result = harness.run_data_engine_subprocess(
        runner_script="tests/integration/simbox/runners/simbox_split_aloha_nav2_detour_runner.py",
        interpreter=ISAAC_TEST_PYTHON,
        timeout=1800,
    )

    assert result.returncode == 0, f"simbox split_aloha nav2 detour smoke test failed: {result.returncode}"

    print("✓ simbox split_aloha nav2 detour smoke test completed successfully")


if __name__ == "__main__":
    # Run all tests when script is executed directly
    test_simbox_plan()
    test_simbox_render()
    test_simbox_pipeline()
