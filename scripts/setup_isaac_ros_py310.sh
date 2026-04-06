#!/usr/bin/env bash
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/mnt/exdisk1/isaac-sim-4.1}"
ISAAC_ROS_HUMBLE_ROOT="${ISAAC_SIM_ROOT}/exts/omni.isaac.ros2_bridge/humble"

if [[ ! -d "${ISAAC_ROS_HUMBLE_ROOT}" ]]; then
  echo "[setup_isaac_ros_py310] ERROR: missing ${ISAAC_ROS_HUMBLE_ROOT}" >&2
  return 1 2>/dev/null || exit 1
fi

_strip_opt_ros_paths() {
  local input="${1:-}"
  local out=""
  local IFS=':'
  read -r -a parts <<< "${input}"
  for p in "${parts[@]}"; do
    [[ -z "${p}" ]] && continue
    [[ "${p}" == /opt/ros/* ]] && continue
    if [[ -z "${out}" ]]; then
      out="${p}"
    else
      out="${out}:${p}"
    fi
  done
  echo "${out}"
}

# Prevent Python 3.12 Jazzy packages from polluting Isaac Python 3.10 runtime.
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH || true

CLEAN_PYTHONPATH="$(_strip_opt_ros_paths "${PYTHONPATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(_strip_opt_ros_paths "${LD_LIBRARY_PATH:-}")"

if [[ -n "${CLEAN_PYTHONPATH}" ]]; then
  export PYTHONPATH="${ISAAC_ROS_HUMBLE_ROOT}/rclpy:${CLEAN_PYTHONPATH}"
else
  export PYTHONPATH="${ISAAC_ROS_HUMBLE_ROOT}/rclpy"
fi

if [[ -n "${CLEAN_LD_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${ISAAC_ROS_HUMBLE_ROOT}/lib:${CLEAN_LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${ISAAC_ROS_HUMBLE_ROOT}/lib"
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

echo "[setup_isaac_ros_py310] Activated Isaac ROS env"
echo "  ISAAC_ROS_HUMBLE_ROOT=${ISAAC_ROS_HUMBLE_ROOT}"
echo "  PYTHONPATH prefix=${ISAAC_ROS_HUMBLE_ROOT}/rclpy"
echo "  LD_LIBRARY_PATH prefix=${ISAAC_ROS_HUMBLE_ROOT}/lib"
