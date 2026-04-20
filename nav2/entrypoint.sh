#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
export PYTHONPATH="/workspace/InterndataEngine:${PYTHONPATH}"

cd /workspace/InterndataEngine

pids=()

if [ "${INTERNDATA_AUTOSTART_ROS_STACK:-1}" = "1" ]; then
  NAV2_STACK_CONFIG_DIR="${INTERNDATA_NAV2_STACK_CONFIG_DIR:-output/nav2_runtime/bootstrap}"
  NAV2_BOOTSTRAP_MAP_DIR="${INTERNDATA_NAV2_BOOTSTRAP_MAP_DIR:-output/nav2_runtime/bootstrap_map}"
  NAV2_PARAMS_FILE="/workspace/InterndataEngine/${NAV2_STACK_CONFIG_DIR}/nav2_params.yaml"

  python3 -m nav2.prepare_nav2_stack \
    --robot-config "${INTERNDATA_NAV2_ROBOT_CONFIG:-workflows/simbox/core/configs/robots/split_aloha.yaml}" \
    --output-dir "${NAV2_STACK_CONFIG_DIR}" \
    --map-dir "${NAV2_BOOTSTRAP_MAP_DIR}"

  python3 -m nav2.nav2_bridge_adapter \
    --robot-name "${INTERNDATA_NAV2_ROBOT_NAME:-split_aloha}" \
    --map-update-topic "${INTERNDATA_NAV2_BRIDGE_MAP_UPDATE_TOPIC:-/simbox/nav_bridge/map_update}" \
    --goal-topic "${INTERNDATA_NAV2_BRIDGE_GOAL_TOPIC:-/simbox/nav_bridge/goal}" \
    --cancel-topic "${INTERNDATA_NAV2_BRIDGE_CANCEL_TOPIC:-/simbox/nav_bridge/cancel}" \
    --status-topic "${INTERNDATA_NAV2_BRIDGE_STATUS_TOPIC:-/simbox/nav_bridge/status}" \
    --result-topic "${INTERNDATA_NAV2_BRIDGE_RESULT_TOPIC:-/simbox/nav_bridge/result}" \
    --odom-topic "${INTERNDATA_NAV2_ODOM_TOPIC:-/odom}" \
    --action-name "${INTERNDATA_NAV2_ACTION_NAME:-/navigate_to_pose}" \
    --load-map-service "${INTERNDATA_NAV2_LOAD_MAP_SERVICE:-/map_server/load_map}" \
    --clear-global-costmap-service "${INTERNDATA_NAV2_CLEAR_GLOBAL_COSTMAP_SERVICE:-/global_costmap/clear_entirely_global_costmap}" \
    --clear-local-costmap-service "${INTERNDATA_NAV2_CLEAR_LOCAL_COSTMAP_SERVICE:-/local_costmap/clear_entirely_local_costmap}" &
  pids+=($!)

  ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 --frame-id map --child-frame-id odom &
  pids+=($!)

  ros2 run nav2_map_server map_server \
    --ros-args \
    --params-file "${NAV2_PARAMS_FILE}" &
  pids+=($!)

  ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
    -p use_sim_time:=true \
    -p autostart:=true \
    -p node_names:="['map_server']" &
  pids+=($!)

  ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=true \
    autostart:=true \
    use_composition:=False \
    params_file:="${NAV2_PARAMS_FILE}" &
  pids+=($!)
fi

if [ "$#" -gt 0 ]; then
  "$@" &
  pids+=($!)
fi

if [ "${#pids[@]}" -eq 0 ]; then
  exec bash
fi

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}

trap cleanup EXIT INT TERM
wait -n "${pids[@]}"
