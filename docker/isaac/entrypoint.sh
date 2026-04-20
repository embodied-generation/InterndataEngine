#!/usr/bin/env bash
set -e

export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/isaac-sim}"
export CUROBO_PATH="${CUROBO_PATH:-/opt/curobo}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export INTERNDATA_AUTOSTART_LAUNCHER="${INTERNDATA_AUTOSTART_LAUNCHER:-0}"
export INTERNDATA_LAUNCHER_CONFIG="${INTERNDATA_LAUNCHER_CONFIG:-configs/simbox/de_plan_with_render_template.yaml}"

for lib_dir in \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ros2_bridge/humble/lib" \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ros2_bridge/foxy/lib" \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/torch/lib" \
  "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cuda_runtime/lib" \
  "${ISAAC_SIM_PATH}/kit/exts/omni.cuda.libs/bin"
do
  if [ -d "${lib_dir}" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${lib_dir}"
  fi
done

run_default_entry() {
  if [ "${INTERNDATA_AUTOSTART_LAUNCHER}" = "1" ]; then
    cd /workspace
    exec "${ISAAC_SIM_PATH}/python.sh" launcher.py --config "${INTERNDATA_LAUNCHER_CONFIG}"
  fi

  if [ "${LIVESTREAM:-0}" = "1" ]; then
    exec "${ISAAC_SIM_PATH}/isaac-sim.sh" \
      --no-window \
      --/isaac/startup/ros_bridge_extension=omni.isaac.ros2_bridge
  fi

  exec bash
}

if [ -d /workspace ]; then
  if [ -d "${CUROBO_PATH}" ] && [ ! -e /workspace/InternDataAssets/curobo ]; then
    mkdir -p /workspace/InternDataAssets
    ln -s "${CUROBO_PATH}" /workspace/InternDataAssets/curobo
  fi

  if [ -d /workspace/workflows/simbox ] && [ ! -e /workspace/workflows/simbox/curobo ]; then
    ln -s ../../InternDataAssets/curobo /workspace/workflows/simbox/curobo
  fi
fi

if [ -d "${CUROBO_PATH}/src" ]; then
  export PYTHONPATH="${CUROBO_PATH}/src:${PYTHONPATH}"
fi

echo "[isaac] Python version:"
"${ISAAC_SIM_PATH}/python.sh" -c "import sys; print(sys.version)"

echo "[isaac] Test imports..."
"${ISAAC_SIM_PATH}/python.sh" - <<'PY'
mods = [
    "trimesh",
    "open3d",
    "cv2",
    "imageio",
    "plyfile",
    "omegaconf",
    "pydantic",
    "toml",
    "shapely",
    "ray",
    "pympler",
    "skimage",
    "lmdb",
    "numpy",
    "scipy",
    "yaml",
    "sklearn",
    "transforms3d",
    "curobo",
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
]
for module in mods:
    try:
        __import__(module)
        print("[ok]", module)
    except Exception as exc:
        print("[fail]", module, exc)
PY
if [ "$#" -eq 0 ]; then
  run_default_entry
fi

exec "$@"
