# Memory Export

Generated: 2026-04-06

## User Memory (/memories/)

No user memory files found.

## Session Memory (/memories/session/)

No session memory files found.

## Repository Memory (/memories/repo/)

### /memories/repo/simbox_robot_loading.md

- Robot YAMLs are merged into task robot entries in workflows/simbox_dual_workflow.py::_merge_robot_configs using robot_config_file.
- Robot class resolution is by target_class via core.robots.get_robot_cls -> ROBOT_DICT.
- New robot classes must be imported in workflows/simbox/core/robots/__init__.py to ensure decorator registration at runtime.
- workflows/simbox/assets resolves to InternDataAssets/assets (symlink); robot cfg path should be written relative to that asset root (e.g., franka_mobile/robot.usd).
- CuRobo robot configs must use joint/link names exactly as they appear in the URDF; for mobile_fr3_duo_v0_2 use left_fr3v2_* and franka_spine_vertical_joint, not panda_* names.
- If a robot uses >2 CuRobo robot_file entries, workflows/simbox_dual_workflow.py controller dispatch must map filenames to explicit keys (left/right/base) to avoid overwriting non-left files as right.
- SimBoxDualWorkFlow now resolves controller side as: filename contains left/right -> explicit side, otherwise if only one robot_file then defaults to left; multi-file ambiguous names raise ValueError.
- For franka_mobile whole-body planning, skill targets are expressed in reference_prim_path while cuRobo plans in mobile base frame; controller must convert poses between these frames in get_ee_pose/ee_forward/test_* APIs.
- For phase-4 compatibility, pick/manualpick/dynamicpick only relax direction filters when planning_mode=mobile (and relax_mobile_direction_filters=true); fixed mode remains unchanged.
- Phase-5 keeps home/joint_ctrl/heuristic_skill arm-only: for whole-body controllers they compose arm commands with fixed extra joints and default to force_fixed_planning_mode=true.

- For tracer2_franka URDF orientation issues, validate base_link->panda_hand zero-pose direction numerically; target a dominant +Z direction (e.g., ~0.995) before re-importing into Isaac to avoid subjective viewport misreads.

- Tracer2 planning jitter can be caused by unwrapped revolute joint states (e.g., >100 rad equivalent angles); normalize angular joints to [-pi, pi] before planning and before sending actions.

- Tracer2Franka can run mobile-only while keeping skill compatibility by preserving set_planning_mode API and aliasing fixed->mobile; avoid removing planning_mode attribute until skills stop referencing it.

- Regressions can occur if set_planning_mode("fixed") is treated as a no-op: skills rely on fixed mode for reference frame semantics (arm base frame) and filter behavior; keep fixed/mobile semantics even with a single mobile backend.

- In ground pick-place tasks, "robot does not move" can come from all pick plans failing: using test_mode=ik with floor targets often admits infeasible grasps, then ee_forward falls back to current joints; prefer test_mode=forward and add skill-level ignore_substring:["floor"] for floor-heavy scenes.

- For tracer2 single_pick stability, keep controller planning_mode semantics active (fixed/mobile) and default to fixed; forcing planning_mode to always mobile can relax pick filters and cause repeated plan failures that appear as "robot not moving".

- Keep robot class name aligned with robot YAML target_class and robots/__init__.py imports; a mismatch (e.g., class FrankaMobile in tracer2_franka.py while importing Tracer2Franka) causes ImportError during workflow bootstrap.

- Other controllers (FR3/FrankaRobotiq85/Lift2/SplitAloha/Genie1) do not override ee_forward; tracer2-specific per-step post-processing in ee_forward can introduce jitter, so keep tracer2 ee_forward minimal (pose conversion + super call + finite-value guard).

- Tracer2Franka instability can come from overly high generic max_joint_velocities; set tracer2-specific caps aligned with URDF limits (base ~[1.0,1.0,1.5], arm ~[2.175..2.61]) to reduce sim-arm divergence while keeping planner targets smooth.

- tracer2 mobile crash triage: controller-side safeguards (cspace finger-joint pruning for mobile planner, invalid prim-path fallback, non-finite joint sanitization, tracer2 default use_cuda_graph=False in code) removed observed torch.cuda.graphs.replay/segfault signatures in long headless soak logs; remaining omni.usd delegate _Get 'prim' warnings were non-fatal.

- SplitAloha ROS base bridge requires workflow lifecycle hooks (initialize after reset, step every sim tick, destroy before task reset); bridge module alone is not sufficient to activate ROS I/O.

### /memories/repo/test_execution_notes.md

- In this workspace, pytest may fail from unrelated system plugins (ROS launch_testing). Use PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 when running isolated repository tests.
- tracer2_franka base collision completeness can be validated via tests/integration/test_tracer2_franka_base_collision.py (checks collision_link_names, collision spheres, and URDF collision geometry for base links).
- SimBoxDualWorkFlow reset mutates task_cfg (e.g., pops arena_file/camera_file/logger_file); init_task should use deepcopy(self.task_cfgs[index]) to avoid cross-scene config pollution and None path crashes.
- tracer2_franka USD has defaultPrim /tracer2_franka with links directly under it; robot config paths should be relative like panda_hand/panda_link0 (no tracer2_franka/ prefix), otherwise controller reset and contact filters hit invalid prim / zero matches.

- output/simbox_plan_with_render/de_time_profile_*.log only records high-level pipeline events; Isaac PhysX/RTX plugin errors (e.g., Illegal BroadPhaseUpdateData) may only appear in terminal stdout, so root-cause tracing must follow workflow return paths (wf.plan_with_render -> obs_num<=0) plus control-action checks.

- For mobile robots randomized by RandomRegionSampler.A_on_B_region_sampler, invalid USD bbox values can propagate to placement z via obj_z_min and produce extreme world coordinates (e.g., -3.4e38) leading to PhysX Illegal BroadPhaseUpdateData; add bbox validity guards before using bbox min/max.

- In Isaac smoke scripts that instantiate SimBoxDualWorkFlow directly, call init_task(0) (not reset()) so self.task_cfg is initialized; reset() expects task_cfg already set by base workflow init_task.
- Isaac python.sh forwards argv into Kit; strip script-only args from sys.argv before SimulationApp(...) to avoid collisions with Kit CLI options and misleading early shutdown behavior.
- For diagnostics, print errors before simulation_app.close(); close may still end with EXIT_CODE:0 even when Python exceptions occurred, so rely on explicit [smoke] FAILED/traceback output.
- If arm/gripper appear completely static, check workflows/simbox/core/robots/template_robot.py::apply_action: set_joint_position_targets must run unconditionally; if placed inside if bad_mask.any() block, all valid actions are dropped.

- On Ubuntu 24.04 with Isaac Sim 4.1 (Python 3.10), avoid sourcing system ROS Jazzy Python paths (/opt/ros/jazzy/python3.12). Use Isaac bundled ROS2 Humble path exts/omni.isaac.ros2_bridge/humble/rclpy + humble/lib; otherwise rclpy/tf2 ABI mismatches occur.

- If integration requires behavior changes to external ROS driver repos (e.g., ranger_ros2), keep driver source untouched and add a local adapter/controller class in InternDataEngine instead.

- SplitAloha base root prim /World/.../split_aloha can stay static even when mobile_translate_x/y virtual joints move child links; odom based on robot.get_world_pose() may report near-zero displacement.
- For ROS bridge diagnostics, avoid attaching smoke subscriptions on the same node that publishes/consumes commands; callback processing can starve command subscriptions in tight spin_once(0) loops.
- SplitAloha wheel/steering targets alone may not move base in this asset; bridge should drive virtual joints (mobile_translate_x, mobile_translate_y, mobile_rotate) from cmd_vel-derived velocities.

- PlaneObject previously had visual geometry only; adding an optional hidden thin-cuboid child with UsdPhysics.CollisionAPI (e.g., collision_enabled: true, collision_thickness) is a practical way to give floor planes collision volume in arena YAMLs.

- For Isaac Sim scripts that import from isaacsim import SimulationApp, run via /mnt/exdisk1/isaac-sim-4.1/python.sh (not raw kit/python/bin/python3), otherwise ModuleNotFoundError: isaacsim can occur.
- To reduce SplitAloha ROS motion jitter in diagnostics scripts, reuse workflow-managed _ros_base_bridges/_ros_base_command_controllers and avoid creating a second bridge/controller pair on the same robot.

- In the current Isaac Sim 4.1 runtime used here, nav2_msgs is not available by default (ModuleNotFoundError via /mnt/exdisk1/isaac-sim-4.1/python.sh), so Nav2 integration should be guarded behind config flags and fail-soft initialization.

- Isaac Sim PhysX scene query (omni.physx.get_physx_scene_query_interface()) supports raycast_closest(origin, dir, distance) and returns a dict containing at least hit (bool) and when hit distance/collision/rigidBody; useful for synthetic /scan generation from world ground truth.

- For point-navigation demos on SplitAloha Ackermann base, strict final yaw can cause false negatives (reached_target=false) despite small position error; make yaw requirement optional (e.g., --require-yaw) and default success to position-only.

- For scripts/simbox/record_collaborate_topdown_mp4.py, persist world-scan outputs per run (*_scan.jsonl + *_scan.npz) and auto-create date-stamped run directories (e.g., output/camera_probe/YYYYMMDD_HHMMSS_*) to avoid artifact mix-ups.
- If users report "base not moving" while virtual navigation progresses, compare world_planar_displacement vs virtual_planar_displacement; a tiny ratio indicates root/world pose stays near-static even when virtual joints change.

- In SplitAlohaIsaacBaseBridge, if virtual odom is enabled, publish /tf from the same virtual pose as /odom; mixing virtual odom with world-root TF creates frame inconsistency and confusing “moving vs not moving” diagnostics.

### /memories/repo/virtual_joint_cleanup_notes.md

- For strict "remove all" requests, run repo-wide grep across workflows/scripts/tests/nimbus/configs, not just touched files.
- In this repo, legacy virtual-joint remnants can also hide in skills modules (e.g., mobile_*), even after bridge/robot/config cleanup.
