# 基座机械臂接入 CuRobo 的待办清单

本文档用于整理当前仓库接入“基座 + 机械臂”一体规划的实施步骤。

## 目标

- 让 `franka_mobile` 这一类机器人支持基座与机械臂联合规划
- 保持现有以末端位姿为输入的技能尽量不改
- 仅在必要时修改直接操作关节空间的技能
- 保证 `IK`、图搜索、轨迹优化、仿真关节映射使用同一套关节定义

## 当前确认的执行思路

- 不新增对外的 `sample_frame` 配置字段
- 对外只增加一个动态技能参数：`planning_mode`
- `planning_mode` 取值：
  - `fixed`：锁死底盘，只让机械臂参与规划
  - `mobile`：底盘与机械臂一起参与规划
- 保留当前 skill 侧 `get_ee_poses("armbase")` 的调用形式
- `armbase` 视为当前规划参考基座
- controller 根据当前 `planning_mode` 决定 `armbase` 实际指向：
  - 机械臂安装基座
  - 或整机基座
- 第一版优先新增移动基座专用 controller 子类，通过 override 跑通最小链路
- 第一版允许继续复用 `arm_indices` 和 `arm_action` 字段名

## 第一阶段：新增机器人配置与机器人类

- 在 [workflows/simbox/core/configs/robots](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/configs/robots) 下新增一个机器人配置文件
  建议命名为 `franka_mobile.yaml`
- 在该配置中，把 `robot_file` 指向 CuRobo 的 [`franka_mobile.yml`](/mnt/exdisk1/project/InternDataEngine/InternDataAssets/curobo/src/curobo/content/configs/robot/franka_mobile.yml#L32)
- 在新的机器人配置中补齐以下字段：
  - `body_indices`：基座关节在 articulation 中的索引
  - `left_joint_indices`：机械臂关节索引
  - `left_gripper_indices`：夹爪关节索引
  - `body_home`
  - `left_joint_home`
  - `left_gripper_home`
  - `fl_base_path`
  - `fl_ee_path`
- 增加机器人形态字段：
  - `robot_type`
- 核对 articulation 的 `dof_names` 与预期映射是否一致，至少确认以下关节：
  - `base_x`
  - `base_y`
  - `base_z` 或者场景里实际使用的基座偏航关节名
  - `panda_joint1` 到 `panda_joint7`
  - 夹爪关节
- 在 [workflows/simbox/core/robots](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/robots) 下新增机器人类
  建议命名为 `franka_mobile.py`
- 让新机器人类继承 [`TemplateRobot`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/robots/template_robot.py#L20)
- 复用 `body_indices` 与 `body_home`

## 第二阶段：新增基座机械臂控制器

- 在 [workflows/simbox/core/controllers](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers) 下新增控制器
  建议命名为 `franka_mobile_controller.py`
- 让其继承 [`TemplateController`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py#L46)
- 第一版通过子类 override 跑通
- 在 `_configure_joint_indices(...)` 中明确整机关节顺序：`["base_x", "base_y", "base_z", "panda_joint1", ..., "panda_joint7"]`
- 在同一函数中补齐：
  - `cmd_js_names`
  - `arm_indices = body_indices + left_joint_indices`
  - `gripper_indices = left_gripper_indices`
- controller 内部维护：
  - 机械臂安装基座对应的参考 prim
  - 整机基座对应的参考 prim
- controller 根据当前 `planning_mode` 选择当前生效的参考 prim
- 视情况 override 以下方法：
  - `_load_kin_model()`
  - `get_ee_pose()`
  - `forward_kinematic()`
  - `forward()`
  - `pre_forward()`
- `lr_name` 保持现有约定

## 第三阶段：验证现有 TemplateController 公共逻辑是否可直接复用

- 验证以下公共路径：
  - [`TemplateController.plan()`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py#L309)
  - [`TemplateController.plan_batch()`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py#L286)
  - [`TemplateController.ee_forward()`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py#L359)
  - [`TemplateController.get_ee_pose()`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py#L450)
  - [`TemplateController.forward_kinematic()`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py#L462)
  - [`TemplateController.pre_forward()`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py#L534)
- 重点检查：
  - `_load_kin_model()` 是否能正确加载 whole-body 运动学
  - `get_ee_pose()` 的输入关节维度是否与 whole-body `FK` 一致
  - `ee_forward()` 输出的 `joint_positions` 与 `joint_indices` 是否包含底盘关节
  - `reference_prim_path` 在 `fixed/mobile` 模式切换时是否始终一致

## 第四阶段：尽量保持末端位姿驱动技能不变

- 验证下列技能是否可以不改结构直接工作：
  - [`pick.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/pick.py)
  - [`manualpick.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/manualpick.py)
  - [`dynamicpick.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/dynamicpick.py)
  - [`place.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/place.py)
  - [`goto_pose.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/goto_pose.py)
  - [`move.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/move.py)
- 对抓取类技能补充检查：
  - [`pick.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/pick.py#L196)
  - [`manualpick.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/manualpick.py#L287)
  - [`dynamicpick.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/dynamicpick.py#L195)
- 这些技能当前把抓取候选位姿采样写死在 `armbase` 语义上
- skill 动态参数增加 `planning_mode`
- skill 在初始化时把当前 `planning_mode` 传给 controller
- controller 根据 `planning_mode` 决定当前 `armbase` 的参考基座
- 保留 `get_ee_poses("armbase")` 调用形式
- 当 `planning_mode=mobile` 时，`armbase` 的实际语义应切换为整机规划参考基座
- 当 `planning_mode=fixed` 时，`armbase` 保持当前机械臂安装基座语义
- 复核以下筛选项是否需要放宽：
  - `direction_to_obj`
  - `filter_x_dir`
  - `filter_y_dir`
  - `filter_z_dir`
- 第一版优先放宽上述方向筛选

## 第五阶段：修正默认只支持机械臂关节的技能

- 处理下列技能：
  - [`home.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/home.py#L20)
  - [`joint_ctrl.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/joint_ctrl.py#L37)
  - [`heuristic_skill.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/heuristic_skill.py#L24)
- 决定保持机械臂专用还是升级为整机关节控制
- 如果升级为整机控制，需要修改：
  - 使用 `controller.plan_indices`
  - 使用整机 home 关节向量
  - 把 `arm_action` 语义改成 `plan_action`
- 如果继续保持机械臂专用，写清作用范围

## 第六阶段：把基座原语技能与整机规划分开

- 保留 [`mobile_translate.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/mobile_translate.py#L18) 与 [`mobile_rotate.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/mobile_rotate.py#L18)
- 这两个技能只作为显式底盘原语使用，不作为默认抓取流程的一部分

## 第七阶段：补齐 CuRobo 中的基座碰撞模型

- 扩展 CuRobo 机器人配置中的碰撞部分，让基座也进入碰撞检测
- 更新内容包括：
  - `collision_link_names`
  - 碰撞球配置
  - 必要的基座几何包围表示

## 第八阶段：增加验证测试

- 至少新增一个集成级测试，覆盖以下场景：
  - 目标末端位姿必须依赖基座移动才能到达
  - 单纯机械臂规划会失败
  - 整机规划成功，且输出中确实包含基座关节变化
- 增加控制器级检查，至少验证：
  - `cmd_js_names` 顺序
  - CuRobo 关节名与 articulation 关节名是否对齐
  - `kin_model` 与 `motion_gen` 的正运动学是否一致
- 可参考 CuRobo 自带测试 [`motion_gen_api_test.py`](/mnt/exdisk1/project/InternDataEngine/InternDataAssets/curobo/tests/motion_gen_api_test.py#L64)，尤其是锁关节更新的用法

## 第九阶段：实施顺序建议

- 第一轮只支持一个单臂基座机械臂，不要同时处理双臂扩展
- 先打通：
  - 一个机器人配置
  - 一个控制器
  - 一个任务
  - 一条 `pick -> place` 的技能链
- 在最小链路跑通前，不优先重构 `TemplateController`
- 在最小链路跑通后，再决定哪些公共逻辑值得回收上提到 `TemplateController`
- 不与双臂重构、夹爪行为调整、新技能设计混改

## 最小可行改动集

- 新增 `franka_mobile.yaml`
- 新增 `FrankaMobile` 机器人类
- 新增 `FrankaMobileController`
- 在 `FrankaMobileController` 中通过 override 跑通 `planning_mode` 与参考基座切换
- 验证 `pick` 与 `goto_pose`
- 补齐基座碰撞球
- 新增一个回归测试

## 待后续实现

- 如果移动基座 controller 需要重写的公共逻辑越来越多，再统一回收并重构 [`TemplateController`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py)
- 统一清理 `arm_indices` 和 `arm_action` 等旧命名
- 重新评估 [`TemplateController._load_kin_model()`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/controllers/template_controller.py#L115) 是否需要统一改为 `RobotConfig.from_dict(...)`
- 为 logger 补充整机动作记录支持：
  - [`workflows/simbox/core/loggers/utils.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/loggers/utils.py)
- 升级下列关节空间 skill：
  - [`home.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/home.py#L38)
  - [`joint_ctrl.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/joint_ctrl.py#L30)
  - [`heuristic_skill.py`](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/skills/heuristic_skill.py#L103)
- 重新检查以下辅助逻辑是否需要抽象上提：
  - `dummy_forward`
  - `in_plane_rotation`
  - 任何默认假设“最后一个 arm joint 是 wrist”的逻辑

## 已知风险

- 当前仓库中很多地方把“机械臂”当成默认受控对象，重命名时要非常谨慎
- 如果 `kin_model` 与 `motion_gen` 使用的关节定义不同，正运动学会悄悄出错
- 基座关节名必须与 articulation 完全一致，否则重排关节时会出错
- 如果没有补齐基座碰撞体，整机规划只是在运动学上看起来正确，无法保证场景中可执行

## 完成标准

- 基座机械臂可以在一个末端位姿驱动的抓取技能中，让基座与机械臂一起由 CuRobo 规划
- 控制器输出的动作包含整机关节与夹爪关节
- `FK`、`IK`、`MotionGen` 三者使用同一套关节顺序
- CuRobo 已启用基座碰撞模型
- 原有机械臂专用技能要么继续可用，要么被明确标注为“仅机械臂适用”
