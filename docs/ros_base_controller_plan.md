# ROS Driver, Nav2, and Isaac Sim Integration Plan

## Goal

构建一条完整的底盘 ROS 集成链路：

```text
/cmd_vel
  -> ROS base driver
  -> Isaac Sim bridge
  -> SplitAloha real base joints

Nav2
  -> /cmd_vel
  -> ROS base driver
  -> Isaac Sim bridge
  -> SplitAloha real base joints
```

本次实现范围分为两层：

- 底盘控制集成层
  - 将 ROS 底盘驱动与 Isaac Sim 中的真实底盘 joints 接通
- 导航层
  - 采用 Nav2 作为导航框架

本次不重写 ROS 底盘驱动算法，只负责：

- ROS 驱动与 Isaac Sim 的接口对齐
- Isaac Sim 侧命令执行
- Isaac Sim 状态回传给 ROS
- Nav2 所需状态与传感器接口打通

## Target Architecture

### Base Control Path

```text
geometry_msgs/msg/Twist
  -> ROS base driver
  -> base command topic
  -> Isaac Sim bridge
  -> steering joints + wheel joints
```

### Navigation Path

```text
Nav2
  -> /cmd_vel
  -> ROS base driver
  -> Isaac Sim bridge
  -> SplitAloha base

Isaac Sim
  -> /joint_states
  -> /odom
  -> /tf
  -> laser / depth / occupancy-related sensor topics
  -> Nav2
```

## Scope

### In Scope

- 接入 ROS 底盘驱动输出
- 在 Isaac Sim 中执行真实底盘 joint 命令
- 从 Isaac Sim 回传：
  - `/joint_states`
  - `/odom`
  - `/tf`
- 为 Nav2 提供最小可用接口：
  - `/cmd_vel`
  - `/odom`
  - `/tf`
  - 传感器 topic
- 基础联调与 smoke test

### Out of Scope

- 重写 ROS 底盘驱动本体
- 自研导航框架
- 自研底盘轨迹跟踪器
- 多模式运动规划器
- 机械臂与底盘联合任务调度

## Core Assumptions

- ROS 侧已有可用的底盘驱动文件
- ROS 驱动至少支持：
  - 订阅 `/cmd_vel`
  - 输出底盘控制命令
- `SplitAloha` 在 Isaac Sim 中存在真实底盘结构：
  - 4 个 steering joints
  - 4 个 wheel joints
- Nav2 将作为唯一导航框架

## Interfaces to Be Fixed First

在开始编码前，先固定以下接口。

### 1. ROS Driver Output Interface

需要明确：

- 控制命令 topic 名称
- 消息类型
- 字段语义
- 控制频率
- 是否带时间戳

可接受的两类接口：

1. 关节级控制命令
   - steering 目标角
   - wheel 目标速度
2. 车体级控制命令
   - 例如车体线速度、角速度或模式化底盘命令

第一种更容易直接接 Isaac Sim。

### 2. Isaac Sim State Output Interface

需要发布：

- `/joint_states`
- `/odom`
- `/tf`

最小坐标树：

- `odom -> base_link`

### 3. Nav2 Sensor Interface

至少需要一类导航感知输入：

- `/scan`
- 或点云转激光
- 或已知地图 + 局部障碍数据

第一阶段建议优先选：

- 2D 激光雷达 topic `/scan`

这样接 Nav2 最直接。

## Robot YAML Extension

在 [split_aloha.yaml](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/configs/robots/split_aloha.yaml) 中新增 `base` 配置块。

建议字段：

```yaml
base:
  steering_joint_names: []
  wheel_joint_names: []
  wheel_base: 0.0
  track_width: 0.0
  wheel_radius: 0.0
  steering_limit: 0.0
  steering_rate_limit: 0.0
  wheel_velocity_limit: 0.0
  command_timeout: 0.25
  ros:
    command_topic: ""
    command_type: ""
    joint_state_topic: "/joint_states"
    odom_topic: "/odom"
    tf_enabled: true
    base_frame: "base_link"
    odom_frame: "odom"
    scan_topic: "/scan"
```

要求：

- 所有底盘参数只从 YAML 读取
- 所有 ROS topic/frame 只从 YAML 读取
- 不在 bridge 中写死 joint 名称或 topic

## Implementation Plan

### Phase 1: Real Joint Interface Confirmation

先在 Isaac Sim 运行时确认 `SplitAloha` 的真实底盘控制接口。

需要确认：

- steering joint names
- wheel joint names
- 对应 DOF index
- 哪些 joint 适合 position target
- 哪些 joint 适合 velocity target

输出：

- 一份 joint mapping 表
- 一份 YAML 初始化参数

### Phase 2: Isaac Sim Bridge

在 Isaac Sim 侧新增 bridge 模块。

职责：

- 订阅 ROS 驱动输出的底盘控制 topic
- 将控制命令映射到真实底盘 joints
- steering joints 下发 position target
- wheel joints 下发 velocity target
- 处理：
  - steering limit
  - steering rate limit
  - wheel velocity clamp
  - command timeout

要求：

- 不使用虚拟底盘关节作为最终执行接口
- 不在 bridge 中写特判逻辑
- 统一按配置和消息语义执行

### Phase 3: ROS State Publishing

在 Isaac Sim 中实时发布：

- `/joint_states`
- `/odom`
- `/tf`

里程计来源：

- 机器人在 Isaac Sim 中的真实位姿
- 机器人在 Isaac Sim 中的真实速度

要求：

- 先确保 `odom -> base_link` 正确
- 时间戳与 ROS 时钟语义一致

### Phase 4: Nav2 Integration

接入 Nav2，作为唯一导航框架。

需要具备：

- `/cmd_vel`
- `/odom`
- `/tf`
- `/scan` 或等价障碍传感器输入

建议接入顺序：

1. 先完成 teleop `/cmd_vel` 控制
2. 再接入 Nav2 local planner/controller
3. 再接入全局地图与完整导航任务

Nav2 在本方案中的职责：

- 接收目标点
- 规划导航路径
- 输出 `/cmd_vel`

ROS base driver 在本方案中的职责：

- 接收 `/cmd_vel`
- 解算底盘控制命令

Isaac Sim bridge 在本方案中的职责：

- 执行 ROS 驱动输出
- 回传状态

### Phase 5: Smoke Tests and Integration Tests

#### Base Control Smoke Test

验证：

- ROS 侧发送命令
- Isaac Sim 真实 joints 收到命令
- 机器人底盘产生对应运动
- `/odom` 连续更新

#### Nav2 Smoke Test

验证：

- Nav2 输出 `/cmd_vel`
- ROS base driver 正常响应
- Isaac Sim 中机器人可向目标点运动

#### Regression Tests

验证：

- joint interface 不漂移
- topic/frame 配置变更后仍能运行
- command timeout 生效

## Execution Order

严格按下面顺序实施：

1. 运行时确认真实底盘 joints
2. 补充 `split_aloha.yaml` 的 `base` 配置
3. 固定 ROS 驱动输出接口
4. 实现 Isaac Sim bridge
5. 实现 `/joint_states`、`/odom`、`/tf`
6. 跑 teleop `/cmd_vel` smoke test
7. 接入 Nav2
8. 跑 Nav2 smoke test

## Risks

### 1. Dummy Base Chain May Interfere

如果资产中的虚拟底盘链仍参与真实物理链，可能会和轮组控制冲突。

这项需要在 Phase 1 就确认。

### 2. Real Wheel Physics May Be Unstable

如果资产的质量、碰撞或关节驱动设置不健康，可能出现：

- 轮子转但车不动
- 底盘抖动
- 里程计不稳定

这属于资产物理问题，需要单独排查。

### 3. ROS Driver Output May Be Custom

如果 ROS 驱动输出的是自定义消息，Isaac Sim 侧需要明确增加对应依赖和消息支持。

### 4. Nav2 Sensor Input May Need Simplification

如果当前 Isaac Sim 场景没有稳定的 `/scan`，Nav2 接入会被传感器链路阻塞。

第一阶段建议优先保证：

- 先跑底盘控制闭环
- 后补导航传感器链路

## Acceptance Criteria

满足以下条件即认为该方案完成第一阶段：

- ROS 驱动输出的底盘命令能进入 Isaac Sim
- Isaac Sim 中真实底盘 joints 收到命令
- 机器人在仿真中产生对应底盘运动
- ROS 侧能收到 `/joint_states`
- ROS 侧能收到 `/odom`
- `odom -> base_link` 可用

满足以下条件即认为导航链路完成：

- Nav2 能输出 `/cmd_vel`
- ROS base driver 能消费 Nav2 输出
- Isaac Sim 中机器人能在导航任务下运动

## Deliverables

最终交付物应包括：

- 更新后的 [split_aloha.yaml](/mnt/exdisk1/project/InternDataEngine/workflows/simbox/core/configs/robots/split_aloha.yaml)
- Isaac Sim bridge 模块
- ROS 状态发布模块
- Nav2 接入配置
- smoke test 脚本或 runner
