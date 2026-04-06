## 虚拟底盘位姿关节导航 API 模块文档

### 1. 模块总览

移动导航模块采用以下结构：

```text
workflows/simbox/mobile/
  occupancy_map/
    occupancy_map_manager.py
  planner/
    base_planner.py
    grid_planner.py
  controllers/
    mobile_controller.py
    split_aloha_controller.py
  utils/
    geometry.py
    grid.py
    path.py
```

模块职责：

- `occupancy_map/`：生成 3D 占据结果并维护 2D 投影地图
- `planner/`：执行 2D 路径搜索并输出 waypoint
- `controllers/`：调度地图、路径、路径跟踪与位姿关节控制
- `utils/`：提供几何、栅格、路径相关工具函数
- 机器人配置中的 `mobile` 配置块：提供虚拟底盘位姿关节与导航执行参数

### 2. 核心数据结构

#### 2.1 Pose2D

```python
from dataclasses import dataclass


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
```

用途：

- 表示底盘当前位姿
- 表示导航目标位姿
- 表示路径中的 waypoint

#### 2.2 GridCoord

```python
from dataclasses import dataclass


@dataclass
class GridCoord:
    row: int
    col: int
```

用途：

- 表示 occupancy map 中的栅格坐标

#### 2.3 PathPlanResult

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class PathPlanResult:
    success: bool
    grid_path: List[GridCoord] = field(default_factory=list)
    world_path: List[Pose2D] = field(default_factory=list)
    message: str = ""
```

用途：

- 表示路径规划输出
- 统一承载路径规划状态和路径数据

#### 2.4 NavigationState

```python
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class NavigationState:
    current_pose: Optional[Pose2D] = None
    goal_pose: Optional[Pose2D] = None
    current_path: List[Pose2D] = field(default_factory=list)
    waypoint_index: int = 0
    goal_reached: bool = False
```

用途：

- 表示控制器内部导航状态

### 3. Occupancy Map 模块

文件建议：

- `workflows/simbox/mobile/occupancy_map/occupancy_map_manager.py`

类定义：

```python
class OccupancyMapManager:
    def __init__(
        self,
        cell_size: float,
        map_origin: tuple[float, float],
        map_size: tuple[int, int],
        stage,
        collision_root: str | None = None,
    ) -> None: ...
```

职责：

- 调用 `omni.isaac.occupancy_map.bindings._occupancy_map.Generator`
- 管理地图尺寸、原点、分辨率和 z 范围
- 维护 3D 占据结果到 2D 地图的投影
- 维护世界坐标与栅格坐标转换
- 提供占用查询与地图刷新接口

实现约束：

- 通过 `omni.physx.get_physx_interface()` 获取 `physx`
- 通过 `omni.usd.get_context().get_stage_id()` 获取 `stage_id`
- 通过 `_occupancy_map.Generator(physx, stage_id)` 创建地图生成器
- 通过 `update_settings(cell_size, occupied_value, free_value, unknown_value)` 设置地图生成参数
- 通过 `set_transform(origin, min_bound, max_bound)` 设置地图范围
- 通过 `generate3d()`、`get_occupied_positions()`、`get_buffer()`、`get_dimensions()` 获取地图结果
- 通过占据体素投影生成 planner 使用的 2D 地图
- 地图原始值语义固定为：
  - `1.0` 表示 occupied
  - `0.0` 表示 free
  - `0.5` 表示 unknown

公开接口：

```python
def build_map(self) -> None: ...
def update_map(self) -> None: ...
def get_map(self): ...
def get_raw_map(self): ...
def get_raw_volume(self): ...
def get_volume_dims(self) -> tuple[int, int, int]: ...
def world_to_grid(self, x: float, y: float) -> GridCoord: ...
def grid_to_world(self, row: int, col: int) -> tuple[float, float]: ...
def is_occupied(self, row: int, col: int) -> bool: ...
def get_neighbors(self, row: int, col: int) -> list[GridCoord]: ...
```

### 4. 路径规划模块

文件建议：

- `workflows/simbox/mobile/planner/base_planner.py`
- `workflows/simbox/mobile/planner/grid_planner.py`

#### 4.1 BasePlanner

```python
class BasePlanner:
    def plan(
        self,
        start: Pose2D,
        goal: Pose2D,
        occupancy_map: OccupancyMapManager,
    ) -> PathPlanResult:
        raise NotImplementedError
```

#### 4.2 GridPlanner

```python
class GridPlanner(BasePlanner):
    def __init__(
        self,
        algorithm: str = "astar",
        allow_diagonal: bool = True,
        heuristic_weight: float = 1.0,
    ) -> None: ...
```

职责：

- 基于 occupancy map 执行 2D 栅格路径规划
- 支持 A*、Dijkstra 等算法配置
- 输出栅格路径和世界路径

公开接口：

```python
def plan(
    self,
    start: Pose2D,
    goal: Pose2D,
    occupancy_map: OccupancyMapManager,
) -> PathPlanResult: ...

def smooth_path(self, world_path: list[Pose2D]) -> list[Pose2D]: ...
```

### 5. 控制器模块

文件建议：

- `workflows/simbox/mobile/controllers/mobile_controller.py`
- `workflows/simbox/mobile/controllers/split_aloha_controller.py`

#### 5.1 MobileController

```python
class MobileController:
    def __init__(
        self,
        robot,
        occupancy_map_manager: OccupancyMapManager,
        planner: BasePlanner,
        cfg: dict,
    ) -> None: ...
```

职责：

- 统一调度地图刷新、路径规划、路径跟踪和位姿关节控制
- 维护导航状态
- 对上层暴露统一导航接口

公开接口：

```python
def get_base_pose(self) -> Pose2D: ...
def set_goal(self, goal_pose: Pose2D) -> None: ...
def update_map(self) -> None: ...
def plan_path(self, goal_pose: Pose2D | None = None) -> PathPlanResult: ...
def get_next_waypoint(self) -> Pose2D | None: ...
def step(self): ...
def reset(self) -> None: ...
def is_goal_reached(self) -> bool: ...
def get_navigation_state(self) -> NavigationState: ...
```

内部流程：

```python
def step(self):
    current_pose = self.get_base_pose()
    self.update_map()
    if not self.state.current_path:
        self.plan_path()
    waypoint = self.get_next_waypoint()
    target = self.track_waypoint(current_pose, waypoint)
    action = self.compute_virtual_base_action(target)
    return action
```

#### 5.2 SplitAlohaController

```python
class SplitAlohaController(MobileController):
    def __init__(self, robot, occupancy_map_manager, planner, cfg: dict) -> None: ...
```

职责：

- 适配具体机器人
- 读取虚拟底盘位姿关节名称、索引、`mobile_base_path`、`base_frame`
- 输出 articulation action

公开接口：

```python
def get_virtual_base_joint_indices(self) -> list[int]: ...
def get_virtual_base_joint_names(self) -> list[str]: ...
def compute_virtual_base_action(self, target_pose: Pose2D): ...
def apply_action(self, action) -> None: ...
```

### 6. 路径跟踪接口

`MobileController` 内部封装路径跟踪逻辑。

建议封装接口：

```python
def track_waypoint(self, current_pose: Pose2D, waypoint: Pose2D) -> Pose2D:
    ...
```

输入：

- 当前底盘位姿
- 当前路径目标点

输出：

- 虚拟底盘位姿关节目标

说明：

- 跟踪逻辑维护 waypoint 到达判断
- waypoint 切换规则由控制器内部维护

### 7. 位姿关节控制接口

`MobileController` 内部封装虚拟底盘位姿关节控制逻辑。

建议封装接口：

```python
def compute_virtual_base_action(self, target_pose: Pose2D):
    ...
```

输入：

- `x`
- `y`
- `yaw`

输出：

- articulation joint action

### 8. 配置接口

#### 8.1 NavigationConfig

```python
navigation_cfg = {
    "cell_size": 0.05,
    "map_origin": [-5.0, -5.0],
    "map_size": [200, 200],
    "planner_algorithm": "astar",
    "allow_diagonal": True,
    "heuristic_weight": 1.0,
    "path_sample_step": 0.1,
    "goal_tolerance_xy": 0.05,
    "goal_tolerance_yaw": 0.1,
    "max_step_translation": 0.1,
    "max_step_rotation": 0.2,
}
```

#### 8.2 RobotConfig

```python
robot_mobile_cfg = {
    "mobile": {
        "virtual_base_joint_names": [
            "mobile_translate_x",
            "mobile_translate_y",
            "mobile_rotate",
        ],
        "virtual_base_joint_indices": [0, 1, 2],
        "mobile_base_path": "split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/base_link",
        "base_frame": "base_link",
        "odom_source": "articulation",
        "goal_tolerance_xy": 0.05,
        "goal_tolerance_yaw": 0.1,
        "max_step_translation": 0.1,
        "max_step_rotation": 0.2,
    },
}
```

对应的 YAML 结构建议：

```yaml
mobile:
  virtual_base_joint_names:
    - mobile_translate_x
    - mobile_translate_y
    - mobile_rotate
  virtual_base_joint_indices: [0, 1, 2]
  mobile_base_path: "split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/base_link"
  base_frame: "base_link"
  odom_source: "articulation"
  goal_tolerance_xy: 0.05
  goal_tolerance_yaw: 0.1
  max_step_translation: 0.1
  max_step_rotation: 0.2
```

### 9. 上层调用示例

```python
goal = Pose2D(x=2.0, y=1.0, yaw=1.57)
mobile_controller.set_goal(goal)

while not mobile_controller.is_goal_reached():
    action = mobile_controller.step()
    mobile_controller.apply_action(action)
```

### 10. 实现依赖

Isaac Sim 侧依赖：

- `omni.isaac.occupancy_map`
- `omni.isaac.occupancy_map.bindings._occupancy_map.Generator`
- `omni.physx`
- `omni.usd`

项目内依赖：

- 机器人 articulation
- 机器人配置
- stage collision geometry

### 11. 文档约束

- 所有导航目标使用 `Pose2D`
- 所有路径规划结果统一使用 `PathPlanResult`
- 所有地图访问统一通过 `OccupancyMapManager`
- 所有上层导航调用统一通过 `MobileController`
