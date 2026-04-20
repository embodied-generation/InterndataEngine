# `default_nav.yaml` 当前说明

对应文件：

- `workflows/simbox/core/configs/nav/default_nav.yaml`

当前这份 YAML 已经收紧到只保留静态地图 Nav2 链路需要的字段，不再包含任何激光定位或世界真值 LaserScan 相关配置。

## 当前保留字段

### `ros.localization.map_output_dir`

- 含义：静态地图导出目录
- 使用位置：`workflows/simbox/core/mobile/nav2/runtime.py`
- 实际作用：`IsaacStaticMapExporter` 会把导出的 `map.pgm` 和 `map.yaml` 写到这个目录下

### `ros.nav2.enabled`

- 含义：是否启用 `Nav2Navigator`
- 使用位置：`workflows/simbox/core/mobile/nav2/nav2_navigator.py`

### `nav2_skill.bt_navigator.plugin_lib_names`

- 含义：Nav2 BT Navigator 需要加载的插件库列表
- 使用位置：`workflows/simbox/core/mobile/nav2/runtime.py`
- 说明：当前实现会把这一项直接写进 Nav2 参数文件，通常不建议删空

## 已删除内容

下面这些内容已经不再属于当前导航链路：

- `ros.scan_topic`
- `ros.world_scan.*`
- `ros.localization.scan_topic`
- AMCL 相关字段

原因是当前系统的导航模式已经固定为：

- 使用 `IsaacStaticMapExporter` 从场景碰撞几何导出静态地图
- 使用 `map_server` 加载静态地图
- 使用 `odom / truth pose` 作为位姿来源

而不是：

- 发布仿真 LaserScan
- 用激光做定位或建图

## 当前地图来源

静态地图不是来自 Isaac Sim 自带 occupied map 扩展接口，而是项目内自定义导出器直接读取 USD / collision geometry 后生成的：

- `workflows/simbox/core/mobile/nav2/localization_stack.py`

运行时入口在：

- `workflows/simbox/core/mobile/nav2/runtime.py`
