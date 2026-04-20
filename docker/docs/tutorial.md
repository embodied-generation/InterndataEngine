可以。下面这套我按你的目标定成了：

* **容器 1：Isaac Sim 4.1.0 + cuRobo + 你列的 Python 包**
* **容器 2：ROS 2 Humble + Nav2**
* 两边都用 `host network`
* **Isaac 容器不装完整 ROS 2**，而是用 **Isaac Sim 自带的内部 ROS 2 库** 来跑 bridge；官方 4.5 文档明确说，如果没有系统级 ROS 2，Isaac Sim 可以加载自带的最小 ROS 2 库，并与同机或同网段的 ROS 2 节点通信。官方也推荐 Isaac Sim 侧优先使用内置的 Python 环境 `python.sh` 来安装额外包。Nav2 这边则需要完整 ROS 2、`colcon`、`rosdep`、`vcstool`。 ([Isaac Sim 文档][1])

先说一个我建议你接受的设计决定：

**`usd-core==24.11` 不要直接装进 Isaac Sim 主环境。**
这不是 NVIDIA 文档里一句明文禁止，而是基于 Isaac Sim 本身就是一个带自有 Python 环境和 Omniverse/Isaac 扩展运行时的应用做出的工程建议；把 PyPI 的 `usd-core` 再塞进去，最容易把 `pxr`/USD 相关模块覆盖掉。你如果只是某些离线脚本需要 `usd-core`，建议把它放到单独 utility 容器或单独 venv；不要混进 Isaac 主服务。这个判断是推断，不是官方硬性规则。 ([Isaac Sim 文档][2])

另外，你列的大部分包对 **Python 3.10/Linux** 是可行的；例如：

* `open3d-cpu==0.19.0` 有 `cp310 manylinux` wheel。 ([PyPI][3])
* `ray==2.54.1` 存在，并且有 `cp310 manylinux2014` wheel，要求 Python `>=3.10`。 ([PyPI][4])
* `drake==1.51.1` 存在，要求 Python `>=3.10`。 ([PyPI][5])

---

## 目录结构

```text
project/
├── docker-compose.yml
├── .env
├── isaac/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── requirements.isaac.txt
│   └── requirements.usd-extra.txt   # 可选，不默认装
├── nav2/
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── maps/
│   │   ├── office.yaml
│   │   └── office.pgm
│   ├── params/
│   │   └── nav2_params.yaml
│   └── rviz/
│       └── nav2.rviz
└── app/
    └── your_isaac_scripts.py
```

---

## 1) `docker-compose.yml`

```yaml
name: isaac-nav2-stack

services:
  isaac-sim:
    build:
      context: ./isaac
      dockerfile: Dockerfile
    image: local/isaac-sim-4.1.0-curobo-app:latest
    container_name: isaac-sim
    network_mode: host
    gpus: all
    privileged: false
    stdin_open: true
    tty: true
    shm_size: "8gb"
    environment:
      ACCEPT_EULA: "Y"
      PRIVACY_CONSENT: "Y"
      OMNI_KIT_ACCEPT_EULA: "YES"

      # Isaac 内部 ROS 2 Humble 库
      ISAAC_SIM_PATH: /isaac-sim
      RMW_IMPLEMENTATION: rmw_fastrtps_cpp
      ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}
      FASTDDS_BUILTIN_TRANSPORTS: DEFAULT

      # 如果你要跨机器通信，可再配 FASTRTPS_DEFAULT_PROFILES_FILE
      # FASTRTPS_DEFAULT_PROFILES_FILE: /workspace/config/fastdds.xml

      # 可选：Nucleus
      # OMNI_SERVER: omniverse://your-nucleus-host/NVIDIA/Assets/Isaac/4.1

      # 可选：headless streaming
      LIVESTREAM: "1"
    volumes:
      - ${ISAAC_CACHE_MAIN:-~/docker/isaac-sim/cache/main}:/root/.cache/ov:rw
      - ${ISAAC_CACHE_COMPUTE:-~/docker/isaac-sim/cache/computecache}:/root/.nv/ComputeCache:rw
      - ${ISAAC_LOGS:-~/docker/isaac-sim/logs}:/root/.nvidia-omniverse/logs:rw
      - ${ISAAC_CONFIG:-~/docker/isaac-sim/config}:/root/.nvidia-omniverse/config:rw
      - ${ISAAC_DATA:-~/docker/isaac-sim/data}:/root/.local/share/ov/data:rw
      - ${ISAAC_PKGS:-~/docker/isaac-sim/pkg}:/root/.local/share/ov/pkg:rw
      - ../app:/workspace/app:rw
    working_dir: /workspace
    entrypoint: ["/entrypoint.sh"]
    command:
      - bash
      - -lc
      - |
        if [ "${LIVESTREAM}" = "1" ]; then
          /isaac-sim/isaac-sim.sh \
            --no-window \
            --/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge
        else
          exec bash
        fi

  nav2:
    build:
      context: ./nav2
      dockerfile: Dockerfile
    image: local/ros2-humble-nav2:latest
    container_name: nav2
    network_mode: host
    stdin_open: true
    tty: true
    environment:
      ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}
      RMW_IMPLEMENTATION: rmw_fastrtps_cpp
      TURTLEBOT3_MODEL: waffle
    volumes:
      - ./nav2/maps:/workspace/maps:rw
      - ./nav2/params:/workspace/params:rw
      - ./nav2/rviz:/workspace/rviz:rw
    working_dir: /workspace
    entrypoint: ["/entrypoint.sh"]
    command:
      - bash
      - -lc
      - |
        source /opt/ros/humble/setup.bash
        ros2 launch nav2_bringup bringup_launch.py \
          use_sim_time:=True \
          map:=/workspace/maps/office.yaml \
          params_file:=/workspace/params/nav2_params.yaml

  rviz:
    image: osrf/ros:humble-desktop
    container_name: rviz
    network_mode: host
    stdin_open: true
    tty: true
    environment:
      DISPLAY: ${DISPLAY}
      QT_X11_NO_MITSHM: "1"
      ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}
      RMW_IMPLEMENTATION: rmw_fastrtps_cpp
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./nav2/rviz:/workspace/rviz:rw
    profiles: ["gui"]
    command: >
      bash -lc "source /opt/ros/humble/setup.bash &&
                rviz2 -d /workspace/rviz/nav2.rviz"
```

### `.env`

```dotenv
ROS_DOMAIN_ID=0
ISAAC_CACHE_MAIN=~/docker/isaac-sim/cache/main
ISAAC_CACHE_COMPUTE=~/docker/isaac-sim/cache/computecache
ISAAC_LOGS=~/docker/isaac-sim/logs
ISAAC_CONFIG=~/docker/isaac-sim/config
ISAAC_DATA=~/docker/isaac-sim/data
ISAAC_PKGS=~/docker/isaac-sim/pkg
DISPLAY=:0
```

官方当前容器文档强调了容器部署里要把 cache/config/log/data 持久化挂出来，而且 Docker Compose 方案本身就是官方支持的做法之一；同机 ROS 2 通信时，默认 Fast DDS 配置即可，能用共享内存传输，性能最好。 ([Isaac Sim 文档][6])

---

## 2) `isaac/Dockerfile`

```dockerfile
FROM nvcr.io/nvidia/isaac-sim:4.1.0

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ISAAC_SIM_PATH=/isaac-sim

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y \
    git git-lfs curl wget ca-certificates \
    build-essential cmake pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 用 Isaac 自带 Python 装 pip 包
COPY requirements.isaac.txt /tmp/requirements.isaac.txt

RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.isaac.txt

# 安装 cuRobo
RUN git lfs install && \
    cd /opt && \
    git clone https://github.com/NVlabs/curobo.git && \
    cd /opt/curobo && \
    /isaac-sim/python.sh -m pip install --no-cache-dir -e .[isaacsim] --no-build-isolation

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
```

这部分遵循了两条官方路径：Isaac Sim 官方建议通过内置 `python.sh` 安装额外 pip 包；cuRobo 官方也明确建议在 Isaac Sim 场景下用 `python.sh`，先装 `tomli wheel ninja`，再 `pip install -e .[isaacsim] --no-build-isolation`。 ([Isaac Sim 文档][2])

---

## 3) `isaac/requirements.isaac.txt`

我按你的清单整理成了 **默认安装版**。
**我把 `usd-core==24.11` 刻意拿掉了。**

```txt
trimesh==4.11.5
open3d-cpu==0.19.0
opencv-python==4.10.0.84
pathfinding==1.0.20
imageio[ffmpeg]==2.37.3
plyfile==1.1.3
omegaconf==2.3.0
pydantic==2.8.0
toml==0.10.2
shapely==2.1.2
ray==2.54.1
pympler==1.1
scikit-image==0.25.2
lmdb==2.2.0
setuptools==65.5.1
wheel==0.46.3
drake==1.51.1
colored==2.3.2
transforms3d==0.4.2
concave-hull==0.1.2
tomli==2.4.1
ninja==1.13.0
numpy==1.26.0
scipy==1.14.1
pyyaml==6.0.3
scikit-learn==1.7.2
```

### 可选：`isaac/requirements.usd-extra.txt`

```txt
usd-core==24.11
```

如果你**必须**在 Isaac 容器里测试它，可以手动加一层：

```dockerfile
COPY requirements.usd-extra.txt /tmp/requirements.usd-extra.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.usd-extra.txt
```

但我不建议默认这么做。

---

## 4) `isaac/entrypoint.sh`

```bash
#!/usr/bin/env bash
set -e

export ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-/isaac-sim}

# Isaac 内部 ROS 2 Humble 库
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ISAAC_SIM_PATH}/exts/isaacsim.ros2.bridge/humble/lib

# 避免某些脚本错误以为没 ROS 环境
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

echo "[isaac] Python version:"
${ISAAC_SIM_PATH}/python.sh -c "import sys; print(sys.version)"

echo "[isaac] Test imports..."
${ISAAC_SIM_PATH}/python.sh - <<'PY'
mods = [
    "trimesh", "open3d", "cv2", "imageio", "plyfile",
    "omegaconf", "pydantic", "toml", "shapely", "ray",
    "pympler", "skimage", "lmdb", "numpy", "scipy",
    "yaml", "sklearn", "transforms3d"
]
for m in mods:
    try:
        __import__(m)
        print("[ok]", m)
    except Exception as e:
        print("[fail]", m, e)
PY

exec "$@"
```

Isaac 4.5 文档明确给了“无系统级 ROS 2 安装时”的配置方式：设置 `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`，并把 `.../exts/isaacsim.ros2.bridge/humble/lib` 加到 `LD_LIBRARY_PATH`。文档也说明这套内部库足够支撑 ROS2 Bridge 里自带的 publisher/subscriber 节点。 ([Isaac Sim 文档][1])

---

## 5) `nav2/Dockerfile`

```dockerfile
FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-turtlebot3-msgs \
    ros-humble-turtlebot3-description \
    ros-humble-rviz2 \
    ros-humble-tf-transformations \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update || true

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
```

Nav2 官方文档写得很直接：要构建/使用 Nav2，先要有完整 ROS 2，以及 `colcon`、`rosdep`、`vcstool`。 ([Nav2][7])

---

## 6) `nav2/entrypoint.sh`

```bash
#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

exec "$@"
```

---

## 7) 启动方式

先准备持久化目录：

```bash
mkdir -p ~/docker/isaac-sim/{cache/main,cache/computecache,config,data,logs,pkg}
```

如果你要用本机 X11 跑 `rviz` profile：

```bash
xhost +local:docker
```

构建并启动：

```bash
docker compose build
docker compose up -d isaac-sim nav2
```

如果还要 RViz：

```bash
docker compose --profile gui up -d rviz
```

看日志：

```bash
docker compose logs -f isaac-sim
docker compose logs -f nav2
```

---

## 8) 这套方案的工作方式

### Isaac 容器负责

* 加载场景、机器人、传感器
* ROS 2 Bridge
* 发布 `/tf`、`/odom`、`/scan`、`/camera/*`
* 订阅 `/cmd_vel`
* 跑 cuRobo / 你的自定义 Python 逻辑

### Nav2 容器负责

* `map_server`
* `amcl`
* `planner_server`
* `controller_server`
* `bt_navigator`
* `behavior_server`
* lifecycle manager

这正符合 Isaac Sim 官方给的方向：Isaac Sim 通过 ROS 2 Bridge 与外部 ROS 系统集成；官方还有专门的 ROS2 Navigation / Multiple Robot ROS2 Navigation 教程。 ([Isaac Sim 文档][8])

---

## 9) 你还需要在 Isaac 场景里做的事

Compose 只把容器拉起来，**不会自动让 Nav2 跑起来**。
你还需要在 Isaac 场景里保证至少这些 topic/TF 是对的：

* `/clock`
* `/tf`
* `/tf_static`
* `/odom`
* `/scan` 或 `/pointcloud`
* `/cmd_vel`

最关键的是：

1. 机器人底盘的 TF 树要完整
   至少 `map -> odom -> base_link` 这一套要合理。
2. 激光雷达要真正发布到 ROS 2
   Isaac Sim 官方有 RTX Lidar 发布到 `LaserScan` / `PointCloud2` 的教程。 ([Isaac Sim 文档][9])
3. 底盘控制要消费 `cmd_vel`
   无论你是 OmniGraph 还是 Python，都要把 `/cmd_vel` 接到你的移动底盘控制器。

---

## 10) 我建议你保留的“风险清单”

### 风险 1：`usd-core==24.11`

默认不要装进 Isaac 主环境。原因上面说过。

### 风险 2：覆盖 Isaac 自带依赖

像 `numpy`、`scipy`、`pydantic` 这种大包，虽然能装，但有可能把 Isaac 内置扩展依赖带偏。
所以我建议你第一次构建后，先执行：

```bash
docker compose run --rm isaac-sim bash -lc "/isaac-sim/python.sh -c 'import omni, isaacsim, pxr; print(\"ok\")'"
```

如果这一步过了，再继续测你自己的脚本。

### 风险 3：`drake`

能装，但体量大、依赖重。你如果只是某些离线几何/规划工具要用 Drake，更推荐单独 utility 容器，而不是长期放在 Isaac 主服务里。
它本身是支持 Python `>=3.10` 的。 ([PyPI][5])

### 风险 4：跨机器网络

如果 Isaac 和 Nav2 不在同一台机器，Fast DDS 需要额外 profile；官方文档也专门给了多机通信时配置 UDP 传输的说明。 ([Isaac Sim 文档][1])

---

## 11) 我给你的最终建议

这套 compose 先这样落地最稳：

* **Isaac 容器只用内部 ROS 2 库，不装完整 ROS**
* **Nav2 容器用完整 ROS 2 Humble**
* **Python 包装进 Isaac 的 `python.sh` 环境**
* **`usd-core` 单独隔离**
* 当前默认链路不再依赖仓库内自定义 ROS plugin workspace
* 所有 Isaac 脚本都放到 `app/`

这样你能同时满足：

* Isaac Sim 4.1.0
* cuRobo
* 你这一串 Python 依赖
* Nav2 多容器运行
* 后续继续沿官方 Nav2 + Isaac Bridge 主链路扩展

下一步最有用的是，我直接把 **Nav2 需要的 `nav2_params.yaml`** 和 **Isaac 侧 `/cmd_vel + lidar + odom + tf` 的最小对接清单** 也给你补出来。

[1]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_ros.html "ROS and ROS 2 Installation — Isaac Sim Documentation"
[2]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_python.html "Python Environment Installation — Isaac Sim Documentation"
[3]: https://pypi.org/project/open3d-cpu/0.19.0/ "open3d-cpu · PyPI"
[4]: https://pypi.org/project/ray/2.54.1/ "ray · PyPI"
[5]: https://pypi.org/project/drake/1.51.1/ "drake · PyPI"
[6]: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html "Container Installation — Isaac Sim Documentation"
[7]: https://docs.nav2.org/development_guides/build_docs/index.html "Build and Install — Nav2 1.0.0 documentation"
[8]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/ros2_tutorials/ros2_landing_page.html?utm_source=chatgpt.com "ROS and ROS 2 — Isaac Sim Documentation"
[9]: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/ros2_tutorials/tutorial_ros2_rtx_lidar.html?utm_source=chatgpt.com "RTX Lidar Sensors — Isaac Sim Documentation"
