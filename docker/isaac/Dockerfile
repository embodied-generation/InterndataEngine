ARG ISAAC_SIM_VERSION=4.1.0
ARG BASE_DIST=ubuntu22.04
ARG CUDA_VERSION=11.8.0
ARG TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9

FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION} AS isaac-sim-base
FROM nvidia/cuda:${CUDA_VERSION}-devel-${BASE_DIST} AS curobo-builder

ARG TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ISAAC_SIM_PATH=/isaac-sim
ENV CUROBO_PATH=/opt/curobo
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/isaac-sim/kit/python/bin:${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

SHELL ["/bin/bash", "-lc"]

RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|https://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g; s|https://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list; \
    fi && \
    find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) -print0 | xargs -0 -r sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' \
      -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g'

RUN mkdir -p /root/.pip && \
    cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF


RUN apt-get update && apt-get install -y \
    git curl wget ca-certificates \
    build-essential cmake pkg-config \
    libatomic1 libegl1 libglu1-mesa libgomp1 libxi6 libxrandr2 libxt6 \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    libfreetype-dev libfontconfig1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=isaac-sim-base /isaac-sim /isaac-sim
COPY --from=isaac-sim-base /root/.nvidia-omniverse/config /root/.nvidia-omniverse/config
COPY --from=isaac-sim-base /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY --from=isaac-sim-base /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

COPY docker/isaac/requirements.isaac.txt /tmp/requirements.isaac.txt
COPY InternDataAssets/curobo ${CUROBO_PATH}

RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.isaac.txt

RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
      pybind11 \
      yourdfpy \
      warp-lang \
      tqdm \
      importlib_resources \
      tomli \
      wheel \
      ninja && \
    nvcc --version && \
    cd "${CUROBO_PATH}" && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0 /isaac-sim/python.sh setup.py build_ext --inplace

FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ISAAC_SIM_PATH=/isaac-sim
ENV CUROBO_PATH=/opt/curobo
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

SHELL ["/bin/bash", "-lc"]

RUN mkdir -p /root/.pip && \
    cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

COPY docker/isaac/requirements.isaac.txt /tmp/requirements.isaac.txt
COPY --from=curobo-builder ${CUROBO_PATH} ${CUROBO_PATH}

RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.isaac.txt && \
    /isaac-sim/python.sh -m pip install --no-cache-dir \
      pybind11 \
      yourdfpy \
      warp-lang \
      tqdm \
      importlib_resources

COPY docker/isaac/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
