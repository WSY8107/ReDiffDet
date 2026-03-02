# 1. 基础环境
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /workspace

# 2. 系统依赖
RUN apt-get update && apt-get install -y \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 安装框架 (Step 4 & 5)
RUN pip install --no-cache-dir -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mim install "mmcv==2.2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mim install "mmdet>=3.0.0rc6,<3.4.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mim install "mmsegmentation>=1.2.2" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 自动修复版本断言 (直接搜索路径，避免 Python 导入触发报错)
RUN MMDET_PATH=$(pip show mmdet | grep Location | awk '{print $2}')/mmdet && \
    sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/g" $MMDET_PATH/__init__.py

RUN MMSEG_PATH=$(pip show mmsegmentation | grep Location | awk '{print $2}')/mmseg && \
    if [ -f $MMSEG_PATH/__init__.py ]; then \
        sed -i "s/mmcv_max_version = '2.2.0'/mmcv_max_version = '2.3.0'/g" $MMSEG_PATH/__init__.py || \
        sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/g" $MMSEG_PATH/__init__.py; \
    fi