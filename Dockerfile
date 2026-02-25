# 1. 使用 PyTorch 官方提供的镜像，同样是 2.4.0, CUDA 12.1
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /workspace

# 2. 合并仓库和 PAI 所需的系统依赖
RUN apt-get update && apt-get install -y \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 安装核心组件
RUN pip install --no-cache-dir -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mim install "mmcv==2.2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mim install "mmdet>=3.0.0,<3.4.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN mim install "mmsegmentation>=1.2.2" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 解决版本 AssertionError
RUN MMDET_PATH=$(python -c "import mmdet; import os; print(os.path.dirname(mmdet.__file__))") && \
    sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/g" $MMDET_PATH/__init__.py

RUN MMSEG_PATH=$(python -c "import mmseg; import os; print(os.path.dirname(mmseg.__file__))") && \
    sed -i "s/mmcv_max_version = '2.2.0'/mmcv_max_version = '2.3.0'/g" $MMSEG_PATH/__init__.py \
    || sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/g" $MMSEG_PATH/__init__.py

