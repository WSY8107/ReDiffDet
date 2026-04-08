#!/bin/bash

# --- 1. 确定工作根目录 ---
# 我们统一使用 /mnt/workspace 作为代码存放地
WORK_DIR="/mnt/workspace"
mkdir -p $WORK_DIR
cd $WORK_DIR

echo "=== [1/5] GitHub 代码同步 ==="
PROJECT_NAME="ReDiffDet"
# 替换为你的真实 GitHub 地址，建议带上 Token 防止交互式报错
GIT_URL="https://github.com/WSY8107/ReDiffDet.git" 

if [ -d "$PROJECT_NAME/.git" ]; then
    echo "检测到已有代码库，正在更新..."
    cd $PROJECT_NAME
    git fetch --all && git reset --hard origin/$(git branch --show-current)
    git pull
else
    echo "代码库不存在，正在克隆..."
    # 这里的 --depth 1 可以加快克隆速度
    git clone --depth 1 $GIT_URL
    cd $PROJECT_NAME
fi

# 此时我们已经在项目根目录了
CODE_DIR=$(pwd)
echo "[OK] 当前工作目录: $CODE_DIR"

# --- 2. 标注文件本地化 (解决 I/O 瓶颈) ---
echo "=== [2/5] 同步标注到本地 SSD ==="
LOCAL_ANN_DIR="/root/data/RSAR/train/annfiles"
mkdir -p $LOCAL_ANN_DIR
# 确保环境变量 AK/SK 已配置
ossutil cp -rf oss://rediffdet/data/RSAR/train/annfiles/ $LOCAL_ANN_DIR/ --parallel=50 --sign-version v4

# --- 3. 路径映射与环境刷新 ---
echo "=== [3/5] 建立路径映射与预热 ==="
OSS_DIR="/mnt/rediffdet"
mkdir -p data checkpoints
ln -snf $OSS_DIR/checkpoint/resnet50.pth ./checkpoints/resnet50.pth
ln -snf $OSS_DIR/data/RSAR ./data/RSAR
# 强制刷新 OSS 挂载
ls -R ./data/RSAR | head -n 5 > /dev/null

# --- 4. 准备输出路径 ---
TIMESTAMP=$(date +%Y%m%d_%H%M)
OSS_WORK_DIR="$OSS_DIR/work_dirs/rsar_$TIMESTAMP"
mkdir -p $OSS_WORK_DIR
echo "=== [4/5] 准备输出路径: $OSS_WORK_DIR ==="

# --- 5. 启动分布式训练 ---
echo "=== [5/5] 启动分布式训练 (2卡 P100/T4) ==="
export PYTHONPATH=$PYTHONPATH:$CODE_DIR
pkill -9 python || true

# 检查关键文件是否存在
if [ ! -f "tools/train.py" ]; then
    echo "[FATAL] 在 $(pwd) 下找不到 tools/train.py！"
    ls -F
    exit 1
fi

torchrun --nproc_per_node=2 \
    tools/train.py \
    projects/GSDet_baseline/configs/GSDet_r50_b900_h2h4_h2r1_r2r1_1x_rsar.py \
    --launcher pytorch \
    --work-dir $OSS_WORK_DIR \
    --cfg-options \
    train_dataloader.dataset.ann_file=$LOCAL_ANN_DIR \
    train_dataloader.dataset.serialize_data=False \
    train_dataloader.dataset.lazy_init=True \
    train_dataloader.batch_size=8 \
    train_dataloader.num_workers=4 \
    train_dataloader.persistent_workers=True