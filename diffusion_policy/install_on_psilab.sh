#!/bin/bash
# 在 psilab 环境基础上安装 diffusion_policy 依赖
# 注意：psilab 使用 Python 3.10 + PyTorch 2.5.1，与原始 diffusion_policy 的要求不同
# 此脚本会安装兼容版本的依赖

set -e

echo "=========================================="
echo "在 psilab 环境上安装 diffusion_policy 依赖"
echo "=========================================="

# 检查是否在正确的 conda 环境中
if [[ "$CONDA_DEFAULT_ENV" != "psilab" ]]; then
    echo "[WARNING] 当前不在 psilab 环境，请先激活："
    echo "  conda activate psilab"
    echo ""
    read -p "是否继续安装？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "[INFO] 安装核心依赖..."
pip install --upgrade pip

# Hydra 配置管理（核心依赖）
pip install hydra-core==1.3.2 omegaconf

# 数据存储和处理
pip install zarr numcodecs h5py

# 深度学习工具
pip install einops accelerate diffusers

# 可视化和日志
pip install wandb tensorboardX scikit-video imageio imageio-ffmpeg

# 机器人环境（使用兼容版本）
echo ""
echo "[INFO] 安装机器人仿真环境..."
pip install pygame pymunk shapely

# gym 环境（与 gymnasium 可能有冲突，优先使用 gymnasium）
# 如果需要旧版 gym，取消下面的注释
# pip install gym==0.21.0

# dm_control（MuJoCo 环境）
pip install dm_control

# robomimic（机器人模仿学习）
pip install robomimic

# PyTorch Video
pip install pytorchvideo

# Ray（分布式训练）
pip install "ray[default,tune]"

# 图像编解码
pip install imagecodecs

# MuJoCo Python 绑定（如果需要 mujoco-py）
echo ""
echo "[INFO] 安装 MuJoCo 相关..."
pip install mujoco

# 可选：free-mujoco-py（需要系统依赖：libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf）
# pip install free-mujoco-py==2.1.6

# 可选：robosuite（需要特定版本）
# pip install robosuite

echo ""
echo "[INFO] 安装 diffusion_policy 本身（可编辑模式）..."
cd "$(dirname "$0")"
pip install -e .

# 安装 psi_dp 扩展（如果存在 setup.py）
if [ -f "psi_dp/setup.py" ]; then
    echo "[INFO] 安装 psi_dp 扩展..."
    pip install -e psi_dp/
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "测试导入："
echo "  python -c \"import diffusion_policy; print('diffusion_policy OK')\""
echo ""
echo "运行训练示例："
echo "  python train.py --config-name=train_diffusion_unet_lowdim_workspace"
echo ""

