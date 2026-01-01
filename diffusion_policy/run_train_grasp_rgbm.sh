#!/bin/bash
# ============================================
# DP:Train-Grasp-RGBM 训练脚本
# 对应 launch.json 中的 "DP:Train-Grasp-RGBM" 配置
# 
# 使用方法: 
#   bash run_train_grasp_rgbm.sh [前台/后台]
# 
# 示例:
#   bash run_train_grasp_rgbm.sh           # 前台运行（默认）
#   bash run_train_grasp_rgbm.sh fg        # 前台运行
#   bash run_train_grasp_rgbm.sh bg        # 后台运行
# ============================================

set -e  # 遇到错误立即退出

# ============================================
# 配置参数（对应 launch.json）
# ============================================
GPU_ID=7
CONFIG_NAME="train_diffusion_transformer_rgbm_workspace"
TASK="task=grasp_rgbm"
MODE=${1:-"fg"}  # 默认前台运行

# ============================================
# 工作目录
# ============================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================
# 激活conda环境
# ============================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate psilab

# ============================================
# 显示配置
# ============================================
echo "============================================"
echo "DP:Train-Grasp-RGBM 训练"
echo "============================================"
echo "GPU ID:      $GPU_ID"
echo "配置文件:    $CONFIG_NAME"
echo "任务:        $TASK"
echo "运行模式:    $MODE"
echo "============================================"
echo ""

# ============================================
# 启动训练
# ============================================
if [ "$MODE" == "bg" ]; then
    # 后台运行
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="logs/train_grasp_rgbm_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/gpu${GPU_ID}.log"
    
    echo "后台运行模式"
    echo "日志文件:    $LOG_FILE"
    echo ""
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python train.py \
        --config-name "$CONFIG_NAME" \
        $TASK \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "$PID" > "$LOG_DIR/pid.txt"
    
    echo "训练已启动！"
    echo "  PID: $PID"
    echo "  日志: $LOG_FILE"
    echo ""
    echo "============================================"
    echo "常用命令:"
    echo "============================================"
    echo ""
    echo "  # 实时查看日志"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "  # 查看GPU使用情况"
    echo "  nvidia-smi"
    echo ""
    echo "  # 停止训练"
    echo "  kill $PID"
    echo ""
    echo "============================================"
else
    # 前台运行
    echo "前台运行模式（按 Ctrl+C 停止训练）"
    echo ""
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --config-name "$CONFIG_NAME" \
        $TASK
fi

