#!/bin/bash
# OpenVLA 抓取任务评估启动脚本
# 参考 LIBERO 评估方式

# 设置默认值
MODEL_FAMILY="openvla"
PRETRAINED_CHECKPOINT=""
TASK_CONFIG=""
OBS_MODE="rgb"
MAX_EPISODE=50
CENTER_CROP=True
USE_WANDB=False
WANDB_PROJECT="openvla-grasp-eval"
WANDB_ENTITY="xiongdi"
DEVICE="cuda:0"
LOAD_IN_8BIT=False
LOAD_IN_4BIT=False
RUN_ID_NOTE=""

# 显示使用说明
show_usage() {
    cat << EOF
使用方法: ./run_openvla_eval.sh [选项]

参考 LIBERO 评估方式的 OpenVLA 抓取任务评估脚本

必需参数:
  --checkpoint PATH         预训练模型路径（必需）
  --task_config PATH        任务配置文件路径（必需）

可选参数:
  --obs_mode MODE          观测模式: rgb, rgbm, nd, rgbnd (默认: rgb)
  --max_episode N          最大测试 episode 数 (默认: 50)
  --center_crop BOOL       是否使用 center crop (默认: True)
  --device DEVICE          设备 (默认: cuda:0)
  --load_in_8bit           使用 8-bit 量化
  --load_in_4bit           使用 4-bit 量化
  --use_wandb              启用 W&B 日志记录
  --wandb_project NAME     W&B 项目名 (默认: openvla-grasp-eval)
  --wandb_entity NAME      W&B entity
  --run_id_note NOTE       运行 ID 备注
  -h, --help               显示此帮助信息

示例:
  # 基础评估
  ./run_openvla_eval.sh \\
      --checkpoint /path/to/openvla-7b \\
      --task_config config_grasp.yaml \\
      --max_episode 100

  # 使用量化和 W&B
  ./run_openvla_eval.sh \\
      --checkpoint openvla/openvla-7b-finetuned-grasp \\
      --task_config config_grasp.yaml \\
      --load_in_8bit \\
      --use_wandb \\
      --wandb_project my-project \\
      --wandb_entity my-team

  # RGBM 模式评估
  ./run_openvla_eval.sh \\
      --checkpoint /path/to/checkpoint \\
      --task_config config.yaml \\
      --obs_mode rgbm \\
      --max_episode 50

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            PRETRAINED_CHECKPOINT="$2"
            shift 2
            ;;
        --task_config)
            TASK_CONFIG="$2"
            shift 2
            ;;
        --obs_mode)
            OBS_MODE="$2"
            shift 2
            ;;
        --max_episode)
            MAX_EPISODE="$2"
            shift 2
            ;;
        --center_crop)
            CENTER_CROP="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --load_in_8bit)
            LOAD_IN_8BIT=True
            shift
            ;;
        --load_in_4bit)
            LOAD_IN_4BIT=True
            shift
            ;;
        --use_wandb)
            USE_WANDB=True
            shift
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --run_id_note)
            RUN_ID_NOTE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "错误: 必须提供 --checkpoint 参数"
    show_usage
    exit 1
fi

if [ -z "$TASK_CONFIG" ]; then
    echo "错误: 必须提供 --task_config 参数"
    show_usage
    exit 1
fi

# 构建 Python 命令
CMD="python grasp_il_openvla.py \
    --model_family $MODEL_FAMILY \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --task_config $TASK_CONFIG \
    --obs_mode $OBS_MODE \
    --max_episode $MAX_EPISODE \
    --center_crop $CENTER_CROP \
    --device $DEVICE \
    --load_in_8bit $LOAD_IN_8BIT \
    --load_in_4bit $LOAD_IN_4BIT \
    --use_wandb $USE_WANDB \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY"

# 添加可选参数
if [ -n "$RUN_ID_NOTE" ]; then
    CMD="$CMD --run_id_note $RUN_ID_NOTE"
fi

# 打印命令
echo "=========================================="
echo "OpenVLA 抓取任务评估"
echo "=========================================="
echo "模型: $PRETRAINED_CHECKPOINT"
echo "任务配置: $TASK_CONFIG"
echo "观测模式: $OBS_MODE"
echo "最大 Episode: $MAX_EPISODE"
echo "设备: $DEVICE"
echo "量化: 8-bit=$LOAD_IN_8BIT, 4-bit=$LOAD_IN_4BIT"
echo "W&B: $USE_WANDB"
if [ "$USE_WANDB" = "True" ]; then
    echo "  项目: $WANDB_PROJECT"
    echo "  Entity: $WANDB_ENTITY"
fi
echo "=========================================="
echo ""
echo "执行命令:"
echo "$CMD"
echo ""
echo "=========================================="
echo ""

# 执行命令
eval $CMD

