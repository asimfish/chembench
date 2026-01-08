#!/bin/bash
# OpenVLA 抓取评估启动器
# 使用 Isaac Lab 的 Python 环境运行 OpenVLA 评估脚本

# Isaac Lab 路径
ISAACLAB_PATH="/home/psibot/chembench/psilab/source/isaaclab"

# 脚本路径
SCRIPT_PATH="/home/psibot/chembench/psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp/grasp_il_openvla.py"

# 使用 Isaac Lab 的 Python 环境运行脚本
cd $ISAACLAB_PATH
./isaaclab.sh -p $SCRIPT_PATH "$@"

