# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Auto-generated for Diffusion Policy Testing
# Date: 2025-12-02

"""
Diffusion Policy 测试脚本

用法 (VSCode launch.json):
    {
        "name": "DP-Grasp-Lego-v1:Test",
        "type": "debugpy",
        "request": "launch",
        "args": [
            "--task", "Psi-DP-Grasp-Lego-v1",
            "--num_envs", "1",
            "--scene", "room_cfg:PSI_DC_02_CFG",
            "--enable_eval",
            "--checkpoint", "/home/psibot/diffusion_policy/data/outputs/2025.12.02/14.22.05_train_diffusion_transformer_isaaclab_grasp_lego_isaaclab/checkpoints/latest.ckpt"
        ],
        "program": "${workspaceFolder}/scripts_psi/workflows/imitation_learning/dp_play.py",
        "console": "integratedTerminal",
        "justMyCode": false
    }

命令行用法:
    cd /home/psibot/psi-lab-v2
    python scripts_psi/workflows/imitation_learning/dp_play.py \
        --task Psi-DP-Grasp-Lego-v1 \
        --num_envs 1 \
        --scene room_cfg:PSI_DC_02_CFG \
        --enable_eval \
        --checkpoint /path/to/checkpoint.ckpt
"""

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Diffusion Policy testing script for Grasp Lego task.")

parser.add_argument("--task", type=str, default="Psi-DP-Grasp-Lego-v1", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")

# Psi Lab arguments
parser.add_argument("--enable_wandb", action="store_true", default=False, help="Whether update data to wandb or not.")
parser.add_argument("--enable_json", action="store_true", default=False, help="Whether create scene from json or not.")
parser.add_argument("--scene", type=str, default="room_cfg:PSI_DC_02_CFG", help="Scene.")
parser.add_argument("--json_file", type=str, default=None, help="Scene json file.")
parser.add_argument("--enable_output", action="store_true", default=False, help="Whether output data to hdf5 files or not.")
parser.add_argument("--enable_eval", action="store_true", default=False, help="Enable print evalutation result or not.")
parser.add_argument("--output_folder", type=str, default=None, help="Hdf5 files folder.")
parser.add_argument("--sample_step", type=int, default=1, help="Simulation steps per sample step") 
parser.add_argument("--async_reset", action="store_true", default=False, help="Whether reset envs asynchronous.")
parser.add_argument("--enable_random", action="store_true", default=False, help="Whether enable random when envs reset.")
parser.add_argument("--enable_marker", action="store_true", default=False, help="Whether show marker or not.")
parser.add_argument("--checkpoint", type=str, 
                    default="/home/psibot/diffusion_policy/data/outputs/2025.12.02/14.22.05_train_diffusion_transformer_isaaclab_grasp_lego_isaaclab/checkpoints/latest.ckpt",
                    help="The checkpoint to load and predict") 
parser.add_argument("--max_step", type=int, default=None, help="The max step to reset") 
parser.add_argument("--max_episode", type=int, default=100, help="The max episode to run") 


""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# store args before create app as it will pop some arg from args_cli
enable_cameras = args_cli.enable_cameras

# launch omniverse app
app_launcher = AppLauncher(args_cli)


""" Common Modules """ 
import sys
import torch
import gymnasium as gym

# 添加 diffusion_policy 到 path
sys.path.insert(0, "/home/psibot/diffusion_policy")

""" Isaac Lab Modules """ 
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401

""" Psi Lab Modules """ 
import psilab_tasks
from psilab.utils.timer_utils import Timer
from psilab_tasks.utils import parse_scene_cfg, parse_il_env_cfg


# parse arguments for isaac lab env config
env_cfg = parse_env_cfg(
    args_cli.task, 
    device=args_cli.device,
    num_envs=args_cli.num_envs,
)

# parse arguments for psi lab il env config
env_cfg = parse_il_env_cfg(
    env_cfg,
    args_cli.seed,
    args_cli.enable_wandb,
    args_cli.enable_output,
    args_cli.output_folder,
    args_cli.sample_step,
    args_cli.async_reset,
    args_cli.enable_random,
    args_cli.enable_marker,
    args_cli.enable_eval,
    args_cli.checkpoint,
    args_cli.max_step,
    args_cli.max_episode,
    args_cli.task,
    args_cli.scene
)

# parse arguments for psi lab scene config
env_cfg.scene = parse_scene_cfg(
    args_cli.task, 
    args_cli.enable_json,
    args_cli.scene,
    args_cli.json_file,
    args_cli.num_envs,
)

# clear camera configs in scene while "enable_cameras" flag is False
if enable_cameras is False:
    env_cfg.scene.cameras_cfg = {}
    env_cfg.scene.tiled_cameras_cfg = {}
    for robot_cfg in env_cfg.scene.robots_cfg.values():
        robot_cfg.cameras = {}  # type: ignore
        robot_cfg.tiled_cameras = {}  # type: ignore

# clear random configs while "enable_random" flag is false
if not env_cfg.enable_random:
    env_cfg.scene.random = None

# clear marker configs while "enable_marker" flag is false
if not env_cfg.enable_marker:
    env_cfg.scene.marker_cfg = None

# get timer
timer = Timer() 

print("\n" + "=" * 60)
print("Diffusion Policy Test")
print("=" * 60)
print(f"Task: {args_cli.task}")
print(f"Checkpoint: {args_cli.checkpoint}")
print(f"Num envs: {args_cli.num_envs}")
print(f"Max episodes: {args_cli.max_episode}")
print("=" * 60 + "\n")

# create env      
env = gym.make(args_cli.task, cfg=env_cfg)

# reset env
env.reset()

# main loop
print("Test started! Press Ctrl+C to stop.\n")

try:
    while True:
        env.step(torch.zeros(1))
        # break loop if reach max_episode
        if env.env.episode_num >= env_cfg.max_episode:  # type: ignore
            break
except KeyboardInterrupt:
    print("\nTest interrupted by user.")
    
# result log
print("\n" + "#" * 17 + " Statistics " + "#" * 17)

record_time = timer.run_time() / 60.0

print(f"运行时长: {record_time:.2f} 分钟")
print(f"运行次数: {env.env.episode_num} 次")  # type: ignore
print(f"成功次数: {env.env.episode_success_num} 次")  # type: ignore

if env.env.episode_num > 0:  # type: ignore
    success_rate = env.env.episode_success_num / env.env.episode_num * 100  # type: ignore
    print(f"成功率: {success_rate:.1f}%")

if env_cfg.enable_output and record_time > 0:
    record_rate = env.env.episode_success_num / record_time  # type: ignore
    print(f"采集效率: {record_rate:.2f} 条/分钟")

print("#" * 17 + " Statistics " + "#" * 17 + "\n")

# close app
app_launcher.app.close()

