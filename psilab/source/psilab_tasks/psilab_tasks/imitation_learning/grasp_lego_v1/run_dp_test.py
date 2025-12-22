#!/usr/bin/env python3
"""
运行 Diffusion Policy 测试

用法:
    # 在 Isaac Lab 环境中运行
    cd /home/psibot/psi-lab-v2
    ./isaaclab.sh -p source/psilab_tasks/psilab_tasks/imitation_learning/grasp_lego_v1/run_dp_test.py
    
    # 或者指定 checkpoint
    ./isaaclab.sh -p source/psilab_tasks/psilab_tasks/imitation_learning/grasp_lego_v1/run_dp_test.py \
        --checkpoint /path/to/checkpoint.ckpt
"""

import argparse
import sys
import os

# 添加 diffusion_policy 到 path
sys.path.insert(0, "/home/psibot/diffusion_policy")


def main():
    parser = argparse.ArgumentParser(description="Run Diffusion Policy test in Isaac Lab")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="/home/psibot/diffusion_policy/data/outputs/2025.12.02/14.22.05_train_diffusion_transformer_isaaclab_grasp_lego_isaaclab/checkpoints/latest.ckpt",
        help="Path to the trained checkpoint"
    )
    parser.add_argument(
        "--num_envs", "-n",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no rendering)"
    )
    
    args = parser.parse_args()
    
    # 检查 checkpoint 是否存在
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    print("="*60)
    print("Diffusion Policy Isaac Lab Test")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num envs: {args.num_envs}")
    print(f"Headless: {args.headless}")
    print("="*60)
    
    # 导入环境配置
    from psilab_tasks.imitation_learning.grasp_lego_v1.scenes.room_cfg import PSI_DC_02_CFG
    from psilab_tasks.imitation_learning.grasp_lego_v1.grasp_lego_dp_env import (
        GraspLegoDPEnvCfg, 
        GraspLegoDPEnv
    )
    
    # 更新场景配置
    scene_cfg = PSI_DC_02_CFG.replace(num_envs=args.num_envs)
    
    # 创建环境配置
    env_cfg = GraspLegoDPEnvCfg(
        scene=scene_cfg,
        checkpoint=args.checkpoint,
        enable_eval=True,
        enable_log=True,
        enable_output=False,
    )
    
    # 如果 headless 模式，修改渲染设置
    if args.headless:
        env_cfg.sim.render_interval = 0
    
    # 创建环境
    print("\nInitializing environment...")
    env = GraspLegoDPEnv(env_cfg, render_mode="rgb_array" if args.headless else None)
    
    # 重置环境
    env.reset()
    
    print("\n" + "="*60)
    print("Test started! Press Ctrl+C to stop.")
    print("="*60 + "\n")
    
    # 运行测试
    step_count = 0
    try:
        while True:
            env.step(None)
            step_count += 1
            if step_count % 1000 == 0:
                print(f"Step: {step_count}")
    except KeyboardInterrupt:
        print(f"\n\nTest stopped after {step_count} steps")
    finally:
        env.close()
        
    print("\n" + "="*60)
    print("Final Results:")
    print(f"  Total episodes: {env._episode_num}")
    print(f"  Successful episodes: {env._episode_success_num}")
    if env._episode_num > 0:
        print(f"  Success rate: {env._episode_success_num / env._episode_num * 100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()

