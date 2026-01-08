#!/usr/bin/env python3
"""
调试关节配置问题
检查机器人关节数量、索引等信息
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "psilab/source")))

def debug_robot_joints():
    """调试机器人关节配置"""
    print("="*80)
    print("机器人关节配置调试")
    print("="*80)
    
    try:
        # 导入必要的模块
        from psilab_tasks.imitation_learning.grasp.scenes.room_cfg import GraspBottleSceneCfg
        from psilab_tasks.imitation_learning.grasp.grasp_il_act import GraspBottleEnvCfg, GraspBottleEnv
        
        # 创建配置
        cfg = GraspBottleEnvCfg()
        cfg.scene = GraspBottleSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=False)
        
        # 设置检查点路径（假设存在）
        checkpoint_path = "/home/psibot/chembench/act/test_act/grasp_beaker_kl2_chunk8_bs64_lr1e5_hd512_20260107_215647"
        cfg.checkpoint = checkpoint_path
        
        print(f"\n创建环境...")
        env = GraspBottleEnv(cfg, render_mode=None)
        
        robot = env._robot
        
        print(f"\n{'='*80}")
        print(f"机器人信息")
        print(f"{'='*80}")
        print(f"机器人名称: {robot.cfg.prim_path}")
        print(f"总关节数: {robot.num_joints}")
        print(f"总身体数: {robot.num_bodies}")
        
        print(f"\n{'='*80}")
        print(f"关节名称列表")
        print(f"{'='*80}")
        for i, name in enumerate(robot.joint_names):
            print(f"  [{i:2d}] {name}")
        
        print(f"\n{'='*80}")
        print(f"执行器配置")
        print(f"{'='*80}")
        for actuator_name, actuator in robot.actuators.items():
            print(f"\n执行器: {actuator_name}")
            print(f"  类型: {type(actuator).__name__}")
            print(f"  关节索引: {actuator.joint_indices}")
            print(f"  关节数量: {len(actuator.joint_indices)}")
            print(f"  关节名称: {actuator.joint_names}")
        
        print(f"\n{'='*80}")
        print(f"关节限制")
        print(f"{'='*80}")
        joint_limits = robot.data.joint_limits[0]  # 第一个环境
        print(f"  形状: {joint_limits.shape}")
        for i, (lower, upper) in enumerate(joint_limits):
            if i < len(robot.joint_names):
                print(f"  [{i:2d}] {robot.joint_names[i]:<30s}: [{lower:8.4f}, {upper:8.4f}]")
        
        print(f"\n{'='*80}")
        print(f"内部缓冲区")
        print(f"{'='*80}")
        print(f"  joint_pos_target 形状: {robot.data.joint_pos_target.shape}")
        print(f"  joint_vel_target 形状: {robot.data.joint_vel_target.shape}")
        print(f"  joint_effort_target 形状: {robot.data.joint_effort_target.shape}")
        print(f"  _joint_effort_target_sim 形状: {robot._joint_effort_target_sim.shape}")
        
        print(f"\n{'='*80}")
        print(f"检查关节目标设置")
        print(f"{'='*80}")
        
        # 尝试设置关节目标
        arm_indices = robot.actuators["arm2"].joint_indices
        hand_indices = robot.actuators["hand2"].joint_indices[:6]
        
        print(f"  arm2 索引: {arm_indices}")
        print(f"  hand2 索引: {hand_indices[:6]}")
        
        # 创建测试目标
        test_arm_target = torch.zeros((1, len(arm_indices)), device=robot.device)
        test_hand_target = torch.zeros((1, len(hand_indices)), device=robot.device)
        
        print(f"\n  设置 arm2 目标 (形状: {test_arm_target.shape})...")
        robot.set_joint_position_target(test_arm_target, arm_indices)
        
        print(f"  设置 hand2 目标 (形状: {test_hand_target.shape})...")
        robot.set_joint_position_target(test_hand_target, hand_indices)
        
        print(f"\n  ✓ 关节目标设置成功")
        
        print(f"\n{'='*80}")
        print(f"检查 write_data_to_sim")
        print(f"{'='*80}")
        
        try:
            print(f"  尝试调用 write_data_to_sim...")
            robot.write_data_to_sim()
            print(f"  ✓ write_data_to_sim 成功")
        except Exception as e:
            print(f"  ✗ write_data_to_sim 失败:")
            print(f"    错误: {e}")
            
            # 打印更多调试信息
            print(f"\n  调试信息:")
            print(f"    _joint_effort_target_sim 值:")
            print(f"      形状: {robot._joint_effort_target_sim.shape}")
            print(f"      数据类型: {robot._joint_effort_target_sim.dtype}")
            print(f"      设备: {robot._joint_effort_target_sim.device}")
            print(f"      是否包含 NaN: {torch.isnan(robot._joint_effort_target_sim).any()}")
            print(f"      是否包含 Inf: {torch.isinf(robot._joint_effort_target_sim).any()}")
            print(f"      最小值: {robot._joint_effort_target_sim.min()}")
            print(f"      最大值: {robot._joint_effort_target_sim.max()}")
            
            # 检查 PhysX view
            print(f"\n    PhysX View 信息:")
            print(f"      view 对象: {robot.root_physx_view}")
            print(f"      _ALL_INDICES: {robot._ALL_INDICES}")
            
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"调试完成")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_robot_joints()

