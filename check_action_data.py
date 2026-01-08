#!/usr/bin/env python3
"""
检查action数据的详细内容
"""

import zarr
import numpy as np

def check_action_data(zarr_path):
    print("=" * 80)
    print("检查 Action 数据")
    print("=" * 80)
    
    root = zarr.open(zarr_path, mode='r')
    action = root['data']['action'][:]
    
    print(f"\n【基本信息】")
    print(f"形状: {action.shape}")
    print(f"数据类型: {action.dtype}")
    print(f"总帧数: {action.shape[0]}")
    print(f"Action维度: {action.shape[1]}")
    
    print(f"\n【数值范围统计】")
    for i in range(action.shape[1]):
        col = action[:, i]
        print(f"维度 {i:2d}: min={np.min(col):8.4f}, max={np.max(col):8.4f}, "
              f"mean={np.mean(col):8.4f}, std={np.std(col):7.4f}")
    
    print(f"\n【前5帧数据】")
    for i in range(min(5, len(action))):
        print(f"帧 {i}: {action[i]}")
    
    print(f"\n【中间5帧数据（帧 {len(action)//2-2} 到 {len(action)//2+2}）】")
    mid = len(action) // 2
    for i in range(max(0, mid-2), min(len(action), mid+3)):
        print(f"帧 {i}: {action[i]}")
    
    print(f"\n【最后5帧数据】")
    for i in range(max(0, len(action)-5), len(action)):
        print(f"帧 {i}: {action[i]}")
    
    # 检查是否有异常值
    print(f"\n【异常值检查】")
    nan_count = np.isnan(action).sum()
    inf_count = np.isinf(action).sum()
    print(f"NaN值数量: {nan_count}")
    print(f"Inf值数量: {inf_count}")
    
    # 检查变化率
    print(f"\n【变化率分析】")
    if len(action) > 1:
        diffs = np.diff(action, axis=0)
        for i in range(action.shape[1]):
            col_diff = diffs[:, i]
            print(f"维度 {i:2d} 变化率: min={np.min(col_diff):8.4f}, max={np.max(col_diff):8.4f}, "
                  f"mean={np.mean(np.abs(col_diff)):8.4f}")
    
    # 检查是否有维度是常量
    print(f"\n【常量维度检查】")
    for i in range(action.shape[1]):
        col = action[:, i]
        if np.std(col) < 1e-6:
            print(f"⚠️  维度 {i} 几乎是常量: {np.mean(col):.6f}")
        else:
            unique_values = len(np.unique(col))
            print(f"✓ 维度 {i} 有变化，唯一值数量: {unique_values}")
    
    # 与其他状态数据对比
    print(f"\n【与机器人状态对比】")
    if 'arm2_pos' in root['data']:
        arm_pos = root['data']['arm2_pos'][:]
        print(f"arm2_pos 形状: {arm_pos.shape}")
        print(f"arm2_pos 前3帧:")
        for i in range(min(3, len(arm_pos))):
            print(f"  {arm_pos[i]}")
    
    if 'hand2_pos' in root['data']:
        hand_pos = root['data']['hand2_pos'][:]
        print(f"\nhand2_pos 形状: {hand_pos.shape}")
        print(f"hand2_pos 前3帧:")
        for i in range(min(3, len(hand_pos))):
            print(f"  {hand_pos[i]}")
    
    # 检查action是否可能包含arm+hand
    if action.shape[1] == 13:
        print(f"\n【Action组成分析（假设13维 = 7关节 + 6手指）】")
        print(f"可能的组成:")
        print(f"  - 前7维: 机器人关节位置")
        print(f"  - 后6维: 手指/夹爪位置")
        
        arm_part = action[:, :7]
        hand_part = action[:, 7:]
        
        print(f"\n前7维（疑似关节）统计:")
        for i in range(7):
            col = arm_part[:, i]
            print(f"  关节 {i}: [{np.min(col):7.4f}, {np.max(col):7.4f}]")
        
        print(f"\n后6维（疑似手指）统计:")
        for i in range(6):
            col = hand_part[:, i]
            print(f"  手指 {i}: [{np.min(col):7.4f}, {np.max(col):7.4f}]")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    zarr_path = "/home/psibot/chembench/data/zarr_point_cloud/motion_plan/grasp/100ml玻璃烧杯.zarr"
    check_action_data(zarr_path)




