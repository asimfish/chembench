#!/usr/bin/env python3
"""
修复 Grasp 数据转换问题
将 Grasp 数据强制转换为单手任务（移除左手数据）
"""

import zarr
import numpy as np
import os
from pathlib import Path

def convert_bimanual_to_single_hand(zarr_path: str, output_path: str = None):
    """
    将包含双手数据的 Zarr 文件转换为只包含右手数据的版本
    
    Args:
        zarr_path: 输入 Zarr 文件路径
        output_path: 输出 Zarr 文件路径（默认为 zarr_path + "_single_hand"）
    """
    if output_path is None:
        zarr_path_obj = Path(zarr_path)
        output_path = str(zarr_path_obj.parent / (zarr_path_obj.stem + "_single_hand.zarr"))
    
    print(f"正在转换: {zarr_path}")
    print(f"输出到: {output_path}")
    
    # 打开原始 Zarr
    src_root = zarr.open(zarr_path, mode='r')
    
    # 创建新 Zarr
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    dst_root = zarr.open(output_path, mode='w')
    
    # 复制 meta 组
    meta_src = src_root['meta']
    meta_dst = dst_root.create_group('meta')
    meta_dst.create_dataset('episode_ends', data=meta_src['episode_ends'][:])
    
    # 创建 data 组
    data_src = src_root['data']
    data_dst = dst_root.create_group('data')
    
    # 检查 action 维度
    action_data = data_src['action'][:]
    print(f"\n原始 action 形状: {action_data.shape}")
    
    if action_data.shape[1] == 26:
        print("  检测到双手 action (26维)，转换为单手 (13维)")
        # 只保留右手: arm2(7) + hand2(6)
        action_single = action_data[:, :13]
        data_dst.create_dataset('action', data=action_single)
        print(f"  新 action 形状: {action_single.shape}")
    else:
        print(f"  action 已是单手 ({action_data.shape[1]}维)，直接复制")
        data_dst.create_dataset('action', data=action_data)
    
    # 需要复制的键（右手 + 通用数据）
    keys_to_copy = [
        # 右手状态
        'arm2_pos', 'arm2_vel', 'hand2_pos', 'hand2_vel',
        'arm2_eef_pos', 'arm2_eef_quat',
        # 物体位姿
        'target_pose',
        # 相机图像
        'chest_camera_rgb', 'head_camera_rgb', 'third_camera_rgb',
        # 可选数据
        'chest_camera_mask', 'head_camera_mask', 'third_camera_mask',
        'chest_camera_depth', 'head_camera_depth', 'third_camera_depth',
        'chest_camera_normals', 'head_camera_normals', 'third_camera_normals',
        'chest_camera_pointcloud', 'head_camera_pointcloud', 'third_camera_pointcloud',
        # 时间戳
        'timestamps'
    ]
    
    # 需要跳过的键（左手数据）
    keys_to_skip = [
        'arm1_pos', 'arm1_vel', 'hand1_pos', 'hand1_vel',
        'arm1_eef_pos', 'arm1_eef_quat'
    ]
    
    print("\n复制数据:")
    for key in data_src.keys():
        if key == 'action':
            continue  # 已处理
        
        if key in keys_to_skip:
            print(f"  跳过 (左手数据): {key}")
            continue
        
        if key in keys_to_copy or key in data_src:
            try:
                data = data_src[key][:]
                data_dst.create_dataset(key, data=data)
                print(f"  ✓ {key}: {data.shape}")
            except Exception as e:
                print(f"  ✗ {key}: 复制失败 - {e}")
    
    print(f"\n✅ 转换完成！")
    print(f"   输出文件: {output_path}")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python fix_grasp_zarr_conversion.py <zarr_path> [output_path]")
        print("\n示例:")
        print("  python fix_grasp_zarr_conversion.py /path/to/grasp_data.zarr")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_bimanual_to_single_hand(zarr_path, output_path)

