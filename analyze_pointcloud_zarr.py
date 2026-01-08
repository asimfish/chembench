#!/usr/bin/env python3
"""
分析点云zarr数据集
"""

import zarr
import numpy as np
from pathlib import Path

def analyze_zarr_dataset(zarr_path):
    """分析zarr数据集结构和内容"""
    print("=" * 80)
    print(f"分析 Zarr 数据集: {zarr_path}")
    print("=" * 80)
    
    # 打开zarr数据集
    root = zarr.open(zarr_path, mode='r')
    
    # 分析meta信息
    print("\n【1. Meta信息】")
    print("-" * 80)
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    print(f"Episode ends: {episode_ends}")
    print(f"总轨迹数: {len(episode_ends)}")
    if len(episode_ends) > 0:
        print(f"总时间步数: {episode_ends[-1]}")
        episode_lengths = np.diff([0] + list(episode_ends))
        print(f"每条轨迹长度: {episode_lengths}")
        print(f"平均轨迹长度: {np.mean(episode_lengths):.1f}")
        print(f"最短轨迹: {np.min(episode_lengths)}")
        print(f"最长轨迹: {np.max(episode_lengths)}")
    
    # 分析data内容
    print("\n【2. Data内容】")
    print("-" * 80)
    data = root['data']
    
    data_info = {}
    for key in sorted(data.keys()):
        dataset = data[key]
        data_info[key] = {
            'shape': dataset.shape,
            'dtype': dataset.dtype,
            'chunks': dataset.chunks,
            'size_mb': dataset.nbytes / (1024**2),
        }
        
        print(f"\n{key}:")
        print(f"  形状: {dataset.shape}")
        print(f"  类型: {dataset.dtype}")
        print(f"  分块: {dataset.chunks}")
        print(f"  大小: {dataset.nbytes / (1024**2):.2f} MB")
        
        # 显示数据范围（采样部分数据）
        if len(dataset) > 0:
            try:
                # 采样第一个和最后一个数据点
                if len(dataset.shape) == 2:
                    sample = dataset[0]
                    print(f"  第一帧: min={np.min(sample):.4f}, max={np.max(sample):.4f}, mean={np.mean(sample):.4f}")
                elif len(dataset.shape) == 3:
                    sample = dataset[0]
                    print(f"  第一帧: shape={sample.shape}, min={np.min(sample):.4f}, max={np.max(sample):.4f}")
                elif len(dataset.shape) == 4:
                    sample = dataset[0]
                    print(f"  第一帧: shape={sample.shape}, dtype={sample.dtype}")
                    if 'pointcloud' in key.lower():
                        # 点云数据特殊处理
                        print(f"    点数: {sample.shape[0]}")
                        if sample.shape[-1] >= 3:
                            xyz = sample[..., :3]
                            print(f"    XYZ范围: x=[{np.min(xyz[...,0]):.3f}, {np.max(xyz[...,0]):.3f}], "
                                  f"y=[{np.min(xyz[...,1]):.3f}, {np.max(xyz[...,1]):.3f}], "
                                  f"z=[{np.min(xyz[...,2]):.3f}, {np.max(xyz[...,2]):.3f}]")
            except Exception as e:
                print(f"  无法采样数据: {e}")
    
    # 分析点云数据
    print("\n【3. 点云数据详细分析】")
    print("-" * 80)
    
    pointcloud_keys = [k for k in data.keys() if 'pointcloud' in k.lower()]
    for key in pointcloud_keys:
        print(f"\n{key}:")
        dataset = data[key]
        print(f"  总形状: {dataset.shape}")
        
        if len(dataset) > 0:
            # 分析第一个和中间的点云
            indices = [0, len(dataset) // 2, -1] if len(dataset) > 2 else [0]
            for idx in indices:
                try:
                    pc = dataset[idx]
                    print(f"\n  时间步 {idx}:")
                    print(f"    形状: {pc.shape}")
                    
                    if len(pc.shape) >= 2 and pc.shape[-1] >= 3:
                        xyz = pc[..., :3].reshape(-1, 3)
                        # 移除无效点
                        valid_mask = ~np.any(np.isnan(xyz), axis=1) & ~np.any(np.isinf(xyz), axis=1)
                        valid_xyz = xyz[valid_mask]
                        
                        print(f"    总点数: {len(xyz)}")
                        print(f"    有效点数: {len(valid_xyz)}")
                        if len(valid_xyz) > 0:
                            print(f"    X范围: [{np.min(valid_xyz[:,0]):.3f}, {np.max(valid_xyz[:,0]):.3f}]")
                            print(f"    Y范围: [{np.min(valid_xyz[:,1]):.3f}, {np.max(valid_xyz[:,1]):.3f}]")
                            print(f"    Z范围: [{np.min(valid_xyz[:,2]):.3f}, {np.max(valid_xyz[:,2]):.3f}]")
                            print(f"    中心: ({np.mean(valid_xyz[:,0]):.3f}, {np.mean(valid_xyz[:,1]):.3f}, {np.mean(valid_xyz[:,2]):.3f})")
                            
                            # 检查是否有额外的特征
                            if pc.shape[-1] > 3:
                                print(f"    额外特征维度: {pc.shape[-1] - 3}")
                                for i in range(3, pc.shape[-1]):
                                    feat = pc[..., i].flatten()
                                    print(f"      特征{i-3}: [{np.min(feat):.3f}, {np.max(feat):.3f}]")
                except Exception as e:
                    print(f"    读取失败: {e}")
    
    # 分析ground_truth_pointcloud
    if 'ground_truth_pointcloud' in data:
        print("\n【4. Ground Truth Pointcloud】")
        print("-" * 80)
        gt_pc = data['ground_truth_pointcloud']
        print(f"形状: {gt_pc.shape}")
        print(f"这是物体的真值点云轨迹")
        
        if len(gt_pc) > 0:
            # 分析几个关键帧
            print("\n关键帧分析:")
            for i in [0, len(gt_pc)//4, len(gt_pc)//2, len(gt_pc)*3//4, -1]:
                if i < len(gt_pc):
                    pc = gt_pc[i]
                    print(f"\n  帧 {i}:")
                    print(f"    点数: {len(pc)}")
                    if len(pc) > 0:
                        xyz = pc[:, :3] if pc.shape[-1] >= 3 else pc
                        valid_mask = ~np.any(np.isnan(xyz), axis=1)
                        valid_xyz = xyz[valid_mask]
                        if len(valid_xyz) > 0:
                            print(f"    中心: ({np.mean(valid_xyz[:,0]):.3f}, {np.mean(valid_xyz[:,1]):.3f}, {np.mean(valid_xyz[:,2]):.3f})")
                            print(f"    范围: X[{np.min(valid_xyz[:,0]):.3f}, {np.max(valid_xyz[:,0]):.3f}] "
                                  f"Y[{np.min(valid_xyz[:,1]):.3f}, {np.max(valid_xyz[:,1]):.3f}] "
                                  f"Z[{np.min(valid_xyz[:,2]):.3f}, {np.max(valid_xyz[:,2]):.3f}]")
    
    # 分析target_pose
    if 'target_pose' in data:
        print("\n【5. Target Pose】")
        print("-" * 80)
        poses = data['target_pose']
        print(f"形状: {poses.shape}")
        
        if len(poses) > 0:
            print("\n轨迹分析:")
            pose_first = poses[0]
            pose_last = poses[-1]
            print(f"起始位置: ({pose_first[0]:.3f}, {pose_first[1]:.3f}, {pose_first[2]:.3f})")
            print(f"结束位置: ({pose_last[0]:.3f}, {pose_last[1]:.3f}, {pose_last[2]:.3f})")
            
            # 计算位移
            positions = poses[:, :3]
            displacement = np.linalg.norm(positions[-1] - positions[0])
            print(f"总位移: {displacement:.3f} m")
            
            # 计算轨迹长度
            deltas = np.diff(positions, axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            total_distance = np.sum(distances)
            print(f"轨迹长度: {total_distance:.3f} m")
    
    # 统计总览
    print("\n【6. 数据集统计总览】")
    print("-" * 80)
    total_size = sum([info['size_mb'] for info in data_info.values()])
    print(f"数据集总大小: {total_size:.2f} MB")
    print(f"数据项数量: {len(data_info)}")
    print(f"包含数据类型:")
    
    categories = {
        '机器人状态': [],
        '相机RGB': [],
        '相机深度': [],
        '相机法线': [],
        '相机掩码': [],
        '点云': [],
        '姿态': [],
        '其他': []
    }
    
    for key in data_info.keys():
        if any(x in key for x in ['arm', 'hand', 'pos', 'vel', 'eef']) and 'camera' not in key:
            categories['机器人状态'].append(key)
        elif 'rgb' in key:
            categories['相机RGB'].append(key)
        elif 'depth' in key:
            categories['相机深度'].append(key)
        elif 'normal' in key:
            categories['相机法线'].append(key)
        elif 'mask' in key:
            categories['相机掩码'].append(key)
        elif 'pointcloud' in key:
            categories['点云'].append(key)
        elif 'pose' in key or 'action' in key:
            categories['姿态'].append(key)
        else:
            categories['其他'].append(key)
    
    for cat, items in categories.items():
        if items:
            print(f"\n  {cat}: {len(items)} 项")
            for item in items:
                print(f"    - {item}: {data_info[item]['shape']}")


def main():
    zarr_path = "/home/psibot/chembench/data/zarr_point_cloud/motion_plan/grasp/100ml玻璃烧杯.zarr"
    
    if not Path(zarr_path).exists():
        print(f"错误: Zarr数据集不存在: {zarr_path}")
        return 1
    
    analyze_zarr_dataset(zarr_path)
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()




