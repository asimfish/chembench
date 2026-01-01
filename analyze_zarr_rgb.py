#!/usr/bin/env python3
"""
分析 Zarr 格式的 chest_camera_rgb 数据
"""
import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_zarr_rgb_data(zarr_path):
    """
    分析 Zarr 格式的 RGB 相机数据
    
    Args:
        zarr_path: Zarr 数据集的路径
    """
    print("=" * 80)
    print(f"分析 Zarr 数据集: 100ml玻璃烧杯")
    print("=" * 80)
    
    # 打开 zarr 数据集
    try:
        root = zarr.open(zarr_path, mode='r')
    except Exception as e:
        print(f"错误: 无法打开 Zarr 数据集: {e}")
        return
    
    # 分析 chest_camera_rgb 数据
    print("\n【1. RGB 相机数据基本信息】")
    print("-" * 80)
    
    if 'data' not in root:
        print("错误: 找不到 'data' 组")
        return
    
    data_group = root['data']
    
    if 'chest_camera_rgb' not in data_group:
        print("错误: 找不到 'chest_camera_rgb' 数据")
        return
    
    rgb_data = data_group['chest_camera_rgb']
    
    print(f"数据类型: {rgb_data.dtype}")
    print(f"数据形状: {rgb_data.shape}")
    print(f"存储形状 (chunks): {rgb_data.chunks}")
    print(f"压缩器: {rgb_data.compressor}")
    print(f"数据大小: {rgb_data.nbytes / (1024**2):.2f} MB")
    print(f"磁盘占用 (估算): {rgb_data.nbytes_stored / (1024**2):.2f} MB")
    print(f"压缩比: {rgb_data.nbytes / rgb_data.nbytes_stored:.2f}x")
    
    # 解析形状
    if len(rgb_data.shape) == 4:
        num_episodes, num_frames, height, width = rgb_data.shape[:4]
        channels = rgb_data.shape[4] if len(rgb_data.shape) == 5 else 3
    else:
        print(f"警告: 数据形状不符合预期 {rgb_data.shape}")
        return
    
    print(f"\n数据结构:")
    print(f"  - 轨迹数 (episodes): {num_episodes}")
    print(f"  - 每轨迹帧数: {num_frames}")
    print(f"  - 图像分辨率: {height} x {width}")
    print(f"  - 通道数: {channels}")
    
    # 分析数值范围
    print("\n【2. 数值统计】")
    print("-" * 80)
    
    # 读取第一个轨迹的数据来分析
    sample_data = rgb_data[0]
    
    print(f"第一个轨迹数据:")
    print(f"  - 最小值: {np.min(sample_data)}")
    print(f"  - 最大值: {np.max(sample_data)}")
    print(f"  - 平均值: {np.mean(sample_data):.2f}")
    print(f"  - 标准差: {np.std(sample_data):.2f}")
    print(f"  - 数据范围: {'[0-255] (uint8)' if rgb_data.dtype == np.uint8 else '[0-1] (float)'}")
    
    # 分析其他数据维度
    print("\n【3. 其他数据维度】")
    print("-" * 80)
    
    data_dims = {}
    for key in data_group.keys():
        if key.startswith('chest_camera') or key.startswith('head_camera'):
            data_dims[key] = data_group[key].shape
        elif key in ['action', 'arm2_pos', 'arm2_vel', 'arm2_eef_pos', 'arm2_eef_quat', 
                     'hand2_pos', 'hand2_vel', 'target_pose']:
            data_dims[key] = data_group[key].shape
    
    for key, shape in sorted(data_dims.items()):
        print(f"  {key:30s}: {shape}")
    
    # 读取 episode_ends 元数据
    print("\n【4. 轨迹元数据】")
    print("-" * 80)
    
    if 'meta' in root and 'episode_ends' in root['meta']:
        episode_ends = root['meta']['episode_ends'][:]
        print(f"轨迹结束索引: {episode_ends}")
        print(f"总轨迹数: {len(episode_ends)}")
        
        if len(episode_ends) > 1:
            episode_lengths = np.diff([0] + list(episode_ends))
            print(f"轨迹长度: min={np.min(episode_lengths)}, max={np.max(episode_lengths)}, mean={np.mean(episode_lengths):.1f}")
    
    # 可视化示例帧
    print("\n【5. 生成可视化】")
    print("-" * 80)
    
    # 选择几个关键帧进行可视化
    frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'100ml玻璃烧杯 - Chest Camera RGB (轨迹 0)', fontsize=14, fontweight='bold')
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]
        
        # 读取图像数据
        img = rgb_data[0, frame_idx]
        
        # 如果是float类型且在[0,1]范围，转换为uint8
        if img.dtype == np.float32 or img.dtype == np.float64:
            if np.max(img) <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        ax.set_title(f'Frame {frame_idx}/{num_frames-1}')
        ax.axis('off')
    
    plt.tight_layout()
    output_path = '/home/psibot/chembench/zarr_rgb_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化图像已保存: {output_path}")
    
    # 分析时间序列数据
    print("\n【6. 时间序列数据分析】")
    print("-" * 80)
    
    if 'arm2_eef_pos' in data_group:
        eef_pos = data_group['arm2_eef_pos'][0]  # 第一个轨迹
        print(f"末端执行器位置 (arm2_eef_pos):")
        print(f"  - 形状: {eef_pos.shape}")
        print(f"  - X范围: [{np.min(eef_pos[:, 0]):.4f}, {np.max(eef_pos[:, 0]):.4f}]")
        print(f"  - Y范围: [{np.min(eef_pos[:, 1]):.4f}, {np.max(eef_pos[:, 1]):.4f}]")
        print(f"  - Z范围: [{np.min(eef_pos[:, 2]):.4f}, {np.max(eef_pos[:, 2]):.4f}]")
    
    if 'action' in data_group:
        action = data_group['action'][0]
        print(f"\nAction 数据:")
        print(f"  - 形状: {action.shape}")
        print(f"  - 维度: {action.shape[-1]} (可能是关节位置/速度命令)")
    
    if 'hand2_pos' in data_group:
        hand_pos = data_group['hand2_pos'][0]
        print(f"\n手部位置 (hand2_pos):")
        print(f"  - 形状: {hand_pos.shape}")
        print(f"  - 初始状态: {hand_pos[0]}")
        print(f"  - 最终状态: {hand_pos[-1]}")
    
    # 数据完整性检查
    print("\n【7. 数据完整性检查】")
    print("-" * 80)
    
    issues = []
    
    # 检查是否有NaN或Inf
    if np.any(np.isnan(sample_data)):
        issues.append("⚠ 发现 NaN 值")
    if np.any(np.isinf(sample_data)):
        issues.append("⚠ 发现 Inf 值")
    
    # 检查数据范围
    if rgb_data.dtype == np.uint8:
        if np.min(sample_data) < 0 or np.max(sample_data) > 255:
            issues.append(f"⚠ uint8 数据超出范围 [0, 255]: [{np.min(sample_data)}, {np.max(sample_data)}]")
    
    # 检查是否所有轨迹长度一致
    if num_episodes > 1:
        shapes_match = all(rgb_data[i].shape == rgb_data[0].shape for i in range(min(10, num_episodes)))
        if not shapes_match:
            issues.append("⚠ 不同轨迹的数据形状不一致")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✓ 数据完整性检查通过")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    zarr_path = "/home/psibot/chembench/data/zarr_final/motion_plan/grasp/part1/100ml玻璃烧杯.zarr"
    analyze_zarr_rgb_data(zarr_path)

