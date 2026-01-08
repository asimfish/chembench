#!/usr/bin/env python3
"""
可视化真值点云数据
支持多种可视化方式：matplotlib 3D、open3d交互式、动画等
"""

import numpy as np
import zarr
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def load_ground_truth_pointcloud(zarr_path):
    """加载真值点云数据"""
    print(f"加载点云数据: {zarr_path}")
    
    if Path(zarr_path).is_dir():
        # 如果是目录，尝试读取data/ground_truth_pointcloud
        root = zarr.open(zarr_path, mode='r')
        if 'data' in root and 'ground_truth_pointcloud' in root['data']:
            pc_data = root['data']['ground_truth_pointcloud'][:]
        elif 'ground_truth_pointcloud' in root:
            pc_data = root['ground_truth_pointcloud'][:]
        else:
            raise ValueError(f"在 {zarr_path} 中找不到 ground_truth_pointcloud 数据")
    else:
        raise ValueError(f"路径不存在: {zarr_path}")
    
    print(f"点云数据形状: {pc_data.shape}")
    print(f"  时间步数: {pc_data.shape[0]}")
    print(f"  每帧点数: {pc_data.shape[1]}")
    
    return pc_data


def visualize_static_matplotlib(pc_data, frame_indices=None, save_path=None):
    """
    使用matplotlib静态显示多个关键帧
    
    Args:
        pc_data: (T, N, 3) 或 (T, N, 6) 点云序列
                 3通道: [x, y, z]
                 6通道: [x, y, z, r, g, b]
        frame_indices: 要显示的帧索引列表，None则显示均匀分布的4帧
        save_path: 保存图片路径
    """
    if frame_indices is None:
        # 显示开始、1/3、2/3、结束四个关键帧
        num_frames = len(pc_data)
        frame_indices = [0, num_frames//3, 2*num_frames//3, num_frames-1]
    
    num_plots = len(frame_indices)
    fig = plt.figure(figsize=(5*num_plots, 5))
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(1, num_plots, i+1, projection='3d')
        
        pc = pc_data[frame_idx]
        
        # 提取 XYZ 和可选的 RGB
        xyz = pc[:, :3]  # [N, 3]
        rgb = pc[:, 3:6] if pc.shape[1] >= 6 else None  # [N, 3] or None
        
        # 智能颜色处理（与 visualize_pointcloud_debug 一致）
        if rgb is not None and np.any(rgb != 0):
            # 如果 RGB 不全是零，使用 RGB 着色
            colors = rgb
            # 归一化到 [0, 1]（如果需要）
            if colors.max() > 1.0:
                colors = colors / 255.0
            use_colorbar = False
        else:
            # 使用 Z 值着色
            colors = xyz[:, 2]
            use_colorbar = True
        
        # 绘制点云（点更大，与 visualize_pointcloud_debug 一致）
        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                           c=colors, cmap='viridis', s=5, alpha=0.6)
        
        # 设置标题和标签
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # 设置等比例坐标范围（与 visualize_pointcloud_debug 一致）
        max_range = np.array([xyz[:, 0].max()-xyz[:, 0].min(),
                             xyz[:, 1].max()-xyz[:, 1].min(),
                             xyz[:, 2].max()-xyz[:, 2].min()]).max() / 2.0
        
        mid_x = (xyz[:, 0].max()+xyz[:, 0].min()) * 0.5
        mid_y = (xyz[:, 1].max()+xyz[:, 1].min()) * 0.5
        mid_z = (xyz[:, 2].max()+xyz[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 添加颜色条（仅当使用标量着色时）
        if use_colorbar:
            plt.colorbar(scatter, ax=ax, label='Z value', shrink=0.5, pad=0.1)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()


def visualize_animation_matplotlib(pc_data, skip_frames=10, save_path=None):
    """
    使用matplotlib创建动画
    
    Args:
        pc_data: (T, N, 3) 点云序列
        skip_frames: 跳帧数（加快动画速度）
        save_path: 保存动画路径（.gif或.mp4）
    """
    print("创建动画...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算全局坐标范围
    all_points = pc_data.reshape(-1, 3)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    # 添加边距
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim([x_min - margin*x_range, x_max + margin*x_range])
    ax.set_ylim([y_min - margin*y_range, y_max + margin*y_range])
    ax.set_zlim([z_min - margin*z_range, z_max + margin*z_range])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 初始化散点图
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=1, alpha=0.6)
    title = ax.set_title('')
    
    def init():
        scatter._offsets3d = ([], [], [])
        return scatter, title
    
    def update(frame):
        idx = frame * skip_frames
        if idx >= len(pc_data):
            idx = len(pc_data) - 1
        
        pc = pc_data[idx]
        scatter._offsets3d = (pc[:, 0], pc[:, 1], pc[:, 2])
        scatter.set_array(pc[:, 2])
        title.set_text(f'Frame {idx}/{len(pc_data)-1}')
        
        return scatter, title
    
    num_frames = len(pc_data) // skip_frames
    anim = FuncAnimation(fig, update, frames=num_frames, 
                        init_func=init, blit=False, interval=50)
    
    if save_path:
        print(f"保存动画到: {save_path}")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=20)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=20)
        print("动画已保存!")
    
    plt.show()


def visualize_trajectory_path(pc_data, sample_rate=50):
    """
    可视化物体中心的运动轨迹
    
    Args:
        pc_data: (T, N, 3) 点云序列
        sample_rate: 采样率（显示物体的频率）
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算每帧的中心点
    centers = np.mean(pc_data, axis=1)  # (T, 3)
    
    # 绘制轨迹线
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 
           'r-', linewidth=2, label='Trajectory', alpha=0.7)
    
    # 标记起点和终点
    ax.scatter(*centers[0], c='green', s=200, marker='o', 
              label='Start', edgecolors='black', linewidths=2)
    ax.scatter(*centers[-1], c='red', s=200, marker='o', 
              label='End', edgecolors='black', linewidths=2)
    
    # 在关键位置显示物体点云
    indices = range(0, len(pc_data), sample_rate)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))
    
    for i, (idx, color) in enumerate(zip(indices, colors)):
        pc = pc_data[idx]
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], 
                  c=[color], s=0.5, alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Object Movement Trajectory')
    ax.legend()
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()


def visualize_open3d(pc_data, frame_indices=None):
    """
    使用Open3D交互式可视化（如果可用）
    
    Args:
        pc_data: (T, N, 3) 点云序列
        frame_indices: 要显示的帧索引列表
    """
    try:
        import open3d as o3d
    except ImportError:
        print("未安装Open3D，请使用: pip install open3d")
        return
    
    if frame_indices is None:
        # 显示几个关键帧
        num_frames = len(pc_data)
        frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
    
    print("Open3D 交互式可视化")
    print("说明：")
    print("  - 鼠标左键拖动：旋转")
    print("  - 鼠标滚轮：缩放")
    print("  - 鼠标右键拖动：平移")
    print("  - 按 'q' 退出")
    
    # 创建点云对象
    pcds = []
    colors = plt.cm.rainbow(np.linspace(0, 1, len(frame_indices)))
    
    for idx, color in zip(frame_indices, colors):
        pc = pc_data[idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.paint_uniform_color(color[:3])
        pcds.append(pcd)
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0])
    pcds.append(coord_frame)
    
    # 可视化
    o3d.visualization.draw_geometries(
        pcds,
        window_name="Ground Truth Pointcloud",
        width=1280,
        height=720,
        left=50,
        top=50
    )


def compare_with_camera_pointcloud(zarr_path, frame_idx=0):
    """
    对比真值点云和相机点云
    
    Args:
        zarr_path: zarr数据路径
        frame_idx: 要对比的帧索引
    """
    root = zarr.open(zarr_path, mode='r')
    
    # 读取数据
    gt_pc = root['data']['ground_truth_pointcloud'][frame_idx]
    
    # 检查是否有相机点云
    camera_pcs = {}
    for key in ['chest_camera_pointcloud', 'head_camera_pointcloud', 'third_camera_pointcloud']:
        if key in root['data']:
            camera_pcs[key] = root['data'][key][frame_idx, :, :3]  # 只取XYZ
    
    if not camera_pcs:
        print("未找到相机点云数据")
        return
    
    # 创建子图
    num_plots = len(camera_pcs) + 1
    fig = plt.figure(figsize=(5*num_plots, 5))
    
    # 绘制真值点云
    ax = fig.add_subplot(1, num_plots, 1, projection='3d')
    ax.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2], 
              c='red', s=1, alpha=0.6, label='Ground Truth')
    ax.set_title('Ground Truth Pointcloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.view_init(elev=20, azim=45)
    
    # 绘制相机点云
    for i, (name, pc) in enumerate(camera_pcs.items()):
        ax = fig.add_subplot(1, num_plots, i+2, projection='3d')
        
        # 过滤无效点
        valid_mask = ~np.any(np.isnan(pc), axis=1) & ~np.any(np.isinf(pc), axis=1)
        pc_valid = pc[valid_mask]
        
        ax.scatter(pc_valid[:, 0], pc_valid[:, 1], pc_valid[:, 2], 
                  c='blue', s=1, alpha=0.6)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()


def compare_animation_with_camera(zarr_path, skip_frames=10, save_path=None):
    """
    创建真值点云和相机点云的动态对比动画
    
    Args:
        zarr_path: zarr数据路径
        skip_frames: 跳帧数
        save_path: 动画保存路径（可选）
    """
    root = zarr.open(zarr_path, mode='r')
    
    # 读取数据
    gt_pc_data = root['data']['ground_truth_pointcloud'][:]
    
    # 检查是否有相机点云
    camera_names = []
    camera_data = {}
    for key in ['chest_camera_pointcloud', 'head_camera_pointcloud', 'third_camera_pointcloud']:
        if key in root['data']:
            camera_names.append(key)
            camera_data[key] = root['data'][key][:, :, :3]  # 只取XYZ
    
    if not camera_names:
        print("未找到相机点云数据")
        return
    
    num_frames = gt_pc_data.shape[0]
    frame_indices = range(0, num_frames, skip_frames)
    
    # 创建图形
    num_plots = len(camera_names) + 1
    fig = plt.figure(figsize=(5*num_plots, 5))
    
    # 创建子图
    axes = []
    for i in range(num_plots):
        ax = fig.add_subplot(1, num_plots, i+1, projection='3d')
        axes.append(ax)
    
    # 计算全局坐标范围
    all_points = [gt_pc_data.reshape(-1, 3)]
    for cam_data in camera_data.values():
        valid_mask = ~np.any(np.isnan(cam_data), axis=2) & ~np.any(np.isinf(cam_data), axis=2)
        valid_points = cam_data[valid_mask]
        all_points.append(valid_points)
    
    all_points = np.vstack(all_points)
    x_min, x_max = np.percentile(all_points[:, 0], [1, 99])
    y_min, y_max = np.percentile(all_points[:, 1], [1, 99])
    z_min, z_max = np.percentile(all_points[:, 2], [1, 99])
    
    def update(frame_idx):
        for ax in axes:
            ax.clear()
        
        # 绘制真值点云
        ax = axes[0]
        gt_pc = gt_pc_data[frame_idx]
        ax.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2], 
                  c='red', s=1, alpha=0.6, label='Ground Truth')
        ax.set_title(f'Ground Truth (Frame {frame_idx})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.legend()
        ax.view_init(elev=20, azim=45)
        
        # 绘制相机点云
        for i, cam_name in enumerate(camera_names):
            ax = axes[i+1]
            pc = camera_data[cam_name][frame_idx]
            
            # 过滤无效点
            valid_mask = ~np.any(np.isnan(pc), axis=1) & ~np.any(np.isinf(pc), axis=1)
            pc_valid = pc[valid_mask]
            
            ax.scatter(pc_valid[:, 0], pc_valid[:, 1], pc_valid[:, 2], 
                      c='blue', s=1, alpha=0.6)
            ax.set_title(f'{cam_name.replace("_", " ").title()} (Frame {frame_idx})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return axes
    
    # 创建动画
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=list(frame_indices), 
                        interval=200, repeat=True)
    
    # 保存或显示
    if save_path:
        print(f"保存动画到: {save_path}")
        anim.save(save_path, writer='pillow', fps=5)
        print("保存完成!")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='可视化真值点云数据')
    parser.add_argument('--zarr', type=str, required=True, 
                       help='Zarr数据路径')
    parser.add_argument('--mode', type=str, default='static',
                       choices=['static', 'animation', 'trajectory', 'open3d', 'compare', 'compare_animation'],
                       help='可视化模式')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                       help='要显示的帧索引（例如：--frames 0 100 200）')
    parser.add_argument('--skip_frames', type=int, default=10,
                       help='动画跳帧数')
    parser.add_argument('--sample_rate', type=int, default=50,
                       help='轨迹模式的采样率')
    parser.add_argument('--save', type=str, default=None,
                       help='保存路径')
    
    args = parser.parse_args()
    
    # 加载点云数据
    pc_data = load_ground_truth_pointcloud(args.zarr)
    
    print(f"\n可视化模式: {args.mode}")
    
    # 根据模式选择可视化方法
    if args.mode == 'static':
        print("显示静态关键帧...")
        visualize_static_matplotlib(pc_data, args.frames, args.save)
    
    elif args.mode == 'animation':
        print("创建动画...")
        visualize_animation_matplotlib(pc_data, args.skip_frames, args.save)
    
    elif args.mode == 'trajectory':
        print("显示运动轨迹...")
        visualize_trajectory_path(pc_data, args.sample_rate)
    
    elif args.mode == 'open3d':
        print("Open3D交互式可视化...")
        visualize_open3d(pc_data, args.frames)
    
    elif args.mode == 'compare':
        print("对比真值点云和相机点云...")
        frame_idx = args.frames[0] if args.frames else 0
        compare_with_camera_pointcloud(args.zarr, frame_idx)
    
    elif args.mode == 'compare_animation':
        print("创建动态对比动画...")
        compare_animation_with_camera(args.zarr, args.skip_frames, args.save)
    
    print("\n完成!")


if __name__ == '__main__':
    main()

