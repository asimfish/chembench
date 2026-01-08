#!/usr/bin/env python3
"""
可视化相机点云数据
支持多种可视化方式：单相机、多相机对比、与真值对比等
"""

import numpy as np
import zarr
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def load_camera_pointclouds(zarr_path):
    """加载所有相机点云数据"""
    print(f"加载相机点云数据: {zarr_path}")
    
    root = zarr.open(zarr_path, mode='r')
    
    camera_names = ['chest_camera_pointcloud', 'head_camera_pointcloud', 'third_camera_pointcloud']
    camera_data = {}
    
    for name in camera_names:
        if 'data' in root and name in root['data']:
            data = root['data'][name]
            camera_data[name] = data
            print(f"  {name}: {data.shape}")
        elif name in root:
            data = root[name]
            camera_data[name] = data
            print(f"  {name}: {data.shape}")
    
    if not camera_data:
        raise ValueError("未找到相机点云数据")
    
    return root, camera_data


def process_pointcloud(pc_data, use_rgb=False):
    """
    处理点云数据，提取XYZ和颜色
    
    Args:
        pc_data: (N, 6) 点云数据，前3维是XYZ，后3维可能是RGB或其他特征
        use_rgb: 是否使用RGB颜色
    
    Returns:
        xyz: (N, 3) XYZ坐标
        colors: (N, 3) RGB颜色或None
    """
    # 提取XYZ
    xyz = pc_data[:, :3]
    
    # 过滤无效点
    valid_mask = ~np.any(np.isnan(xyz), axis=1) & ~np.any(np.isinf(xyz), axis=1)
    valid_mask &= np.any(xyz != 0, axis=1)  # 过滤全0点
    
    xyz = xyz[valid_mask]
    # xyz = xyz
    
    # 提取颜色
    colors = None
    if use_rgb and pc_data.shape[1] >= 6:
        colors = pc_data[valid_mask, 3:6]
        # 检查颜色范围并归一化
        if colors.max() > 1.0:
            colors = colors / 255.0
        # 确保在[0,1]范围内
        colors = np.clip(colors, 0, 1)
    
    return xyz, colors


def visualize_single_camera(zarr_path, camera_name='chest_camera_pointcloud', 
                           frame_idx=0, use_rgb=True, save_path=None):
    """
    可视化单个相机的点云
    
    Args:
        zarr_path: zarr数据路径
        camera_name: 相机名称
        frame_idx: 帧索引
        use_rgb: 是否使用RGB颜色
        save_path: 保存路径
    """
    root, camera_data = load_camera_pointclouds(zarr_path)
    
    if camera_name not in camera_data:
        print(f"未找到相机: {camera_name}")
        print(f"可用相机: {list(camera_data.keys())}")
        return
    
    pc_data = camera_data[camera_name][frame_idx]
    xyz, colors = process_pointcloud(pc_data, use_rgb)
    
    print(f"\n{camera_name} - Frame {frame_idx}")
    print(f"  有效点数: {len(xyz)}")
    print(f"  XYZ范围:")
    print(f"    X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
    print(f"    Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
    print(f"    Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
    
    # 可视化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if use_rgb and colors is not None:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                  c=colors, s=1, alpha=0.8)
    else:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                  c=xyz[:, 2], cmap='viridis', s=1, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{camera_name.replace("_", " ").title()} - Frame {frame_idx}')
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()


def visualize_all_cameras(zarr_path, frame_idx=0, use_rgb=True, save_path=None):
    """
    对比显示所有相机的点云
    
    Args:
        zarr_path: zarr数据路径
        frame_idx: 帧索引
        use_rgb: 是否使用RGB颜色
        save_path: 保存路径
    """
    root, camera_data = load_camera_pointclouds(zarr_path)
    
    num_cameras = len(camera_data)
    fig = plt.figure(figsize=(6*num_cameras, 5))
    
    for i, (name, data) in enumerate(sorted(camera_data.items())):
        pc_data = data[frame_idx]
        xyz, colors = process_pointcloud(pc_data, use_rgb)
        
        ax = fig.add_subplot(1, num_cameras, i+1, projection='3d')
        
        if use_rgb and colors is not None:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                      c=colors, s=1, alpha=0.8)
        else:
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                      c=xyz[:, 2], cmap='viridis', s=1, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name.replace("_", " ").title()}\n{len(xyz)} points')
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle(f'All Cameras - Frame {frame_idx}', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()


def visualize_camera_animation(zarr_path, camera_name='chest_camera_pointcloud',
                               skip_frames=10, use_rgb=True, save_path=None):
    """
    创建相机点云动画
    
    Args:
        zarr_path: zarr数据路径
        camera_name: 相机名称
        skip_frames: 跳帧数
        use_rgb: 是否使用RGB颜色
        save_path: 保存路径
    """
    root, camera_data = load_camera_pointclouds(zarr_path)
    
    if camera_name not in camera_data:
        print(f"未找到相机: {camera_name}")
        return
    
    data = camera_data[camera_name]
    
    print("创建动画...")
    print(f"总帧数: {len(data)}")
    print(f"跳帧: {skip_frames}, 动画帧数: {len(data) // skip_frames}")
    
    # 预统计所有帧的点云数量
    print("\n统计点云数量...")
    point_counts = []
    for i in range(0, len(data), skip_frames):
        xyz, _ = process_pointcloud(data[i], False)
        point_counts.append(len(xyz))
    
    print(f"点云数量统计:")
    print(f"  最小: {min(point_counts)} 点")
    print(f"  最大: {max(point_counts)} 点")
    print(f"  平均: {np.mean(point_counts):.0f} 点")
    print(f"  中位数: {np.median(point_counts):.0f} 点")
    
    fig = plt.figure(figsize=(14, 10))
    
    # 主图：3D点云
    ax = fig.add_subplot(121, projection='3d')
    
    # 计算全局坐标范围（采样部分帧）
    sample_indices = np.linspace(0, len(data)-1, min(10, len(data)), dtype=int)
    all_xyz = []
    for idx in sample_indices:
        xyz, _ = process_pointcloud(data[idx], False)
        if len(xyz) > 0:
            all_xyz.append(xyz)
    
    if all_xyz:
        all_xyz = np.vstack(all_xyz)
        x_min, x_max = all_xyz[:, 0].min(), all_xyz[:, 0].max()
        y_min, y_max = all_xyz[:, 1].min(), all_xyz[:, 1].max()
        z_min, z_max = all_xyz[:, 2].min(), all_xyz[:, 2].max()
        
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
    
    scatter = ax.scatter([], [], [], s=1, alpha=0.8)
    title = ax.set_title('')
    
    # 副图：点云数量统计曲线
    ax2 = fig.add_subplot(122)
    frame_indices = list(range(0, len(data), skip_frames))
    ax2.plot(frame_indices, point_counts, 'b-', linewidth=2, label='Point Count')
    ax2.axhline(y=np.mean(point_counts), color='r', linestyle='--', 
                label=f'Mean: {np.mean(point_counts):.0f}')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Number of Points')
    ax2.set_title('Point Cloud Density Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 当前帧标记
    current_marker, = ax2.plot([], [], 'ro', markersize=10, label='Current')
    
    def init():
        scatter._offsets3d = ([], [], [])
        current_marker.set_data([], [])
        return scatter, title, current_marker
    
    def update(frame):
        idx = frame * skip_frames
        if idx >= len(data):
            idx = len(data) - 1
        
        xyz, colors = process_pointcloud(data[idx], use_rgb)
        num_points = len(xyz)
        
        if num_points > 0:
            scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            if use_rgb and colors is not None:
                scatter.set_color(colors)
            else:
                scatter.set_array(xyz[:, 2])
                scatter.set_cmap('viridis')
        
        # 更新标题，显示点云数量
        title.set_text(f'{camera_name}\nFrame {idx}/{len(data)-1} | Points: {num_points}')
        
        # 更新当前帧标记
        current_marker.set_data([idx], [num_points])
        
        # 在控制台输出（可选）
        if frame % 10 == 0:  # 每10帧输出一次
            print(f"Frame {idx:4d}: {num_points:4d} points")
        
        return scatter, title, current_marker
    
    num_frames = len(data) // skip_frames
    anim = FuncAnimation(fig, update, frames=num_frames,
                        init_func=init, blit=False, interval=50)
    
    plt.tight_layout()
    
    if save_path:
        print(f"\n保存动画到: {save_path}")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=20)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=20)
        print("动画已保存!")
    
    plt.show()
    
    # 动画结束后，显示详细统计
    print("\n" + "="*60)
    print("点云数量详细统计")
    print("="*60)
    print(f"总帧数: {len(data)}")
    print(f"分析帧数: {len(point_counts)}")
    print(f"点云数量:")
    print(f"  最小值: {min(point_counts)} 点 (帧 {frame_indices[np.argmin(point_counts)]})")
    print(f"  最大值: {max(point_counts)} 点 (帧 {frame_indices[np.argmax(point_counts)]})")
    print(f"  平均值: {np.mean(point_counts):.2f} 点")
    print(f"  标准差: {np.std(point_counts):.2f} 点")
    print(f"  中位数: {np.median(point_counts):.0f} 点")
    
    # 检测异常帧
    mean_count = np.mean(point_counts)
    std_count = np.std(point_counts)
    outliers = []
    for i, count in enumerate(point_counts):
        if abs(count - mean_count) > 2 * std_count:
            outliers.append((frame_indices[i], count))
    
    if outliers:
        print(f"\n异常帧 (偏离均值2个标准差以上):")
        for frame_idx, count in outliers:
            print(f"  帧 {frame_idx}: {count} 点")
    else:
        print(f"\n✓ 所有帧的点云数量都在正常范围内")


def visualize_camera_vs_groundtruth(zarr_path, camera_name='chest_camera_pointcloud',
                                    frame_idx=0, use_rgb=True, save_path=None):
    """
    对比相机点云和真值点云
    
    Args:
        zarr_path: zarr数据路径
        camera_name: 相机名称
        frame_idx: 帧索引
        use_rgb: 相机点云是否使用RGB颜色
        save_path: 保存路径
    """
    root = zarr.open(zarr_path, mode='r')
    
    # 加载相机点云
    if 'data' in root and camera_name in root['data']:
        camera_pc = root['data'][camera_name][frame_idx]
    else:
        print(f"未找到相机点云: {camera_name}")
        return
    
    # 加载真值点云
    gt_pc = None
    if 'data' in root and 'ground_truth_pointcloud' in root['data']:
        gt_pc = root['data']['ground_truth_pointcloud'][frame_idx]
    elif 'ground_truth_pointcloud' in root:
        gt_pc = root['ground_truth_pointcloud'][frame_idx]
    
    # 处理点云
    camera_xyz, camera_colors = process_pointcloud(camera_pc, use_rgb)
    
    # 创建图形
    if gt_pc is not None:
        fig = plt.figure(figsize=(18, 6))
        
        # 相机点云
        ax1 = fig.add_subplot(131, projection='3d')
        if use_rgb and camera_colors is not None:
            ax1.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2],
                       c=camera_colors, s=1, alpha=0.8)
        else:
            ax1.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2],
                       c=camera_xyz[:, 2], cmap='viridis', s=1, alpha=0.8)
        ax1.set_title(f'{camera_name}\n{len(camera_xyz)} points')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.view_init(elev=20, azim=45)
        
        # 真值点云
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2],
                   c='red', s=1, alpha=0.6)
        ax2.set_title(f'Ground Truth\n{len(gt_pc)} points')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.view_init(elev=20, azim=45)
        
        # 叠加显示
        ax3 = fig.add_subplot(133, projection='3d')
        if use_rgb and camera_colors is not None:
            ax3.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2],
                       c=camera_colors, s=1, alpha=0.5, label='Camera')
        else:
            ax3.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2],
                       c='blue', s=1, alpha=0.5, label='Camera')
        ax3.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2],
                   c='red', s=1, alpha=0.5, label='Ground Truth')
        ax3.set_title('Overlay')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        ax3.view_init(elev=20, azim=45)
        
        plt.suptitle(f'Frame {frame_idx}', fontsize=14)
    else:
        # 只显示相机点云
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if use_rgb and camera_colors is not None:
            ax.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2],
                      c=camera_colors, s=1, alpha=0.8)
        else:
            ax.scatter(camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2],
                      c=camera_xyz[:, 2], cmap='viridis', s=1, alpha=0.8)
        
        ax.set_title(f'{camera_name} - Frame {frame_idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()


def visualize_open3d(zarr_path, camera_names=None, frame_indices=None, use_rgb=True):
    """
    使用Open3D交互式可视化相机点云
    
    Args:
        zarr_path: zarr数据路径
        camera_names: 相机名称列表
        frame_indices: 帧索引列表
        use_rgb: 是否使用RGB颜色
    """
    try:
        import open3d as o3d
    except ImportError:
        print("未安装Open3D，请使用: pip install open3d")
        return
    
    root, camera_data = load_camera_pointclouds(zarr_path)
    
    if camera_names is None:
        camera_names = list(camera_data.keys())
    
    if frame_indices is None:
        # 显示第一帧
        frame_indices = [0]
    
    print("Open3D 交互式可视化")
    print("说明：")
    print("  - 鼠标左键拖动：旋转")
    print("  - 鼠标滚轮：缩放")
    print("  - 鼠标右键拖动：平移")
    print("  - 按 'q' 退出")
    
    # 创建点云对象
    pcds = []
    colors_map = plt.cm.rainbow(np.linspace(0, 1, len(camera_names) * len(frame_indices)))
    color_idx = 0
    
    for camera_name in camera_names:
        if camera_name not in camera_data:
            continue
        
        for frame_idx in frame_indices:
            pc_data = camera_data[camera_name][frame_idx]
            xyz, colors = process_pointcloud(pc_data, use_rgb)
            
            if len(xyz) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                
                if use_rgb and colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                else:
                    pcd.paint_uniform_color(colors_map[color_idx][:3])
                
                pcds.append(pcd)
                color_idx += 1
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0])
    pcds.append(coord_frame)
    
    # 可视化
    o3d.visualization.draw_geometries(
        pcds,
        window_name="Camera Pointcloud",
        width=1280,
        height=720,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(description='可视化相机点云数据')
    parser.add_argument('--zarr', type=str, required=True,
                       help='Zarr数据路径')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'all', 'animation', 'compare', 'open3d'],
                       help='可视化模式')
    parser.add_argument('--camera', type=str, default='chest_camera_pointcloud',
                       choices=['chest_camera_pointcloud', 'head_camera_pointcloud', 
                               'third_camera_pointcloud'],
                       help='相机名称')
    parser.add_argument('--frames', type=int, nargs='+', default=[0],
                       help='要显示的帧索引')
    parser.add_argument('--skip_frames', type=int, default=10,
                       help='动画跳帧数')
    parser.add_argument('--use_rgb', action='store_true', default=False,
                       help='是否使用RGB颜色')
    parser.add_argument('--save', type=str, default=None,
                       help='保存路径')
    
    args = parser.parse_args()
    
    print(f"可视化模式: {args.mode}")
    print(f"使用RGB颜色: {args.use_rgb}")
    
    # 根据模式选择可视化方法
    if args.mode == 'single':
        print(f"显示单个相机: {args.camera}, 帧 {args.frames[0]}")
        visualize_single_camera(args.zarr, args.camera, args.frames[0], 
                               args.use_rgb, args.save)
    
    elif args.mode == 'all':
        print(f"显示所有相机，帧 {args.frames[0]}")
        visualize_all_cameras(args.zarr, args.frames[0], args.use_rgb, args.save)
    
    elif args.mode == 'animation':
        print(f"创建动画: {args.camera}")
        visualize_camera_animation(args.zarr, args.camera, args.skip_frames,
                                  args.use_rgb, args.save)
    
    elif args.mode == 'compare':
        print(f"对比相机与真值: {args.camera}, 帧 {args.frames[0]}")
        visualize_camera_vs_groundtruth(args.zarr, args.camera, args.frames[0],
                                       args.use_rgb, args.save)
    
    elif args.mode == 'open3d':
        print("Open3D交互式可视化")
        visualize_open3d(args.zarr, [args.camera], args.frames, args.use_rgb)
    
    print("\n完成!")


if __name__ == '__main__':
    main()

