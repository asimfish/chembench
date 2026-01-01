import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2

def save_image_samples(data_group, output_dir, num_samples=5):
    """保存相机图片和mask样本用于检查"""
    
    # 相机图片键（RGB 图片）
    camera_keys = ['head_camera_rgb', 'chest_camera_rgb', 'third_camera_rgb']
    # Mask 图片键
    mask_keys = ['head_camera_mask', 'chest_camera_mask', 'third_camera_mask']
    # Depth 和 Normals 图片键
    depth_keys = ['head_camera_depth', 'chest_camera_depth', 'third_camera_depth']
    normals_keys = ['head_camera_normals', 'chest_camera_normals', 'third_camera_normals']
    
    # 保存 RGB 图片
    for camera_key in camera_keys:
        if camera_key not in data_group:
            continue
            
        camera_data = data_group[camera_key][:]
        camera_dir = os.path.join(output_dir, camera_key)
        if not os.path.exists(camera_dir):
            os.makedirs(camera_dir)
        
        total_frames = camera_data.shape[0]
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        print(f"Saving {num_samples} RGB samples for {camera_key}...")
        print(f"  Shape: {camera_data.shape}, dtype: {camera_data.dtype}")
        
        for idx in sample_indices:
            frame = camera_data[idx]
            # RGB 3通道
            if frame.shape[-1] == 3:
                rgb_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                rgb_path = os.path.join(camera_dir, f'rgb_frame_{idx:04d}.png')
                cv2.imwrite(rgb_path, rgb_bgr)
            else:
                print(f"  Warning: Unexpected shape {frame.shape} for RGB data")
        
        print(f"  Saved to {camera_dir}/")
    
    # 保存 Mask 图片
    for mask_key in mask_keys:
        if mask_key not in data_group:
            continue
            
        mask_data = data_group[mask_key][:]
        mask_dir = os.path.join(output_dir, mask_key)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        total_frames = mask_data.shape[0]
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        print(f"Saving {num_samples} mask samples for {mask_key}...")
        print(f"  Shape: {mask_data.shape}, dtype: {mask_data.dtype}")
        
        for idx in sample_indices:
            mask_frame = mask_data[idx]
            mask_path = os.path.join(mask_dir, f'mask_frame_{idx:04d}.png')
            cv2.imwrite(mask_path, mask_frame)
        
        print(f"  Saved to {mask_dir}/")

    # 保存 Depth 图片
    for depth_key in depth_keys:
        if depth_key not in data_group:
            continue
            
        depth_data = data_group[depth_key][:]
        depth_dir = os.path.join(output_dir, depth_key)
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)
        
        total_frames = depth_data.shape[0]
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        print(f"Saving {num_samples} depth samples for {depth_key}...")
        print(f"  Shape: {depth_data.shape}, dtype: {depth_data.dtype}")
        
        for idx in sample_indices:
            # 假设 shape 为 (H, W, 1) 或 (H, W)
            d_frame = depth_data[idx]
            if d_frame.ndim == 3 and d_frame.shape[-1] == 1:
                d_frame = d_frame[:, :, 0]
            
            # 归一化以便可视化: (value - min) / (max - min) * 255
            # 处理 NaN/Inf
            d_frame = np.nan_to_num(d_frame, nan=0.0, posinf=0.0, neginf=0.0)
            
            d_min, d_max = d_frame.min(), d_frame.max()
            if d_max > d_min:
                d_norm = ((d_frame - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                d_norm = np.zeros_like(d_frame, dtype=np.uint8)
            
            # 应用伪彩色 (JET Colormap: 蓝-青-黄-红)
            d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
            
            save_path = os.path.join(depth_dir, f'depth_frame_{idx:04d}.png')
            cv2.imwrite(save_path, d_color)
            
        print(f"  Saved to {depth_dir}/")

    # 保存 Normals 图片
    for normals_key in normals_keys:
        if normals_key not in data_group:
            continue
            
        normals_data = data_group[normals_key][:]
        normals_dir = os.path.join(output_dir, normals_key)
        if not os.path.exists(normals_dir):
            os.makedirs(normals_dir)
        
        total_frames = normals_data.shape[0]
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        print(f"Saving {num_samples} normals samples for {normals_key}...")
        print(f"  Shape: {normals_data.shape}, dtype: {normals_data.dtype}")
        
        for idx in sample_indices:
            # 假设 shape 为 (H, W, 3)
            n_frame = normals_data[idx]
            
            # 如果是 uint8 类型，直接显示；如果是 float，映射到 [0, 255]
            if np.issubdtype(n_frame.dtype, np.floating):
                # 法线通常是单位向量，范围 [-1, 1]，映射到 [0, 1] 再到 [0, 255]
                n_vis = ((n_frame + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            else:
                # 已经是 uint8，直接使用
                n_vis = n_frame

            # 转换为 BGR 用于 OpenCV 保存
            n_bgr = cv2.cvtColor(n_vis, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(normals_dir, f'normals_frame_{idx:04d}.png')
            cv2.imwrite(save_path, n_bgr)
            
        print(f"  Saved to {normals_dir}/")

def analyze_zarr(zarr_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Opening Zarr file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    data_group = root['data']
    
    keys_to_plot = ['action', 'arm2_pos', 'arm2_vel', 'hand2_pos', 'hand2_vel','arm2_eef_pos']
    
    for key in keys_to_plot:
        if key not in data_group:
            print(f"Warning: Key '{key}' not found in Zarr data group.")
            continue
            
        print(f"Plotting {key}...")
        data = data_group[key][:]
        
        # Determine shape
        T = data.shape[0]
        dims = data.shape[1] if len(data.shape) > 1 else 1
        
        # Create plot
        plt.figure(figsize=(15, 8))
        
        # Create time axis
        time_steps = np.arange(T)
        
        if dims > 1:
            for i in range(dims):
                plt.plot(time_steps, data[:, i], label=f'Dim {i}')
        else:
            plt.plot(time_steps, data, label=f'Dim 0')
            
        plt.title(f'{key} over time')
        plt.xlabel('Time steps')
        plt.ylabel('Value')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{key}.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Saved plot to {output_file}")

    # 保存相机图片和mask样本
    save_image_samples(data_group, output_dir, num_samples=10)

    print("Analysis complete.")

if __name__ == "__main__":
    default_path = "/home/psibot/chembench/data/zarr_rgb/motion_plan/grasp/100ml玻璃烧杯_20251220_183642.zarr"
    
    parser = argparse.ArgumentParser(description="Analyze Zarr dataset plots.")
    parser.add_argument("--path", type=str, default=default_path, help="Path to Zarr file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots")
    
    args = parser.parse_args()
    
    # 如果没有指定输出目录，则根据 zarr 文件名自动生成
    if args.output is None:
        zarr_name = os.path.basename(args.path.rstrip('/'))
        if zarr_name.endswith('.zarr'):
            zarr_name = zarr_name[:-5]
        output_dir = os.path.join("analysis_results", zarr_name)
    else:
        output_dir = args.output
    
    analyze_zarr(args.path, output_dir)

