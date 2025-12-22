import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2

def save_mask_samples(data_group, output_dir, num_samples=5):
    """保存mask样本图片用于检查"""
    mask_keys = ['head_camera_mask', 'chest_camera_mask']
    rgb_only_keys = ['head_camera_rgb_only', 'chest_camera_rgb_only']
    
    for mask_key, rgb_only_key in zip(mask_keys, rgb_only_keys):
        # 保存单独的RGB图片（如果存在）
        if rgb_only_key in data_group:
            rgb_only_data = data_group[rgb_only_key][:]
            rgb_dir = os.path.join(output_dir, rgb_only_key)
            if not os.path.exists(rgb_dir):
                os.makedirs(rgb_dir)
            
            total_frames = rgb_only_data.shape[0]
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            
            print(f"Saving {num_samples} RGB samples for {rgb_only_key}...")
            
            for idx in sample_indices:
                rgb_frame = rgb_only_data[idx]
                rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                rgb_path = os.path.join(rgb_dir, f'rgb_frame_{idx:04d}.png')
                cv2.imwrite(rgb_path, rgb_bgr)
            
            print(f"  Saved to {rgb_dir}/")
        
        # 保存mask图片
        if mask_key not in data_group:
            print(f"Warning: '{mask_key}' not found, skipping mask visualization.")
            continue
            
        mask_data = data_group[mask_key][:]
        
        # 创建mask输出目录
        mask_dir = os.path.join(output_dir, mask_key)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        # 保存前几帧
        total_frames = mask_data.shape[0]
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        print(f"Saving {num_samples} mask samples for {mask_key}...")
        
        for idx in sample_indices:
            mask_frame = mask_data[idx]
            mask_path = os.path.join(mask_dir, f'mask_frame_{idx:04d}.png')
            cv2.imwrite(mask_path, mask_frame)
        
        print(f"  Saved to {mask_dir}/")

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

    # 保存mask样本图片
    save_mask_samples(data_group, output_dir, num_samples=5)

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

