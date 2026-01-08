"""
可视化6通道图像 (RGB + masked RGB)
Visualize 6-channel images (RGB + masked RGB)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def visualize_6channel_image(hdf5_path, frame_idx=0, output_dir=None):
    """
    可视化6通道图像数据
    
    Args:
        hdf5_path: HDF5文件路径
        frame_idx: 帧索引
        output_dir: 输出目录
    """
    print(f"Loading: {os.path.basename(hdf5_path)}")
    print(f"Frame: {frame_idx}")
    print("=" * 60)
    
    with h5py.File(hdf5_path, 'r') as f:
        # 获取相机列表
        camera_names = list(f['observations/images'].keys())
        print(f"Cameras: {camera_names}\n")
        
        num_cameras = len(camera_names)
        
        # 创建图表：每个相机3行（RGB, masked RGB, overlay）
        fig, axes = plt.subplots(num_cameras, 3, figsize=(15, 5 * num_cameras))
        if num_cameras == 1:
            axes = axes.reshape(1, -1)
        
        for cam_idx, cam_name in enumerate(camera_names):
            img_6ch = f[f'observations/images/{cam_name}'][frame_idx]
            
            # 分离RGB和masked RGB
            rgb = img_6ch[:, :, :3]  # 前3通道
            masked_rgb = img_6ch[:, :, 3:]  # 后3通道
            
            # 显示RGB
            axes[cam_idx, 0].imshow(rgb)
            axes[cam_idx, 0].set_title(f'{cam_name}\nRGB (channels 0-2)')
            axes[cam_idx, 0].axis('off')
            
            # 显示masked RGB
            axes[cam_idx, 1].imshow(masked_rgb)
            axes[cam_idx, 1].set_title(f'{cam_name}\nMasked RGB (channels 3-5)')
            axes[cam_idx, 1].axis('off')
            
            # 显示overlay（半透明叠加）
            overlay = np.copy(rgb).astype(float)
            mask_binary = (masked_rgb.sum(axis=2) > 0).astype(float)
            overlay_masked = masked_rgb.astype(float) * 0.7 + rgb.astype(float) * 0.3
            
            # 创建混合图像
            overlay_img = np.copy(rgb).astype(float)
            for c in range(3):
                overlay_img[:, :, c] = np.where(
                    mask_binary > 0,
                    overlay_masked[:, :, c],
                    rgb[:, :, c]
                )
            overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
            
            axes[cam_idx, 2].imshow(overlay_img)
            axes[cam_idx, 2].set_title(f'{cam_name}\nOverlay')
            axes[cam_idx, 2].axis('off')
            
            print(f"{cam_name}:")
            print(f"  RGB shape: {rgb.shape}")
            print(f"  RGB range: [{rgb.min()}, {rgb.max()}]")
            print(f"  Masked RGB range: [{masked_rgb.min()}, {masked_rgb.max()}]")
            print(f"  Mask coverage: {(mask_binary.sum() / mask_binary.size * 100):.1f}%")
            print()
        
        plt.suptitle(f"6-Channel Image Visualization\n{os.path.basename(hdf5_path)} - Frame {frame_idx}", 
                     fontsize=16, y=0.995)
        plt.tight_layout()
        
        # 保存
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, 
                f"{os.path.basename(hdf5_path).replace('.hdf5', '')}_frame{frame_idx}_6ch.png"
            )
        else:
            output_path = hdf5_path.replace('.hdf5', f'_frame{frame_idx}_6ch.png')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}")
        
        plt.show()


def compare_original_vs_6channel(zarr_path, hdf5_path, frame_idx=0):
    """
    对比原始Zarr数据和转换后的6通道HDF5数据
    """
    import zarr
    
    print("Comparing original Zarr vs converted HDF5")
    print("=" * 60)
    
    # 打开Zarr
    z = zarr.open(zarr_path, mode='r')
    episode_ends = z['meta/episode_ends'][:]
    
    # 打开HDF5
    with h5py.File(hdf5_path, 'r') as f:
        # 提取episode编号
        episode_idx = int(os.path.basename(hdf5_path).replace('episode_', '').replace('.hdf5', ''))
        
        # 计算在Zarr中的帧索引
        episode_start = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
        zarr_frame_idx = episode_start + frame_idx
        
        print(f"Episode: {episode_idx}")
        print(f"Frame in episode: {frame_idx}")
        print(f"Frame in Zarr: {zarr_frame_idx}\n")
        
        # 比较第一个相机
        cam_name = 'chest_camera'
        
        # 从Zarr读取
        rgb_zarr = z['data'][f'{cam_name}_rgb'][zarr_frame_idx]
        mask_zarr = z['data'][f'{cam_name}_mask'][zarr_frame_idx]
        
        # 从HDF5读取
        img_6ch = f[f'observations/images/{cam_name}'][frame_idx]
        rgb_hdf5 = img_6ch[:, :, :3]
        masked_rgb_hdf5 = img_6ch[:, :, 3:]
        
        # 重建masked RGB用于对比
        mask_3ch = np.stack([mask_zarr, mask_zarr, mask_zarr], axis=-1)
        masked_rgb_expected = (rgb_zarr * (mask_3ch / 255.0)).astype(np.uint8)
        
        # 可视化对比
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Zarr原始数据
        axes[0, 0].imshow(rgb_zarr)
        axes[0, 0].set_title('Zarr: Original RGB')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_zarr, cmap='gray')
        axes[0, 1].set_title('Zarr: Mask')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(masked_rgb_expected)
        axes[0, 2].set_title('Zarr: RGB × Mask (computed)')
        axes[0, 2].axis('off')
        
        # HDF5转换后数据
        axes[1, 0].imshow(rgb_hdf5)
        axes[1, 0].set_title('HDF5: RGB (ch 0-2)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.mean(img_6ch[:, :, 3:], axis=2), cmap='gray')
        axes[1, 1].set_title('HDF5: Masked RGB mean (ch 3-5)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(masked_rgb_hdf5)
        axes[1, 2].set_title('HDF5: Masked RGB (ch 3-5)')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Comparison: Zarr vs HDF5\n{cam_name} - Frame {frame_idx}', fontsize=16)
        plt.tight_layout()
        
        # 验证一致性
        print("Verification:")
        print(f"  RGB match: {np.allclose(rgb_zarr, rgb_hdf5)}")
        print(f"  Masked RGB match: {np.allclose(masked_rgb_expected, masked_rgb_hdf5)}")
        print(f"  RGB difference: max={np.abs(rgb_zarr.astype(int) - rgb_hdf5.astype(int)).max()}")
        print(f"  Masked RGB difference: max={np.abs(masked_rgb_expected.astype(int) - masked_rgb_hdf5.astype(int)).max()}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize 6-channel images')
    parser.add_argument('hdf5_path', type=str, help='Path to HDF5 file')
    parser.add_argument('--frame', type=int, default=0, help='Frame index (default: 0)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for images')
    parser.add_argument('--compare_zarr', type=str, default=None, 
                        help='Path to original Zarr file for comparison')
    
    args = parser.parse_args()
    
    if args.compare_zarr:
        compare_original_vs_6channel(args.compare_zarr, args.hdf5_path, args.frame)
    else:
        visualize_6channel_image(args.hdf5_path, args.frame, args.output_dir)


if __name__ == '__main__':
    main()


