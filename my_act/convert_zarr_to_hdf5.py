"""
Convert Zarr dataset to HDF5 format for ACT training.
This script converts data from Zarr format (used in your chembench dataset) 
to HDF5 format expected by ACT.
"""

import zarr
import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm


def convert_zarr_to_hdf5(zarr_path, output_dir, camera_names=None, use_rgb_mask=True):
    """
    Convert Zarr dataset to HDF5 episodes for ACT.
    
    Args:
        zarr_path: Path to the .zarr file
        output_dir: Directory to save HDF5 episodes
        camera_names: List of camera names to use (e.g., ['head_camera', 'chest_camera', 'third_camera'])
        use_rgb_mask: If True, concatenate RGB (3ch) + masked RGB (3ch) = 6 channels per camera
    """
    # Default camera names if not specified
    if camera_names is None:
        camera_names = ['head_camera', 'chest_camera', 'third_camera']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open Zarr dataset
    print(f"Opening Zarr dataset: {zarr_path}")
    store = zarr.open(zarr_path, mode='r')
    
    # Get episode information
    episode_ends = store['meta']['episode_ends'][:]
    num_episodes = len(episode_ends)
    print(f"Found {num_episodes} episodes")
    
    # Get data dimensions
    print(f"\nDataset information:")
    print(f"  Action shape: {store['data']['action'].shape}")
    print(f"  Arm position shape: {store['data']['arm2_pos'].shape}")
    print(f"  Hand position shape: {store['data']['hand2_pos'].shape}")
    print(f"  Arm velocity shape: {store['data']['arm2_vel'].shape}")
    print(f"  Hand velocity shape: {store['data']['hand2_vel'].shape}")
    for cam in camera_names:
        cam_key = f"{cam}_rgb"
        if cam_key in store['data']:
            print(f"  {cam} RGB shape: {store['data'][cam_key].shape}")
        mask_key = f"{cam}_mask"
        if mask_key in store['data']:
            print(f"  {cam} Mask shape: {store['data'][mask_key].shape}")
    
    # Convert each episode
    episode_start = 0
    for episode_idx in tqdm(range(num_episodes), desc="Converting episodes"):
        episode_end = episode_ends[episode_idx]
        episode_len = episode_end - episode_start
        
        # Extract data for this episode
        action = store['data']['action'][episode_start:episode_end]  # Shape: (T, 13)
        
        # State: arm_pos (7) + hand_pos (6) = 13 dimensions
        arm_pos = store['data']['arm2_pos'][episode_start:episode_end]  # Shape: (T, 7)
        hand_pos = store['data']['hand2_pos'][episode_start:episode_end]  # Shape: (T, 6)
        qpos = np.concatenate([arm_pos, hand_pos], axis=1)  # Shape: (T, 13)
        
        # Velocity: arm_vel (7) + hand_vel (6) = 13 dimensions
        arm_vel = store['data']['arm2_vel'][episode_start:episode_end]  # Shape: (T, 7)
        hand_vel = store['data']['hand2_vel'][episode_start:episode_end]  # Shape: (T, 6)
        qvel = np.concatenate([arm_vel, hand_vel], axis=1)  # Shape: (T, 13)
        
        # Get camera images with RGB + masked RGB (6 channels)
        camera_images = {}
        for cam_name in camera_names:
            cam_rgb_key = f"{cam_name}_rgb"
            cam_mask_key = f"{cam_name}_mask"
            
            if cam_rgb_key in store['data'] and cam_mask_key in store['data']:
                rgb = store['data'][cam_rgb_key][episode_start:episode_end]  # (T, H, W, 3)
                mask = store['data'][cam_mask_key][episode_start:episode_end]  # (T, H, W)
                
                # Convert to uint8 if not already
                if rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.uint8)
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                
                if use_rgb_mask:
                    # Expand mask to 3 channels: (T, H, W) -> (T, H, W, 3)
                    mask_3ch = np.stack([mask, mask, mask], axis=-1)
                    
                    # Apply mask to RGB: masked RGB = RGB * (mask / 255)
                    masked_rgb = (rgb * (mask_3ch / 255.0)).astype(np.uint8)
                    
                    # Concatenate RGB (3ch) + masked RGB (3ch) = 6 channels
                    rgb_masked = np.concatenate([rgb, masked_rgb], axis=-1)  # (T, H, W, 6)
                    camera_images[cam_name] = rgb_masked
                else:
                    # Use only RGB (3 channels)
                    camera_images[cam_name] = rgb
        
        # Save as HDF5
        output_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(output_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as hdf5_file:
            # Set attributes (match the sim data format)
            hdf5_file.attrs['sim'] = True
            
            # Create observations group first
            obs_group = hdf5_file.create_group('observations')
            
            # Create images group
            images_group = obs_group.create_group('images')
            for cam_name, images in camera_images.items():
                # Create dataset with chunking for better I/O
                img_shape = images.shape
                images_group.create_dataset(
                    cam_name, 
                    data=images,
                    dtype='uint8',
                    chunks=(1, img_shape[1], img_shape[2], img_shape[3])
                )
            
            # Save qpos and qvel
            obs_group.create_dataset('qpos', data=qpos.astype(np.float32))
            obs_group.create_dataset('qvel', data=qvel.astype(np.float32))
            
            # Save action (at root level, not in observations)
            hdf5_file.create_dataset('action', data=action.astype(np.float32))
        
        # Update start for next episode
        episode_start = episode_end
    
    print(f"\nConversion complete! {num_episodes} episodes saved to {output_dir}")
    print(f"\nDataset statistics:")
    print(f"  Action dimension: {action.shape[1]}")
    print(f"  State dimension (qpos): {qpos.shape[1]}")
    print(f"  State dimension (qvel): {qvel.shape[1]}")
    print(f"  Episode length: {episode_len} timesteps")
    print(f"  Camera names: {list(camera_images.keys())}")
    if camera_images:
        img_shape = list(camera_images.values())[0].shape[1:]
        print(f"  Image shape: {img_shape}")
        print(f"  Image channels: {img_shape[-1]} ({'RGB+masked_RGB' if img_shape[-1] == 6 else 'RGB only'})")
    
    return num_episodes, episode_len, list(camera_images.keys())


def main():
    parser = argparse.ArgumentParser(description='Convert Zarr dataset to HDF5 for ACT training')
    parser.add_argument('--zarr_path', type=str, required=True,
                        help='Path to input .zarr file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save HDF5 episodes')
    parser.add_argument('--camera_names', type=str, nargs='+', 
                        default=['head_camera', 'chest_camera', 'third_camera'],
                        help='List of camera names to use')
    parser.add_argument('--no_rgb_mask', action='store_true',
                        help='Use only RGB (3ch) instead of RGB+masked_RGB (6ch)')
    
    args = parser.parse_args()
    
    convert_zarr_to_hdf5(args.zarr_path, args.output_dir, args.camera_names, 
                         use_rgb_mask=not args.no_rgb_mask)


if __name__ == '__main__':
    main()

