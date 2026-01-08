"""
Quick data inspection script to visualize converted HDF5 data.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def inspect_episode(hdf5_path, frame_idx=0):
    """
    Inspect a single episode and display information.
    
    Args:
        hdf5_path: Path to HDF5 episode file
        frame_idx: Frame index to visualize (default: 0)
    """
    print(f"Inspecting: {os.path.basename(hdf5_path)}")
    print("=" * 60)
    
    with h5py.File(hdf5_path, 'r') as f:
        # Print structure
        print("\nFile structure:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        
        f.visititems(print_structure)
        
        # Print attributes
        if f.attrs:
            print(f"\nAttributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
        
        # Get data
        action = f['action'][:]
        qpos = f['observations/qpos'][:]
        qvel = f['observations/qvel'][:]
        
        # Print statistics
        print(f"\nData statistics:")
        print(f"  Episode length: {len(action)} timesteps")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"  Qpos range: [{qpos.min():.3f}, {qpos.max():.3f}]")
        print(f"  Qvel range: [{qvel.min():.3f}, {qvel.max():.3f}]")
        
        # Print first frame
        print(f"\nFrame {frame_idx}:")
        print(f"  Action: {action[frame_idx]}")
        print(f"  Qpos: {qpos[frame_idx]}")
        print(f"  Qvel: {qvel[frame_idx]}")
        
        # Get cameras
        images_group = f['observations/images']
        camera_names = list(images_group.keys())
        
        print(f"\nCameras: {camera_names}")
        
        # Visualize images
        num_cameras = len(camera_names)
        if num_cameras > 0:
            fig, axes = plt.subplots(1, num_cameras, figsize=(6 * num_cameras, 6))
            if num_cameras == 1:
                axes = [axes]
            
            for idx, cam_name in enumerate(camera_names):
                img = images_group[cam_name][frame_idx]
                axes[idx].imshow(img)
                axes[idx].set_title(f"{cam_name}\nFrame {frame_idx}")
                axes[idx].axis('off')
                
                print(f"  {cam_name}: shape={img.shape}, range=[{img.min()}, {img.max()}]")
            
            plt.suptitle(f"Episode: {os.path.basename(hdf5_path)}", fontsize=16)
            plt.tight_layout()
            
            # Save figure
            output_path = hdf5_path.replace('.hdf5', f'_frame_{frame_idx}.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {output_path}")
            
            # Show plot
            plt.show()
        
        # Plot action and state trajectories
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Action trajectory
        axes[0].plot(action)
        axes[0].set_title('Action Trajectory')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Action Value')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend([f'dim_{i}' for i in range(action.shape[1])], 
                       loc='upper right', ncol=7, fontsize=8)
        
        # Qpos trajectory
        axes[1].plot(qpos)
        axes[1].set_title('Joint Position (qpos) Trajectory')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Position')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend([f'joint_{i}' for i in range(qpos.shape[1])], 
                       loc='upper right', ncol=7, fontsize=8)
        
        # Qvel trajectory
        axes[2].plot(qvel)
        axes[2].set_title('Joint Velocity (qvel) Trajectory')
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Velocity')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend([f'joint_{i}' for i in range(qvel.shape[1])], 
                       loc='upper right', ncol=7, fontsize=8)
        
        plt.suptitle(f"Episode: {os.path.basename(hdf5_path)}", fontsize=16)
        plt.tight_layout()
        
        # Save trajectory plot
        traj_output_path = hdf5_path.replace('.hdf5', '_trajectories.png')
        plt.savefig(traj_output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Trajectories saved to: {traj_output_path}")
        
        plt.show()


def compare_episodes(hdf5_path1, hdf5_path2):
    """
    Compare two episodes side by side.
    """
    print(f"Comparing episodes:")
    print(f"  Episode 1: {os.path.basename(hdf5_path1)}")
    print(f"  Episode 2: {os.path.basename(hdf5_path2)}")
    print("=" * 60)
    
    with h5py.File(hdf5_path1, 'r') as f1, h5py.File(hdf5_path2, 'r') as f2:
        # Compare lengths
        len1 = len(f1['action'])
        len2 = len(f2['action'])
        print(f"\nEpisode lengths: {len1} vs {len2}")
        
        # Compare dimensions
        action_dim1 = f1['action'].shape[1]
        action_dim2 = f2['action'].shape[1]
        print(f"Action dimensions: {action_dim1} vs {action_dim2}")
        
        # Compare cameras
        cams1 = list(f1['observations/images'].keys())
        cams2 = list(f2['observations/images'].keys())
        print(f"Cameras: {cams1} vs {cams2}")
        
        # Visualize first frame from both
        fig, axes = plt.subplots(2, max(len(cams1), len(cams2)), 
                                 figsize=(6 * max(len(cams1), len(cams2)), 12))
        
        # Episode 1
        for idx, cam_name in enumerate(cams1):
            img = f1['observations/images'][cam_name][0]
            axes[0, idx].imshow(img)
            axes[0, idx].set_title(f"Ep1: {cam_name}")
            axes[0, idx].axis('off')
        
        # Episode 2
        for idx, cam_name in enumerate(cams2):
            img = f2['observations/images'][cam_name][0]
            axes[1, idx].imshow(img)
            axes[1, idx].set_title(f"Ep2: {cam_name}")
            axes[1, idx].axis('off')
        
        plt.suptitle("Episode Comparison (Frame 0)", fontsize=16)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inspect HDF5 episode data')
    parser.add_argument('hdf5_path', type=str, 
                        help='Path to HDF5 episode file')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to visualize (default: 0)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to second HDF5 file for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_episodes(args.hdf5_path, args.compare)
    else:
        inspect_episode(args.hdf5_path, args.frame)


if __name__ == '__main__':
    main()


