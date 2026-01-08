#!/usr/bin/env python3
"""
Convert grasp dataset from raw zarr format to DP3 training format.

This script extracts:
- point_cloud: third_camera_pointcloud (T, 2048, 6)
- state: arm2_pos(7) + hand2_pos(6) = (T, 13)
- action: arm2_action(7) + hand2_action(6) from action[:, 13:26] or action[:, :13] = (T, 13)
- episode_ends: from meta/episode_ends

Usage:
    python scripts/convert_grasp_to_dp3.py \
        --input /path/to/100ml玻璃烧杯.zarr \
        --output /path/to/output/grasp_100ml.zarr
"""

import argparse
import zarr
import numpy as np
from pathlib import Path
from termcolor import cprint


def convert_grasp_data(input_path: str, output_path: str):
    """
    Convert grasp data from raw format to DP3 format.
    
    Args:
        input_path: Path to input zarr file
        output_path: Path to output zarr file
    """
    cprint(f'[ConvertGraspData] Loading data from {input_path}', 'cyan')
    
    # Open input zarr
    input_zarr = zarr.open(input_path, 'r')
    
    # Print input data info
    cprint('\n=== Input Data Info ===', 'yellow')
    print(f"Available keys in data:")
    for key in input_zarr['data'].keys():
        print(f"  - {key}: {input_zarr['data'][key].shape}")
    
    # Check if we have point cloud data
    if 'third_camera_pointcloud' not in input_zarr['data']:
        cprint('❌ Error: third_camera_pointcloud not found in input data!', 'red')
        cprint('   Available keys:', 'red')
        for key in input_zarr['data'].keys():
            print(f"     - {key}")
        return
    
    episode_ends = input_zarr['meta']['episode_ends'][:]
    print(f"\nepisode_ends: {episode_ends.shape}, {len(episode_ends)} episodes")
    
    # Extract data
    cprint('\n[ConvertGraspData] Extracting data...', 'cyan')
    
    # Point cloud: use third camera only
    point_cloud = input_zarr['data']['third_camera_pointcloud'][:]  # (T, 2048, 6)
    cprint(f'✓ point_cloud: {point_cloud.shape}', 'green')
    
    # State: arm2_pos(7) + hand2_pos(6) = 13
    arm2_pos = input_zarr['data']['arm2_pos'][:]  # (T, 7)
    hand2_pos = input_zarr['data']['hand2_pos'][:]  # (T, 6)
    state = np.concatenate([arm2_pos, hand2_pos], axis=-1)  # (T, 13)
    cprint(f'✓ state: {state.shape} = arm2_pos(7) + hand2_pos(6)', 'green')
    
    # Action: 需要判断是单手数据还是双手数据
    action_full = input_zarr['data']['action'][:]  # (T, 13) or (T, 26)
    
    if action_full.shape[-1] == 26:
        # 双手数据：提取右手动作 action[:, 13:26]
        action = action_full[:, 13:26]  # (T, 13)
        cprint(f'✓ action: {action.shape} = arm2_action(7) + hand2_action(6) [from 26-dim]', 'green')
    elif action_full.shape[-1] == 13:
        # 单手数据：直接使用
        action = action_full  # (T, 13)
        cprint(f'✓ action: {action.shape} = arm2_action(7) + hand2_action(6) [from 13-dim]', 'green')
    else:
        cprint(f'❌ Error: Unexpected action dimension: {action_full.shape[-1]}', 'red')
        cprint(f'   Expected 13 (single hand) or 26 (two hands)', 'red')
        return
    
    # Optional: RGB image for visualization
    if 'third_camera_rgb' in input_zarr['data']:
        third_camera_rgb = input_zarr['data']['third_camera_rgb'][:]  # (T, H, W, 3)
        cprint(f'✓ img: {third_camera_rgb.shape}', 'green')
    else:
        third_camera_rgb = None
        cprint('⚠ Warning: third_camera_rgb not found, skipping...', 'yellow')
    
    # Print data statistics
    cprint('\n=== Data Statistics ===', 'yellow')
    print(f"point_cloud: min={point_cloud.min():.3f}, max={point_cloud.max():.3f}")
    print(f"state: min={state.min():.3f}, max={state.max():.3f}")
    print(f"action: min={action.min():.3f}, max={action.max():.3f}")
    print(f"Total steps: {point_cloud.shape[0]}")
    print(f"Total episodes: {len(episode_ends)}")
    
    # Verify episode consistency
    assert point_cloud.shape[0] == state.shape[0] == action.shape[0]
    if third_camera_rgb is not None:
        assert point_cloud.shape[0] == third_camera_rgb.shape[0]
    assert episode_ends[-1] == point_cloud.shape[0], \
        f"Last episode_end {episode_ends[-1]} != total steps {point_cloud.shape[0]}"
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create output zarr
    cprint(f'\n[ConvertGraspData] Creating output zarr at {output_path}', 'cyan')
    output_zarr = zarr.open(str(output_path), 'w')
    
    # Save data
    cprint('[ConvertGraspData] Saving data...', 'cyan')
    
    data_group = output_zarr.create_group('data')
    data_group.create_dataset('point_cloud', data=point_cloud, 
                             chunks=(41, 2048, 6), dtype=np.float32)
    data_group.create_dataset('state', data=state, 
                             chunks=(52, 13), dtype=np.float32)
    data_group.create_dataset('action', data=action, 
                             chunks=(52, 13), dtype=np.float32)
    
    if third_camera_rgb is not None:
        data_group.create_dataset('img', data=third_camera_rgb, 
                                 chunks=(14, 224, 224, 3), dtype=np.uint8)
    
    meta_group = output_zarr.create_group('meta')
    meta_group.create_dataset('episode_ends', data=episode_ends, dtype=np.int64)
    
    cprint(f'\n✅ Successfully converted data to {output_path}', 'green')
    
    # Print output data info
    cprint('\n=== Output Data Info ===', 'yellow')
    print(f"point_cloud: {output_zarr['data']['point_cloud'].shape}")
    print(f"state: {output_zarr['data']['state'].shape}")
    print(f"action: {output_zarr['data']['action'].shape}")
    if third_camera_rgb is not None:
        print(f"img: {output_zarr['data']['img'].shape}")
    print(f"episode_ends: {output_zarr['meta']['episode_ends'].shape}")
    
    # Verify data integrity
    cprint('\n[ConvertGraspData] Verifying data integrity...', 'cyan')
    output_data = zarr.open(str(output_path), 'r')
    
    assert np.allclose(output_data['data']['point_cloud'][:], point_cloud)
    assert np.allclose(output_data['data']['state'][:], state)
    assert np.allclose(output_data['data']['action'][:], action)
    if third_camera_rgb is not None:
        assert np.array_equal(output_data['data']['img'][:], third_camera_rgb)
    assert np.array_equal(output_data['meta']['episode_ends'][:], episode_ends)
    
    cprint('✅ Data integrity verified!', 'green')
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert grasp data to DP3 format')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input zarr file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output zarr file'
    )
    
    args = parser.parse_args()
    
    convert_grasp_data(args.input, args.output)


if __name__ == '__main__':
    main()




