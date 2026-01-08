"""
Verify the converted HDF5 data structure and quality.
"""

import h5py
import numpy as np
import os
import argparse


def verify_hdf5_episode(hdf5_path, verbose=False):
    """
    Verify a single HDF5 episode file.
    
    Returns:
        dict: Statistics and validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check required structure
            if 'action' not in f:
                results['errors'].append("Missing 'action' dataset")
                results['valid'] = False
            
            if 'observations' not in f:
                results['errors'].append("Missing 'observations' group")
                results['valid'] = False
                return results
            
            obs = f['observations']
            
            if 'qpos' not in obs:
                results['errors'].append("Missing 'observations/qpos' dataset")
                results['valid'] = False
            
            if 'qvel' not in obs:
                results['errors'].append("Missing 'observations/qvel' dataset")
                results['valid'] = False
            
            if 'images' not in obs:
                results['errors'].append("Missing 'observations/images' group")
                results['valid'] = False
            
            # Get dimensions
            action = f['action'][:]
            qpos = obs['qpos'][:]
            qvel = obs['qvel'][:]
            
            results['stats']['episode_length'] = len(action)
            results['stats']['action_dim'] = action.shape[1]
            results['stats']['qpos_dim'] = qpos.shape[1]
            results['stats']['qvel_dim'] = qvel.shape[1]
            
            # Check dimensions match
            if len(action) != len(qpos) or len(action) != len(qvel):
                results['errors'].append(
                    f"Length mismatch: action={len(action)}, qpos={len(qpos)}, qvel={len(qvel)}"
                )
                results['valid'] = False
            
            if action.shape[1] != qpos.shape[1]:
                # This is OK - action and qpos can have different dimensions
                # For example: action=13, qpos=13 (single arm + hand)
                # or action=14, qpos=14 (dual arm)
                pass
            
            # Check images
            if 'images' in obs:
                images_group = obs['images']
                camera_names = list(images_group.keys())
                results['stats']['camera_names'] = camera_names
                results['stats']['num_cameras'] = len(camera_names)
                
                for cam_name in camera_names:
                    img = images_group[cam_name]
                    if len(img) != len(action):
                        results['errors'].append(
                            f"Image length mismatch for {cam_name}: {len(img)} != {len(action)}"
                        )
                        results['valid'] = False
                    
                    # Check image channels (can be 3 for RGB or 6 for RGB+masked_RGB)
                    if img.ndim == 4:
                        num_channels = img.shape[-1]
                        if num_channels not in [3, 6]:
                            results['warnings'].append(
                                f"Image channels for {cam_name} is {num_channels}, expected 3 or 6"
                            )
                    
                    if img.dtype != np.uint8:
                        results['warnings'].append(
                            f"Image dtype for {cam_name} is {img.dtype}, expected uint8"
                        )
                    
                    if len(camera_names) > 0:
                        results['stats']['image_shape'] = img.shape[1:]
            
            # Check data types
            if action.dtype != np.float32:
                results['warnings'].append(f"Action dtype is {action.dtype}, expected float32")
            if qpos.dtype != np.float32:
                results['warnings'].append(f"Qpos dtype is {qpos.dtype}, expected float32")
            if qvel.dtype != np.float32:
                results['warnings'].append(f"Qvel dtype is {qvel.dtype}, expected float32")
            
            # Check for NaN or Inf
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                results['warnings'].append("Action contains NaN or Inf values")
            if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
                results['warnings'].append("Qpos contains NaN or Inf values")
            if np.any(np.isnan(qvel)) or np.any(np.isinf(qvel)):
                results['warnings'].append("Qvel contains NaN or Inf values")
            
            # Check attributes
            if 'sim' in f.attrs:
                results['stats']['sim'] = f.attrs['sim']
            
            if verbose:
                print(f"\n{os.path.basename(hdf5_path)}:")
                print(f"  Episode length: {results['stats']['episode_length']}")
                print(f"  Action dim: {results['stats']['action_dim']}")
                print(f"  State dim: {results['stats']['qpos_dim']}")
                print(f"  Cameras: {results['stats']['camera_names']}")
                if 'image_shape' in results['stats']:
                    print(f"  Image shape: {results['stats']['image_shape']}")
                
                if results['errors']:
                    print(f"  ❌ Errors: {len(results['errors'])}")
                    for error in results['errors']:
                        print(f"    - {error}")
                
                if results['warnings']:
                    print(f"  ⚠️  Warnings: {len(results['warnings'])}")
                    for warning in results['warnings']:
                        print(f"    - {warning}")
                
                if results['valid'] and not results['warnings']:
                    print(f"  ✓ Valid")
    
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Exception: {str(e)}")
    
    return results


def verify_dataset(dataset_dir, verbose=False):
    """
    Verify all episodes in a dataset directory.
    """
    print(f"Verifying dataset: {dataset_dir}")
    print("=" * 60)
    
    # Find all HDF5 files
    hdf5_files = sorted([
        f for f in os.listdir(dataset_dir) 
        if f.endswith('.hdf5')
    ])
    
    if not hdf5_files:
        print(f"No HDF5 files found in {dataset_dir}")
        return
    
    print(f"Found {len(hdf5_files)} episodes\n")
    
    all_results = []
    valid_count = 0
    error_count = 0
    warning_count = 0
    
    for hdf5_file in hdf5_files:
        hdf5_path = os.path.join(dataset_dir, hdf5_file)
        results = verify_hdf5_episode(hdf5_path, verbose=verbose)
        all_results.append(results)
        
        if results['valid']:
            valid_count += 1
        else:
            error_count += 1
        
        if results['warnings']:
            warning_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {len(hdf5_files)}")
    print(f"Valid: {valid_count}")
    print(f"Errors: {error_count}")
    print(f"Warnings: {warning_count}")
    
    # Aggregate statistics
    if all_results and all_results[0]['stats']:
        stats = all_results[0]['stats']
        print(f"\nDataset configuration:")
        print(f"  Episode length: {stats.get('episode_length', 'N/A')} (varies by episode)")
        print(f"  Action dim: {stats.get('action_dim', 'N/A')}")
        print(f"  State dim: {stats.get('qpos_dim', 'N/A')}")
        print(f"  Cameras: {stats.get('camera_names', [])}")
        if 'image_shape' in stats:
            print(f"  Image shape: {stats['image_shape']}")
    
    if error_count > 0:
        print(f"\n⚠️  {error_count} episodes have errors!")
    elif warning_count > 0:
        print(f"\n⚠️  {warning_count} episodes have warnings")
    else:
        print(f"\n✓ All episodes are valid!")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Verify converted HDF5 dataset')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing HDF5 episodes')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information for each episode')
    
    args = parser.parse_args()
    
    verify_dataset(args.dataset_dir, args.verbose)


if __name__ == '__main__':
    main()

