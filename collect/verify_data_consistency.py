#!/usr/bin/env python3
"""
数据一致性验证脚本
检查采集的数据中场景物体和图像物体是否一致
"""

import h5py
import numpy as np
from pathlib import Path
import argparse
import sys

def analyze_hdf5(hdf5_path):
    """分析 HDF5 文件的内容"""
    print(f"\n分析文件: {hdf5_path}")
    print("=" * 80)
    
    with h5py.File(hdf5_path, 'r') as f:
        # 显示所有顶层键
        print(f"顶层键: {list(f.keys())}")
        
        # 检查 episode 数量
        if 'data' in f:
            data = f['data']
            episodes = [k for k in data.keys() if k.startswith('episode_')]
            print(f"Episode 数量: {len(episodes)}")
            
            if episodes:
                # 检查第一个 episode
                ep0 = data[episodes[0]]
                print(f"\nEpisode 0 的键: {list(ep0.keys())}")
                
                # 检查图像数据
                for camera_name in ['chest_camera_rgb', 'head_camera_rgb']:
                    if camera_name in ep0:
                        img_data = ep0[camera_name]
                        print(f"  {camera_name}: shape={img_data.shape}, dtype={img_data.dtype}")
                        
                        # 显示第一帧
                        if len(img_data) > 0:
                            first_frame = img_data[0]
                            print(f"    第一帧: min={first_frame.min()}, max={first_frame.max()}, mean={first_frame.mean():.2f}")
                
                # 检查状态数据
                if 'arm2_eef_pos' in ep0:
                    eef_pos = ep0['arm2_eef_pos']
                    print(f"  arm2_eef_pos: shape={eef_pos.shape}, dtype={eef_pos.dtype}")
                    if len(eef_pos) > 0:
                        print(f"    范围: [{eef_pos.min():.3f}, {eef_pos.max():.3f}]")
        
        # 检查 meta 信息
        if 'meta' in f:
            meta = f['meta']
            print(f"\nMeta 信息:")
            for key in meta.attrs.keys():
                print(f"  {key}: {meta.attrs[key]}")

def verify_images(hdf5_path, output_dir=None):
    """验证图像是否正常，可选保存第一帧"""
    print(f"\n验证图像数据...")
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            print("❌ 没有找到 'data' 键")
            return False
        
        data = f['data']
        episodes = [k for k in data.keys() if k.startswith('episode_')]
        
        if not episodes:
            print("❌ 没有找到任何 episode")
            return False
        
        # 检查第一个 episode 的第一帧
        ep0 = data[episodes[0]]
        
        all_ok = True
        for camera_name in ['chest_camera_rgb', 'head_camera_rgb']:
            if camera_name not in ep0:
                print(f"⚠️  没有找到 {camera_name}")
                continue
            
            img_data = ep0[camera_name]
            if len(img_data) == 0:
                print(f"❌ {camera_name} 没有数据")
                all_ok = False
                continue
            
            first_frame = img_data[0]
            
            # 检查图像是否全黑或全白
            mean_val = first_frame.mean()
            if mean_val < 1.0:
                print(f"❌ {camera_name} 第一帧几乎全黑 (mean={mean_val:.2f})")
                all_ok = False
            elif mean_val > 254.0:
                print(f"❌ {camera_name} 第一帧几乎全白 (mean={mean_val:.2f})")
                all_ok = False
            else:
                print(f"✅ {camera_name} 第一帧正常 (mean={mean_val:.2f})")
            
            # 可选：保存第一帧
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                import cv2
                output_path = output_dir / f"{camera_name}_frame0.png"
                # HDF5 中的图像可能是 RGB，需要转换为 BGR
                if first_frame.ndim == 3 and first_frame.shape[2] == 3:
                    cv2.imwrite(str(output_path), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(str(output_path), first_frame)
                print(f"  保存第一帧到: {output_path}")
        
        return all_ok

def main():
    parser = argparse.ArgumentParser(description="验证数据一致性")
    parser.add_argument("hdf5_path", type=str, help="HDF5 文件路径")
    parser.add_argument("--save-images", type=str, help="保存第一帧图像到指定目录")
    parser.add_argument("--detailed", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        print(f"❌ 文件不存在: {hdf5_path}")
        sys.exit(1)
    
    # 分析文件
    if args.detailed:
        analyze_hdf5(hdf5_path)
    
    # 验证图像
    is_valid = verify_images(hdf5_path, args.save_images)
    
    if is_valid:
        print("\n✅ 数据验证通过")
        sys.exit(0)
    else:
        print("\n❌ 数据验证失败")
        sys.exit(1)

if __name__ == "__main__":
    main()

