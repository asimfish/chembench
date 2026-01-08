#!/usr/bin/env python3
"""
数据格式快速检查工具
Quick data format checker
"""

import h5py
import numpy as np
import sys
import os


def check_episode(hdf5_path):
    """检查单个episode的数据格式"""
    
    print(f"\n{'='*70}")
    print(f"检查文件: {os.path.basename(hdf5_path)}")
    print(f"{'='*70}\n")
    
    issues = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查Action
            if 'action' not in f:
                issues.append("❌ 缺少 'action' 数据集")
            else:
                action = f['action']
                print(f"✓ Action: shape={action.shape}, dtype={action.dtype}")
                if action.shape[1] != 13:
                    issues.append(f"⚠️  Action维度是 {action.shape[1]}，期望13")
                if action.dtype != np.float32:
                    issues.append(f"⚠️  Action类型是 {action.dtype}，期望float32")
            
            # 检查Observations
            if 'observations' not in f:
                issues.append("❌ 缺少 'observations' 组")
                return issues
            
            obs = f['observations']
            
            # 检查qpos
            if 'qpos' not in obs:
                issues.append("❌ 缺少 'observations/qpos'")
            else:
                qpos = obs['qpos']
                print(f"✓ Qpos: shape={qpos.shape}, dtype={qpos.dtype}")
                if qpos.shape[1] != 13:
                    issues.append(f"⚠️  Qpos维度是 {qpos.shape[1]}，期望13")
                if qpos.dtype != np.float32:
                    issues.append(f"⚠️  Qpos类型是 {qpos.dtype}，期望float32")
            
            # 检查qvel
            if 'qvel' not in obs:
                issues.append("❌ 缺少 'observations/qvel'")
            else:
                qvel = obs['qvel']
                print(f"✓ Qvel: shape={qvel.shape}, dtype={qvel.dtype}")
                if qvel.shape[1] != 13:
                    issues.append(f"⚠️  Qvel维度是 {qvel.shape[1]}，期望13")
                if qvel.dtype != np.float32:
                    issues.append(f"⚠️  Qvel类型是 {qvel.dtype}，期望float32")
            
            # 检查Images
            if 'images' not in obs:
                issues.append("❌ 缺少 'observations/images'")
            else:
                images = obs['images']
                camera_names = list(images.keys())
                print(f"✓ Cameras: {camera_names}")
                
                expected_cameras = ['head_camera', 'chest_camera', 'third_camera']
                for cam in expected_cameras:
                    if cam not in camera_names:
                        issues.append(f"⚠️  缺少相机: {cam}")
                
                for cam_name in camera_names:
                    img = images[cam_name]
                    print(f"  - {cam_name}: shape={img.shape}, dtype={img.dtype}")
                    
                    if img.dtype != np.uint8:
                        issues.append(f"⚠️  {cam_name}类型是{img.dtype}，期望uint8")
                    
                    if img.ndim != 4:
                        issues.append(f"⚠️  {cam_name}维度数是{img.ndim}，期望4 (T,H,W,C)")
                    else:
                        channels = img.shape[-1]
                        if channels == 6:
                            print(f"    ✓ 6通道 (RGB + masked_RGB)")
                        elif channels == 3:
                            print(f"    ℹ️  3通道 (仅RGB)")
                        else:
                            issues.append(f"⚠️  {cam_name}通道数是{channels}，期望3或6")
    
    except Exception as e:
        issues.append(f"❌ 读取文件出错: {str(e)}")
    
    # 打印问题
    print()
    if issues:
        print("发现问题:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ 所有检查通过！")
        return True


def check_dataset(dataset_dir):
    """检查整个数据集"""
    
    print(f"\n{'='*70}")
    print(f"检查数据集: {dataset_dir}")
    print(f"{'='*70}")
    
    # 查找所有HDF5文件
    hdf5_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    hdf5_files = sorted(hdf5_files)
    
    if not hdf5_files:
        print(f"\n❌ 在 {dataset_dir} 中未找到HDF5文件")
        return False
    
    print(f"\n找到 {len(hdf5_files)} 个episodes")
    
    # 检查第一个和最后一个
    print(f"\n检查第一个episode...")
    first_ok = check_episode(os.path.join(dataset_dir, hdf5_files[0]))
    
    if len(hdf5_files) > 1:
        print(f"\n检查最后一个episode...")
        last_ok = check_episode(os.path.join(dataset_dir, hdf5_files[-1]))
    else:
        last_ok = True
    
    # 汇总
    print(f"\n{'='*70}")
    print("数据集检查汇总")
    print(f"{'='*70}\n")
    print(f"Episodes数量: {len(hdf5_files)}")
    print(f"第一个episode: {'✅ 通过' if first_ok else '❌ 失败'}")
    print(f"最后一个episode: {'✅ 通过' if last_ok else '❌ 失败'}")
    
    if first_ok and last_ok:
        print("\n✅ 数据集格式正确！")
        print("\n可以用于训练。记得:")
        print("  1. 网络需支持6通道输入 (如果使用RGB+mask)")
        print("  2. Action/State维度是13")
        print("  3. 图像归一化: img / 255.0")
        return True
    else:
        print("\n❌ 数据集存在问题，请检查上述错误")
        return False


def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  检查单个文件:")
        print("    python check_data_format.py path/to/episode_0.hdf5")
        print()
        print("  检查整个数据集:")
        print("    python check_data_format.py dataset/grasp_100ml_beaker_v2")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if not os.path.exists(path):
        print(f"❌ 路径不存在: {path}")
        sys.exit(1)
    
    if os.path.isfile(path):
        # 检查单个文件
        success = check_episode(path)
    else:
        # 检查整个目录
        success = check_dataset(path)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()


