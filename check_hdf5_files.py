#!/usr/bin/env python3
"""
检查 HDF5 文件的完整性
找出损坏的文件并列出它们
"""

import os
import sys
import h5py
from pathlib import Path

def check_hdf5_file(filepath):
    """检查单个 HDF5 文件"""
    try:
        with h5py.File(filepath, 'r') as f:
            # 尝试读取文件的基本信息
            keys = list(f.keys())
            return True, f"✅ OK - {len(keys)} keys"
    except Exception as e:
        return False, f"❌ ERROR: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("用法: python check_hdf5_files.py <hdf5文件夹路径>")
        sys.exit(1)
    
    h5_dir = sys.argv[1]
    
    if not os.path.exists(h5_dir):
        print(f"错误：目录不存在: {h5_dir}")
        sys.exit(1)
    
    # 查找所有 HDF5 文件
    hdf5_files = []
    for file in os.listdir(h5_dir):
        if file.endswith('.hdf5') or file.endswith('.h5'):
            hdf5_files.append(file)
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    print("=" * 80)
    
    good_files = []
    bad_files = []
    
    for idx, filename in enumerate(sorted(hdf5_files), 1):
        filepath = os.path.join(h5_dir, filename)
        is_ok, message = check_hdf5_file(filepath)
        
        status = "✅" if is_ok else "❌"
        print(f"[{idx}/{len(hdf5_files)}] {status} {filename}")
        print(f"    {message}")
        
        if is_ok:
            good_files.append(filename)
        else:
            bad_files.append((filename, message))
    
    print("=" * 80)
    print(f"\n总结:")
    print(f"  ✅ 正常文件: {len(good_files)}")
    print(f"  ❌ 损坏文件: {len(bad_files)}")
    
    if bad_files:
        print(f"\n损坏的文件列表:")
        for filename, error in bad_files:
            print(f"  - {filename}")
            print(f"    错误: {error}")
        
        print(f"\n建议的删除命令:")
        for filename, _ in bad_files:
            print(f"  rm {os.path.join(h5_dir, filename)}")

if __name__ == "__main__":
    main()

