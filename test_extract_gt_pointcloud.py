#!/usr/bin/env python3
"""
测试脚本：提取100ml玻璃烧杯的真值点云
"""

import subprocess
import sys
from pathlib import Path

# 设置路径
USD_PATH = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd"
ZARR_PATH = "/home/psibot/chembench/data/zarr_final/motion_plan/grasp/part1/100ml玻璃烧杯.zarr"
OUTPUT_PATH = "/home/psibot/chembench/data/zarr_final/motion_plan/grasp/part1/100ml玻璃烧杯_ground_truth.zarr"

def main():
    print("=" * 60)
    print("测试: 从USD提取点云并生成真值轨迹")
    print("=" * 60)
    
    # 检查输入文件
    if not Path(USD_PATH).exists():
        print(f"错误: USD文件不存在: {USD_PATH}")
        return 1
    
    if not Path(ZARR_PATH).exists():
        print(f"错误: Zarr文件不存在: {ZARR_PATH}")
        return 1
    
    print(f"\nUSD文件: {USD_PATH}")
    print(f"Zarr文件: {ZARR_PATH}")
    print(f"输出路径: {OUTPUT_PATH}\n")
    
    # 运行提取脚本
    cmd = [
        "/home/psibot/chembench/psilab/isaaclab.sh",
        "-p",
        "/home/psibot/chembench/extract_ground_truth_pointcloud.py",
        "--usd", USD_PATH,
        "--zarr", ZARR_PATH,
        "--num_points", "10000",
        "--output", OUTPUT_PATH,
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd, cwd="/home/psibot/chembench")
    
    if result.returncode == 0:
        print("\n✓ 成功!")
        
        # 验证输出
        if Path(OUTPUT_PATH).exists():
            print(f"✓ 输出文件已创建: {OUTPUT_PATH}")
            
            # 尝试读取验证
            try:
                import zarr
                z = zarr.open(OUTPUT_PATH, mode='r')
                if 'ground_truth_pointcloud' in z:
                    data = z['ground_truth_pointcloud']
                    print(f"✓ 数据形状: {data.shape}")
                    print(f"✓ 时间步数: {data.shape[0]}")
                    print(f"✓ 点云点数: {data.shape[1]}")
                    print(f"✓ 元数据: {dict(z.attrs)}")
                else:
                    print("⚠ 未找到 ground_truth_pointcloud 数据集")
            except Exception as e:
                print(f"⚠ 读取输出文件时出错: {e}")
        else:
            print(f"⚠ 输出文件未创建: {OUTPUT_PATH}")
    else:
        print(f"\n✗ 失败 (返回码: {result.returncode})")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

