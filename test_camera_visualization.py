#!/usr/bin/env python3
"""
快速测试相机点云可视化
"""

import subprocess
import sys
from pathlib import Path

ZARR_PATH = "/home/psibot/chembench/data/zarr_point_cloud/motion_plan/grasp/100ml玻璃烧杯.zarr"

def test_single_camera():
    """测试单相机可视化"""
    print("\n" + "="*60)
    print("测试: 单相机点云可视化（胸部相机，RGB）")
    print("="*60)
    
    cmd = [
        sys.executable,
        "visualize_camera_pointcloud.py",
        "--zarr", ZARR_PATH,
        "--mode", "single",
        "--camera", "chest_camera_pointcloud",
        "--frames", "0",
        "--use_rgb"
    ]
    
    print("命令:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def test_all_cameras():
    """测试所有相机对比"""
    print("\n" + "="*60)
    print("测试: 所有相机对比")
    print("="*60)
    
    cmd = [
        sys.executable,
        "visualize_camera_pointcloud.py",
        "--zarr", ZARR_PATH,
        "--mode", "all",
        "--frames", "0",
        "--use_rgb"
    ]
    
    print("命令:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def test_compare():
    """测试对比模式"""
    print("\n" + "="*60)
    print("测试: 对比相机点云与真值点云")
    print("="*60)
    
    cmd = [
        sys.executable,
        "visualize_camera_pointcloud.py",
        "--zarr", ZARR_PATH,
        "--mode", "compare",
        "--camera", "chest_camera_pointcloud",
        "--frames", "0",
        "--use_rgb"
    ]
    
    print("命令:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def main():
    print("="*60)
    print("相机点云可视化测试")
    print("="*60)
    
    if not Path(ZARR_PATH).exists():
        print(f"错误: Zarr数据不存在: {ZARR_PATH}")
        return 1
    
    print(f"数据路径: {ZARR_PATH}")
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 单相机可视化（胸部相机，RGB）")
    print("2. 所有相机对比")
    print("3. 对比相机与真值")
    print("4. 运行所有测试")
    
    choice = input("\n输入选择 (1-4): ").strip()
    
    results = []
    
    if choice == "1":
        results.append(("单相机", test_single_camera()))
    elif choice == "2":
        results.append(("所有相机", test_all_cameras()))
    elif choice == "3":
        results.append(("对比真值", test_compare()))
    elif choice == "4":
        print("\n运行所有测试...")
        results.append(("单相机", test_single_camera()))
        results.append(("所有相机", test_all_cameras()))
        results.append(("对比真值", test_compare()))
    else:
        print("无效选择")
        return 1
    
    # 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{name}: {status}")
    
    all_success = all(r[1] for r in results)
    
    if all_success:
        print("\n✓ 所有测试通过!")
        return 0
    else:
        print("\n✗ 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())




