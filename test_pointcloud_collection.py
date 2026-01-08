#!/usr/bin/env python3
"""
测试点云采集功能

使用方法:
    python test_pointcloud_collection.py
"""

import torch
import numpy as np


def test_camera_pointcloud_utils():
    """测试 camera_pointcloud_utils 模块"""
    print("=" * 60)
    print("测试 1: camera_pointcloud_utils 导入")
    print("=" * 60)
    
    try:
        from psilab.utils.camera_pointcloud_utils import (
            add_pointcloud_method_to_camera,
            get_pointcloud_from_camera
        )
        print("✅ 成功导入 camera_pointcloud_utils")
        print(f"  - add_pointcloud_method_to_camera: {add_pointcloud_method_to_camera}")
        print(f"  - get_pointcloud_from_camera: {get_pointcloud_from_camera}")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_isaac_lab_utils():
    """测试 Isaac Lab 点云工具"""
    print("\n" + "=" * 60)
    print("测试 2: Isaac Lab 点云工具")
    print("=" * 60)
    
    try:
        from isaaclab.sensors.camera.utils import (
            create_pointcloud_from_rgbd,
            create_pointcloud_from_depth
        )
        print("✅ 成功导入 Isaac Lab 点云工具")
        print(f"  - create_pointcloud_from_rgbd: {create_pointcloud_from_rgbd}")
        print(f"  - create_pointcloud_from_depth: {create_pointcloud_from_depth}")
        
        # 测试基本功能
        print("\n测试基本功能...")
        
        # 创建模拟数据
        intrinsic = torch.tensor([
            [400.0, 0, 320.0],
            [0, 400.0, 240.0],
            [0, 0, 1.0]
        ], dtype=torch.float32)
        
        depth = torch.rand(480, 640) * 2.0  # 随机深度 0-2米
        rgb = torch.randint(0, 255, (480, 640, 3), dtype=torch.float32)
        
        # 生成点云
        points, colors = create_pointcloud_from_rgbd(
            intrinsic_matrix=intrinsic,
            depth=depth,
            rgb=rgb,
            normalize_rgb=True,
            device="cpu"
        )
        
        print(f"✅ 点云生成成功")
        print(f"  - Points shape: {points.shape}")
        print(f"  - Colors shape: {colors.shape}")
        print(f"  - Points range: [{points.min():.3f}, {points.max():.3f}]")
        print(f"  - Colors range: [{colors.min():.3f}, {colors.max():.3f}]")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_data_collect_utils():
    """测试 data_collect_utils 集成"""
    print("\n" + "=" * 60)
    print("测试 3: data_collect_utils 集成")
    print("=" * 60)
    
    try:
        from psilab.utils.data_collect_utils import (
            POINTCLOUD_UTILS_AVAILABLE,
        )
        print(f"✅ data_collect_utils 导入成功")
        print(f"  - POINTCLOUD_UTILS_AVAILABLE: {POINTCLOUD_UTILS_AVAILABLE}")
        
        if POINTCLOUD_UTILS_AVAILABLE:
            print("  - ✅ 点云工具可用，数据采集时会自动保存点云")
        else:
            print("  - ⚠️  点云工具不可用，点云采集将被跳过")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_zarr_utils():
    """测试 zarr_utils 点云处理"""
    print("\n" + "=" * 60)
    print("测试 4: zarr_utils 点云处理")
    print("=" * 60)
    
    try:
        from psilab.utils.zarr_utils import (
            depth_to_pointcloud,
            furthest_point_sampling,
            ISAACLAB_AVAILABLE
        )
        print(f"✅ zarr_utils 导入成功")
        print(f"  - ISAACLAB_AVAILABLE: {ISAACLAB_AVAILABLE}")
        print(f"  - depth_to_pointcloud: {depth_to_pointcloud}")
        print(f"  - furthest_point_sampling: {furthest_point_sampling}")
        
        # 测试 FPS 算法
        print("\n测试最远点采样...")
        points = np.random.randn(10000, 6).astype(np.float32)
        sampled = furthest_point_sampling(points, n_samples=1024)
        print(f"✅ FPS 成功")
        print(f"  - 原始点数: {points.shape[0]}")
        print(f"  - 采样点数: {sampled.shape[0]}")
        print(f"  - 采样形状: {sampled.shape}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    print("🧪 点云采集功能测试套件\n")
    
    results = []
    
    # 测试 1
    results.append(("camera_pointcloud_utils", test_camera_pointcloud_utils()))
    
    # 测试 2
    results.append(("Isaac Lab 工具", test_isaac_lab_utils()))
    
    # 测试 3
    results.append(("data_collect_utils", test_data_collect_utils()))
    
    # 测试 4
    results.append(("zarr_utils", test_zarr_utils()))
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}  {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！点云采集功能已就绪。")
        print("\n📝 下一步:")
        print("  1. 确保相机配置包含 'depth' 和 'rgb'")
        print("  2. 运行数据采集，点云会自动保存到 HDF5")
        print("  3. 使用 zarr_utils.py 转换数据时，会自动使用 HDF5 中的点云")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

