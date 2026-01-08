#!/usr/bin/env python3
"""
调试点云变换问题 - 检查x轴反向的原因
"""

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def test_rotation_matrix_convention():
    """测试旋转矩阵的正确用法"""
    print("=" * 80)
    print("测试旋转矩阵约定")
    print("=" * 80)
    
    # 创建一个简单的测试点云（沿x轴的点）
    points = torch.tensor([
        [1.0, 0.0, 0.0],  # 沿x轴
        [0.0, 1.0, 0.0],  # 沿y轴
        [0.0, 0.0, 1.0],  # 沿z轴
    ], dtype=torch.float64)
    
    # 绕z轴旋转90度（逆时针，从上往下看）
    # 预期：(1,0,0) -> (0,1,0), (0,1,0) -> (-1,0,0)
    angle = 90  # degrees
    quat_xyzw = R.from_euler('z', angle, degrees=True).as_quat()
    quat_wxyz = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    print(f"\n旋转：绕Z轴 {angle}°")
    print(f"四元数(wxyz): {quat_wxyz}")
    
    # 构建旋转矩阵（与 grasp_mp.py 中相同的公式）
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    Rot = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    ])
    
    print(f"\n旋转矩阵 R:")
    print(Rot)
    
    # 测试两种乘法
    print("\n" + "=" * 80)
    print("方法1: points @ R (点云作为行向量)")
    result1 = torch.matmul(points, Rot)
    print(f"原点云:\n{points}")
    print(f"变换后:\n{result1}")
    print(f"X轴点 (1,0,0) -> {result1[0]}")
    print(f"Y轴点 (0,1,0) -> {result1[1]}")
    
    print("\n" + "=" * 80)
    print("方法2: points @ R.T (点云作为行向量，用转置)")
    result2 = torch.matmul(points, Rot.T)
    print(f"原点云:\n{points}")
    print(f"变换后:\n{result2}")
    print(f"X轴点 (1,0,0) -> {result2[0]}")
    print(f"Y轴点 (0,1,0) -> {result2[1]}")
    
    print("\n" + "=" * 80)
    print("方法3: R @ points.T (点云作为列向量)")
    result3 = torch.matmul(Rot, points.T).T
    print(f"原点云:\n{points}")
    print(f"变换后:\n{result3}")
    print(f"X轴点 (1,0,0) -> {result3[0]}")
    print(f"Y轴点 (0,1,0) -> {result3[1]}")
    
    # 对比scipy的结果（作为ground truth）
    print("\n" + "=" * 80)
    print("Ground Truth (scipy):")
    rot_scipy = R.from_euler('z', angle, degrees=True)
    result_scipy = rot_scipy.apply(points.numpy())
    print(f"变换后:\n{result_scipy}")
    print(f"X轴点 (1,0,0) -> {result_scipy[0]}")
    print(f"Y轴点 (0,1,0) -> {result_scipy[1]}")


def test_x_axis_flip():
    """测试x轴翻转的情况"""
    print("\n\n" + "=" * 80)
    print("测试X轴翻转问题")
    print("=" * 80)
    
    # 创建测试点云
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=torch.float64)
    
    # 无旋转，只测试是否x轴方向反了
    quat_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)  # 单位四元数（无旋转）
    
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    R_identity = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    ])
    
    print("\n单位四元数的旋转矩阵（应该是单位矩阵）:")
    print(R_identity)
    
    result = torch.matmul(points, R_identity.T)
    print(f"\n原点云:\n{points}")
    print(f"变换后(R.T):\n{result}")
    
    result2 = torch.matmul(points, R_identity)
    print(f"变换后(R):\n{result2}")
    
    # 测试x轴镜像（手性变换）
    print("\n" + "=" * 80)
    print("如果是坐标系手性问题，可能需要翻转x轴：")
    flip_x = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)
    result_flip = torch.matmul(points, flip_x)
    print(f"应用x轴翻转矩阵:\n{result_flip}")


def test_180_degree_rotation():
    """测试绕Z轴旋转180度的修正（物体朝向相差180度）"""
    print("\n\n" + "=" * 80)
    print("测试绕Z轴旋转180度修正")
    print("=" * 80)
    
    # 模拟烧杯点云：口朝向+X方向
    # 假设物体中心在原点，口在+X方向
    center = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)  # 物体中心
    beaker_points = torch.tensor([
        [0.05, 0.00, 0.5],   # 口边缘（+X方向）
        [0.03, 0.03, 0.5],   # 口边缘（+X+Y方向）
        [0.00, 0.05, 0.5],   # 口边缘（+Y方向）
        [-0.03, 0.03, 0.5],  # 口边缘（-X+Y方向）
        [0.00, 0.00, 0.45],  # 底部中心
    ], dtype=torch.float64)
    
    print(f"\n原始点云（烧杯口朝向+X）:")
    print(beaker_points)
    
    # 应用旋转180度修正（相对于物体中心）
    centered = beaker_points - center.unsqueeze(0)
    print(f"\n相对于中心 {center.tolist()}:")
    print(centered)
    
    # 绕Z轴旋转180度: x' = -x, y' = -y, z' = z
    rotated_centered = centered.clone()
    rotated_centered[:, 0] = -rotated_centered[:, 0]
    rotated_centered[:, 1] = -rotated_centered[:, 1]
    
    print(f"\n旋转180度后（相对于中心）:")
    print(rotated_centered)
    
    # 移回世界坐标
    rotated_points = rotated_centered + center.unsqueeze(0)
    
    print(f"\n修正后的点云（烧杯口应该朝向-X）:")
    print(rotated_points)
    
    print("\n验证:")
    print(f"  原口边缘点 (0.05, 0, 0.5) -> 修正后 {rotated_points[0].tolist()}")
    print(f"  预期: (-0.05, 0, 0.5) ✓" if torch.allclose(rotated_points[0], torch.tensor([-0.05, 0.0, 0.5], dtype=torch.float64)) else "  预期: (-0.05, 0, 0.5) ✗")
    print(f"  物体中心保持不变: {torch.allclose(beaker_points.mean(dim=0), rotated_points.mean(dim=0))}")


if __name__ == "__main__":
    test_rotation_matrix_convention()
    test_x_axis_flip()
    test_180_degree_rotation()
    
    print("\n\n" + "=" * 80)
    print("结论建议:")
    print("=" * 80)
    print("1. 如果方法2(points @ R.T)与scipy匹配 -> 使用 R.T（原代码正确）")
    print("2. 如果方法1(points @ R)与scipy匹配 -> 使用 R（新代码正确）")
    print("3. 如果都不匹配但符号相反 -> 可能是坐标系手性问题")
    print("4. 如果物体朝向相差180度 -> 应用绕Z轴旋转180度修正（x'=-x, y'=-y, z'=z）")

