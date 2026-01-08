#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 USD 提取物体点云，并根据 target_pose 生成真值点云轨迹（写入 zarr）。

主要特性：
- 遍历所有 UsdGeom.Mesh，支持 n-gon 三角化
- 使用 XformCache 把每个 mesh bake 到 root_prim 坐标系（或世界系）
- 支持按 UsdGeom.GetStageMetersPerUnit(stage) 转成米（可关闭）
- 支持 quat 顺序显式指定：wxyz 或 xyzw
- 支持流式写 zarr（避免爆内存）
"""

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
import zarr
from tqdm import tqdm

from pxr import Usd, UsdGeom


# -----------------------------
# 数学工具
# -----------------------------
def apply_transform_points(points: np.ndarray, mat4: np.ndarray) -> np.ndarray:
    """points: (N,3), mat4: (4,4)"""
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    out = (mat4 @ pts_h.T).T
    return out[:, :3]


def quaternion_to_rotation_matrix_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """
    输入 q = [w, x, y, z]，返回 3x3 旋转矩阵
    """
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = q

    # 归一化
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 1e-12:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    else:
        # 退化情况：当作单位四元数
        w, x, y, z = 1.0, 0.0, 0.0, 0.0

    # 标准公式（右手系）
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def pose_to_transform_matrix(pose: np.ndarray, quat_order: str = "wxyz") -> np.ndarray:
    """
    pose: [x,y,z,qw,qx,qy,qz] if quat_order='wxyz'
          [x,y,z,qx,qy,qz,qw] if quat_order='xyzw'
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    t = pose[:3]
    q = pose[3:]

    if quat_order.lower() == "wxyz":
        q_wxyz = q
    elif quat_order.lower() == "xyzw":
        q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    else:
        raise ValueError("quat_order must be 'wxyz' or 'xyzw'")

    R = quaternion_to_rotation_matrix_wxyz(q_wxyz)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# -----------------------------
# USD -> trimesh
# -----------------------------
def triangulate_ngon(face_vertex_indices, face_vertex_counts, vertex_offset: int):
    """
    用 fan triangulation 将 n-gon 三角化，返回 (F,3) faces
    """
    faces = []
    idx = 0
    for c in face_vertex_counts:
        c = int(c)
        poly = face_vertex_indices[idx:idx + c]
        idx += c
        if c < 3:
            continue
        # fan: (0, i, i+1)
        v0 = int(poly[0]) + vertex_offset
        for i in range(1, c - 1):
            faces.append([v0, int(poly[i]) + vertex_offset, int(poly[i + 1]) + vertex_offset])
    if len(faces) == 0:
        return None
    return np.asarray(faces, dtype=np.int64)


def load_usd_mesh_as_trimesh(
    usd_path: str,
    root_prim_path: str | None = None,
    time_code: float | None = None,
    convert_to_meters: bool = True,
    process: bool = False,
):
    """
    从 USD 加载所有 Mesh，合并为一个 trimesh.Trimesh（如果没有面则 PointCloud）。
    - root_prim_path: 将所有 mesh 顶点变换到该 root prim 坐标系；None 表示世界系
    - time_code: 指定 USD 时间采样；None 用 Default
    - convert_to_meters: True 时使用 Stage metersPerUnit 将点转成米
    """
    usd_path = str(usd_path)
    print(f"加载 USD: {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"无法打开 USD: {usd_path}")

    # 时间
    if time_code is None:
        tc = Usd.TimeCode.Default()
    else:
        tc = Usd.TimeCode(float(time_code))

    # 单位：1 stage unit = meters_per_unit meters
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if meters_per_unit is None:
        meters_per_unit = 1.0
    meters_per_unit = float(meters_per_unit)
    unit_scale = meters_per_unit if convert_to_meters else 1.0

    if convert_to_meters:
        print(f"USD metersPerUnit = {meters_per_unit}，将顶点缩放到米单位（乘以 {unit_scale}）")
    else:
        print("不使用 USD metersPerUnit 缩放（保持 USD 原单位）")

    xcache = UsdGeom.XformCache(tc)

    # root 的 world transform（用于把 mesh 变换到 root 坐标系）
    if root_prim_path is not None:
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim or not root_prim.IsValid():
            raise ValueError(f"root_prim_path 不存在或无效：{root_prim_path}")
        root_to_world = np.array(xcache.GetLocalToWorldTransform(root_prim), dtype=np.float64)
        world_to_root = np.linalg.inv(root_to_world)
        print(f"使用 root_prim: {root_prim_path}（把 mesh 统一到该坐标系）")
    else:
        world_to_root = np.eye(4, dtype=np.float64)
        print("未指定 root_prim_path：把 mesh 统一到世界坐标系")

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    # 检测 instancer（仅提示）
    has_point_instancer = False
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.PointInstancer):
            has_point_instancer = True
            break
    if has_point_instancer:
        print("⚠️ 检测到 UsdGeom.PointInstancer（实例化）。本脚本不会展开实例，可能导致缺失几何。")

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)

        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get(tc) if points_attr else None
        if points is None or len(points) == 0:
            continue

        verts = np.asarray(points, dtype=np.float64) * unit_scale

        # mesh local -> world
        mesh_to_world = np.array(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)

        # mesh local -> root
        mesh_to_root = world_to_root @ mesh_to_world
        verts_root = apply_transform_points(verts, mesh_to_root)

        all_vertices.append(verts_root)

        # faces
        fvi_attr = mesh.GetFaceVertexIndicesAttr()
        fvc_attr = mesh.GetFaceVertexCountsAttr()
        fvi = fvi_attr.Get(tc) if fvi_attr else None
        fvc = fvc_attr.Get(tc) if fvc_attr else None

        if fvi is not None and fvc is not None and len(fvi) > 0 and len(fvc) > 0:
            faces = triangulate_ngon(fvi, fvc, vertex_offset=vertex_offset)
            if faces is not None and len(faces) > 0:
                all_faces.append(faces)

        vertex_offset += verts_root.shape[0]

    if len(all_vertices) == 0:
        raise ValueError("USD 文件中没有找到有效的 UsdGeom.Mesh 点数据")

    vertices = np.vstack(all_vertices).astype(np.float64)
    faces = np.vstack(all_faces).astype(np.int64) if len(all_faces) > 0 else None

    print(f"合并 mesh 完成：{vertices.shape[0]} 顶点，{0 if faces is None else faces.shape[0]} 三角面")

    if faces is None or faces.shape[0] == 0:
        pc = trimesh.points.PointCloud(vertices=vertices)
        return pc

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=process)
    return tm


def sample_pointcloud(mesh, num_points: int, seed: int | None = None) -> np.ndarray:
    """
    从 Trimesh 表面采样点云；若没有 faces，则从 vertices 采样
    """
    if seed is not None:
        np.random.seed(int(seed))

    if isinstance(mesh, trimesh.Trimesh) and mesh.faces is not None and len(mesh.faces) > 0:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points.astype(np.float32)

    # PointCloud 或无面
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.shape[0] <= num_points:
        return verts
    idx = np.random.choice(verts.shape[0], num_points, replace=False)
    return verts[idx]


# -----------------------------
# Zarr 读取/写入
# -----------------------------
def load_target_poses(zarr_path: str, dataset_path: str = "data/target_pose") -> np.ndarray:
    print(f"加载 target_pose: {zarr_path} -> {dataset_path}")
    z = zarr.open(zarr_path, mode="r")

    # 支持 "data/target_pose" 这种路径
    grp = z
    parts = dataset_path.strip("/").split("/")
    for p in parts[:-1]:
        grp = grp[p]
    arr = grp[parts[-1]]

    poses = arr[:]
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"target_pose 形状应为 (T,7)，但得到 {poses.shape}")
    print(f"加载姿态数量: {poses.shape[0]}")
    return poses


def save_pointcloud_to_ply(points: np.ndarray, output_path: str, colors=None):
    """
    保存点云为 PLY 格式（Blender 可打开）
    points: (N,3) 点坐标
    colors: (N,3) 可选颜色 RGB，范围 [0,1] 或 [0,255]
    """
    try:
        import trimesh
    except ImportError:
        print("⚠️ 未安装 trimesh，跳过 PLY 导出")
        return
    
    points = np.asarray(points, dtype=np.float64)
    pc = trimesh.points.PointCloud(vertices=points)
    
    if colors is not None:
        colors = np.asarray(colors)
        # 确保颜色在 [0,255] 范围
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        pc.colors = colors
    
    pc.export(output_path)
    print(f"✓ 已保存 PLY: {output_path}")


def write_zarr_sequence(
    output_path: str,
    dataset_name: str,
    T: int,
    N: int,
    generator_fn,
    chunks=(1, 8192, 3),
    dtype="f4",
    overwrite=True,
    attrs: dict | None = None,
):
    """
    流式写入 zarr：generator_fn(i) -> (N,3)
    """
    output_path = str(output_path)
    store_mode = "w" if overwrite else "a"
    root = zarr.open(output_path, mode=store_mode)

    # 创建 dataset
    if dataset_name in root:
        if overwrite:
            del root[dataset_name]
        else:
            raise ValueError(f"dataset 已存在：{dataset_name}（overwrite=False）")

    dset = root.create_dataset(
        dataset_name,
        shape=(T, N, 3),
        chunks=chunks,
        dtype=dtype,
    )

    if attrs:
        for k, v in attrs.items():
            root.attrs[k] = v

    for i in tqdm(range(T), desc="写入 zarr"):
        pc = generator_fn(i)
        dset[i, :, :] = pc.astype(np.float32)


# -----------------------------
# 主流程
# -----------------------------
def generate_ground_truth_pointclouds(
    usd_path: str,
    zarr_path: str | None,
    num_points: int,
    output_path: str | None,
    apply_transform: bool,
    quat_order: str,
    root_prim_path: str | None,
    target_pose_dataset: str,
    convert_to_meters: bool,
    time_code: float | None,
    seed: int | None,
    stream_to_zarr: bool,
):
    # 1) 加载 USD mesh（统一到 root 坐标系 或 世界系）
    mesh = load_usd_mesh_as_trimesh(
        usd_path=usd_path,
        root_prim_path=root_prim_path,
        time_code=time_code,
        convert_to_meters=convert_to_meters,
        process=False,
    )

    # 2) 采样物体“局部点云”（此处局部= root 坐标系下的静态几何）
    print(f"从 mesh 采样 {num_points} 点...")
    base_pc = sample_pointcloud(mesh, num_points=num_points, seed=seed)
    print(f"base_pc: {base_pc.shape}, dtype={base_pc.dtype}")

    if not apply_transform:
        # 只输出单帧
        seq = base_pc[None, :, :]  # (1,N,3)
        if output_path:
            print(f"保存原始点云到 zarr: {output_path}")
            root = zarr.open(output_path, mode="w")
            root.create_dataset(
                "ground_truth_pointcloud",
                data=seq.astype(np.float32),
                chunks=(1, min(num_points, 8192), 3),
                dtype="f4",
            )
            root.attrs.update({
                "num_timesteps": 1,
                "num_points": int(num_points),
                "source_usd": str(usd_path),
                "source_zarr": str(zarr_path) if zarr_path else "None",
                "transformed": False,
                "quat_order": str(quat_order),
                "root_prim_path": str(root_prim_path) if root_prim_path else "None",
                "convert_to_meters": bool(convert_to_meters),
                "time_code": float(time_code) if time_code is not None else "Default",
            })
            
            # 保存 PLY 格式
            # ply_path = str(Path(output_path).with_suffix('.ply'))
            # save_pointcloud_to_ply(base_pc, ply_path)
        return seq

    # 3) 读取姿态轨迹
    if zarr_path is None:
        raise ValueError("apply_transform=True 但未提供 zarr_path")
    poses = load_target_poses(zarr_path, dataset_path=target_pose_dataset)
    T = poses.shape[0]

    def frame_pc(i: int) -> np.ndarray:
        Ti = pose_to_transform_matrix(poses[i], quat_order=quat_order)
        out = apply_transform_points(base_pc, Ti)
        return out.astype(np.float32)

    # 4) 写出 / 返回
    if output_path and stream_to_zarr:
        print(f"流式写入点云序列到: {output_path}")
        attrs = {
            "num_timesteps": int(T),
            "num_points": int(num_points),
            "source_usd": str(usd_path),
            "source_zarr": str(zarr_path),
            "transformed": True,
            "quat_order": str(quat_order),
            "root_prim_path": str(root_prim_path) if root_prim_path else "None",
            "convert_to_meters": bool(convert_to_meters),
            "time_code": float(time_code) if time_code is not None else "Default",
            "target_pose_dataset": str(target_pose_dataset),
        }
        write_zarr_sequence(
            output_path=output_path,
            dataset_name="ground_truth_pointcloud",
            T=T,
            N=num_points,
            generator_fn=frame_pc,
            chunks=(1, min(num_points, 8192), 3),
            dtype="f4",
            overwrite=True,
            attrs=attrs,
        )
        
        # 保存关键帧到 PLY（第一帧、最后一帧）
        # ply_base = Path(output_path).with_suffix('')
        # print(f"保存关键帧到 PLY...")
        # save_pointcloud_to_ply(frame_pc(0), f"{ply_base}_frame_000.ply")
        # save_pointcloud_to_ply(frame_pc(T-1), f"{ply_base}_frame_{T-1:03d}.ply")
        
        # 不把全量返回（避免占内存），返回 None 或返回空数组都行
        return None

    # 不流式：直接堆内存
    seq = np.zeros((T, num_points, 3), dtype=np.float32)
    for i in tqdm(range(T), desc="生成点云序列"):
        seq[i] = frame_pc(i)

    if output_path:
        print(f"保存到 zarr: {output_path}")
        root = zarr.open(output_path, mode="w")
        root.create_dataset(
            "ground_truth_pointcloud",
            data=seq,
            chunks=(1, min(num_points, 8192), 3),
            dtype="f4",
        )
        root.attrs.update({
            "num_timesteps": int(T),
            "num_points": int(num_points),
            "source_usd": str(usd_path),
            "source_zarr": str(zarr_path),
            "transformed": True,
            "quat_order": str(quat_order),
            "root_prim_path": str(root_prim_path) if root_prim_path else "None",
            "convert_to_meters": bool(convert_to_meters),
            "time_code": float(time_code) if time_code is not None else "Default",
            "target_pose_dataset": str(target_pose_dataset),
        })
        
        # 保存关键帧到 PLY
        # ply_base = Path(output_path).with_suffix('')
        # print(f"保存关键帧到 PLY...")
        # save_pointcloud_to_ply(seq[0], f"{ply_base}_frame_000.ply")
        # save_pointcloud_to_ply(seq[-1], f"{ply_base}_frame_{T-1:03d}.ply")

    return seq


def visualize_trajectory(pointcloud_sequence: np.ndarray, skip_frames: int = 10):
    """
    Open3D 可视化（可选）
    pointcloud_sequence: (T,N,3)
    """
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 open3d 或 matplotlib，跳过可视化。")
        return

    seq = np.asarray(pointcloud_sequence)
    if seq.ndim != 3:
        raise ValueError(f"pointcloud_sequence 应为 (T,N,3)，但得到 {seq.shape}")

    T = seq.shape[0]
    k = math.ceil(T / max(1, skip_frames))
    colors = plt.cm.rainbow(np.linspace(0, 1, k))

    pcds = []
    j = 0
    for i in range(0, T, skip_frames):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seq[i])
        pcd.paint_uniform_color(colors[j][:3])
        pcds.append(pcd)
        j += 1

    o3d.visualization.draw_geometries(pcds)


def main():
    parser = argparse.ArgumentParser(description="USD -> 点云，并根据 target_pose 生成真值点云轨迹（zarr）")

    parser.add_argument("--usd", type=str, required=True, help="USD 文件路径")
    parser.add_argument("--zarr", type=str, default=None, help="输入 zarr 路径（包含 target_pose）；不提供则只输出原始点云")
    parser.add_argument("--target-pose-dataset", type=str, default="data/target_pose", help="target_pose 数据集路径（默认 data/target_pose）")

    parser.add_argument("--num-points", type=int, default=10000, help="每帧采样点数")
    parser.add_argument("--output", type=str, default=None, help="输出 zarr 路径（默认自动推断）")

    parser.add_argument("--no-transform", action="store_true", help="不应用 pose 变换（只输出原始点云）")
    parser.add_argument("--quat-order", type=str, default="wxyz", choices=["wxyz", "xyzw"], help="target_pose 四元数顺序")
    parser.add_argument("--root-prim", type=str, default=None, help="把 USD mesh 统一到该 root prim 坐标系（例如 /World/Object）；不填则用世界系")
    parser.add_argument("--timecode", type=float, default=None, help="USD 读取的时间（不填用 Default）")

    parser.add_argument("--convert-to-meters", action="store_true", help="使用 USD metersPerUnit 把顶点转成米（推荐当 pose 用米时开启）")
    parser.add_argument("--no-convert-to-meters", action="store_true", help="不使用 metersPerUnit（保持 USD 原单位）")

    parser.add_argument("--seed", type=int, default=None, help="采样随机种子（可复现）")

    parser.add_argument("--stream-to-zarr", action="store_true", help="流式写入 zarr（不把整段序列放内存，推荐长序列）")

    parser.add_argument("--visualize", action="store_true", help="是否可视化（需要 open3d+matplotlib）")
    parser.add_argument("--skip-frames", type=int, default=50, help="可视化时跳帧数")
    
    parser.add_argument("--export-all-frames-ply", action="store_true", help="导出所有帧为单独的 PLY 文件（会产生大量文件）")

    args = parser.parse_args()

    # convert_to_meters 逻辑：默认关闭，用户可显式开；若同时指定两个 flag，以 no-convert 优先
    if args.no_convert_to_meters:
        convert_to_meters = False
    else:
        convert_to_meters = bool(args.convert_to_meters)

    # 变换模式：未给 zarr 时自动 no-transform
    if (not args.no_transform) and (args.zarr is None):
        print("⚠️ 未指定 --zarr，自动启用 --no-transform（只输出原始点云）")
        args.no_transform = True

    # 默认输出路径
    if args.output is None:
        if args.zarr:
            zarr_path = Path(args.zarr)
            args.output = str(zarr_path.parent / f"{zarr_path.stem}_ground_truth.zarr")
        else:
            usd_path = Path(args.usd)
            args.output = str(usd_path.parent / f"{usd_path.stem}_pointcloud.zarr")

    seq = generate_ground_truth_pointclouds(
        usd_path=args.usd,
        zarr_path=args.zarr,
        num_points=args.num_points,
        output_path=args.output,
        apply_transform=not args.no_transform,
        quat_order=args.quat_order,
        root_prim_path=args.root_prim,
        target_pose_dataset=args.target_pose_dataset,
        convert_to_meters=convert_to_meters,
        time_code=args.timecode,
        seed=args.seed,
        stream_to_zarr=args.stream_to_zarr,
    )

    # 导出所有帧为 PLY（如果用户请求）
    if args.export_all_frames_ply and not args.no_transform and seq is not None:
        ply_base = Path(args.output).with_suffix('')
        ply_dir = Path(f"{ply_base}_frames")
        ply_dir.mkdir(exist_ok=True)
        print(f"\n导出所有帧到 PLY: {ply_dir}/")
        for i in tqdm(range(seq.shape[0]), desc="导出 PLY"):
            save_pointcloud_to_ply(seq[i], str(ply_dir / f"frame_{i:04d}.ply"))
        print(f"✓ 已导出 {seq.shape[0]} 个 PLY 文件")

    print("\n完成!")
    print(f"输出文件: {args.output}")
    if args.no_transform:
        print("模式: 原始点云（未应用 pose 变换）")
    else:
        print("模式: 真值点云序列（应用 pose 变换）")
        if args.stream_to_zarr:
            print("写入方式: 流式写 zarr（未在内存中保留完整序列）")
        else:
            print(f"内存中序列形状: {None if seq is None else seq.shape}")

    # 可视化（仅当 seq 在内存里）
    if args.visualize:
        if seq is None:
            print("⚠️ 你开启了 --stream-to-zarr，内存里没有完整序列；如需可视化请关闭 --stream-to-zarr")
        else:
            visualize_trajectory(seq, skip_frames=max(1, args.skip_frames))


if __name__ == "__main__":
    main()
