import trimesh
import numpy as np
import open3d as o3d
from .transform_utils import transform_points, random_point

def compute_sdf_from_obj_surface(mesh, resolution=2):         # 2mm
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError("Invalid mesh provided.")
    bounds_min, bounds_max = mesh.bounds  #  网格点上面的这个有点问题。。。
    vertices = o3d.core.Tensor(mesh.vertices, dtype=o3d.core.Dtype.Float32)
    triangles = o3d.core.Tensor(mesh.faces, dtype=o3d.core.Dtype.UInt32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)
    shape = np.ceil((bounds_max - bounds_min) / resolution).astype(int)

    grid = np.mgrid[
        bounds_min[0]:bounds_max[0]:complex(0, shape[0]),
        bounds_min[1]:bounds_max[1]:complex(0, shape[1]),
        bounds_min[2]:bounds_max[2]:complex(0, shape[2])
    ]
    grid = grid.reshape(3, -1).T

    sdf_voxels = scene.compute_signed_distance(grid.astype(np.float32))

    sdf_voxels = -sdf_voxels.cpu().numpy()
    sdf_voxels = sdf_voxels.reshape(shape)

    return grid, sdf_voxels


def compute_sdf_from_obj(obj_mes,bounds_max,bounds_min, resolution=2):         # 2mm
    # 读取 OBJ 文件并转换为 Trimesh 对象

    
    # 将 Trimesh 对象转换为 Open3D TriangleMesh
    vertices = o3d.core.Tensor(obj_mes.vertices, dtype=o3d.core.Dtype.Float32)
    triangles = o3d.core.Tensor(obj_mes.faces, dtype=o3d.core.Dtype.UInt32)
    
    # 创建 RaycastingScene
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(vertices, triangles)  # 物体都添加计算

    # 创建 3D 网格以进行采样
    shape = np.ceil((np.array(bounds_max) - np.array(bounds_min)) / resolution).astype(int)
    steps = (np.array(bounds_max) - np.array(bounds_min)) / shape
    grid = np.mgrid[
        bounds_min[0]:bounds_max[0]:steps[0],
        bounds_min[1]:bounds_max[1]:steps[1],
        bounds_min[2]:bounds_max[2]:steps[2]
    ]
    grid = grid.reshape(3, -1).T

    # 计算 SDF
    sdf_voxels = scene.compute_signed_distance(grid.astype(np.float32))

    # 转换为 NumPy 数组并调整形状
    sdf_voxels = sdf_voxels.cpu().numpy()
    sdf_voxels = -sdf_voxels  # 翻转符号
    sdf_voxels = sdf_voxels.reshape(shape)

    return grid,sdf_voxels



# cost function
def get_distance_with_sdf(collision_points, pose, sdf_func):
    transformed_pcs = transform_points(collision_points, pose)   #基于不同pose下point 坐标
    transformed_pcs_flatten = transformed_pcs.reshape(-1, 3)  # [num_poses * num_points, 3]
    signed_distance = sdf_func(transformed_pcs_flatten)  # [num_poses * num_points]
    return signed_distance


def calculate_collision_cost_sdf(transformed_pcs_active, passive_pose, sdf_func, threshold):
    assert passive_pose.shape == (4, 4)
    
    # Convert transformed points to the passive object's coordinate system
    inv_passive_pose = np.linalg.inv(passive_pose)
    transformed_pcs_in_passive = transform_points(transformed_pcs_active[0], inv_passive_pose[None])
    # Flatten the points for SDF function
    transformed_pcs_flatten = transformed_pcs_in_passive.reshape(-1, 3)
    # # Calculate signed distances
    signed_distance = sdf_func(transformed_pcs_flatten)
    # Penalize if any point is closer than the minimum distance
    collision_cost = -np.sum(np.maximum(0, threshold - signed_distance))


    return collision_cost