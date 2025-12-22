import trimesh
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R

def random_point(points,num):
    # 随机选择三个不同的点
    random_indices = np.random.choice(points.shape[0], num, replace=False)
    random_points = points[random_indices]

    # 计算选中点坐标的均值
    x_mean = np.mean(random_points[:, 0])
    y_mean = np.mean(random_points[:, 1])
    z_mean = np.mean(random_points[:, 2])
    selected_points = np.array([x_mean, y_mean, z_mean])
    return selected_points


def transform_points(points, transform_matrix):
    # 确保输入的点是一个 Nx3 的数组
    assert points.shape[1] == 3, "输入的点必须是 Nx3 的数组"
    assert transform_matrix.shape == (4, 4), "变换矩阵必须是 4x4 的"

    # 将点扩展为齐次坐标 (N x 4)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))

    # 应用变换矩阵
    transformed_points_homogeneous = points_homogeneous @ transform_matrix.T

    # 转换回非齐次坐标 (N x 3)
    transformed_points = transformed_points_homogeneous[:, :3]

    return transformed_points





def farthest_point_sampling(pc, num_points):
    """
    Given a point cloud, sample num_points points that are the farthest apart.
    Use o3d farthest point sampling.
    """
    assert pc.ndim == 2 and pc.shape[1] == 3, "pc must be a (N, 3) numpy array"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    downpcd_farthest = pcd.farthest_point_down_sample(num_points)
    return np.asarray(downpcd_farthest.points)




def get_bott_up_point(points,obj_size,descending,):

    # 按照 Z 值从小到大排序
    ascending_indices  = np.argsort(points[:, 2])
    if descending:
        # 反转索引实现降序
        ascending_indices = ascending_indices[::-1]
    sorted_points = points[ascending_indices]
    # print("sorted_points",sorted_points)
    threshold = 0.03* obj_size
    z_m = sorted_points[0][-1]  # 
    while True:
        top_surface_points = sorted_points[np.abs(sorted_points[:, 2] - z_m) < threshold]
        if len(top_surface_points) >= 15:
            break
        # 增加阈值以获取更多点
        threshold += 0.01 * obj_size
    # 获取顶部/底部表面的点
    top_surface_points = sorted_points[np.abs(sorted_points[:, 2] - z_m) < threshold]
    # print("sorted_points",len(top_surface_points))

    # # 使用 KMeans 进行聚类，以保证点的均匀分布
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(top_surface_points[:, :2])  # 仅使用 X 和 Y 坐标进行聚类
    # 获取每个聚类中心最近的点
    centers = kmeans.cluster_centers_
    selected_points = []

    for center in centers:
        # 计算每个中心点到所有点的距离
        distances = np.linalg.norm(top_surface_points[:, :2] - center, axis=1)
        # 找到最近的点
        closest_point_idx = np.argmin(distances)
        selected_points.append(top_surface_points[closest_point_idx])
    selected_points = np.array(selected_points)
    selected_points[:, 2] = z_m
    #  modify
    return selected_points






# ===============================================
# = optimization utils
# ===============================================
def normalize_vars(vars, og_bounds):
    """
    Given 1D variables and bounds, normalize the variables to [-1, 1] range.
    """
    normalized_vars = np.empty_like(vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        if b_max != b_min:
            normalized_vars[i] = (vars[i] - b_min) / (b_max - b_min) * 2 - 1
        else:
            # 处理 b_max 等于 b_min 的情况
            normalized_vars[i] = 0  # 或者其他合适的默认值
    return normalized_vars

def unnormalize_vars(normalized_vars, og_bounds):
    """
    Given 1D variables in [-1, 1] and original bounds, denormalize the variables to the original range.
    """
    vars = np.empty_like(normalized_vars)
    # import pdb;pdb.set_trace()
    for i, (b_min, b_max) in enumerate(og_bounds):
        vars[i] = (normalized_vars[i] + 1) / 2.0  * (b_max - b_min) + b_min
    return vars


def euler2quat(euler):
    """
    Converts euler angles into quaternion form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: (x,y,z,w) float quaternion angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    return R.from_euler("xyz", euler).as_quat()

def quat2euler(quat):
    """
    Converts euler angles into quaternion form

    Args:
        quat (np.array): (x,y,z,w) float quaternion angles

    Returns:
        np.array: (r,p,y) angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    return R.from_quat(quat).as_euler("xyz")


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (..., 4) (x,y,z,w) float quaternion angles

    Returns:
        np.array: (..., 3, 3) rotation matrix
    """
    return R.from_quat(quaternion).as_matrix()

def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        np.array: 4x4 homogeneous matrix
    """

    if isinstance(pose[0], np.ndarray):
        translation = pose[0]
    else:
        raise TypeError("Unsupported data type for translation")

    if isinstance(pose[1], np.ndarray):
        quaternion = pose[1]
    else:
        raise TypeError("Unsupported data type for quaternion")
    
    homo_pose_mat = np.zeros((4, 4), dtype=translation.dtype)
    homo_pose_mat[:3, :3] = quat2mat(quaternion)
    homo_pose_mat[:3, 3] = translation
    homo_pose_mat[3, 3] = 1.0
    
    return homo_pose_mat

