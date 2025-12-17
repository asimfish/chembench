import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation as R

def calculate_rotation_matrix2(v1, v2):
    """ Calculate the rotation matrix that aligns v1 to v2 """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    rot_vec = np.cross(v1, v2)
    rot_angle = np.arccos(np.dot(v1, v2))
    # 检查是否共线相反
    if np.allclose(v1, -v2):
        # 选择一个垂直于v1和v2的向量作为旋转轴
        # 这里选择z轴方向的向量(0, 0, 1)
        # 如果v1和v2是z轴方向的，可以选择x轴或y轴
        rot_vec = np.cross(v1, [0, 0, 1])
        if np.linalg.norm(rot_vec) < 1e-10:  # 处理特殊情况
            rot_vec = np.cross(v1, [1, 0, 0])  # 选择x轴作为旋转轴
        rot_vec = rot_vec / np.linalg.norm(rot_vec)
        rot_angle = np.pi  # 180度旋转

    else:
        rot_vec = np.cross(v1, v2)
        rot_angle = np.arccos(np.dot(v1, v2))
    # Calculate the rotation matrix
    rotation_matrix = R.from_rotvec(rot_vec * rot_angle).as_matrix()
    return rotation_matrix

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]], 
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat

def calculate_rotation_matrix(v1, v2):
    scale = np.linalg.norm(v2)
    v2 = v2/ scale
    # must ensure pVec_Arr is also a unit vec. 
    z_mat = get_cross_prod_mat(v1)

    z_c_vec = np.matmul(z_mat, v2)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(v1, v2) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(v1, v2) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                    z_c_vec_mat)/(1 + np.dot(v1, v2))

    qTrans_Mat *= scale
    return qTrans_Mat


def transform_coordinates_3d(coordinates: ndarray, sRT: ndarray):
    """Apply 3D affine transformation to pointcloud.

    :param coordinates: ndarray of shape [3, N]
    :param sRT: ndarray of shape [4, 4]

    :returns: new pointcloud of shape [3, N]
    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack(
        [coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)]
    )
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d: ndarray, intrinsics_K: ndarray):
    """
    :param coordinates_3d: [3, N]
    :param intrinsics_K: K matrix [3, 3] (the return value of :func:`.data_types.CameraIntrinsicsBase.to_matrix`)

    :returns: projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics_K @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates



def rotate_around_axis(pose, P1, vector, angle_delta):
    """
    让一个物体绕着世界坐标系中的一个轴旋转。
    
    参数:
    pose : np.ndarray
        4x4的物体姿态矩阵
    P1 : np.ndarray
        旋转轴的起点，形状为(3,)
    P2 : np.ndarray
        旋转轴的终点，形状为(3,)
    theta : float
        旋转角度（弧度）

    返回:
    np.ndarray
        旋转后的4x4姿态矩阵
    """
    
    # 计算旋转轴方向向量
    v = vector
    # 归一化方向向量
    u = v / np.linalg.norm(v)
    theta = np.radians(angle_delta)
    
    # 计算Rodrigues' rotation formula的矩阵K
    ux, uy, uz = u
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    
    # 计算旋转矩阵R
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    # 将R转换为4x4形式
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = R
    
    # 构建平移矩阵T1
    T1 = np.eye(4)
    T1[:3, 3] = -P1
    
    # 构建平移矩阵T2
    T2 = np.eye(4)
    T2[:3, 3] = P1
    
    # 组合变换矩阵
    M = T2 @ R_4x4 @ T1
    
    # 应用变换到原始姿态矩阵
    new_pose = M @ pose
    
    return new_pose