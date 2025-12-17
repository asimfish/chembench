import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation as R, Slerp


def interpolate_rotation_matrices(rot_matrix1, rot_matrix2, num_interpolations):
    # Convert the rotation matrices to rotation objects
    rot1 = R.from_matrix(rot_matrix1)
    rot2 = R.from_matrix(rot_matrix2)
    
    # Convert the rotation objects to quaternions
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()
    
    # Define the times of the known rotations
    times = [0, 1]
    
    # Create the Slerp object
    slerp = Slerp(times, R.from_quat([quat1, quat2]))
    
    # Define the times of the interpolations
    interp_times = np.linspace(0, 1, num_interpolations)
    
    # Perform the interpolation
    interp_rots = slerp(interp_times)
    
    # Convert the interpolated rotations to matrices
    interp_matrices = interp_rots.as_matrix()
    
    return interp_matrices



def is_y_axis_up(pose_matrix):
    """
    判断物体在世界坐标系中的 y 轴是否朝上。

    参数:
    pose_matrix (numpy.ndarray): 4x4 齐次变换矩阵。

    返回:
    bool: True 如果 y 轴朝上，False 如果 y 轴朝下。
    """
    # 提取旋转矩阵的第二列
    y_axis_vector = pose_matrix[:3, 1]

    # 世界坐标系的 y 轴
    world_y_axis = np.array([0, 1, 0])

    # 计算点积
    dot_product = np.dot(y_axis_vector, world_y_axis)

    # 返回 True 如果朝上，否则返回 False
    return dot_product > 0

def is_local_axis_facing_world_axis(pose_matrix, local_axis='y', world_axis='z'):
    # 定义局部坐标系的轴索引
    local_axis_index = {'x': 0, 'y': 1, 'z': 2}
    
    # 定义世界坐标系的轴向量
    world_axes = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }
    
    # 提取局部坐标系的指定轴
    local_axis_vector = pose_matrix[:3, local_axis_index[local_axis]]
    
    # 获取世界坐标系的指定轴向量
    world_axis_vector = world_axes[world_axis]
    
    # 计算点积
    dot_product = np.dot(local_axis_vector, world_axis_vector)
    
    # 返回 True 如果朝向指定世界轴，否则返回 False
    return dot_product > 0

    
def fix_gripper_rotation(source_affine, target_affine, rot_axis='z'):
    '''
      gripper是对称结构，绕Z轴旋转180度前后是等效的。选择离当前pose更近的target pose以避免不必要的旋转
    '''
    if rot_axis == 'z':
        R_180 = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    elif rot_axis == 'y':
        R_180 = np.array([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1]])
    elif rot_axis == 'x':
        R_180 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    else:
        assert False, "Invalid rotation axis. Please choose from 'x', 'y', 'z'."
    
    # 提取source和target的旋转部分（3x3矩阵）
    source_rotation = source_affine[:3, :3]
    target_rotation = target_affine[:3, :3]
    # 将target_rotation绕其自身的Z轴转180度，得到target_rotation_2
    target_rotation_2 = np.dot(target_rotation, R_180)
    # 定义一个函数来计算两个旋转矩阵之间的距离
    def rotation_matrix_distance(R1, R2):
        # 使用奇异值分解来计算两个旋转矩阵之间的距离
        U, _, Vh = np.linalg.svd(np.dot(R1.T, R2))
        # 确保旋转矩阵的行列式为1，即旋转而不是反射
        det_check = np.linalg.det(U) * np.linalg.det(Vh)
        if det_check < 0:
            Vh = -Vh
        return np.arccos(np.trace(Vh) / 2)
    
    # 计算source_rotation与target_rotation之间的距离
    distance_target_rotation = rotation_matrix_distance(source_rotation, target_rotation)
    # 计算source_rotation与target_rotation_2之间的距离
    distance_target_rotation_2 = rotation_matrix_distance(source_rotation, target_rotation_2)
    # 比较两个距离，确定哪个旋转更接近source_rotation
    if distance_target_rotation < distance_target_rotation_2:
        return target_affine
    else:
        # 重新组合旋转矩阵target_rotation_2和原始的平移部分
        target_affine_2 = np.eye(4)
        target_affine_2[:3, :3] = target_rotation_2
        target_affine_2[:3, 3] = target_affine[:3, 3]  # 保留原始的平移部分
        return target_affine_2
    

def rotate_180_along_axis(pose, rot_axis='z'):
    '''
      gripper是对称结构，绕Z轴旋转180度前后是等效的。选择离当前pose更近的target pose以避免不必要的旋转
    '''
    if rot_axis == 'z':
        R_180 = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    elif rot_axis == 'y':
        R_180 = np.array([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1]])
    elif rot_axis == 'x':
        R_180 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    else:
        assert False, "Invalid rotation axis. Please choose from 'x', 'y', 'z'."
    


    single_mode = pose.ndim == 2
    if single_mode:
        pose = pose[np.newaxis, :, :]
    R_180 = np.tile(R_180[np.newaxis, :, :], (pose.shape[0], 1, 1))
    pose[:,:3,:3] = pose[:, :3,:3] @ R_180

    if single_mode:
        pose = pose[0]

    return pose



def translate_along_axis(pose, shift, axis='z', use_local=True):
    '''
    根据指定的角度和旋转轴来旋转target_affine。
    参数:
    - target_affine: 4x4 仿射变换矩阵
    - angle_degrees: 旋转角度（以度为单位）
    - rot_axis: 旋转轴，'x'、'y' 或 'z'
    '''
    pose = pose.copy()
    translation = np.zeros(3)
    if axis == 'z':
        translation[2] = shift
    elif axis == 'y':
        translation[1] = shift
    elif axis == 'x':
        translation[0] = shift
    if len(pose.shape)==3:
        for i in range(pose.shape[0]):
            if use_local:
                pose[i][:3, 3] += pose[i][:3, :3] @ translation
            else:
                pose[i][:3, 3] += translation
    else:
        if use_local:
            pose[:3, 3] += pose[:3, :3] @ translation
        else:
            pose[:3, 3] += translation
    
    return pose


def rotate_along_axis(target_affine, angle_degrees, rot_axis='z', use_local=True):
    '''
    根据指定的角度和旋转轴来旋转target_affine。
    参数:
    - target_affine: 4x4 仿射变换矩阵
    - angle_degrees: 旋转角度（以度为单位）
    - rot_axis: 旋转轴，'x'、'y' 或 'z'
    '''
    # 将角度转换为弧度
    angle_radians = np.deg2rad(angle_degrees)
    
    # 创建旋转对象
    if rot_axis == 'z':
        rotation_vector = np.array([0, 0, angle_radians])
    elif rot_axis == 'y':
        rotation_vector = np.array([0, angle_radians, 0])
    elif rot_axis == 'x':
        rotation_vector = np.array([angle_radians, 0, 0])
    else:
        raise ValueError("Invalid rotation axis. Please choose from 'x', 'y', 'z'.")
    
    # 生成旋转矩阵
    R_angle = R.from_rotvec(rotation_vector).as_matrix()
    
    # 提取旋转部分（3x3矩阵）
    target_rotation = target_affine[:3, :3]
    
    # 将 target_rotation 绕指定轴旋转指定角度，得到 target_rotation_2
    if use_local:
        target_rotation_2 = np.dot(target_rotation, R_angle)
    else:
        target_rotation_2 = np.dot(R_angle, target_rotation)
    
    # 重新组合旋转矩阵 target_rotation_2 和原始的平移部分
    target_affine_2 = np.eye(4)
    target_affine_2[:3, :3] = target_rotation_2
    target_affine_2[:3, 3] = target_affine[:3, 3]  # 保留原始的平移部分
    
    return target_affine_2



def rotation_matrix_to_quaternion(R):
    assert R.shape == (3, 3)
    
    # 计算四元数分量
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

def quat_wxyz_to_rotation_matrix(quat):
    qw, qx, qy, qz = quat
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def estimate_affine_3d_fixed_scale(src_points, dst_points):
    ransac = RANSACRegressor()
    ransac.fit(src_points, dst_points)
    inlier_mask = ransac.inlier_mask_
    
    src = src_points[inlier_mask]
    dst = dst_points[inlier_mask]

    # Normalize the input points to ensure uniform scaling
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Estimate rotation using singular value decomposition (SVD)
    U, _, Vt = np.linalg.svd(np.dot(dst_centered.T, src_centered))
    R_est = np.dot(U, Vt)

    # Ensure a right-handed coordinate system
    if np.linalg.det(R_est) < 0:
        Vt[2, :] *= -1
        R_est = np.dot(U, Vt)

    # Calculate the uniform scale
    scale = np.sum(dst_centered * (R_est @ src_centered.T).T) / np.sum(src_centered ** 2)

    # Construct the affine transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = scale * R_est
    transform[:3, 3] = dst_mean - scale * R_est @ src_mean

    return transform