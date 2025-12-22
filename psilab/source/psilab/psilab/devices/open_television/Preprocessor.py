import math
import numpy as np

from .constants_vuer import grd_yup2grd_zup, hand2inspire,hand_vr2urdf,head_vr2urdf,hand_vr2urdf_dc
from .motion_utils import mat_update, fast_mat_inv


class VuerPreprocessor:

    def __init__(self):
        self.vuer_head_mat = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 1.5],
                                  [0, 0, 1, -0.2],
                                  [0, 0, 0, 1]])
        self.vuer_right_wrist_mat = np.array([[1, 0, 0, 0.5],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, -0.5],
                                         [0, 0, 0, 1]])
        self.vuer_left_wrist_mat = np.array([[1, 0, 0, -0.5],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, -0.5],
                                        [0, 0, 0, 1]])

    # 针对PsiRobot DC 01 修改
    def process(self, tv):
        # 读取原始数据，坐标系为右手坐标系，Y轴向上
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())

        # 将原始数据转换为右手坐标系，Z轴县三个行
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

        # 右乘旋转矩阵的原因与计算方法：
        # 由于vr中双手在胸前位置，食指、中指、无名指、小拇指向前，手掌向下的放松状态为初始姿态，此时姿态角(roll,pitch,yaw)为(0,0,0)
        # 但是USD和URDF中双手的腕部坐标系初始姿态往往不同于VR，因此需要将VR的腕部旋转矩阵转换为USD/URDF坐标系下的旋转矩阵
        # 以OY手为例，其初始姿态为食指、中指、无名指、小拇指指向Z轴正方向，手掌向X轴正方向
        # 因此原始数据(手腕部相对于基坐标系的旋转矩阵)需要右乘旋转矩阵（以XYZ为例，roll:0, pitch:1.57, yaw:0）
        rel_left_wrist_mat = left_wrist_mat @ hand_vr2urdf_dc  # 腕部相对头部的旋转矩阵
        rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]  #腕部相对头部的偏移

        rel_right_wrist_mat = right_wrist_mat @ hand_vr2urdf_dc # 腕部相对头部的旋转矩阵
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]    #腕部相对头部的偏移

        # homogeneous
        left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        # Todo：梳理手指关节的坐标变换过程
        left_fingers = grd_yup2grd_zup @ left_fingers 
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand_vr2urdf.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand_vr2urdf.T @ rel_right_fingers)[0:3, :].T

        # 和双手一样，头部也需要根据机器人相机位置进行调整
        head_mat = head_mat @ head_vr2urdf

        # 返回头部齐次变换矩阵，腕部相对头部齐次变换矩阵，手指相对头部齐次变换(待定)
        return head_mat, rel_left_wrist_mat, rel_right_wrist_mat, rel_left_fingers, rel_right_fingers

    # 未修改，未使用
    def get_hand_gesture(self, tv):
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())

        # change of basis
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

        left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire.T @ rel_right_fingers)[0:3, :].T
        all_fingers = np.concatenate([rel_left_fingers, rel_right_fingers], axis=0)

        return all_fingers

