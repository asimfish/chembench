# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
import sys
import math
import torch
import numpy
import time
import threading
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from pytransform3d import rotations

""" Omniverse Modules  """ 
import carb
import omni

""" IsaacLab Modules  """ 
from isaaclab.utils.math import quat_from_matrix

""" PsiLab Modules  """ 
from psilab.devices.open_television.TeleVision import OpenTeleVision
from psilab.devices.open_television.vuer_tp_cfg import VuerTpCfg
from psilab.devices.teleop_base import TeleOperateDeviceBase

class VuerTp(TeleOperateDeviceBase):
    """
    Vuer device for teleoperation
    """

    cfg: VuerTpCfg

    def __init__(self, cfg:VuerTpCfg):

        super().__init__()
        #
        self.cfg = cfg
        #
        self.device_init(cfg) 
        # 注册键盘句柄
        self.register_keyboard_handler()
        # buffer 
        self.left_wrist_pose = torch.zeros(7,device=self.cfg.device)
        self.right_wrist_pose = torch.zeros(7,device=self.cfg.device)
        self.left_hand_joint_pos = torch.zeros(len(cfg.hand_retarget_indexs),device=self.cfg.device)
        self.right_hand_joint_pos = torch.zeros(len(cfg.hand_retarget_indexs),device=self.cfg.device)
        self.left_eye_pose = torch.zeros(7,device=self.cfg.device)
        self.right_eye_pose = torch.zeros(7,device=self.cfg.device)
        # variables
        self.dis_thumbs = 1.0
        self.dis_pinkys = 1.0
        self.dis_thumb_middle = 1.0
        self.dis_thumb_ring = 1.0


    def device_init(self, cfg:VuerTpCfg):
    
        # eye camera resolution
        self.resolution = self.cfg.resolution
        # 图像剪裁尺寸？
        self.crop_size_w = 0
        self.crop_size_h = 0
        # 图像剪裁规则？
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)
        # 图像尺寸，双眼图像沿width方向拼接，三通道rgb
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        #
        self.img_height, self.img_width = self.resolution_cropped[:2]
        # 根据图像尺寸创建共享内存
        self.shm = shared_memory.SharedMemory(create=True, size=int(numpy.prod(self.img_shape)* numpy.uint8().itemsize))
        # image 数组
        self.img_array = numpy.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=numpy.uint8, buffer=self.shm.buf)
        # 
        image_queue = Queue()
        # 
        toggle_streaming = Event()
        # 初始化 open television,证书为相对路径，也可使用绝对路径
        self.tv = OpenTeleVision(self.resolution_cropped, 
                                 self.shm.name, 
                                 image_queue, 
                                 toggle_streaming,
                                 cert_file=self.cfg.cert_file, 
                                 key_file=self.cfg.key_file)

        # 根据配置生成默认输出
        self.head_mat_default = numpy.eye(4)
        self.head_mat_default[:,3] = numpy.array(self.cfg.head_pos + tuple([1]))
        self.left_wrist_mat_default = numpy.eye(4)
        self.left_wrist_mat_default[:,3] = numpy.array(self.cfg.hand_right_offset + tuple([1]))
        self.right_wrist_mat_default = numpy.eye(4)
        self.right_wrist_mat_default[:,3] = numpy.array(self.cfg.hand_left_offset + tuple([1]))

        # 初始化 retarget 配置
        left_hand_cfg = self.cfg.left_hand_retarget_cfg
        right_hand_cfg = self.cfg.right_hand_retarget_cfg
        
        # 根据配置文件创建retarget实例
        self.left_retargeting = left_hand_cfg.build()
        self.right_retargeting = right_hand_cfg.build()

    def reset(self):
        #
        super().reset()
        # reset variables
        self.dis_thumbs = 1.0
        self.dis_pinkys = 1.0
        self.dis_thumb_middle = 1.0
        self.dis_thumb_ring = 1.0

    def is_connected(self)->bool:
        return True
    
    def register_keyboard_handler(self):
        """
        Sets up the keyboard callback functionality with omniverse
        """
        appwindow = omni.appwindow.get_default_app_window() # type: ignore
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, self.keyboard_event_handler)
    
    def keyboard_event_handler(self, event, *args, **kwargs):
        
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            or event.type == carb.input.KeyboardEventType.KEY_REPEAT
        ):
            # Z键：重置
            if event.input == carb.input.KeyboardInput.Z:
                self.bReset = True

            # X键：开始控制
            if event.input == carb.input.KeyboardInput.X:
                self.bControl = True

            # C键：开始录制
            if event.input == carb.input.KeyboardInput.C:
                self.bRecording = True

            # V键：结束录制
            if event.input == carb.input.KeyboardInput.V:
                self.bFinished = True
           
            # Q键：结束程序
            if event.input == carb.input.KeyboardInput.Q:
                self.bQuit = True
        
        # If we release a key, clear the active action and keypress
        # elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            
            

        # Callback always needs to return True
        return True
    
    def set_camera_image(self, left_img:torch.Tensor, right_img:torch.Tensor):
        # # 获取Vr输出
        # head_rmat, left_pose, right_pose, left_qpos, right_qpos = self.veur_step()

        # 将图像拷贝至共享内存
        numpy.copyto(self.img_array, numpy.hstack((left_img.cpu(), right_img.cpu())))


    def update(self):

        # 读取vuer原始数据，坐标系为右手坐标系，Y轴向上，X轴向右，Z轴向后
        self.vuer_head_mat = mat_update(self.head_mat_default, self.tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(self.right_wrist_mat_default, self.tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.left_wrist_mat_default, self.tv.left_hand.copy())

        # 将原始数据转换为右手坐标系，X轴向前，Y轴向左，Z轴向上
        head_mat = self.cfg.grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(self.cfg.grd_yup2grd_zup)
        right_wrist_mat = self.cfg.grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(self.cfg.grd_yup2grd_zup)
        left_wrist_mat = self.cfg.grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(self.cfg.grd_yup2grd_zup)

        # 右乘旋转矩阵的原因与计算方法：
        # 由于vr中双手在胸前位置，食指、中指、无名指、小拇指向前，手掌向下的放松状态为初始姿态，此时姿态角(roll,pitch,yaw)为(0,0,0)
        # 但是USD和URDF中双手的腕部坐标系初始姿态往往不同于VR，因此需要将VR的腕部旋转矩阵转换为USD/URDF坐标系下的旋转矩阵
        # 以OY手为例，其初始姿态为食指、中指、无名指、小拇指指向Z轴正方向，手掌向X轴正方向
        # 因此原始数据(手腕部相对于基坐标系的旋转矩阵)需要右乘旋转矩阵（以XYZ为例，roll:0, pitch:1.57, yaw:0）
        rel_left_wrist_mat = left_wrist_mat @ self.cfg.hand_vr2usd  # 腕部相对头部的旋转矩阵
        rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]  #腕部相对头部的偏移
        rel_right_wrist_mat = right_wrist_mat @ self.cfg.hand_vr2usd # 腕部相对头部的旋转矩阵
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]    #腕部相对头部的偏移

        # homogeneous
        left_fingers = numpy.concatenate([self.tv.left_landmarks.copy().T, numpy.ones((1, self.tv.left_landmarks.shape[0]))])
        right_fingers = numpy.concatenate([self.tv.right_landmarks.copy().T, numpy.ones((1, self.tv.right_landmarks.shape[0]))])

        #TODO：梳理手指关节的坐标变换过程
        left_fingers = self.cfg.grd_yup2grd_zup @ left_fingers 
        right_fingers = self.cfg.grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (self.cfg.finger_vr2usd.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (self.cfg.finger_vr2usd.T @ rel_right_fingers)[0:3, :].T

        # 和双手一样，头部也需要根据机器人相机位置进行调整
        head_mat = head_mat @ self.cfg.head_optimize

        # # 返回头部齐次变换矩阵，腕部相对头部齐次变换矩阵，手指相对头部齐次变换(待定)
        # return head_mat, rel_left_wrist_mat, rel_right_wrist_mat, rel_left_fingers, rel_right_fingers

        # 头部旋转矩阵
        head_rmat = head_mat[:3, :3]

        # left_wrist_mat 手腕相对头的齐次变换矩阵,增加偏移方便操作，后续改为动作缩放
        left_wrist_position = rel_left_wrist_mat[:3, 3] + numpy.array(self.cfg.hand_left_offset)
        # left_wrist_position = rel_left_wrist_mat[:3, 3] + np.array(self.vuer_teleop_cfg.head_pos)
        left_wrist_quaternion = rotations.quaternion_from_matrix(rel_left_wrist_mat[:3, :3])[[0,1,2,3]]  # x,y,z,w

        right_wrist_position = rel_right_wrist_mat[:3, 3] + numpy.array(self.cfg.hand_right_offset)
        right_wrist_quaternion = rotations.quaternion_from_matrix(rel_right_wrist_mat[:3, :3])[[0,1,2,3]] #x,y,z,w

        left_pose = numpy.concatenate([left_wrist_position,left_wrist_quaternion])
        right_pose = numpy.concatenate([right_wrist_position,right_wrist_quaternion])

        # retarget 关节顺序为 1_1,1_2,1_3,2_1,2_2,3_1,3_2,4_1,4_2,5_1,5_2
        # 但是usd加载到lab后joint顺序为 1_1,2_1,3_1,4_1,5_1,1_2,2_2,3_2,4_2,5_2,1_3，所以要做映射
        # left_qpos = self.left_retargeting.retarget(rel_left_fingers[self.cfg.tip_indices])[[0,3,5,7,9,1,4,6,8,10,2]]
        # right_qpos = self.right_retargeting.retarget(rel_right_fingers[self.cfg.tip_indices])[[0,3,5,7,9,1,4,6,8,10,2]]
        left_qpos = self.left_retargeting.retarget(rel_left_fingers[self.cfg.tip_indices])[self.cfg.hand_retarget_indexs]
        right_qpos = self.right_retargeting.retarget(rel_right_fingers[self.cfg.tip_indices])[self.cfg.hand_retarget_indexs]
        
        # scale position in XY plane
        left_pose[:2] = self.cfg.hand_scale[:2] * left_pose[:2]
        right_pose[:2] = self.cfg.hand_scale[:2] * right_pose[:2]

        # refresh wrist pose
        self.left_wrist_pose = torch.tensor(left_pose, device=self.cfg.device,dtype=torch.float)
        self.right_wrist_pose = torch.tensor(right_pose, device=self.cfg.device,dtype=torch.float)

        # refresh hand joint position
        self.left_hand_joint_pos = torch.tensor(left_qpos, device=self.cfg.device,dtype=torch.float)
        self.right_hand_joint_pos = torch.tensor(right_qpos, device=self.cfg.device,dtype=torch.float)

        # refresh eye pose
        camera_left_pos = numpy.array(self.cfg.head_pos)+numpy.array(self.cfg.eye_left_offset) @ head_rmat.T
        camera_right_pos = numpy.array(self.cfg.head_pos)+numpy.array(self.cfg.eye_right_offset) @ head_rmat.T
        head_rmat = torch.tensor(head_rmat,device="cuda:0",dtype=torch.float)

        self.left_eye_pose = torch.cat(
            (torch.tensor(camera_left_pos,device=self.cfg.device,dtype=torch.float),
            quat_from_matrix(head_rmat))
        ,0)
        self.right_eye_pose = torch.cat(
            (torch.tensor(camera_right_pos,device=self.cfg.device,dtype=torch.float),
            quat_from_matrix(head_rmat))
        ,0)

        # 计算中指和大拇指的距离
        thumb_pos_left = numpy.array(rel_left_fingers[4])
        thumb_pos_right = numpy.array(rel_right_fingers[4])
        pinky_pos_left = numpy.array(rel_left_fingers[24])
        pinky_pos_right = numpy.array(rel_right_fingers[24])
        middle_pos_right = numpy.array(rel_right_fingers[14])
        ring_pos_right = numpy.array(rel_right_fingers[19])

        self.dis_thumbs = numpy.sqrt(numpy.sum(numpy.square(thumb_pos_left - thumb_pos_right)))
        self.dis_pinkys = numpy.sqrt(numpy.sum(numpy.square(pinky_pos_left - pinky_pos_right)))
        self.dis_thumb_middle = numpy.sqrt(numpy.sum(numpy.square(middle_pos_right - thumb_pos_right)))
        self.dis_thumb_ring = numpy.sqrt(numpy.sum(numpy.square(ring_pos_right - thumb_pos_right)))

        # todo: 默认情况下输出数据相同,dis为0为导致不断重置,因此暂时过滤为零的情况
        if self.dis_thumb_middle >0.0 and self.dis_thumb_middle<0.01:
            self.bControl = True
        # print(self.dis_thumb_ring)
        if self.dis_thumb_ring>0.0 and self.dis_thumb_ring<0.02:
            self.bReset = True


# ************* Helper function **************
def mat_update(mat_defualt, mat):
    if numpy.linalg.det(mat) == 0:
        return mat_defualt
    else:
        return mat

def fast_mat_inv(mat):
    ret = numpy.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret
