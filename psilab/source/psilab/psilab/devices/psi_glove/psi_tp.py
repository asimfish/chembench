# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-10-29
# Vesion: 1.0

""" Common Modules  """ 
import sys
import torch
import numpy
import time
import threading
from typing import Optional


""" Omniverse Modules  """ 
import carb
import omni

""" IsaacLab Modules  """ 
from isaaclab.utils.math import quat_inv,quat_mul

""" PsiLab Modules  """ 
from psilab.devices.open_television.TeleVision import OpenTeleVision
from psilab.devices.teleop_base import TeleOperateDeviceBase


from psilab.devices.psi_glove.tracker.tracker import SteamVRTrackerSDK,TrackerPose
# tracker.tracker import SteamVRTrackerSDK
from psilab.devices.psi_glove.glove.serial_interface import SerialInterface
from psilab.devices.psi_glove.glove.psi_glove_controller import PSIGloveController
from psilab.devices.psi_glove.psi_tp_cfg import PsiTpCfg

from psilab.devices.psi_glove.glove.psi_glove_controller import PSIGloveController, StatusMessage
# from psilab.devices.psi_glove.glove.communication_interface import SerialInterface


class PsiTp(TeleOperateDeviceBase):
    """
    Psi teleoperate device
    """
    cfg: PsiTpCfg

    def __init__(self, cfg:PsiTpCfg):

        super().__init__()

        self.cfg = cfg
        #
        self.device_init()
        # tracker init pose
        self._tracker_pose_init = {
            self.cfg.tracker_serial_left:torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0],device=self.cfg.device),
            self.cfg.tracker_serial_right:torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0],device=self.cfg.device)
        }
        # eef quaternion initillize
        self._eef_quat_init = {
            self.cfg.tracker_serial_left:torch.tensor(self.cfg.eef_left_quat_init,device=self.cfg.device),
            self.cfg.tracker_serial_right:torch.tensor(self.cfg.eef_right_quat_init,device=self.cfg.device),
        }
        # glove adc min/max value
        self._glove_right_min = torch.tensor(self.cfg.glove_right_adc_min,device=self.cfg.device)
        self._glove_right_max = torch.tensor(self.cfg.glove_right_adc_max,device=self.cfg.device)
        self._glove_left_min= torch.tensor(self.cfg.glove_left_adc_min,device=self.cfg.device)
        self._glove_left_max = torch.tensor(self.cfg.glove_left_adc_max,device=self.cfg.device)
        # buffer 
        self._tracker_left_delta_pos = torch.zeros(3,device=self.cfg.device)
        self._tracker_right_delta_pos = torch.zeros(3,device=self.cfg.device)
        self._eef_left_quat = torch.tensor([1,0,0,0],device=self.cfg.device)
        self._eef_right_quat = torch.tensor([1,0,0,0],device=self.cfg.device)
        self._hand_left_pos_norm = torch.zeros(len(self.cfg.glove_index),device=self.cfg.device)
        self._hand_right_pos_norm = torch.zeros(len(self.cfg.glove_index),device=self.cfg.device)
        # variables
        self._delta_pos_scale  = torch.tensor(self.cfg.delta_pos_scale ,device=self.cfg.device)

        
    def device_init(self):
        #
        self.tracker_init()
        #
        self.glove_init()
           
    def tracker_init(self):
        
        print("#" * 15, "Tracker Initialize Start", "#" * 15,"\n")
        self._tracker = SteamVRTrackerSDK()
        if not self._tracker.initialize():
            print("[ERROR] Failed to initialize. Make sure SteamVR is running.")
        # 
        print("[SUCCESS] Connected to SteamVR")
        
        # get all available trackers
        trackers = self._tracker.list_trackers()
        # 
        if not trackers:
            print("[WARN] No trackers found!")
            print("Make sure:")
            print("  - Trackers are powered on")
            print("  - Trackers are paired with SteamVR")
            print("  - Base stations are active")
            self._tracker.shutdown()
            return
        # 
        print(f"[INFO] Found {len(trackers)} tracker(s):")
        for i, tracker in enumerate(trackers, 1):
            print(f"  [{i}] Serial: {tracker['serial']}")
            print(f"      Model: {tracker['model']}")
            print(f"      Index: {tracker['index']}")
            
            # Get battery level if available
            battery = self._tracker.get_tracker_battery(tracker['serial'])
            if battery is not None:
                print(f"      Battery: {battery*100:.1f}%")
        
        print("#" * 15, "Tracker Initialize Finished", "#" * 15,"\n")

    def glove_init(self):

        print("#" * 15, "Glove Initialize Start", "#" * 15,"\n")

        try:
            # 1. 创建串口接口
            # 参数: 串口路径, 波特率, 超时时间(6ms), 自动连接, 模拟模式
            self._glove_left_serial = SerialInterface(
                port=self.cfg.glove_port_left,
                baudrate=self.cfg.glove_baudrate,
                timeout=0.006,
                auto_connect=False,
                mock=False
            )
            self._glove_right_serial = SerialInterface(
                port=self.cfg.glove_port_right,
                baudrate=self.cfg.glove_baudrate,
                timeout=0.006,
                auto_connect=False,
                mock=False
            )
            
            # 2. 创建控制器
            # 参数: 通信接口, 平滑窗口大小(10个样本)
            self._glove_left_controller = PSIGloveController(
                communication_interface=self._glove_left_serial,
                smoothing_window_size=5
            )
            self._glove_right_controller = PSIGloveController(
                communication_interface=self._glove_right_serial,
                smoothing_window_size=5
            )
            
            # 3. 连接设备
            print("[INFO] Psi Glove is connecting...")
            if not self._glove_left_controller.connect():
                print("[ERROR] Can not connect Psi Glove Left")
            if not self._glove_right_controller.connect():
                print("[ERROR] Can not connect Psi Glove Right")
            print("[INFO] Psi Glove connect success...")

            print("#" * 15, "Glove Initialize Finished", "#" * 15,"\n")

        except Exception as e:
            print(f"[ERROR]: {e}")
    
    def reset(self):
        #
        super().reset()

        # reset variables
        self._tracker_pose_init = {
            self.cfg.tracker_serial_left:torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0],device=self.cfg.device),
            self.cfg.tracker_serial_right:torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0],device=self.cfg.device)
        }

    def is_connected(self)->bool:
        return True
    
    def update(self):
        # ########### update htc tracker ############
        #
        poses = self._tracker.get_all_trackers()
        # 
        self._delta_tracker_left_pos, self._wrist_left_quat = self.tracker_pose_process(poses,self.cfg.tracker_serial_left)
        self._delta_tracker_right_pos, self._wrist_right_quat = self.tracker_pose_process(poses,self.cfg.tracker_serial_right)
        
        # ########### update psi glove ############
        # 
        glove_left_adc: Optional[StatusMessage] = self._glove_left_controller.loop()
        glove_right_adc: Optional[StatusMessage] = self._glove_right_controller.loop()
        # 

        # convert to list
        glove_left_adc_list = torch.tensor(glove_left_adc.to_list(),device=self.cfg.device) if glove_left_adc else self._glove_left_min
        glove_right_adc_list = torch.tensor(glove_right_adc.to_list(),device=self.cfg.device) if glove_right_adc else self._glove_right_min

        # normallize
        glove_left_norm = (glove_left_adc_list - self._glove_left_min) / (self._glove_left_max - self._glove_left_min) 
        glove_right_norm = (glove_right_adc_list - self._glove_right_min) / (self._glove_right_max - self._glove_right_min) 

        # TODO：part of joint should processed(大拇指旋转方向取反)??
        # glove_left_status[4] = 1- glove_left_status[4] # type: ignore
        # glove_right_status[4] = 1- glove_right_status[4] # type: ignore


        self._hand_left_pos_norm = glove_left_norm[self.cfg.glove_index] 
        self._hand_right_pos_norm = glove_right_norm[self.cfg.glove_index] 


    def tracker_pose_process(self, poses:dict[str, TrackerPose], serial: str)->tuple[torch.Tensor,torch.Tensor]:
        """
        TODO：确定tracker的坐标系是否和第一个基站位姿有关？
        Tracker原始数据处理：
            前提假设：
                1.真实世界坐标系原点位于使用者中心点在地面投影，X轴正方向为面部朝向，Z轴正方向为垂直地面向上，Y轴方向由右手定则确定
                2.仿真世界坐标系与真实世界坐标系相同
            步骤：
                Step 1. 将Tracker的Position转换到真实(仿真)世界坐标
                Step 2. 将Tracker的Quaternion转换到真实(仿真)世界坐标，这里不用旋转矩阵是因为原始数据就是四元数，
                    而且旋转矩阵的顺序不确定的话容易引起错误，所以直接用四元数计算
                Step 3. 将Tracker的Position和Quaternion拼接为Pose
                Step 4. 判断是否是第一次获得数据，填充初始Pose变量
                Step 5. 根据当前位置和初始位置计算Delta Position
                Step 6. 根据当前姿态和初始姿态计算Delta Quaternion
                Step 7. 根据机械臂末端Link的初始姿态和Delta Quaternion计算新的机械臂末端Link姿态

        Args:
            poses: all tracker poses dict
            serial: pose of which tracker need process
        Return:
            delta_tracker_pos: tracker在仿真世界坐标系下相对于初始位置的位置变化
            wrist_quat: 
                根据tracker相对于初始姿态的姿态变化计算出的手腕在仿真世界坐标系下的绝对姿态
        """
        # 
        if serial not in poses.keys():
            return torch.tensor([0,0,0], device=self.cfg.device),torch.tensor([1,0,0,0], device=self.cfg.device)
        
        # get tracker pose
        tracker_pose = poses[serial]
        # if serial=="LHR-B72584D2":
        #     print(f"{tracker_pose.qw},  {tracker_pose.qx}, {tracker_pose.qy}, {tracker_pose.qz}")
        # get transformation matrix
        tracker_pose_T = tracker_pose.get_transformation_matrix()

        # convert position of tracker to world coordinate system
        tracker_pose_T = self.cfg.grd_yup2grd_zup @ tracker_pose_T @ fast_mat_inv(self.cfg.grd_yup2grd_zup)

        # convert quaternion of tracker to world coordinate system
        tracker_quat = quat_mul(
            torch.tensor(self.cfg.tracker2world_quat,device=self.cfg.device),
            torch.tensor([tracker_pose.qw,tracker_pose.qx, tracker_pose.qy, tracker_pose.qz],device=self.cfg.device),
            )
        
        # get tracker pose in world coordinate system
        tracker_pose = torch.tensor([tracker_pose_T[0,3],tracker_pose_T[1,3],tracker_pose_T[2,3],tracker_quat[0],tracker_quat[1],tracker_quat[2],tracker_quat[3]],device=self.cfg.device)
        
        # initillize pose_init if get data first time
        if torch.equal(self._tracker_pose_init[serial],torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0],device=self.cfg.device)):
            if not torch.equal(tracker_pose,self._tracker_pose_init[serial]):
                self._tracker_pose_init[serial] = tracker_pose # type: ignore

        # compute delta pos
        delta_tracker_pos = self._delta_pos_scale * (tracker_pose[:3] - self._tracker_pose_init[serial][:3] )# type: ignore

        # compute delta quat
        delta_tracker_quat = quat_mul(tracker_pose[3:],quat_inv(self._tracker_pose_init[serial][3:])) # type: ignore
        
        # compute absolute wrist quat 
        eef_quat = quat_mul(
            delta_tracker_quat,
            self._eef_quat_init[serial])

        return delta_tracker_pos,eef_quat

    # def start(self,):
    #     # 启动 控制线程
    #     self.is_running = True
    #     self._thread = threading.Thread(target=self.run, daemon=True)
    #     self._thread.start()
    
    # def run(self):
    #     # loop
    #     while self.is_running:
    #         # 当前时间
    #         current_time = time.time()

    #         #
    #         self.update()

    #         # 控制执行频率
    #         elapsed = time.time() - current_time
    #         # print(elapsed)
    #         if elapsed < self._delta_time:
    #             time.sleep(self._delta_time - elapsed)


# ************* Helper function **************
def fast_mat_inv(mat):
    ret = numpy.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret



