# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from dataclasses import MISSING
from typing import Any
from collections.abc import Sequence

""" Common Modules  """ 
import torch
import numpy as np
import carb
import omni
import cv2
import os
from datetime import datetime

# Ruckig 轨迹平滑 (使用封装类)
import sys
sys.path.append("/home/psibot/psi-lab-v2/fix")
from Ruckig_Interpolator import SmoothJointInterpolator

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils.math import quat_from_matrix,quat_inv,quat_mul


""" Psi Lab Modules  """
from psilab.envs.tp_env import TPEnv 
from psilab.envs.tp_env_cfg import TPEnvCfg


from psilab.utils.timer_utils import Timer
from psilab.devices.configs.psi_glove_cfg import PSIGLOVE_PSI_DC_02_CFG
from psilab import OUTPUT_DIR
from psilab.utils.data_collect_utils import parse_data,save_data
from psilab.eval.grasp_rigid import eval_success_only_height
from psilab.utils.math_utils import unnormalize_v2


@configclass
class GraspBottleEnvCfg(TPEnvCfg):
    """Configuration for Rl environment."""

    # fake params which is useless
    episode_length_s = 1 * 210 / 60.0
    decimation = 1
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # device
    device_type = "psi-glove"
    device_cfg = PSIGLOVE_PSI_DC_02_CFG

    # viewer config
    viewer = ViewerCfg(
        # eye=(0.12, 0.0, 1.5),
        # lookat=(0.9,0.0,0.3)
        # eye=(2.0,0.0,1.2),
        # lookat=(-15.0,0.0,0.3)

        eye=(5.3,0.24,1.2),
        lookat=(5.3,-2.0,1.0)
    )

    # simulation  config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=1,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 32,
            max_velocity_iteration_count = 4,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            gpu_found_lost_pairs_capacity = 137401003
        ),
        render=RenderCfg(
            # enable_translucency=True,
            # enable_reflections=True,
            # enable_global_illumination=True,
            # antialiasing_mode="DLAA",
            # enable_dlssg=True,
            # enable_dl_denoiser=True,
            # samples_per_pixel=16,
                    # render=RenderCfg(
            enable_translucency=True,
        # ),
            # enable_ambient_occlusion=True
        ),
    )

    # 
    sample_step = 1

    # scene config
    scene = MISSING # type: ignore

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/grasp_bottle_img"

    # lift desired height
    lift_height_desired = 0.3

    # --- 视频录制配置 ---
    video_output_folder = OUTPUT_DIR + "/videos"  # 视频输出目录
    video_fps = 30  # 视频帧率
    video_camera_name = "third_person_camera"  # 录制的相机名称

    # --- 轨迹处理选项 ---
    # 是否启用 tracker 位置跳跃过滤
    enable_tracker_filter = True
    tracker_jump_threshold = 0.05  # 位置跳跃阈值 (m)
    
    # 是否启用 Ruckig 轨迹平滑插值
    enable_ruckig_smooth = False
    # Ruckig 运动学限制参数
    ruckig_max_velocity = 3.0      # 最大速度 (m/s)
    ruckig_max_acceleration = 6.0  # 最大加速度 (m/s²)
    ruckig_max_jerk = 12.0         # 最大jerk (m/s³)


class GraspBottleEnv(TPEnv):

    cfg: GraspBottleEnvCfg

    def __init__(self, cfg: GraspBottleEnvCfg, render_mode: str | None = None, **kwargs):

        #
        super().__init__(cfg, render_mode, **kwargs)

        # instances in scene
        # self._robot = self.scene.robots["robot"]
        # self._target = self.scene.rigid_objects["target"]

        # instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["bottle"]
        # self._target = self.scene.articu["bottle"]
        # self._target = self.scene.articulated_objects["bottle"]


        self._contact_sensors = {}
        for key in ["hand2_link_base",
                    "hand2_link_1_1",
                    "hand2_link_1_2",
                    "hand2_link_1_3",
                    "hand2_link_2_1",
                    "hand2_link_2_2",
                    "hand2_link_3_1",
                    "hand2_link_3_2",
                    "hand2_link_4_1",
                    "hand2_link_4_2",
                    "hand2_link_5_1",
                    "hand2_link_5_2"]:
            self._contact_sensors[key] = self.scene.sensors[key]

        # start vuer t
        # start device
        self._device.start() # type: ignore

        # eef link index
        self._arm1_eef_link_index = self._robot.find_bodies("arm1_link7")[0][0]
        self._arm2_eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        # 
        self._arm1_joint_index = self._robot.actuators["arm1"].joint_indices # type: ignore
        self._arm2_joint_index = self._robot.actuators["arm2"].joint_indices # type: ignore
        self._hand1_joint_index = self._robot.actuators["hand1"].joint_indices[:6] # type: ignore
        self._hand2_joint_index = self._robot.actuators["hand2"].joint_indices[:6] # type: ignore


        # joint limit for compute later
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()
        #[x,  y,  z,  qw, qx, qy, qz]
        self._arm1_eef_pose_desired = torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0], device=self.device).unsqueeze(0)
        self._arm2_eef_pose_desired = torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0], device=self.device).unsqueeze(0)

        self._arm1_eef_pose_init = torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0], device=self.device).unsqueeze(0)
        self._arm2_eef_pose_init = torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0], device=self.device).unsqueeze(0)


        # hand real joint index
        self._hand_real_joint_index_left = self._robot.actuators["hand1"].joint_indices[:6] # type: ignore
        self._hand_real_joint_index_right = self._robot.actuators["hand2"].joint_indices[:6] # type: ignore
        # hand virtual joint index
        self._hand_virtual_joint_index_left = self._robot.actuators["hand1"].joint_indices[6:] # type: ignore
        self._hand_virtual_joint_index_right = self._robot.actuators["hand2"].joint_indices[6:] # type: ignore
        
        # initialize Timer
        self._timer = Timer()
        # variables used to store contact flag
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs,3),device=self.device)

        # 获取机器人基座旋转四元数及其逆 (用于坐标变换)
        self._robot_base_quat = self._robot.data.root_state_w[0, 3:7].clone()  # (qw, qx, qy, qz)
        self._robot_base_quat_inv = quat_inv(self._robot_base_quat.unsqueeze(0)).squeeze(0)  # 逆四元数

        # tracker position filter: 用于过滤tracker位置跳跃异常值
        self._prev_delta_tracker_left_pos = None  # 上一帧左手tracker增量位置
        self._prev_delta_tracker_right_pos = None  # 上一帧右手tracker增量位置

        # Ruckig 轨迹平滑插值器 (位置: dof=3)
        self._ruckig_step = self.cfg.sim.dt  # 控制周期，与仿真dt一致
        if self.cfg.enable_ruckig_smooth:
            self._init_ruckig_interpolators()
        
        # keyboard input flag for manual recording trigger
        self._enter_pressed = False
        
        # --- 视频录制相关变量 ---
        self._v_pressed = False  # V键按下标志
        self._video_recording = False  # 是否正在录制视频
        self._video_frames = []  # 存储视频帧
        self._video_start_time = None  # 视频开始录制时间
        
        self._register_keyboard_handler()


        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)



    def _rotate_vector_by_quat(self, vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """使用四元数旋转向量
        Args:
            vec: 3D向量 (3,)
            quat: 四元数 (qw, qx, qy, qz) (4,)
        Returns:
            旋转后的向量 (3,)
        """
        # 确保数据类型一致 (统一为 float32)
        vec = vec.float()
        quat = quat.float()
        
        # 四元数分量
        qw = quat[0]
        quat_xyz = quat[1:4]
        
        # 使用四元数旋转公式: v' = q * v * q^(-1)
        # 简化公式:
        t = 2.0 * torch.cross(quat_xyz, vec)
        return vec + qw * t + torch.cross(quat_xyz, t)

    def _init_ruckig_interpolators(self):
        """初始化 Ruckig 轨迹平滑插值器 (使用 SmoothJointInterpolator 封装类)"""
        # 左手位置插值器 (x, y, z)
        self._ruckig_left = SmoothJointInterpolator(dof=3, step=self._ruckig_step, alpha=0.5)
        # 设置运动学限制 (从配置中读取参数)
        self._ruckig_left.set_kinematic_limits(
            max_velocity=self.cfg.ruckig_max_velocity,
            max_acceleration=self.cfg.ruckig_max_acceleration,
            max_jerk=self.cfg.ruckig_max_jerk
        )

        # 右手位置插值器 (x, y, z)
        self._ruckig_right = SmoothJointInterpolator(dof=3, step=self._ruckig_step, alpha=0.5)
        self._ruckig_right.set_kinematic_limits(
            max_velocity=self.cfg.ruckig_max_velocity,
            max_acceleration=self.cfg.ruckig_max_acceleration,
            max_jerk=self.cfg.ruckig_max_jerk
        )

    def _register_keyboard_handler(self):
        """Register keyboard callback for manual recording trigger"""
        appwindow = omni.appwindow.get_default_app_window()  # type: ignore
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        input_interface.subscribe_to_keyboard_events(keyboard, self._keyboard_event_handler)

    def _keyboard_event_handler(self, event, *args, **kwargs):
        """Handle keyboard events - Enter key triggers recording, V key triggers video recording"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Enter键：手动开启录制
            if event.input == carb.input.KeyboardInput.ENTER:
                self._enter_pressed = True
            # V键：切换视频录制状态
            elif event.input == carb.input.KeyboardInput.V:
                self._v_pressed = True
        return True

    def step(self,actions):
        
        # simulator step
        self.sim_step()
        #
        return super().step(actions)
        
    def sim_step(self):
        
        # hand gesture control
        # change to control mode while index,middel,ring, and little finger is clenched
        # print(self._device._hand_right_pos_norm)
        if not self._device.bControl and self._device._hand_right_pos_norm[1]<0.1 and self._device._hand_right_pos_norm[2]<0.1 and self._device._hand_right_pos_norm[3]<0.1 and self._device._hand_right_pos_norm[4]<0.1:
            self._device.bControl = True
        
        # device control
        if self._device.bControl:

            # 获取当前tracker位置 (设备/世界坐标系)
            delta_left_pos_raw = self._device._delta_tracker_left_pos  # type: ignore
            delta_right_pos_raw = self._device._delta_tracker_right_pos  # type: ignore
            
            # 将位置增量从设备/世界坐标系转换到机器人局部坐标系 (使用逆旋转)
            # 当机器人旋转90度时，设备的"向前"需要映射到机器人的"向前"
            delta_left_pos = self._rotate_vector_by_quat(delta_left_pos_raw, self._robot_base_quat_inv)
            delta_right_pos = self._rotate_vector_by_quat(delta_right_pos_raw, self._robot_base_quat_inv)

            # --- 可选：tracker 位置跳跃过滤 ---
            if self.cfg.enable_tracker_filter:
                # 左手tracker位置跳跃过滤
                if self._prev_delta_tracker_left_pos is not None:
                    left_change = torch.norm(delta_left_pos - self._prev_delta_tracker_left_pos).item()
                    if left_change > self.cfg.tracker_jump_threshold:
                        # print(f"[WARNING] 左手tracker位置跳跃: {left_change:.4f} m, 使用上一帧数据")
                        delta_left_pos = self._prev_delta_tracker_left_pos
                self._prev_delta_tracker_left_pos = delta_left_pos.clone()

                # 右手tracker位置跳跃过滤
                if self._prev_delta_tracker_right_pos is not None:
                    right_change = torch.norm(delta_right_pos - self._prev_delta_tracker_right_pos).item()
                    if right_change > self.cfg.tracker_jump_threshold:
                        # print(f"[WARNING] 右手tracker位置跳跃: {right_change:.4f} m, 使用上一帧数据")
                        delta_right_pos = self._prev_delta_tracker_right_pos
                self._prev_delta_tracker_right_pos = delta_right_pos.clone()

            # 计算目标位置
            target_left_pos = (self._arm1_eef_pose_init[0, 0:3] + delta_left_pos).cpu().numpy()
            target_right_pos = (self._arm2_eef_pose_init[0, 0:3] + delta_right_pos).cpu().numpy()

            # --- 可选：Ruckig 轨迹平滑插值 ---
            if self.cfg.enable_ruckig_smooth:
                # 左手 Ruckig 插值
                if not self._ruckig_left.is_initialized:
                    # 首次初始化：设置当前位置作为起点
                    self._ruckig_left.set_input_param(
                        current_position=target_left_pos.tolist(),
                        current_velocity=[0.0, 0.0, 0.0],
                        current_acceleration=[0.0, 0.0, 0.0]
                    )
                    smooth_left_pos = target_left_pos
                else:
                    # 更新目标位置并获取平滑后的位置
                    smooth_left_pos, _, _, _ = self._ruckig_left.update(target_pos=target_left_pos.tolist())

                # 右手 Ruckig 插值
                if not self._ruckig_right.is_initialized:
                    # 首次初始化：设置当前位置作为起点
                    self._ruckig_right.set_input_param(
                        current_position=target_right_pos.tolist(),
                        current_velocity=[0.0, 0.0, 0.0],
                        current_acceleration=[0.0, 0.0, 0.0]
                    )
                    smooth_right_pos = target_right_pos
                else:
                    # 更新目标位置并获取平滑后的位置
                    smooth_right_pos, _, _, _ = self._ruckig_right.update(target_pos=target_right_pos.tolist())
            else:
                # 不使用平滑，直接使用原始目标位置
                smooth_left_pos = target_left_pos
                smooth_right_pos = target_right_pos

            # arm1 - 使用处理后的位置
            self._arm1_eef_pose_desired[0, 0:3] = torch.from_numpy(smooth_left_pos).to(self.device)
            # 将手腕四元数从设备/世界坐标系转换到机器人局部坐标系: q_local = q_base_inv * q_device
            wrist_left_quat = self._device._wrist_left_quat  # type: ignore
            if wrist_left_quat.dim() == 1:
                wrist_left_quat = wrist_left_quat.unsqueeze(0)
            base_quat_inv = self._robot_base_quat_inv.unsqueeze(0) if self._robot_base_quat_inv.dim() == 1 else self._robot_base_quat_inv
            self._arm1_eef_pose_desired[0, 3:] = quat_mul(base_quat_inv, wrist_left_quat).squeeze(0)
            
            # arm2 - 使用处理后的位置
            self._arm2_eef_pose_desired[0, 0:3] = torch.from_numpy(smooth_right_pos).to(self.device)
            wrist_right_quat = self._device._wrist_right_quat  # type: ignore
            if wrist_right_quat.dim() == 1:
                wrist_right_quat = wrist_right_quat.unsqueeze(0)
            self._arm2_eef_pose_desired[0, 3:] = quat_mul(base_quat_inv, wrist_right_quat).squeeze(0)
            
            self.scene.robots["robot"].set_ik_command({
                    "arm1": self._arm1_eef_pose_desired,
                    "arm2": self._arm2_eef_pose_desired,
                })

        
            self._robot.ik_step()
            #
            hand_left_pos= unnormalize_v2(
                self._device._hand_left_pos_norm,
                self._joint_limit_lower[:,self._hand1_joint_index],
                self._joint_limit_upper[:,self._hand1_joint_index]
            )
            hand_right_pos= unnormalize_v2(
                self._device._hand_right_pos_norm,
                self._joint_limit_lower[:,self._hand2_joint_index],
                self._joint_limit_upper[:,self._hand2_joint_index]
            )
            
            self._robot.set_joint_position_target(hand_left_pos,self._hand1_joint_index) # type: ignore
            self._robot.set_joint_position_target(hand_right_pos,self._hand2_joint_index) # type: ignore

        # automatically determine whether to start recording
        if self._device.bControl and not self._device.bRecording:
            bPrepared = True
            # 开启录制条件1：右手位置与控制输入差距小于阈值
            # state_c = self._device.right_wrist_pose + self._robot.data.body_link_state_w[0][0][:7]
            
            state = self._robot.data.body_link_state_w[:,self._robot.ik_controllers["arm2"].eef_link_index]
            delta_pos_norm = torch.norm((state[0,:3]-self._arm2_eef_pose_desired[0,:3]),p=2)
            if delta_pos_norm>0.1:
                bPrepared = bPrepared and False
            # 开启录制条件2: 右手位姿速度小于阈值
            pos_vel = self._robot.data.body_state_w[0][self._robot.ik_controllers["arm2"].eef_link_index][7:10]
            angle_vel = self._robot.data.body_state_w[0][self._robot.ik_controllers["arm2"].eef_link_index][10:]
            pos_vel_norm = torch.norm(pos_vel,p=2)
            angle_vel_norm = torch.norm(angle_vel,p=2)
            # print(f"delta_pos: {delta_pos_norm}, pos_vel: {pos_vel_norm}, angle_vel: {angle_vel_norm}")
            if pos_vel_norm>0.02 or pos_vel_norm==0 or angle_vel_norm>0.02 or angle_vel_norm == 0:
                bPrepared = bPrepared and False
            
            # 开启录制条件3: 右手四指头伸直
            if self._device._hand_right_pos_norm[1]<0.7 or self._device._hand_right_pos_norm[2]<0.7 or self._device._hand_right_pos_norm[3]<0.7 or self._device._hand_right_pos_norm[4]<0.7:
                bPrepared = bPrepared and False
            
            # 开启录制方式：自动条件满足 或 键盘Enter键手动触发
            if bPrepared or self._enter_pressed:
                self._device.bRecording = True
                self._enter_pressed = False  # reset flag
                # 
                self._voice.say("开始录制")
                print("开始录制")

        # store data accrding to record flag and enable ouput flag and sample step 
        if self._device.bRecording and self.cfg.enable_output and self._sim_step_counter % self.cfg.sample_step == 0:
            # parse sim data
            parse_data(
                sim_time=self._sim_step_counter * self.cfg.sim.dt,
                data = self._data,
                scene = self.scene
            )
        
        # --- 视频录制处理 ---
        self._handle_video_recording()
 
        # reset, 中指和无名指弯曲,食指和小拇指伸直
        if not self._device.bReset and self._device._hand_right_pos_norm[1]>0.7 and self._device._hand_right_pos_norm[2]<0.1 and self._device._hand_right_pos_norm[3]<0.1 and self._device._hand_right_pos_norm[4]>0.7:
            self._device.bReset = True
            self._device.reset()
            # only save data while "enable_output" is true
            if self.cfg.enable_output:
                self._data = save_data(
                    data=self._data,
                    cfg=self.cfg,
                    scene=self.scene,
                    env_indexs=[],
                )
            self.reset()
        
        #
        super().sim_step()
        #
        self._sim_step_counter+=1
        #
        # get dones only when recording
        if self._device.bRecording:
          # get dones
            success, time_out = self._get_dones()
            reset = success | time_out 
            # get ids of envs to reset
            reset_ids = torch.nonzero(reset==True).squeeze()
            # bug: if single index, squeeze will change tensor to torch.Size([])
            reset_ids = reset_ids.unsqueeze(0) if reset_ids.size()==torch.Size([]) else reset_ids
            # get ids of envs completed successfully
            success_ids = torch.nonzero(success==True).squeeze().tolist()
            # bug: if single index, squeeze will change tensor to torch.Size([])
            success_ids = [success_ids] if type(success_ids)==int else success_ids
            # reset envs
            if len(reset_ids) > 0:
                # 
                self._reset_idx(reset_ids,success_ids)  # type: ignore
                # update articulation kinematics
                self.scene.write_data_to_sim()
                self.sim.forward()
                # if sensors are added to the scene, make sure we render to reflect changes in reset
                if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                    self.sim.render()
        
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        #
        super().reset()
        #
        # run 50 step until all rigid is static
        for i in range(50):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            
        # variables used to store contact flag
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init[:,:]=self._target.data.root_link_pos_w[:,:].clone()
        
        # 更新机器人基座旋转四元数及其逆
        self._robot_base_quat = self._robot.data.root_state_w[0, 3:7].clone()
        self._robot_base_quat_inv = quat_inv(self._robot_base_quat.unsqueeze(0)).squeeze(0)
        
        # 初始化末端执行器位姿（相对于机器人基座，在机器人局部坐标系中）
        self._arm1_eef_pose_init = self._robot.data.body_link_state_w[:,self._arm1_eef_link_index,:7].clone()
        self._arm2_eef_pose_init = self._robot.data.body_link_state_w[:,self._arm2_eef_link_index,:7].clone()
        
        # 计算相对于机器人根部的位置偏移（世界坐标系）
        arm1_rel_pos_world = self._arm1_eef_pose_init[:,:3] - self._robot.data.root_state_w[:,:3]
        arm2_rel_pos_world = self._arm2_eef_pose_init[:,:3] - self._robot.data.root_state_w[:,:3]
        
        # 将位置从世界坐标系转换到机器人局部坐标系（使用逆旋转）
        self._arm1_eef_pose_init[0,:3] = self._rotate_vector_by_quat(arm1_rel_pos_world[0], self._robot_base_quat_inv)
        self._arm2_eef_pose_init[0,:3] = self._rotate_vector_by_quat(arm2_rel_pos_world[0], self._robot_base_quat_inv)
        
        # 将姿态四元数也转换到机器人局部坐标系
        base_quat_inv_batch = self._robot_base_quat_inv.unsqueeze(0)
        self._arm1_eef_pose_init[:,3:] = quat_mul(base_quat_inv_batch, self._arm1_eef_pose_init[:,3:])
        self._arm2_eef_pose_init[:,3:] = quat_mul(base_quat_inv_batch, self._arm2_eef_pose_init[:,3:])

        # 重置tracker位置过滤器
        self._prev_delta_tracker_left_pos = None
        self._prev_delta_tracker_right_pos = None

        # 重置 Ruckig 插值器状态 (如果启用)
        if self.cfg.enable_ruckig_smooth:
            self._ruckig_left.close()
            self._ruckig_right.close()
         
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # time_out = self._sim_step_counter //  self.cfg.max_step >= (self._episode_num + 1)
        # task evalutation: success
        bsuccessed= eval_success_only_height(self._target,self._target_pos_init, self.cfg.lift_height_desired) # type: ignore
           
        # update success number
        self._episode_success_num+=len(torch.nonzero(bsuccessed==True).squeeze(1).tolist())

        return bsuccessed, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None, success_ids:Sequence[int]|None=None):
        
        # get env indexs
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if success_ids is None:
            success_ids=[]
        
        # output data
        if self.cfg.enable_output:
            self._data = save_data(
                data=self._data,
                cfg=self.cfg,
                scene=self.scene,
                env_indexs=success_ids,
                reset_env_indexs=env_ids.tolist(),
            )
        # 
        self._device.reset()

        super()._reset_idx(env_ids)   

        # print logs
        if self.cfg.enable_log:
            self._log_info()
       
        # reset variables
        self._target_pos_init[env_ids,:]=self._target.data.root_link_pos_w[env_ids,:].clone()
        self._has_contacted[env_ids] = torch.zeros_like(self._has_contacted[env_ids],device=self.device, dtype=torch.bool) # type: ignore

    def _log_info(self):
        # log data clollection result
        if self.cfg.enable_output and self._episode_num>0:
            # compute data collect result
            record_time = self._timer.run_time() /60.0
            record_rate = self._episode_success_num / record_time
            info = f"采集时长: {record_time} 分钟  "
            info +=f"采集数据: {self._episode_success_num} 条  "
            info += f"采集效率: {record_rate} 条/分钟" 
            print(info)
            # print(info, end='\r')

    def _handle_video_recording(self):
        """处理视频录制逻辑：V键切换录制状态"""
        if self._v_pressed:
            self._v_pressed = False  # 重置标志
            
            if not self._video_recording:
                # 开始录制
                self._start_video_recording()
            else:
                # 停止录制并保存
                self._stop_video_recording()
        
        # 如果正在录制，捕获当前帧
        if self._video_recording:
            self._capture_video_frame()

    def _start_video_recording(self):
        """开始视频录制"""
        self._video_recording = True
        self._video_frames = []
        self._video_start_time = datetime.now()
        self._voice.say("开始录制视频")
        print(f"[VIDEO] 开始录制视频 - 相机: {self.cfg.video_camera_name}")

    def _stop_video_recording(self):
        """停止视频录制并保存"""
        self._video_recording = False
        
        if len(self._video_frames) == 0:
            print("[VIDEO] 没有录制到任何帧，取消保存")
            self._voice.say("没有录制到视频")
            return
        
        # 保存视频
        self._save_video()
        self._voice.say("视频保存完成")

    def _capture_video_frame(self):
        """捕获当前相机帧"""
        try:
            # 获取相机 RGB 数据
            camera = self._robot.tiled_cameras[self.cfg.video_camera_name]
            rgb_data = camera.data.output["rgb"][0, :, :, :3]  # 取第一个环境，去掉 alpha 通道
            
            # 转换为 numpy uint8 格式 (H, W, 3)
            frame = rgb_data.cpu().numpy().astype(np.uint8)
            
            # OpenCV 使用 BGR 格式，需要转换
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            self._video_frames.append(frame_bgr)
        except Exception as e:
            print(f"[VIDEO] 捕获帧失败: {e}")

    def _save_video(self):
        """保存视频到指定目录"""
        # 确保输出目录存在
        os.makedirs(self.cfg.video_output_folder, exist_ok=True)
        
        # 生成文件名 (使用开始录制时间)
        if self._video_start_time is not None:
            timestamp = self._video_start_time.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"video_{timestamp}.avi"
        video_path = os.path.join(self.cfg.video_output_folder, video_filename)
        
        # 获取帧尺寸
        height, width = self._video_frames[0].shape[:2]
        
        # 创建 VideoWriter - 使用 XVID 编码 (兼容性更好)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, self.cfg.video_fps, (width, height))
        
        # 写入所有帧
        for frame in self._video_frames:
            video_writer.write(frame)
        
        video_writer.release()
        
        # 使用 ffmpeg 转换为 mp4 (H.264 编码，兼容性最好)
        mp4_path = video_path.replace('.avi', '.mp4')
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path, 
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                mp4_path
            ], check=True, capture_output=True)
            # 删除临时 avi 文件
            os.remove(video_path)
            video_path = mp4_path
        except Exception as e:
            print(f"[VIDEO] ffmpeg 转换失败，保留 avi 格式: {e}")
        
        # 清空帧列表
        frame_count = len(self._video_frames)
        duration = frame_count / self.cfg.video_fps
        self._video_frames = []
        
        print(f"[VIDEO] 视频已保存: {video_path}")
        print(f"[VIDEO] 帧数: {frame_count}, 时长: {duration:.2f}秒")
