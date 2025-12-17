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
        eye=(2.0,0.0,1.2),
        lookat=(-15.0,0.0,0.3)
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
            samples_per_pixel=16,
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

        # tracker position filter: 用于过滤tracker位置跳跃异常值
        self._prev_delta_tracker_left_pos = None  # 上一帧左手tracker增量位置
        self._prev_delta_tracker_right_pos = None  # 上一帧右手tracker增量位置

        # Ruckig 轨迹平滑插值器 (位置: dof=3)
        self._ruckig_step = self.cfg.sim.dt  # 控制周期，与仿真dt一致
        if self.cfg.enable_ruckig_smooth:
            self._init_ruckig_interpolators()
        
        # keyboard input flag for manual recording trigger
        self._enter_pressed = False
        self._register_keyboard_handler()

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
        """Handle keyboard events - Enter key triggers recording"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Enter键：手动开启录制
            if event.input == carb.input.KeyboardInput.ENTER:
                self._enter_pressed = True
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

            # 获取当前tracker位置
            delta_left_pos = self._device._delta_tracker_left_pos  # type: ignore
            delta_right_pos = self._device._delta_tracker_right_pos  # type: ignore

            # --- 可选：tracker 位置跳跃过滤 ---
            if self.cfg.enable_tracker_filter:
                # 左手tracker位置跳跃过滤
                if self._prev_delta_tracker_left_pos is not None:
                    left_change = torch.norm(delta_left_pos - self._prev_delta_tracker_left_pos).item()
                    if left_change > self.cfg.tracker_jump_threshold:
                        print(f"[WARNING] 左手tracker位置跳跃: {left_change:.4f} m, 使用上一帧数据")
                        delta_left_pos = self._prev_delta_tracker_left_pos
                self._prev_delta_tracker_left_pos = delta_left_pos.clone()

                # 右手tracker位置跳跃过滤
                if self._prev_delta_tracker_right_pos is not None:
                    right_change = torch.norm(delta_right_pos - self._prev_delta_tracker_right_pos).item()
                    if right_change > self.cfg.tracker_jump_threshold:
                        print(f"[WARNING] 右手tracker位置跳跃: {right_change:.4f} m, 使用上一帧数据")
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
            self._arm1_eef_pose_desired[0, 3:] = self._device._wrist_left_quat  # type: ignore
            # arm2 - 使用处理后的位置
            self._arm2_eef_pose_desired[0, 0:3] = torch.from_numpy(smooth_right_pos).to(self.device)
            self._arm2_eef_pose_desired[0, 3:] = self._device._wrist_right_quat  # type: ignore
            
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
        # initiallze 
        
        self._arm1_eef_pose_init = self._robot.data.body_link_state_w[:,self._arm1_eef_link_index,:7].clone()
        self._arm2_eef_pose_init = self._robot.data.body_link_state_w[:,self._arm2_eef_link_index,:7].clone()
        self._arm1_eef_pose_init[:,:3] -= self._robot.data.root_state_w[:,:3]
        self._arm2_eef_pose_init[:,:3] -= self._robot.data.root_state_w[:,:3]

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
 
      