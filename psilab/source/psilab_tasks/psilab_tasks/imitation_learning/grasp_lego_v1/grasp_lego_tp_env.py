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
import carb
import omni

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
class GraspLegoEnvCfg(TPEnvCfg):
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
    output_folder = OUTPUT_DIR + "/grasp_lego"

    # lift desired height
    lift_height_desired = 0.3


class GraspLegoEnv(TPEnv):

    cfg: GraspLegoEnvCfg

    def __init__(self, cfg: GraspLegoEnvCfg, render_mode: str | None = None, **kwargs):

        #
        super().__init__(cfg, render_mode, **kwargs)

        # instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["target"]

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

        # initialize Timer
        self._timer = Timer()
        # variables used to store contact flag
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs,3),device=self.device)
        
        # keyboard input flag for manual recording trigger
        self._enter_pressed = False
        self._register_keyboard_handler()

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

            # arm1
            self._arm1_eef_pose_desired[0,0:3] = self._arm1_eef_pose_init[0,0:3] + self._device._delta_tracker_left_pos # type: ignore
            self._arm1_eef_pose_desired[0,3:] = self._device._wrist_left_quat # type: ignore
            # arm2
            self._arm2_eef_pose_desired[0,0:3] = self._arm2_eef_pose_init[0,0:3] + self._device._delta_tracker_right_pos # type: ignore
            self._arm2_eef_pose_desired[0,3:] = self._device._wrist_right_quat # type: ignore
            
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
 
      