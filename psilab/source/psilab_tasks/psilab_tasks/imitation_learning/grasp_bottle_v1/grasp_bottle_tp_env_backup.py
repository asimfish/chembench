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

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg


""" Psi Lab Modules  """
from psilab.envs.tp_env import TPEnv 
from psilab.envs.tp_env_cfg import TPEnvCfg


from psilab.utils.timer_utils import Timer
from psilab.devices.configs.vuer_cfg import VUER_PSI_DC_01_CFG
rom psilab.devices.configs.psi_glove_cfg import PSIGLOVE_PSI_DC_02_CFG
from psilab import OUTPUT_DIR
from psilab.utils.data_collect_utils import parse_data,save_data
from psilab.eval.grasp_rigid import eval_success,eval_fail


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

    # viewer config
    viewer = ViewerCfg(
        eye=(2.2,0.0,1.2),
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
    output_folder = OUTPUT_DIR + "/tp"

    # lift desired height
    lift_height_desired = 0.3

    #
    # device_type = "vuer"
    # device_cfg = VUER_PSI_DC_01_CFG
        # device
    device_type = "psi-glove"
    device_cfg = PSIGLOVE_PSI_DC_02_CFG


class GraspBottleEnv(TPEnv):

    cfg: GraspBottleEnvCfg

    def __init__(self, cfg: GraspBottleEnvCfg, render_mode: str | None = None, **kwargs):

        # the position of camera(eyes) should be computed before super init according to robot config and vuer device config
        camere_eye_left_pos=(
            cfg.scene.robots_cfg["robot"].init_state.pos[0] + cfg.device_cfg.head_pos[0] + cfg.device_cfg.eye_left_offset[0], # type: ignore
            cfg.scene.robots_cfg["robot"].init_state.pos[1] + cfg.device_cfg.head_pos[1] + cfg.device_cfg.eye_left_offset[1], # type: ignore
            cfg.scene.robots_cfg["robot"].init_state.pos[2] + cfg.device_cfg.head_pos[2] + cfg.device_cfg.eye_left_offset[2], # type: ignore                                
        ),

        camere_eye_right_pos=(
            cfg.scene.robots_cfg["robot"].init_state.pos[0] + cfg.device_cfg.head_pos[0] + cfg.device_cfg.eye_right_offset[0], # type: ignore
            cfg.scene.robots_cfg["robot"].init_state.pos[1] + cfg.device_cfg.head_pos[1] + cfg.device_cfg.eye_right_offset[1], # type: ignore
            cfg.scene.robots_cfg["robot"].init_state.pos[2] + cfg.device_cfg.head_pos[2] + cfg.device_cfg.eye_right_offset[2], # type: ignore                                   
        ),

        cfg.scene.cameras_cfg["eye_left"].offset.pos = camere_eye_left_pos # type: ignore
        cfg.scene.cameras_cfg["eye_right"].offset.pos = camere_eye_right_pos # type: ignore

        #
        super().__init__(cfg, render_mode, **kwargs)

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

        # start vuer threading
        self._device.start()

        # joint limit for compute later
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()
        #
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


    def step(self,actions):
        
        # simulator step
        self.sim_step()
        #
        return super().step(actions)
        
    def sim_step(self):
        
        # store data accrding to record flag and enable ouput flag and sample step 
        if self._device.bRecording and self.cfg.enable_output and self._sim_step_counter % self.cfg.sample_step == 0:
            # parse sim data
            parse_data(
                sim_time=self._sim_step_counter * self.cfg.sim.dt,
                data = self._data,
                scene = self.scene
            )
            # not save eye camera data
            self._data["cameras"]["eye_left.rgb"]=[]
            self._data["cameras"]["eye_right.rgb"]=[]

        # automatically determine whether to start recording
        if self._device.bControl and not self._device.bRecording:
            bPrepared = True
            # 开启录制条件1：右手位置与控制输入差距小于阈值
            state_c = self._device.right_wrist_pose + self._robot.data.body_link_state_w[0][0][:7] # type: ignore
            state = self._robot.data.body_link_state_w[0][self._robot.ik_controllers["arm2"].eef_link_index]
            delta_pos_norm = torch.norm((state[0:3]-state_c[0:3]),p=2)
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
            #
            if bPrepared:
                self._device.bRecording = True
                # wrapped_env.start_record()
                self._voice.say("开始录制")

        # vuer control
        if self._device.bControl:
            # set ik command for arm
            self.scene.robots["robot"].set_ik_command({
                "arm1": self._device.left_wrist_pose, # type: ignore
                "arm2": self._device.right_wrist_pose, # type: ignore
            })
           
            # set joint target for hand
            self.scene.robots["robot"].set_joint_position_target(
                self._device.left_hand_joint_pos, # type: ignore
                self.scene.robots["robot"].actuators["hand1"].joint_indices[:6] # type: ignore
            )
            self.scene.robots["robot"].set_joint_position_target(
                self._device.right_hand_joint_pos, # type: ignore
                self.scene.robots["robot"].actuators["hand2"].joint_indices[:6] # type: ignore
            )

        # reset from vuer
        if self._device.bReset:
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
        
        # set image of vuer from sim
        image_left = (self.scene.cameras["eye_left"].data.output["rgb"])[0]
        image_right = (self.scene.cameras["eye_right"].data.output["rgb"])[0]
        self._device.set_camera_image(image_left,image_right) # type: ignore

        # 根据Vuer输出的头部姿态更新左右眼相机位姿态
        # Tips: Vuer输出相对头部位姿，需要结合robot位置计算世界坐标系位置
        robot_pos = self.scene.robots["robot"].data.root_link_pos_w[0,:3]
        self.scene.cameras["eye_left"].set_world_poses(self._device.left_eye_pose[0:3].add(robot_pos).unsqueeze(0),self._device.left_eye_pose[3:].unsqueeze(0),convention="world") # type: ignore
        self.scene.cameras["eye_right"].set_world_poses(self._device.right_eye_pose[0:3].add(robot_pos).unsqueeze(0),self._device.right_eye_pose[3:].unsqueeze(0),convention="world") # type: ignore

        # 
        self._robot.ik_step()
        #
        super().sim_step()
        #
        self._sim_step_counter+=1

        # get dones only when recording
        if self._device.bRecording:
            # get dones
            success, fail, time_out = self._get_dones()
            reset = success | fail | time_out 
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
            
        # reset variables
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init[:,:]=self._target.data.root_link_pos_w[:,:].clone()
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # time_out = self._sim_step_counter //  self.cfg.max_step >= (self._episode_num + 1)
        # task evalutation
        # failed eval
        bfailed,self._has_contacted = eval_fail(self._target,self._contact_sensors, self._has_contacted) # type: ignore
        # success eval
        bsuccessed= eval_success(self._target, self._contact_sensors,self._target_pos_init, self.cfg.lift_height_desired) # type: ignore
           
        # update success number
        self._episode_success_num+=len(torch.nonzero(bsuccessed==True).squeeze(1).tolist())

        return bsuccessed, bfailed, time_out
    
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
        # log policy evalutation result
        if self.cfg.enable_eval and self._episode_num>0:
            #
            if self.cfg.enable_output:
                # compute data collect result
                record_time = self._timer.run_time() /60.0
                record_rate = self._episode_success_num / record_time
                info = f"采集时长: {record_time} 分钟  "
                info +=f"采集数据: {self._episode_success_num} 条  "
                info += f"采集效率: {record_rate} 条/分钟" 
                print(info)
                # print(info, end='\r')
