# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import Any
from collections.abc import Sequence

""" Common Modules  """ 
import torch

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg


""" Psi Lab Modules  """
from psilab import OUTPUT_DIR
from psilab.envs.mp_env import MPEnv 
from psilab.envs.mp_env_cfg import MPEnvCfg
from psilab.utils.timer_utils import Timer
from psilab.eval.grasp_rigid import eval_fail,eval_success
from psilab.utils.data_collect_utils import parse_data,save_data

@configclass
class GraspBottleEnvCfg(MPEnvCfg):
    """Configuration for Rl environment."""

    # fake params - These parameters are placeholders for episode length, action and observation spaces
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    episode_length_s = 8
    decimation = 2
    sample_step = 1

    # viewer config
    viewer = ViewerCfg(
        eye=(1.2,0.0,1.2),
        lookat=(-15.0,0.0,0.3)
    )

    # simulation  config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=decimation,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 32,
            max_velocity_iteration_count = 4,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            gpu_found_lost_pairs_capacity = 137401003
        ),
 
        render=RenderCfg(),

    )

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/mp"

    # lift desired height
    lift_height_desired = 0.3

class GraspBottleEnv(MPEnv):

    cfg: GraspBottleEnvCfg

    def __init__(self, cfg: GraspBottleEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # instances in scene
        self._robot = self.scene.robots["robot"]
        # self._target = self.scene.rigid_objects["bottle"]
        self._target = self.scene.articulated_objects["bottle"]


        # initialize contact sensor
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

        # joint limit for compute later
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()

        # joint index: order is arm and hand
        self._arm_joint_index = self._robot.find_joints(self._robot.actuators["arm2"].joint_names,preserve_order=True)[0]
        self._hand_joint_index = self._robot.find_joints(self._robot.actuators["hand2"].joint_names,preserve_order=True)[0][:6]
        self._joint_index = self._arm_joint_index + self._hand_joint_index
        # eef link index
        self._eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]

        # 根据 USD 文件中的 prim 名称查找 body 索引
        # 可以先打印所有 body 名称来确认: print(self._target.body_names)
        self._target_point1_index = self._target.find_bodies("point_1_a")[0][0]
        # self._target_point2_index = self._target.find_bodies("point2")[0][0]
        # self._target_point3_index = self._target.find_bodies("point3")[0][0]
        # self._target_point4_index = self._target.find_bodies("point4")[0][0]
        # self._target_point5_index = self._target.find_bodies("point5")[0][0]
        # self._target_point6_index = self._target.find_bodies("point6")[0][0]
        # self._target_point7_index = self._target.find_bodies("point7")[0][0]
        # self._target_point8_index = self._target.find_bodies("point8")[0][0]

        
        # total step number
        self._episode_step = torch.zeros(self.num_envs,device=self.device,dtype=torch.int)
        # 
        self._eef_pose_target = torch.zeros((self.num_envs,self.max_episode_length,7),device=self.device)
        self._hand_pos_target = torch.zeros((self.num_envs,self.max_episode_length,len(self._hand_joint_index)),device=self.device)
        # initialize Timer
        self._timer = Timer()
        # variables
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs,3),device=self.device)

    def create_trajectory(self,env_ids: torch.Tensor | None):
        
        ##获取世界坐标
        # 方法1：获取 body 的位置（仅位置，不含姿态）
        point1_pos_w = self._target.data.body_pos_w[:, self._target_point1_index, :] - self._robot.data.root_link_pos_w[env_ids,:]

        # 方法2：获取 body 的姿态四元数
        # point1_quat_w = self._target.data.body_quat_w[:, self._target_point1_index, :]  # shape: (num_envs, 4)


        

        env_len = env_ids.shape[0]
        #
        k1 = 0.4
        k2 = 0.1
        k1_step = int(k1 * self.max_episode_length)
        k2_step = int(k2 * self.max_episode_length)
        
        # 计算抓取时刻手臂末端期望位姿
        # 0.4244663	-0.1776909	0.9102596	
        # eff_offset = torch.tensor([-0.075,-0.1776909,0.1],device=self.device).unsqueeze(0).repeat(self.num_envs,1)

        eff_offset = torch.tensor([-0.06,-0.1776909,0.1],device=self.device).unsqueeze(0).repeat(env_len,1)
        eff_quat = torch.tensor([0.50605285,-0.2898345,0.7617619,0.28217942],device=self.device).unsqueeze(0).repeat(env_len,1)
        target_position = self._target.data.root_pos_w[env_ids,:]-self._robot.data.root_link_pos_w[env_ids,:]
        eef_pose_target_1 = torch.cat((eff_offset+target_position,eff_quat),dim=1)
        
        # 计算抓取前手指期望位置 全部保持打开
        hand_pos_target_1 = self._joint_limit_upper[env_ids,:][:,self._hand_joint_index] 
        # hand_pos_target_1[:,0] = self._joint_limit_lower[:,self._hand_joint_index[0]]  # 拇指旋转取最小值
        # 计算抓取时手指关节期望位置 除了
        hand_pos_target_2 = self._joint_limit_lower[env_ids,:][:,self._hand_joint_index]
        hand_pos_target_2[:,0] = self._joint_limit_upper[env_ids,:][:,self._hand_joint_index[0]] # 拇指旋转取最大值

        # 计算抓取后手臂末端期望位姿
        lift_pos = torch.tensor([0,0,self.cfg.lift_height_desired],device=self.device).unsqueeze(0).repeat(env_len,1)
        eef_pose_target_2 = torch.cat((eff_offset+target_position+ lift_pos,eff_quat),dim=1)

        # 拼接轨迹
        self._eef_pose_target[env_ids,:k1_step+k2_step,:] = eef_pose_target_1.unsqueeze(1).repeat(1,k1_step+k2_step,1)
        self._eef_pose_target[env_ids,k1_step+k2_step:,:] = eef_pose_target_2.unsqueeze(1).repeat(1,self.max_episode_length - k1_step - k2_step,1)
        self._hand_pos_target[env_ids,:k1_step,:] = hand_pos_target_1.unsqueeze(1).repeat(1,k1_step,1)
        self._hand_pos_target[env_ids,k1_step:,:] = hand_pos_target_2.unsqueeze(1).repeat(1,self.max_episode_length - k1_step,1)

        # 修改 eef 第一阶段轨迹
        delta_eef_pos = (1 / k1_step) * (eef_pose_target_1[:,:3] - self._robot.data.body_link_pos_w[env_ids,self._eef_link_index,:])
        delta_eef_quat = (1 / k1_step) * (eef_pose_target_1[:,3:7] - self._robot.data.body_link_quat_w[env_ids,self._eef_link_index,:])
        # k = torch.tensor([1.0,1.0,1.0], device=self.device).unsqueeze(0).repeat(self.num_envs,1)
        for i in range(int(k1_step * 0.3)):            
            self._eef_pose_target[env_ids,i,1] = self._robot.data.body_link_pos_w[env_ids,self._eef_link_index,1] + i * delta_eef_pos[:,1]
            self._eef_pose_target[env_ids,i,3:7] = self._robot.data.body_link_quat_w[env_ids,self._eef_link_index,:] + i * delta_eef_quat[:,:]

    def step(self,actions):
        
        # set target
        eef_pose_target = torch.tensor([],device=self.device)
        hand_pos_target = torch.tensor([],device=self.device)
        # 
        for i in range(self.num_envs):
            eef_pose_target = torch.cat((eef_pose_target,self._eef_pose_target[i,self._episode_step[i],:].unsqueeze(0)), dim=0)
            hand_pos_target= torch.cat((hand_pos_target,self._hand_pos_target[i,self._episode_step[i],:].unsqueeze(0)), dim=0)

        # eef_pose_target = torch.index_select(self._eef_pose_target, 1, self._episode_step)
        # hand_pos_target = torch.index_select(self._hand_pos_target, 1, self._episode_step)
        # self._eef_pose_target[:,,:]
        self._robot.set_ik_command({"arm2":eef_pose_target})
        # self._robot.set_joint_position_target(self._action[:7],self._robot.actuators["arm2"].joint_indices) # type: ignore
        self._robot.set_joint_position_target(hand_pos_target,self._robot.actuators["hand2"].joint_indices[:6]) # type: ignore


        # sim step according to decimation
        for i in range(self.cfg.decimation):
            # sim step
            self.sim_step()
        
        # update episode step
        self._episode_step+=1
        
        self._episode_step = torch.clamp(
            self._episode_step,
            None,
            (self.max_episode_length-1)*  torch.ones_like(self._episode_step)
        )
            
        return super().step(actions)
        
    def sim_step(self):

        # 
        self._robot.ik_step()
        #
        super().sim_step()
        
        # parse sim data
        if self.cfg.enable_output and self._sim_step_counter % self.cfg.sample_step == 0:
            parse_data(
                sim_time=self._sim_step_counter * self.cfg.sim.dt,
                data = self._data,
                scene = self.scene
            )

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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # time_out = self._sim_step_counter //  self.cfg.max_step >= (self._episode_num + 1)
        # task evalutation
        bfailed,self._has_contacted = eval_fail(self._target,self._contact_sensors, self._has_contacted) # type: ignore
        # success eval
        bsuccessed = eval_success(self._target, self._contact_sensors,self._target_pos_init, self.cfg.lift_height_desired)
        # bsuccessed= eval_success(self._target, self._contact_sensors,self.cfg.lift_height_desired) # type: ignore
     
        # update success number
        self._episode_success_num+=len(torch.nonzero(bsuccessed==True).squeeze(1).tolist())

        return bsuccessed, bfailed, time_out # type: ignore
    
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


        super()._reset_idx(env_ids)   

        # print logs
        if self.cfg.enable_log:
            self._log_info()
        

        # 
        self.create_trajectory(env_ids)
        # reset variables
        self._episode_step[env_ids] = torch.zeros_like(self._episode_step[env_ids])
        self._target_pos_init[env_ids,:]=self._target.data.root_link_pos_w[env_ids,:].clone()
        self._has_contacted[env_ids] = torch.zeros_like(self._has_contacted[env_ids],device=self.device, dtype=torch.bool) # type: ignore

    def _log_info(self):
        # log policy evalutation result
        if self.cfg.enable_eval and self._episode_num>0:
            #
            plocy_success_rate = float(self._episode_success_num) / float(self._episode_num)
            info = f"Policy成功率: {plocy_success_rate * 100.0} % "
            info +=f"成功次数/总次数: {self._episode_success_num}/{self._episode_num}  "
            if self.cfg.enable_output:
                # compute data collect result
                record_time = self._timer.run_time() /60.0
                record_rate = self._episode_success_num / record_time
                info += f"采集效率: {record_rate:.2f} 条/分钟"
            print(info, end='\r')

                



        

       

