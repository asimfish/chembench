# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import Any
from dataclasses import MISSING
from collections.abc import Sequence

""" Common Modules  """ 
import torch

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg


""" Psi Lab Modules  """
from psilab import OUTPUT_DIR
from psilab.scene import SceneCfg
from psilab.envs.il_env import ILEnv 
from psilab.envs.il_env_cfg import ILEnvCfg
from psilab.utils.timer_utils import Timer
from psilab.eval.grasp_rigid import eval_success,eval_fail
from psilab.eval.grasp_rigid import eval_success_only_height
from psilab.utils.data_collect_utils import parse_data,save_data
from psilab.utils.dp_utils import load_diffusion_policy,process_image,process_batch_image

# pi0.5 远程推理客户端
from openpi_client import websocket_client_policy
from openpi_client import image_tools

def create_pi0_client(host: str = "localhost", port: int = 8000):
    """
    创建 Pi0/Pi0.5 远程推理客户端
    
    Args:
        host: 策略服务器地址
        port: 策略服务器端口
    
    Returns:
        client: WebsocketClientPolicy 客户端对象
    """
    print(f"Connecting to Pi0.5 Policy Server at {host}:{port}...")
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"  Connected successfully!")
    print(f"  Server metadata: {client.get_server_metadata()}")
    return client

import numpy as np

@configclass
class GraspBottleEnvCfg(ILEnvCfg):
    """Configuration for Rl environment."""

    # fake params - These parameters are placeholders for episode length, action and observation spaces
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    episode_length_s = 20
    decimation = 4
    sample_step = 1
# 81.68.132.224:18015
    # pi0.5 远程推理配置
    pi0_server_host: str = "81.68.132.224"  # pi0.5 策略服务器地址
    pi0_server_port: int = 18015  # pi0.5 策略服务器端口
    default_prompt: str = "Pick up the green carton of drink from the table."  # 默认任务提示语
    replan_steps: int = 4  # 每隔多少步重新规划动作（类似 main.py 中的 replan_steps）
    image_resize: int = 224  # 图像缩放尺寸（pi0.5 默认 224）

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

    # scene config
    scene :SceneCfg = MISSING # type: ignore

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/il"

    # lift desired height
    lift_height_desired = 0.3

class GraspBottleEnv(ILEnv):

    cfg: GraspBottleEnvCfg

    def __init__(self, cfg: GraspBottleEnvCfg, render_mode: str | None = None, **kwargs):
        #
        cfg.scene.robots_cfg["robot"].diff_ik_controllers = None # type: ignore

        super().__init__(cfg, render_mode, **kwargs)

        # instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["bottle"]

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

        # 连接 pi0.5 远程策略服务器
        self.base_policy = create_pi0_client(
            host=self.cfg.pi0_server_host,
            port=self.cfg.pi0_server_port,
        )
        
        # initialize Timer
        self._timer = Timer()
        # variables used to store contact flag
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs,3),device=self.device)
        
        # pi0.5 action plan queue for each environment (类似 main.py 中的 action_plan)
        from collections import deque
        self._action_plans = [deque() for _ in range(self.num_envs)]
        # 每个环境的 step 计数器（用于 pi0 的 memory 机制）
        self._pi0_step_counters = [0 for _ in range(self.num_envs)]

    def step(self,actions):
        
        # get obs for policy
        eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        eef_state = self._robot.data.body_link_state_w[:,eef_link_index,:7].clone()
        eef_state[:,:3] -= self._robot.data.root_state_w[:,:3]
        
        # 获取原始图像数据 (N, H, W, 4) 格式，需要转换为 numpy uint8 (H, W, 3) 格式
        # 胸部相机
        chest_camera_rgb_batch = self._robot.tiled_cameras["chest_camera"].data.output["rgb"][:,:,:,:3]  # 去掉 alpha 通道
        # 头部相机
        head_camera_rgb_batch = self._robot.tiled_cameras["head_camera"].data.output["rgb"][:,:,:,:3]
        # 第三人称相机
        third_person_camera_rgb_batch = self._robot.tiled_cameras["third_person_camera"].data.output["rgb"][:,:,:,:3]

        # 合并所有状态信息
        arm2_pos = self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices]      # (N, 7)
        arm2_vel = self._robot.data.joint_vel[:,self._robot.actuators["arm2"].joint_indices]      # (N, 7)
        hand2_pos = self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]] # (N, 6)
        hand2_vel = self._robot.data.joint_vel[:,self._robot.actuators["hand2"].joint_indices[:6]] # (N, 6)
        arm2_eef_pos = eef_state[:,:3]   # (N, 3)
        arm2_eef_quat = eef_state[:,3:7] # (N, 4)
        
        state_batch = torch.cat([
            arm2_pos,        # (N, 7)
            arm2_vel,        # (N, 7)
            hand2_pos,       # (N, 6)
            hand2_vel,       # (N, 6)
            arm2_eef_pos,    # (N, 3)
            arm2_eef_quat,   # (N, 4)
        ], dim=1)  # 结果: (N, 33)

        # 为每个环境准备动作
        batch_actions = []
        
        for env_idx in range(self.num_envs):
            # 如果 action_plan 为空，需要重新推理获取新的 action chunk
            if not self._action_plans[env_idx]:
                # 准备单个环境的观察数据（pi0 的输入格式，参考 main.py）
                # 转换图像为 numpy uint8 (H, W, C) 格式，并调整大小以减少网络传输
                resize_size = self.cfg.image_resize
                head_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        head_camera_rgb_batch[env_idx].cpu().numpy(), 
                        resize_size, resize_size
                    )
                )
                chest_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        chest_camera_rgb_batch[env_idx].cpu().numpy(), 
                        resize_size, resize_size
                    )
                )
                third_person_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        third_person_camera_rgb_batch[env_idx].cpu().numpy(), 
                        resize_size, resize_size
                    )
                )
                state = state_batch[env_idx].cpu().numpy()
                
                # 构建 pi0 输入格式（参考 gbimg_policy.py 中的 key 映射）
                element = {
                    "observation/head_camera": head_img,           # 头部相机
                    "observation/chest_camera": chest_img,         # 胸部相机
                    "observation/third_person_camera": third_person_img,  # 第三人称相机
                    "observation/state": state,                    # 状态向量 (33,)
                    "prompt": self.cfg.default_prompt,             # 任务提示语
                }
                
                # 调用 pi0.5 远程策略推理
                result = self.base_policy.infer(element, step=self._pi0_step_counters[env_idx])
                
                # 获取动作块 (action_horizon, action_dim)
                action_chunk = result["actions"]  # numpy array
                
                # 将动作添加到队列中（只取 replan_steps 个动作）
                for act in action_chunk[:self.cfg.replan_steps]:
                    self._action_plans[env_idx].append(act)
            
            # 从队列中取出一个动作
            action = self._action_plans[env_idx].popleft()
            batch_actions.append(action)
            
            # 更新步数计数器
            self._pi0_step_counters[env_idx] += 1
        
        # 将动作转换为 tensor
        batch_actions = torch.tensor(np.stack(batch_actions), dtype=torch.float32, device=self.device)
        
        # sim step according to decimation
        for i in range(self.cfg.decimation):
            self._action = batch_actions
            # sim step
            self.sim_step()
            
        return super().step(actions)
        
    def sim_step(self):

        # set target
        self._robot.set_joint_position_target(self._action[:,:7],self._robot.actuators["arm2"].joint_indices) # type: ignore
        self._robot.set_joint_position_target(self._action[:,7:],self._robot.actuators["hand2"].joint_indices[:6]) # type: ignore

        # write data to sim
        self._robot.write_data_to_sim()
        # sim step
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
        reset_ids = torch.nonzero(reset==True).squeeze()
        # bug: if single index, squeeze will change tensor to torch.Size([])
        reset_ids = reset_ids.unsqueeze(0) if reset_ids.size()==torch.Size([]) else reset_ids

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
        # bsuccessed= eval_success(self._target, self._contact_sensors,self.cfg.lift_height_desired) # type: ignore
        bsuccessed = eval_success_only_height(self._target, self._target_pos_init, self.cfg.lift_height_desired)
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

        
        # if self.cfg.enable_log:
        self._log_info()
        
        # variables used to store contact flag
        self._has_contacted[env_ids] = torch.zeros_like(self._has_contacted[env_ids],device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init[env_ids,:]=self._target.data.root_link_pos_w[env_ids,:].clone()
        
        # 重置 pi0.5 动作队列和步数计数器
        for env_id in env_ids.tolist():
            self._action_plans[env_id].clear()
            self._pi0_step_counters[env_id] = 0
    def _log_info(self):
        # log policy evalutation result
        if self.cfg.enable_eval and self._episode_num > 0:
            # 设置日志打印间隔
            # 单环境或少量环境：每个 episode 都打印
            # 多环境：每 num_envs 个 episode 打印一次（约等于每轮并行完成后打印）
            if self.num_envs <= 4:
                log_interval = 1  # 单环境或少量环境，每次都打印
            else:
                log_interval = self.num_envs  # 多环境，每轮打印一次
            
            # 只在达到打印间隔时输出日志
            if self._episode_num % log_interval == 0 or self._episode_num >= self.cfg.max_episode:
                policy_success_rate = float(self._episode_success_num) / float(self._episode_num)
                print(f"\n{'='*50}")
                print(f"[Episode {self._episode_num}/{self.cfg.max_episode}] 评估统计")
                print(f"  Policy成功率: {policy_success_rate * 100.0:.2f}%")
                print(f"  成功次数/总次数: {self._episode_success_num}/{self._episode_num}")
                
                if self.cfg.enable_output:
                    # compute data collect result
                    record_time = self._timer.run_time() / 60.0
                    record_rate = self._episode_success_num / record_time if record_time > 0 else 0
                    print(f"  采集效率: {record_rate:.2f} 条/分钟")
                print(f"{'='*50}\n")

        

        

       

