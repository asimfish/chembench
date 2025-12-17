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
def load_diffusion_policy_from_checkpoint(checkpoint_path: str, device: str = 'cuda:0'):
    """
    从 checkpoint 加载 Diffusion Policy 模型
    """
    import dill
    import hydra
    from omegaconf import OmegaConf
    
    # 注册 eval resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    print(f"Loading Diffusion Policy from: {checkpoint_path}")
    
    # 加载 checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location='cpu')
    
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取策略 (优先使用 EMA 模型)
    if cfg.training.use_ema and hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("  Using EMA model")
    else:
        policy = workspace.model
        print("  Using regular model")
    
    # policy =  workspace.model

    policy.eval()
    policy.to(device)
    
    # 打印模型信息
    # print(f"  Observation dim: {cfg.shape_meta.obs.state.shape}")
    # print(f"  Action dim: {cfg.shape_meta.action.shape}")
    # print(f"  Action horizon: {cfg.n_action_steps}")
    
    return policy

@configclass
class GraspBottleEnvCfg(ILEnvCfg):
    """Configuration for Rl environment."""

    # fake params - These parameters are placeholders for episode length, action and observation spaces
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    episode_length_s = 10
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
 
        render=RenderCfg(
            enable_translucency=True,
        ),

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

        # load policy
        # self.base_policy = load_diffusion_policy(self.cfg.checkpoint,device=self.device)
        self.base_policy = load_diffusion_policy_from_checkpoint(self.cfg.checkpoint,self.device)
        # initialize Timer
        self._timer = Timer()
        # variables used to store contact flag
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs,3),device=self.device)

        # 设置 RTX 渲染选项: Fractional Cutout Opacity
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)

    def step(self,actions):
        
        # get obs for policy
        eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        eef_state = self._robot.data.body_link_state_w[:,eef_link_index,:7].clone()
        eef_state[:,:3] -= self._robot.data.root_state_w[:,:3]
        
        # process image
        # 胸部相机
        chest_camera_rgb = process_batch_image(self._robot.tiled_cameras["chest_camera"].data.output["rgb"][:,:,:,:])
        # 头部相机
        head_camera_rgb = process_batch_image(self._robot.tiled_cameras["head_camera"].data.output["rgb"][:,:,:,:])
        
        # 目标物体位姿（相对于环境原点，与训练数据保持一致）
        # 注意：训练数据 zarr_utils.py 中使用的是相对于 env_origins 的坐标
        target_pose = self._target.data.root_state_w[:,:7].clone()
        target_pose[:,:3] -= self.scene.env_origins
        
        # create obs dict
        # data shape is Batch_size,1,data_shape...
        current_obs = {
            'chest_camera_rgb': chest_camera_rgb.unsqueeze(1),
            'head_camera_rgb': head_camera_rgb.unsqueeze(1),
            'arm2_pos': self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
            'arm2_vel': self._robot.data.joint_vel[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
            'hand2_pos': self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
            'hand2_vel': self._robot.data.joint_vel[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
            'arm2_eef_pos': eef_state[:,:3].unsqueeze(1),
            'arm2_eef_quat': eef_state[:,3:7].unsqueeze(1),
            'target_pose': target_pose.unsqueeze(1),
        }




        # create obs dict
        # data shape is Batch_size,1,data_shape...
        # obs = torch.cat(
        #     (target_pose.unsqueeze(1),
        #     self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices].clone().unsqueeze(1),
        #     self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices][:,:6].clone().unsqueeze(1)),
        #     dim=2
        # )
        # current_obs = {"state":obs}


        # policy model step
        with torch.no_grad():
            base_act_seq = self.base_policy.predict_action(current_obs)['action']    
        # sim step according to decimation
        for i in range(self.cfg.decimation):
            #
            self._action = base_act_seq[:,i,:]
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

        

        

       

