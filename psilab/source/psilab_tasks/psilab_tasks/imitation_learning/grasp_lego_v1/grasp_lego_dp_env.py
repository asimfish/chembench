# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Auto-generated for Diffusion Policy Testing
# Date: 2025-12-02

"""
使用训练好的 Diffusion Policy 测试 Grasp Lego 任务
"""

from __future__ import annotations
from typing import Any
from dataclasses import MISSING
from collections.abc import Sequence
import sys
import os

import torch

# 添加 diffusion_policy 到 path
sys.path.insert(0, "/home/psibot/diffusion_policy")

from isaaclab.sim import SimulationCfg, PhysxCfg, RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

from psilab import OUTPUT_DIR
from psilab.scene import SceneCfg
from psilab.envs.il_env import ILEnv 
from psilab.envs.il_env_cfg import ILEnvCfg
from psilab.utils.timer_utils import Timer
from psilab.eval.grasp_rigid import eval_success_only_height


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
    
    policy.eval()
    policy.to(device)
    
    # 打印模型信息
    print(f"  Observation dim: {cfg.shape_meta.obs.state.shape}")
    print(f"  Action dim: {cfg.shape_meta.action.shape}")
    print(f"  Action horizon: {cfg.n_action_steps}")
    
    return policy


@configclass
class GraspLegoDPEnvCfg(ILEnvCfg):
    """Configuration for Diffusion Policy testing environment."""

    # Episode parameters
    action_scale = 0.5
    action_space = 18  # arm2(7) + hand2(11)
    observation_space = 25  # arm2_pos(7) + hand2_pos(11) + arm2_eef_pose(7)
    state_space = 25

    episode_length_s = 15  # 15 秒超时
    decimation = 4  # 每 4 步执行一次策略
    sample_step = 1

    # Viewer config
    viewer = ViewerCfg(
        eye=(2.2, 0.0, 1.2),
        lookat=(-15.0, 0.0, 0.3)
    )

    # Simulation config
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=1,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=32,
            max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.002,
            enable_ccd=True,
            gpu_found_lost_pairs_capacity=137401003
        ),
        render=RenderCfg(),
    )

    # Scene config
    scene: SceneCfg = MISSING  # type: ignore

    # Output folder
    output_folder = OUTPUT_DIR + "/dp_eval"

    # Task parameters
    lift_height_desired = 0.3

    # Checkpoint path - 默认值
    checkpoint: str = "/home/psibot/diffusion_policy/data/outputs/2025.12.02/14.22.05_train_diffusion_transformer_isaaclab_grasp_lego_isaaclab/checkpoints/latest.ckpt"


class GraspLegoDPEnv(ILEnv):
    """
    使用 Diffusion Policy 控制的抓取环境
    
    观测空间:
        - arm2_pos: (7,) 机械臂关节位置
        - hand2_pos: (11,) 灵巧手关节位置  
        - arm2_eef_pose: (7,) 末端执行器位姿
        
    动作空间:
        - arm2_target: (7,) 机械臂目标关节位置
        - hand2_target: (11,) 灵巧手目标关节位置
    """

    cfg: GraspLegoDPEnvCfg

    def __init__(self, cfg: GraspLegoDPEnvCfg, render_mode: str | None = None, **kwargs):
        # 禁用 IK 控制器 (使用关节位置控制)
        if cfg.scene.robots_cfg["robot"].diff_ik_controllers is not None:
            cfg.scene.robots_cfg["robot"].diff_ik_controllers = None  # type: ignore

        super().__init__(cfg, render_mode, **kwargs)

        # Scene instances
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["target"]

        # Joint indices
        self._arm2_joint_index = self._robot.actuators["arm2"].joint_indices  # type: ignore
        self._hand2_joint_index = self._robot.actuators["hand2"].joint_indices  # type: ignore
        self._arm2_eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]

        # Joint limits
        self._joint_limit_lower = self._robot.data.joint_limits[:, :, 0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:, :, 1].clone()

        # Load Diffusion Policy
        self.policy = load_diffusion_policy_from_checkpoint(
            self.cfg.checkpoint, 
            str(self.device)
        )

        # Timer for logging
        self._timer = Timer()
        
        # Action buffer (for multi-step execution)
        self._action_buffer = None
        self._action_buffer_idx = 0
        
        # 记录目标物体初始位置 (用于评估成功)
        self._target_pos_init = torch.zeros((self.num_envs, 3), device=self.device)

    def _get_obs(self) -> dict:
        """
        构建观测字典，匹配训练时的数据格式
        """
        # arm2 关节位置 (7,)
        arm2_pos = self._robot.data.joint_pos[:, self._arm2_joint_index].clone()
        
        # hand2 关节位置 (11,)
        hand2_pos = self._robot.data.joint_pos[:, self._hand2_joint_index].clone()
        
        # arm2 末端执行器位姿 (7,) [x, y, z, qx, qy, qz, qw]
        eef_state = self._robot.data.body_link_state_w[:, self._arm2_eef_link_index, :7].clone()
        # 转换为相对于 robot base 的位置
        eef_state[:, :3] -= self._robot.data.root_state_w[:, :3]
        
        # 拼接观测 (batch, 25)
        state = torch.cat([arm2_pos, hand2_pos, eef_state], dim=-1)
        
        # 添加时间维度以匹配模型输入格式 (batch, 1, 25)
        state = state.unsqueeze(1)
        
        return {'state': state}

    def step(self, actions):
        """执行一步环境交互"""
        
        # 获取观测
        obs = self._get_obs()
        
        # 如果 action buffer 为空或用完，重新预测
        if self._action_buffer is None or self._action_buffer_idx >= self._action_buffer.shape[1]:
            with torch.no_grad():
                result = self.policy.predict_action(obs)
                self._action_buffer = result['action']  # (batch, horizon, action_dim)
                self._action_buffer_idx = 0
        
        # 执行 decimation 步
        for i in range(self.cfg.decimation):
            # 如果 action buffer 被清空 (环境被重置)，跳出循环
            if self._action_buffer is None:
                break
                
            # 获取当前动作
            if self._action_buffer_idx < self._action_buffer.shape[1]:
                action = self._action_buffer[:, self._action_buffer_idx, :]
                self._action_buffer_idx += 1
            else:
                # 如果动作用完，使用最后一个动作
                action = self._action_buffer[:, -1, :]
            
            # 分解动作
            arm2_target = action[:, :7]
            hand2_target = action[:, 7:18]
            
            # 设置关节目标
            self._robot.set_joint_position_target(arm2_target, self._arm2_joint_index)
            self._robot.set_joint_position_target(hand2_target, self._hand2_joint_index)
            
            # 执行仿真步
            self.sim_step()
        
        return super().step(actions)

    def sim_step(self):
        """执行仿真步"""
        # 写入数据到仿真
        self._robot.write_data_to_sim()
        
        # 仿真步进
        super().sim_step()
        
        # 检查任务完成
        success, time_out = self._get_dones()
        reset = success | time_out
        reset_ids = torch.nonzero(reset == True).squeeze()
        
        if reset_ids.numel() > 0:
            reset_ids = reset_ids.unsqueeze(0) if reset_ids.dim() == 0 else reset_ids
            success_ids = torch.nonzero(success == True).squeeze().tolist()
            success_ids = [success_ids] if isinstance(success_ids, int) else success_ids
            
            self._reset_idx(reset_ids, success_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """重置环境"""
        super().reset()
        # 清空动作缓冲区
        self._action_buffer = None
        self._action_buffer_idx = 0
        # 记录目标物体初始位置
        self._target_pos_init[:] = self._target.data.root_link_pos_w[:, :].clone()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """检查任务完成条件"""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        success = eval_success_only_height(self._target, self._target_pos_init, self.cfg.lift_height_desired)
        self._episode_success_num += len(torch.nonzero(success == True).squeeze().tolist())
        return success, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None, success_ids: Sequence[int] | None = None):
        """重置指定环境"""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        
        super()._reset_idx(env_ids)
        
        # 清空动作缓冲区
        self._action_buffer = None
        self._action_buffer_idx = 0
        
        # 更新重置环境的目标初始位置
        self._target_pos_init[env_ids, :] = self._target.data.root_link_pos_w[env_ids, :].clone()
        
        if self.cfg.enable_log:
            self._log_info()

    def _log_info(self):
        """打印评估信息"""
        if self.cfg.enable_eval and self._episode_num > 0:
            success_rate = float(self._episode_success_num) / float(self._episode_num)
            print(f"Success rate: {success_rate * 100.0:.1f}%")
            print(f"Success/Total: {self._episode_success_num}/{self._episode_num}")
