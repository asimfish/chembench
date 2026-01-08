"""
Grasp IL Evaluation for ACT Policy - 使用ACT策略评估抓取任务

本文件是 grasp_il.py 的 ACT 版本，用于评估训练好的 ACT 模型在抓取任务上的性能。

使用的 ACT 实现：my_act (基于官方 ACT 实现)

ACT (Action Chunking with Transformers) 特点：
- 使用 ResNet 或其他 CNN backbone 作为视觉编码器
- 基于 Transformer 的动作预测
- 一次预测整个动作序列（action chunking）
- 使用 CVAE 进行动作建模

训练超参数（与训练时保持一致）：
- KL_WEIGHT=10
- CHUNK_SIZE=100 (num_queries)
- HIDDEN_DIM=512
- DIM_FEEDFORWARD=3200
- BATCH_SIZE=8
- NUM_EPOCHS=2000
- LR=1e-5
- SEED=0

支持的观测模式（obs_mode）：
- rgb: 3通道 RGB 图像（ACT 标准输入）
- rgbm: 4通道 RGB + Mask（需要修改 input_channels=4）
- state: 纯状态（无图像）

关键差异（与 Diffusion Policy 版本）：
1. ACT 使用 ACTPolicy 类（my_act/policy.py）
2. 推理接口：policy(qpos, image) 返回 a_hat [B, num_queries, action_dim]
3. 支持 3 通道和 6 通道图像（自动归一化）
4. ACT 一次预测完整序列，需要维护动作队列
"""

import copy
from typing import Literal, Any, Sequence
from dataclasses import MISSING

import torch
import numpy as np
import os

# IsaacLab imports
from isaaclab.sim import SimulationCfg, PhysxCfg, RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg
from psilab import OUTPUT_DIR
from psilab.scene import SceneCfg
from psilab.envs.il_env import ILEnv
from psilab.envs.il_env_cfg import ILEnvCfg
from psilab.utils.timer_utils import Timer
from psilab.eval.grasp_rigid import eval_fail
from psilab.utils.data_collect_utils import parse_data, save_data

# 图像处理工具
from psilab_tasks.imitation_learning.grasp.image_utils import process_batch_image_multimodal


def load_act_policy_and_stats(checkpoint_path: str, camera_names: list[str], device: str = 'cuda:0'):
    """
    从 checkpoint 加载 ACT Policy 模型和数据统计信息 (使用 my_act 实现)
    
    Args:
        checkpoint_path: checkpoint 文件路径（.ckpt 文件）
        camera_names: 相机名称列表（例如 ['chest_camera', 'head_camera', 'third_camera']）
        device: 设备
    
    Returns:
        policy: ACT 策略模型
        stats: 数据统计信息（包含 qpos_mean, qpos_std, action_mean, action_std）
    """
    print(f"\n{'='*50}")
    print(f"[ACT] 加载 ACT Policy (my_act 实现)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Camera names: {camera_names}")
    print(f"{'='*50}\n")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    
    # 导入 my_act 的 ACT 实现
    import sys
    my_act_path = '/home/psibot/chembench/my_act'
    if my_act_path not in sys.path:
        sys.path.insert(0, my_act_path)
    
    try:
        from policy import ACTPolicy
        print("[ACT] ✓ 成功导入 my_act ACTPolicy")
    except ImportError as e:
        raise ImportError(
            f"无法导入 my_act ACTPolicy: {e}\n"
            f"请确保 my_act 在路径: {my_act_path}\n"
            f"当前 sys.path: {sys.path[:3]}"
        )
    
    # ACT 策略配置（与训练时保持一致）
    # 使用用户提供的超参数
    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    state_dim = 13  # arm2(7) + hand2(6)
    
    policy_config = {
        'lr': 1e-5,
        'num_queries': 15,
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'camera_names': camera_names,
        'input_channels': 6,  # 6=RGB+RGB_masked (你的模型是6通道训练的)
        'state_dim': state_dim,
        'action_dim': state_dim,
    }
    
    print(f"[ACT] 策略配置:")
    print(f"    num_queries (chunk_size): {policy_config['num_queries']}")
    print(f"    hidden_dim: {policy_config['hidden_dim']}")
    print(f"    dim_feedforward: {policy_config['dim_feedforward']}")
    print(f"    kl_weight: {policy_config['kl_weight']}")
    print(f"    backbone: {policy_config['backbone']}")
    print(f"    camera_names: {camera_names}")
    print(f"    input_channels: {policy_config['input_channels']}")
    print(f"    state_dim/action_dim: {policy_config['state_dim']}")
    
    # 创建策略实例
    print(f"\n[ACT] 创建模型...")
    policy = ACTPolicy(policy_config)
    
    # 加载 checkpoint
    print(f"[ACT] 加载权重...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    policy.load_state_dict(state_dict)
    print(f"[ACT] ✓ 权重加载成功")
    
    # 移动到设备并设置为评估模式
    policy.eval()
    policy.to(device)
    print(f"[ACT] ✓ 模型已移至 {device} 并设为评估模式\n")
    
    # 加载数据统计信息（用于归一化）
    import pickle
    ckpt_dir = os.path.dirname(checkpoint_path)
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        print(f"[ACT] ✓ 加载数据统计信息: {stats_path}")
        print(f"       qpos_mean: {stats.get('qpos_mean', 'N/A')[:3] if 'qpos_mean' in stats else 'N/A'}...")
        print(f"       action_mean: {stats.get('action_mean', 'N/A')[:3] if 'action_mean' in stats else 'N/A'}...")
    else:
        print(f"[ACT] ⚠️  未找到 dataset_stats.pkl，将不进行数据归一化")
        print(f"       预期路径: {stats_path}")
        stats = None
    
    return policy, stats


@configclass
class GraspBottleEnvCfg(ILEnvCfg):
    """Configuration for ACT IL environment."""

    # fake params - These parameters are placeholders for episode length, action and observation spaces
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    episode_length_s = 8
    decimation = 4
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
    lift_height_desired = 0.2
    
    # 成功判断：朝向偏差阈值（sin²(θ/2)，0=完全一致，1=上下颠倒）
    orientation_threshold: float = 0.05
    
    # 观测模式配置（ACT 支持多种模式）
    # rgb: 3通道 RGB
    # rgbm: 4通道 RGB + Mask
    # rgb_masked_rgb: 6通道 RGB + RGB_masked (你的模型使用的)
    obs_mode: Literal["rgb", "rgbm", "rgb_masked_rgb", "state"] = "rgb_masked_rgb"
    
    # ACT 相机配置
    camera_names: list[str] = ["chest_camera", "head_camera", "third_camera"]
    
    # ACT 动作队列配置
    # ACT 一次预测 num_queries 步动作，但每次环境只执行 decimation 步
    num_queries: int = 8  # ACT 预测的动作序列长度（应与训练时一致）
    temporal_agg: bool = True  # 是否使用时间聚合（temporal aggregation）
    
    # Mask 解耦实验配置（如果使用 RGBM）
    mask_mode: str = "real"


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

        # load ACT policy and stats
        self.base_policy, self.stats = load_act_policy_and_stats(
            self.cfg.checkpoint,
            camera_names=self.cfg.camera_names,
            device=self.device
        )
        
        # 设置归一化/反归一化函数
        if self.stats is not None:
            # 转换为 tensor
            qpos_mean = torch.tensor(self.stats['qpos_mean'], device=self.device, dtype=torch.float32)
            qpos_std = torch.tensor(self.stats['qpos_std'], device=self.device, dtype=torch.float32)
            action_mean = torch.tensor(self.stats['action_mean'], device=self.device, dtype=torch.float32)
            action_std = torch.tensor(self.stats['action_std'], device=self.device, dtype=torch.float32)
            
            self.normalize_qpos = lambda x: (x - qpos_mean) / qpos_std
            self.denormalize_action = lambda x: x * action_std + action_mean
            print(f"[GraspBottleEnv] ✓ 数据归一化已启用")
        else:
            self.normalize_qpos = lambda x: x
            self.denormalize_action = lambda x: x
            print(f"[GraspBottleEnv] ⚠️  数据归一化未启用（无 stats）")
        
        # ACT 配置参数
        self.num_queries = self.cfg.num_queries
        self.temporal_agg = self.cfg.temporal_agg
        self.max_timesteps = int(self.max_episode_length)
        
        # 初始化 Action 队列
        self._action_queue = None
        self._action_queue_ptr = None
        
        # 确保 temporal_agg 是布尔值（防止从配置文件读取时是字符串）
        if isinstance(self.temporal_agg, str):
            self.temporal_agg = (self.temporal_agg.lower() in ['true', '1', 'yes'])
            print(f"[GraspBottleEnv] ⚠️  temporal_agg 从字符串转换为布尔值: {self.temporal_agg}")
        
        # 查询频率：如果启用 temporal_agg，每步都查询；否则按 decimation 查询
        if self.temporal_agg:
            # self.query_frequency = 1  # 每步都查询（参考 ACT 原始实现）
            self.query_frequency = self.cfg.decimation  # 每步都查询（参考 ACT 原始实现）
        else:
            self.query_frequency = self.cfg.decimation
        
        # 初始化 temporal aggregation 相关变量（参考 ACT 原始实现）
        if self.temporal_agg:
            # all_time_actions[t, t:t+num_queries] 存储在时间 t 预测的未来 num_queries 步
            self._all_time_actions = torch.zeros(
                (self.num_envs, self.max_timesteps, self.max_timesteps + self.num_queries, 13),
                device=self.device
            )
            self._timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            print(f"[GraspBottleEnv] ✓ Temporal aggregation 已启用 (k=0.01)")
        else:
            self._all_time_actions = None
            self._timestep = None
            print(f"[GraspBottleEnv] ℹ  Temporal aggregation 未启用，使用简单模式")
        
        print(f"[GraspBottleEnv] ACT Policy 配置:")
        print(f"  num_queries: {self.num_queries}")
        print(f"  temporal_agg: {self.temporal_agg}")
        print(f"  query_frequency: {self.query_frequency}")
        print(f"  decimation: {self.cfg.decimation}")
        
        # initialize Timer
        self._timer = Timer()
        # variables used to store contact flag
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs,3),device=self.device)
        self._target_quat_init = torch.zeros((self.num_envs,4),device=self.device)
        
        # 记录成功时的步数列表
        self._success_steps_list: list[int] = []
        self.sim.set_camera_view([-0.7, -5.2, 1.3], [-1.2, -5.2, 1.1])
        # 设置 RTX 渲染选项
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)

    def step(self, actions):
        """
        ACT Policy 推理步骤
        
        ACT 的推理流程：
        1. 获取当前观测（图像 + 状态）
        2. 调用 policy 预测动作序列（一次预测 num_queries 步）
        3. 从队列中取出 decimation 步执行
        4. 可选：使用 temporal aggregation 平滑动作
        """
        
        # ==================== 获取观测数据 ====================
        # 1. 末端执行器状态
        eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        eef_state = self._robot.data.body_link_state_w[:,eef_link_index,:7].clone()
        eef_state[:,:3] -= self._robot.data.root_state_w[:,:3]
        
        # 2. 获取图像数据
        chest_rgb = self._robot.tiled_cameras["chest_camera"].data.output["rgb"][:, :, :, :]
        head_rgb = self._robot.tiled_cameras["head_camera"].data.output["rgb"][:, :, :, :]
        third_rgb = self._robot.tiled_cameras["third_camera"].data.output["rgb"][:, :, :, :]
        
        # 3. 根据 obs_mode 处理图像
        chest_mask, head_mask, third_mask = None, None, None
        
        # 获取 mask（用于 rgbm 和 rgb_masked_rgb 模式）
        if self.cfg.obs_mode in ["rgbm", "rgb_masked_rgb"]:
            if self.cfg.mask_mode == "real":
                if "instance_segmentation_fast" in self._robot.tiled_cameras["chest_camera"].data.output:
                    chest_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
                if "instance_segmentation_fast" in self._robot.tiled_cameras["head_camera"].data.output:
                    head_mask = self._robot.tiled_cameras["head_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
                if "instance_segmentation_fast" in self._robot.tiled_cameras["third_camera"].data.output:
                    third_mask = self._robot.tiled_cameras["third_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
            elif self.cfg.mask_mode == "all_0":
                chest_mask = torch.zeros_like(chest_rgb[:, :, :, 0])
                head_mask = torch.zeros_like(head_rgb[:, :, :, 0])
                third_mask = torch.zeros_like(third_rgb[:, :, :, 0])
            elif self.cfg.mask_mode == "all_1":
                chest_mask = torch.ones_like(chest_rgb[:, :, :, 0])
                head_mask = torch.ones_like(head_rgb[:, :, :, 0])
                third_mask = torch.ones_like(third_rgb[:, :, :, 0])
        
        # 处理图像
        chest_camera_img = None
        head_camera_img = None
        third_camera_img = None
        
        if self.cfg.obs_mode != 'state':
            chest_camera_img = process_batch_image_multimodal(
                rgb=chest_rgb, 
                mask=chest_mask, 
                depth=None, 
                normal=None, 
                obs_mode=self.cfg.obs_mode
            )
            
            head_camera_img = process_batch_image_multimodal(
                rgb=head_rgb, 
                mask=head_mask, 
                depth=None, 
                normal=None, 
                obs_mode=self.cfg.obs_mode
            )
            
            third_camera_img = process_batch_image_multimodal(
                rgb=third_rgb, 
                mask=third_mask, 
                depth=None, 
                normal=None, 
                obs_mode=self.cfg.obs_mode
            )
        
        # 4. 获取关节状态（qpos）
        arm2_pos = self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices]  # [B, 7]
        hand2_pos = self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]]  # [B, 6]
        qpos = torch.cat([arm2_pos, hand2_pos], dim=-1)  # [B, 13]
        
        # 5. 目标物体位姿（如果需要）
        target_pose = self._target.data.root_state_w[:,:7].clone()
        target_pose[:,:3] -= self.scene.env_origins
        
        # ==================== ACT Policy 推理 ====================
        B = qpos.shape[0]
        
        # 归一化 qpos
        qpos_normalized = self.normalize_qpos(qpos)
        
        # 准备图像输入：将多个相机图像堆叠为 [B, num_cameras, C, H, W]
        images = []
        for cam_name in self.cfg.camera_names:
            if cam_name == "chest_camera":
                images.append(chest_camera_img)
            elif cam_name == "head_camera":
                images.append(head_camera_img)
            elif cam_name == "third_camera":
                images.append(third_camera_img)
            else:
                raise ValueError(f"Unknown camera: {cam_name}")
        
        # Stack: [B, num_cameras, C, H, W]
        image_data = torch.stack(images, dim=1)
        
        # ==================== Temporal Aggregation 模式 ====================
        if self.temporal_agg:
            # 参考 ACT 原始实现的 temporal aggregation
            # 每步都查询 policy
            with torch.no_grad():
                predicted_actions_normalized = self.base_policy(qpos_normalized, image_data)
                all_actions = self.denormalize_action(predicted_actions_normalized)  # [B, num_queries, action_dim]
            
            # 对每个环境独立处理
            actions_to_execute = []
            for env_idx in range(B):
                t = self._timestep[env_idx].item()
                
                if t < self.max_timesteps:
                    # 存储预测：all_time_actions[t, t:t+num_queries] = all_actions
                    end_t = min(t + self.num_queries, self.max_timesteps + self.num_queries)
                    self._all_time_actions[env_idx, t, t:end_t] = all_actions[env_idx, :end_t-t]
                    # (self.num_envs, self.max_timesteps, self.max_timesteps + self.num_queries, 13),

                    # 获取当前时间步的所有预测：actions_for_curr_step = all_time_actions[:, t]
                    actions_for_curr_step = self._all_time_actions[env_idx, :t+1, t]  # [t+1, action_dim]
                    
                    # 过滤掉全零的预测（未预测过的）
                    actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    
                    if len(actions_for_curr_step) > 0:
                        # 指数加权：越近的预测权重越大
                        k = 0.01
                        exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step), device=self.device, dtype=torch.float32))
                        exp_weights = exp_weights / exp_weights.sum()
                        
                        # 加权平均
                        action = (actions_for_curr_step * exp_weights.unsqueeze(1)).sum(dim=0)
                    else:
                        # 如果没有有效预测，使用当前预测的第一步
                        action = all_actions[env_idx, 0]
                else:
                    # 超过最大步数，使用当前预测的第一步
                    action = all_actions[env_idx, 0]
                
                actions_to_execute.append(action)
            
            # Stack 为 [B, action_dim]
            self._action = torch.stack(actions_to_execute, dim=0)
            

            # sim step according to decimation - 只执行前 decimation 步
            # for i in range(self.cfg.decimation):
            #     self._action = torch.stack(actions_to_execute, dim=0)
            #     self.sim_step()
            #     self._timestep += 1


            # 执行一步
            self.sim_step()
            
            # # 更新时间步
            self._timestep += 1
            
        # ==================== 简单模式（无 Temporal Aggregation）====================
        else:
            # 队列模式：如果队列中有足够的动作，直接使用
            if self._action_queue is not None and self._action_queue_ptr is not None:
                # 检查每个环境是否有足够的动作
                remaining_actions = self.num_queries - self._action_queue_ptr
                # 如果任意一个环境剩余动作不足，就重新预测所有环境
                need_prediction = (remaining_actions < self.cfg.decimation).any()
            else:
                need_prediction = True

            if need_prediction:
                # 调用 ACT policy 进行推理
                with torch.no_grad():
                    predicted_actions_normalized = self.base_policy(qpos_normalized, image_data)
                    base_act_seq = self.denormalize_action(predicted_actions_normalized)  # [B, num_queries, action_dim]
                
                # 更新队列（覆盖旧队列）
                self._action_queue = base_act_seq
                self._action_queue_ptr = torch.zeros(B, dtype=torch.long, device=self.device)
            
            # 从队列中取出 decimation 步执行
            for i in range(self.cfg.decimation):
                # 获取每个环境当前的 action index
                action_indices = self._action_queue_ptr
                
                # 安全检查：防止越界
                action_indices = torch.clamp(action_indices, 0, self.num_queries - 1)
                
                # 从队列中取出 action
                # self._action_queue: [B, num_queries, action_dim]
                # action_indices: [B]
                batch_indices = torch.arange(B, device=self.device)
                self._action = self._action_queue[batch_indices, action_indices, :]
                
                # 更新指针
                self._action_queue_ptr += 1
                
                # 执行一步
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
        
        # 重置 temporal aggregation 相关变量
        if self.temporal_agg:
            self._all_time_actions.zero_()
            self._timestep.zero_()

    def _quat_orientation_loss(self, quat_init: torch.Tensor, quat_current: torch.Tensor) -> torch.Tensor:
        """计算两个四元数之间的 pitch+roll 朝向偏差"""
        aw, ax, ay, az = quat_init.unbind(-1)
        bw, bx, by, bz = quat_current.unbind(-1)
        
        cw, cx, cy, cz = aw, -ax, -ay, -az
        
        rw = cw * bw - cx * bx - cy * by - cz * bz
        rx = cw * bx + cx * bw + cy * bz - cz * by
        ry = cw * by - cx * bz + cy * bw + cz * bx
        rz = cw * bz + cx * by - cy * bx + cz * bw
        
        norm = torch.sqrt(rw * rw + rx * rx + ry * ry + rz * rz) + 1e-8
        rx, ry = rx / norm, ry / norm
        
        loss = rx * rx + ry * ry
        return loss

    def _eval_fail_moved_without_contact(self) -> torch.Tensor:
        """评估物体在未接触时是否被移动（被推动/碰撞）"""
        not_contacted = ~self._has_contacted
        current_z = self._target.data.root_pos_w[:, 2]
        height_diff = current_z - self._target_pos_init[:, 2]
        moved_too_much = height_diff > 0.02
        bfailed = not_contacted & moved_too_much
        return bfailed

    def _eval_success_with_orientation(self) -> torch.Tensor:
        """评估抓取是否成功（同时检查高度和朝向）"""
        height_lift = self._target.data.root_pos_w[:, 2] - self._target_pos_init[:, 2]
        height_check = height_lift >= (self.cfg.lift_height_desired * 0.7)
        
        current_quat = self._target.data.root_quat_w
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        bsuccessed = height_check & orientation_check
        return bsuccessed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        bfailed,self._has_contacted = eval_fail(self._target,self._contact_sensors, self._has_contacted) # type: ignore
        
        bfailed_moved = self._eval_fail_moved_without_contact()
        bfailed = bfailed_moved
        
        bsuccessed = self._eval_success_with_orientation()
        
        # 记录成功环境的步数
        success_indices = torch.nonzero(bsuccessed == True).squeeze(1).tolist()
        if isinstance(success_indices, int):
            success_indices = [success_indices]
        for idx in success_indices:
            success_step = self.episode_length_buf[idx].item()
            self._success_steps_list.append(int(success_step))
        
        self._episode_success_num += len(success_indices)

        return bsuccessed, bfailed, time_out # type: ignore
    

    def _reset_idx(self, env_ids: torch.Tensor | None, success_ids:Sequence[int]|None=None):

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
        
        self._log_info()
        
        # 重置特定环境的 temporal aggregation 变量
        if self.temporal_agg and len(env_ids) > 0:
            self._timestep[env_ids] = 0
            self._all_time_actions[env_ids] = 0
        
        # variables used to store contact flag
        self._has_contacted[env_ids] = torch.zeros_like(self._has_contacted[env_ids],device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init[env_ids,:]=self._target.data.root_link_pos_w[env_ids,:].clone()
        self._target_quat_init[env_ids,:]=self._target.data.root_quat_w[env_ids,:].clone()
    
    def _log_info(self):
        # log policy evalutation result
        if self.cfg.enable_eval and self._episode_num > 0:
            if self.num_envs <= 4:
                log_interval = 1
            else:
                log_interval = self.num_envs
            
            if self._episode_num % log_interval == 0 or self._episode_num >= self.cfg.max_episode:
                policy_success_rate = float(self._episode_success_num) / float(self._episode_num)
                
                test_time_sec = self._timer.run_time()
                test_time_min = test_time_sec / 60.0
                test_rate = self._episode_num / test_time_min if test_time_min > 0 else 0
                
                print(f"\n{'='*50}")
                print(f"[ACT Evaluation - Episode {self._episode_num}/{self.cfg.max_episode}]")
                print(f"  Policy成功率: {policy_success_rate * 100.0:.2f}%")
                print(f"  成功次数/总次数: {self._episode_success_num}/{self._episode_num}")
                
                if len(self._success_steps_list) > 0:
                    avg_success_steps = sum(self._success_steps_list) / len(self._success_steps_list)
                    min_steps = min(self._success_steps_list)
                    max_steps = max(self._success_steps_list)
                    print(f"  成功步数: 平均={avg_success_steps:.1f}, 最小={min_steps}, 最大={max_steps}")
                    recent_steps = self._success_steps_list[-10:] if len(self._success_steps_list) > 10 else self._success_steps_list
                    print(f"  最近成功步数: {recent_steps}")
                
                print(f"  测试时间: {test_time_min:.2f} 分钟 ({test_time_sec:.1f} 秒)")
                print(f"  测试效率: {test_rate:.2f} episode/分钟")
                
                if self.cfg.enable_output:
                    record_rate = self._episode_success_num / test_time_min if test_time_min > 0 else 0
                    print(f"  采集效率: {record_rate:.2f} 条/分钟")
                print(f"{'='*50}\n")

