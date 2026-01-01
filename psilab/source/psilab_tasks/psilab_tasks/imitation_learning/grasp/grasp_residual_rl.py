# Task: Residual RL for Dexterous Grasp
# Robot: Robot with two RealMan RM75-6F and a PsiBot G0-R
# Description: 
#   使用 Diffusion Policy 作为基础策略，RL 学习残差来提高成功率
#   
#   核心思想：
#   - Base Policy: Diffusion Policy（从专家演示学习，冻结参数）
#   - Residual RL: 学习对 DP 输出的修正（小幅调整）
#   - Final Action = DP Action + RL Residual
#   
#   优势：
#   - 利用专家知识（DP）作为先验，加速学习
#   - RL 只需学习修正，动作空间更小，更容易训练
#   - 可以弥补 DP 的系统性误差（如模拟到真实的差距）
#   
#   奖励函数：
#   - success_reward: 任务成功的稀疏奖励（+10）
#   - step_penalty: 每步惩罚，鼓励快速完成（-0.01）
#   - residual_penalty: 残差大小惩罚，鼓励最小修正（-0.001 * ||residual||）
#   
#   动作空间：
#   - RL 输出残差 ∈ [-1, 1]^13
#   - 缩放后叠加到 DP 输出：final = dp_action + residual_scale * rl_output
#   
#   观测空间：
#   - 状态观测：关节位置/速度、物体位姿、手指位置、上一步 RL 动作
#   - 可选：相机 RGB/Mask 图像
#   
#   关键差异（相比纯 RL）：
#   1. 动作是残差而非绝对值
#   2. 使用稀疏奖励（DP 已提供好的行为基线）
#   3. 需要加载 DP checkpoint 并冻结参数
#   
#   关键差异（相比 IL）：
#   1. 需要实时推理 DP 获取基础动作
#   2. 成功评估包含朝向检查（与 IL/MP 一致）
#   3. 支持相机观测（可选，用于 RL 策略）

""" Common Modules  """ 
from __future__ import annotations
import torch
import os
import numpy as np
from datetime import datetime
from typing import Tuple
from dataclasses import MISSING

""" Isaac Sim Modules  """ 
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg, PhysxCfg, RenderCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

""" Psi Lab Modules  """
from psilab import OUTPUT_DIR
from psilab.envs.rl_env import RLEnv 
from psilab.envs.rl_env_cfg import RLEnvCfg
from psilab.scene import SceneCfg
from psilab.utils.dp_utils import process_batch_image
from psilab.eval.grasp_rigid import eval_fail
from psilab_tasks.imitation_learning.grasp.image_utils import process_batch_image_multimodal

# ===================== Diffusion Policy 加载函数 =====================

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
    
    return policy


@configclass
class GraspResidualRLEnvCfg(RLEnvCfg):
    """Configuration for Residual RL environment."""

    # episode params
    # max_episode_length = 256
    episode_length_s = 2

    decimation = 4  # 与 IL 一致
    
    # action space: 7 (arm) + 6 (hand) = 13
    action_scale = 0.5  # 残差缩放系数，控制 RL 对 DP 的修正幅度
    action_space = 13
    state_space = 130
    robot_dim = 16
    observation_space = 130

    # RL 特定参数
    vel_obs_scale = 0.2
    act_moving_average = 0.8  # 动作平滑
    
    # 任务参数
    lift_height_target = 0.25  # 目标抬起高度
    
    # 成功判断：朝向偏差阈值（sin²(θ/2)，0=完全一致，1=上下颠倒）
    # 0.1 约等于 37° 的偏差，0.05 约等于 26° 的偏差
    # 注意：Residual RL 使用与 IL 一致的评估标准（高度 + 朝向）
    orientation_threshold: float = 0.1
    
    # Diffusion Policy checkpoint 路径
    dp_checkpoint: str = MISSING  # 必须提供

    # 残差 RL 参数
    residual_scale: float = 0.5  # 残差的缩放系数
    success_reward: float = 10.0  # 成功奖励
    step_penalty: float = -0.01  # 每步惩罚（鼓励快速完成）
    residual_penalty_scale: float = 0.000  # 残差大小惩罚（鼓励最小修正）
    
    # DP 观测模式配置（用于构建 DP 输入）
    # 可选: "rgb", "rgbm", "nd", "rgbnd", "state", "rgb_masked_rgb"
    # 注意：必须与训练 DP 时使用的模式一致
    dp_obs_mode: str = "rgb"
    
    # Mask 解耦实验配置（仅当 dp_obs_mode 包含 mask 时有效）
    # "real": 使用真实的 mask（默认）
    # "all_0": mask 通道填充全0
    # "all_1": mask 通道填充全1
    dp_mask_mode: str = "real"
    
    # 是否使用 RGBM（RGB + Mask）作为 DP 相机输入（已废弃，请使用 dp_obs_mode）
    # True: 使用 4 通道 RGBM 输入（需要相机配置启用 instance_segmentation_fast）
    # False: 使用 3 通道 RGB 输入
    with_mask: bool = False
    
    # RL 观测配置
    # 是否在 RL 观测中包含相机 RGB 图像
    rl_use_camera_rgb: bool = False
    # 是否在 RL 观测中包含相机 mask 图像（需要相机配置启用 instance_segmentation_fast）
    rl_use_camera_mask: bool = False

    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=decimation,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 32,
            max_velocity_iteration_count = 4,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            # gpu_max_rigid_patch_count = 2048 * 2048,
            # gpu_collision_stack_size = 2**29,
            # gpu_found_lost_pairs_capacity = 68600501,
            gpu_found_lost_pairs_capacity = 137401003
        ),
        render=RenderCfg(
            enable_translucency=True,
        ),
    )
    
    # scene config
    scene: SceneCfg = MISSING  # type: ignore
    
    # default output folder
    output_folder = OUTPUT_DIR + "/residual_rl"


class GraspResidualRLEnv(RLEnv):
    """Residual RL Environment: 在 Diffusion Policy 基础上学习残差"""

    cfg: GraspResidualRLEnvCfg

    def __init__(self, cfg: GraspResidualRLEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # ############### 初始化变量 ###################
        self._arm_joint_num = 7
        self._hand_real_joint_num = 6
        self._episodes = 0

        # 获取场景实例
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["bottle"]  # 使用 bottle 与 IL 一致
        self._visualizer = self.scene.visualizer

        # 初始化 contact sensors
        self._contact_sensors = {}
        for key in ["hand2_link_base",
                    "hand2_link_1_1", "hand2_link_1_2", "hand2_link_1_3",
                    "hand2_link_2_1", "hand2_link_2_2",
                    "hand2_link_3_1", "hand2_link_3_2",
                    "hand2_link_4_1", "hand2_link_4_2",
                    "hand2_link_5_1", "hand2_link_5_2"]:
            if key in self.scene.sensors:
                self._contact_sensors[key] = self.scene.sensors[key]

        # 关节索引
        self._arm_joint_index = self._robot.find_joints(
            self._robot.actuators["arm2"].joint_names, preserve_order=True)[0]
        self._hand_joint_index = self._robot.find_joints(
            self._robot.actuators["hand2"].joint_names, preserve_order=True)[0][:6]
        
        self._hand_base_link_index = self._robot.find_bodies(["hand2_link_base"])[0][0]
        self._hand_real_joint_index = self._hand_joint_index[:6]
        
        # 指尖 link 索引
        self._finger_tip_index = self._robot.find_bodies([
            "hand2_link_1_4",
            "hand2_link_2_3",
            "hand2_link_3_3",
            "hand2_link_4_3",
            "hand2_link_5_3",
        ], preserve_order=True)[0]
        
        # 机器人控制索引
        self._robot_index = self._arm_joint_index + self._hand_real_joint_index
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()
        self._curr_targets = self._robot.data.default_joint_pos.clone()
        self._prev_targets = self._robot.data.default_joint_pos.clone()
        
        # 目标物体初始位姿
        self._target_init_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self._target_quat_init = torch.zeros((self.num_envs, 4), device=self.device)
        
        # 单位张量
        self._z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        # ############### 加载 Diffusion Policy ###################
        self._dp_policy = load_diffusion_policy_from_checkpoint(
            self.cfg.dp_checkpoint, device=self.device
        )
        # 冻结 DP 参数
        for param in self._dp_policy.parameters():
            param.requires_grad = False
        
        # DP 输出缓存
        self._dp_action = torch.zeros((self.num_envs, 13), device=self.device)
        self._dp_action_sequence = None  # 存储 DP 输出的动作序列
        self._dp_action_idx = 0  # 当前使用的动作索引
        
        # RL 动作缓存（上一步的 RL 输出）
        self._rl_action = torch.zeros((self.num_envs, 13), device=self.device)
        
        # 任务状态
        self._has_contacted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._successed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._success_count = 0
        self._episode_count = 0
        
        # 统计信息
        self.extras = {
            'success_rate': 0.0, 
            'avg_residual_norm': 0.0,
            'episode_reward': 0.0,
        }
        
        # 打印环境信息
        print("=" * 50)
        print("Residual RL Environment Initialized")
        print(f"  Num envs: {self.num_envs}")
        print(f"  Action space: {self.cfg.action_space}")
        print(f"  Residual scale: {self.cfg.residual_scale}")
        print(f"  DP checkpoint: {self.cfg.dp_checkpoint}")
        print(f"  DP obs mode: {self.cfg.dp_obs_mode}")
        if self.cfg.dp_obs_mode in ["rgbm"]:
            print(f"  DP mask mode: {self.cfg.dp_mask_mode}")
        print("=" * 50)

    def _get_dp_observation(self) -> dict:
        """
        构建 Diffusion Policy 需要的观测
        
        支持多种观测模式：
        - rgb: 3通道 RGB 图像
        - rgbm: 4通道 RGB + Mask
        - nd: 4通道 Normal + Depth
        - rgbnd: 7通道 RGB + Normal + Depth
        - rgb_masked_rgb: 6通道 RGB + RGB*Mask
        - state: 纯状态（无图像）
        """
        # 末端执行器状态
        eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        eef_state = self._robot.data.body_link_state_w[:, eef_link_index, :7].clone()
        eef_state[:, :3] -= self._robot.data.root_state_w[:, :3]
        
        # 目标物体位姿（相对于环境原点）
        target_pose = self._target.data.root_state_w[:, :7].clone()
        target_pose[:, :3] -= self.scene.env_origins
        
        # 兼容旧的 with_mask 参数
        if hasattr(self.cfg, 'with_mask') and self.cfg.with_mask:
            obs_mode = "rgbm"
        else:
            obs_mode = self.cfg.dp_obs_mode
        
        # 如果是 state 模式，不需要图像
        if obs_mode == 'state':
            obs = {
                'arm2_pos': self._robot.data.joint_pos[:, self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
                'hand2_pos': self._robot.data.joint_pos[:, self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1),
                'arm2_eef_pos': eef_state[:, :3].unsqueeze(1),
                'arm2_eef_quat': eef_state[:, 3:7].unsqueeze(1),
                'target_pose': target_pose.unsqueeze(1),
            }
            return obs
        
        # 获取基础 RGB 图像
        chest_rgb = self._robot.tiled_cameras["chest_camera"].data.output["rgb"][:, :, :, :]
        head_rgb = self._robot.tiled_cameras["head_camera"].data.output["rgb"][:, :, :, :]
        
        # 初始化可选通道
        chest_mask, head_mask = None, None
        chest_depth, head_depth = None, None
        chest_normal, head_normal = None, None
        
        # 1. Mask（如果需要）
        if obs_mode in ["rgbm", "rgb_masked_rgb"]:
            if self.cfg.dp_mask_mode == "real":
                if "instance_segmentation_fast" in self._robot.tiled_cameras["chest_camera"].data.output:
                    chest_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
                if "instance_segmentation_fast" in self._robot.tiled_cameras["head_camera"].data.output:
                    head_mask = self._robot.tiled_cameras["head_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
            elif self.cfg.dp_mask_mode == "all_0":
                chest_mask = torch.zeros_like(chest_rgb[:, :, :, 0])
                head_mask = torch.zeros_like(head_rgb[:, :, :, 0])
            elif self.cfg.dp_mask_mode == "all_1":
                chest_mask = torch.ones_like(chest_rgb[:, :, :, 0])
                head_mask = torch.ones_like(head_rgb[:, :, :, 0])
        
        # 2. Depth（如果需要）
        if obs_mode in ["nd", "rgbnd"]:
            if "depth" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_depth = self._robot.tiled_cameras["chest_camera"].data.output["depth"][:, :, :, 0]
            if "depth" in self._robot.tiled_cameras["head_camera"].data.output:
                head_depth = self._robot.tiled_cameras["head_camera"].data.output["depth"][:, :, :, 0]
        
        # 3. Normal（如果需要）
        if obs_mode in ["nd", "rgbnd"]:
            if "normals" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_normal = self._robot.tiled_cameras["chest_camera"].data.output["normals"][:, :, :, :3]
            if "normals" in self._robot.tiled_cameras["head_camera"].data.output:
                head_normal = self._robot.tiled_cameras["head_camera"].data.output["normals"][:, :, :, :3]
        
        # 统一处理图像
        chest_camera_img = process_batch_image_multimodal(
            rgb=chest_rgb,
            mask=chest_mask,
            depth=chest_depth,
            normal=chest_normal,
            obs_mode=obs_mode
        )
        
        head_camera_img = process_batch_image_multimodal(
            rgb=head_rgb,
            mask=head_mask,
            depth=head_depth,
            normal=head_normal,
            obs_mode=obs_mode
        )
        
        # 构建观测字典（与 IL 一致）
        obs = {
            'chest_camera_rgb': chest_camera_img.unsqueeze(1),
            'head_camera_rgb': head_camera_img.unsqueeze(1),
            'arm2_pos': self._robot.data.joint_pos[:, self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
            'hand2_pos': self._robot.data.joint_pos[:, self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1),
            'arm2_eef_pos': eef_state[:, :3].unsqueeze(1),
            'arm2_eef_quat': eef_state[:, 3:7].unsqueeze(1),
            'target_pose': target_pose.unsqueeze(1),
        }
        
        return obs

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """存储 RL 输出的残差动作"""
        # actions 是 RL 网络输出的残差，范围 [-1, 1]
        self.residual_actions = actions.clone()
        # 保存 RL 动作用于观测
        self._rl_action = actions.clone()
    
    def _apply_action(self) -> None:
        """应用动作：DP 动作 + RL 残差"""
        
        # 每个 decimation 周期开始时，获取新的 DP 动作序列
        if self._dp_action_idx == 0 or self._dp_action_sequence is None:
            dp_obs = self._get_dp_observation()
            with torch.no_grad():
                self._dp_action_sequence = self._dp_policy.predict_action(dp_obs)['action']
            self._dp_action_idx = 0
        
        # 获取当前时间步的 DP 动作
        dp_action = self._dp_action_sequence[:, self._dp_action_idx, :]
        self._dp_action_idx = (self._dp_action_idx + 1) % self.cfg.decimation
        
        # 计算残差（缩放后的 RL 输出）
        # RL 输出范围 [-1, 1]，缩放到合理的残差范围
        residual = self.residual_actions * self.cfg.residual_scale
        
        # 最终动作 = DP 动作 + 残差
        final_action = dp_action + residual
        
        # 存储用于奖励计算
        self._dp_action = dp_action.clone()
        self._final_action = final_action.clone()
        
        # 应用动作到机器人
        # Arm: 直接设置关节位置目标
        arm_targets = saturate(
            final_action[:, :self._arm_joint_num],
            self._joint_limit_lower[:, self._arm_joint_index],
            self._joint_limit_upper[:, self._arm_joint_index]
        )
        self._curr_targets[:, self._arm_joint_index] = arm_targets
        
        # Hand: 直接设置关节位置目标
        hand_targets = saturate(
            final_action[:, self._arm_joint_num:],
            self._joint_limit_lower[:, self._hand_real_joint_index],
            self._joint_limit_upper[:, self._hand_real_joint_index]
        )
        # 动作平滑
        self._curr_targets[:, self._hand_real_joint_index] = (
            self.cfg.act_moving_average * hand_targets
            + (1.0 - self.cfg.act_moving_average) * self._prev_targets[:, self._hand_real_joint_index]
        )
        
        self._prev_targets = self._curr_targets.clone()
        self._robot.set_joint_position_target(
            self._curr_targets[:, self._robot_index], joint_ids=self._robot_index
        )
        
    def step(self, action):
        # 调用父类 step
        obs_buf, reward_buf, reset_terminated, reset_time_outs, extras = super().step(action)
        
        # 可视化
        if self.cfg.enable_marker and self._visualizer is not None:
            self._maker_visualizer()

        return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras
    
    def _get_observations(self) -> dict:
        """获取 RL 策略的观测"""
        self._compute_intermediate_values()
        self._get_full_states()
        
        observations = {
            "policy": self._full_state, 
            "critic": self._full_state,
        }
        
        # 添加相机观测（如果配置启用）
        if self.cfg.rl_use_camera_rgb or self.cfg.rl_use_camera_mask:
            camera_obs = self._get_camera_observations()
            observations["policy"] = {
                "state": self._full_state,
                **camera_obs
            }
            observations["critic"] = {
                "state": self._full_state,
                **camera_obs
            }
        
        return observations
    
    def _get_camera_observations(self) -> dict:
        """获取相机观测（RGB 和/或 Mask）"""
        camera_obs = {}
        
        if self.cfg.rl_use_camera_rgb:
            # 获取 RGB 图像
            chest_rgb = self._robot.tiled_cameras["chest_camera"].data.output["rgb"][:, :, :, :]
            head_rgb = self._robot.tiled_cameras["head_camera"].data.output["rgb"][:, :, :, :]
            
            # 处理 RGB 图像
            chest_camera_rgb = process_batch_image(chest_rgb, None, with_mask=False)
            head_camera_rgb = process_batch_image(head_rgb, None, with_mask=False)
            
            camera_obs["chest_camera_rgb"] = chest_camera_rgb
            camera_obs["head_camera_rgb"] = head_camera_rgb
        
        if self.cfg.rl_use_camera_mask:
            # 获取 mask 图像
            chest_mask = None
            head_mask = None
            
            if "instance_segmentation_fast" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
                # 处理 mask：二值化并调整尺寸
                chest_mask_processed = self._process_mask(chest_mask)
                camera_obs["chest_camera_mask"] = chest_mask_processed
                
            if "instance_segmentation_fast" in self._robot.tiled_cameras["head_camera"].data.output:
                head_mask = self._robot.tiled_cameras["head_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
                head_mask_processed = self._process_mask(head_mask)
                camera_obs["head_camera_mask"] = head_mask_processed
        
        return camera_obs
    
    def _process_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """处理 mask 图像：二值化并调整尺寸到 224x224"""
        import torch.nn.functional as F
        batch_size = mask.shape[0]
        mask_processed = torch.zeros((batch_size, 1, 224, 224), 
                                      dtype=torch.float32, device=mask.device)
        
        for i in range(batch_size):
            m = mask[i, ...]
            # 转为二值 mask（非零值为 1.0）
            mask_binary = (m > 0).float()
            # 调整尺寸
            mask_resized = F.interpolate(
                mask_binary.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode='nearest'
            )
            mask_processed[i] = mask_resized.squeeze(0)
        
        return mask_processed
    
    def _get_rewards(self) -> torch.Tensor:
        """
        计算奖励：
        1. 成功奖励：任务成功时给予大的正奖励
        2. 每步惩罚：鼓励快速完成任务
        3. 残差惩罚：鼓励最小化 RL 的干预
        """
        # 检查成功状态
        bsuccessed = self._eval_success_with_orientation()
        
        # 成功奖励
        success_reward = bsuccessed.float() * self.cfg.success_reward
        
        # 每步惩罚
        step_penalty = torch.ones(self.num_envs, device=self.device) * self.cfg.step_penalty
        
        # 残差惩罚（鼓励最小修正）
        residual_norm = torch.norm(self.residual_actions, dim=-1)
        residual_penalty = residual_norm * self.cfg.residual_penalty_scale
        
        # 总奖励
        # total_reward = success_reward + step_penalty - residual_penalty
        total_reward = success_reward 
        
        # 更新统计
        self.extras['avg_residual_norm'] = residual_norm.mean().item()
        
        return total_reward

    def _eval_fail_moved_without_contact(self) -> torch.Tensor:
        """
        评估物体在未接触时是否被移动（被推动/碰撞）
        
        失败条件：物体未被接触，但 z 方向位移超过 2cm
        
        Returns:
            bfailed: 失败标志 [N]
        """
        # 检查物体是否未被接触
        not_contacted = ~self._has_contacted
        
        # 计算 z 方向位移
        current_z = self._target.data.root_pos_w[:, 2]
        height_diff = current_z - self._target_init_pose[:, 2]
        
        # 物体未被接触但 z 方向移动超过 2cm，判定为失败
        moved_too_much = height_diff > 0.02
        
        bfailed = not_contacted & moved_too_much
        
        return bfailed

    def _quat_orientation_loss(self, quat_init: torch.Tensor, quat_current: torch.Tensor) -> torch.Tensor:
        """
        计算两个四元数之间的 pitch+roll 朝向偏差
        返回值 ∈ [0, 1]：0 = 朝向一致，1 = 上下颠倒
        """
        aw, ax, ay, az = quat_init.unbind(-1)
        bw, bx, by, bz = quat_current.unbind(-1)
        
        # 计算 conj(a)
        cw, cx, cy, cz = aw, -ax, -ay, -az
        
        # 计算相对四元数 Δq = conj(a) ⊗ b
        rw = cw * bw - cx * bx - cy * by - cz * bz
        rx = cw * bx + cx * bw + cy * bz - cz * by
        ry = cw * by - cx * bz + cy * bw + cz * bx
        rz = cw * bz + cx * by - cy * bx + cz * bw
        
        # 归一化
        norm = torch.sqrt(rw * rw + rx * rx + ry * ry + rz * rz) + 1e-8
        rx, ry = rx / norm, ry / norm
        
        # pitch+roll 误差
        loss = rx * rx + ry * ry
        return loss

    def _eval_success_with_orientation(self) -> torch.Tensor:
        """
        评估抓取是否成功（检查高度和朝向）
        """
        # 检查抬起高度
        height_lift = self._target.data.root_pos_w[:, 2] - self._target_init_pose[:, 2]
        height_check = height_lift >= (self.cfg.lift_height_target * 0.8)
        
        # 检查朝向偏差
        current_quat = self._target.data.root_quat_w
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        # 综合判断
        bsuccessed = height_check & orientation_check
        
        return bsuccessed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        
        time_out = (self.episode_length_buf >= self.max_episode_length).bool()
        
        # 检查失败（掉落等）
        if self._contact_sensors:
            bfailed, self._has_contacted = eval_fail(
                self._target, self._contact_sensors, self._has_contacted
            )
        else:
            bfailed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 新增：检测物体在未接触时被移动（被推动/碰撞）
        bfailed_moved = self._eval_fail_moved_without_contact()
        # bfailed = bfailed | bfailed_moved
        bfailed = bfailed_moved

        
        # 检查成功
        self._successed = self._eval_success_with_orientation()
        
        # 统计
        success_count = self._successed.sum().item()
        self._success_count += success_count
        
        # 成功也作为终止条件
        resets = self._successed | bfailed
        
        self.extras['success_rate'] = self._successed.float().mean().item() * 100.0
        
        return resets, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES  # type: ignore
        
        # 调用父类 reset
        super()._reset_idx(env_ids)  # type: ignore
        
        # 等待物理稳定
        for _ in range(30):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        
        # 重置机器人
        dof_pos = self._robot.data.default_joint_pos.clone()
        dof_vel = self._robot.data.default_joint_vel.clone()
        self._curr_targets[env_ids, :] = dof_pos[env_ids, :]
        self._prev_targets[env_ids, :] = dof_pos[env_ids, :]
        self._robot.set_joint_position_target(dof_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        
        # 保存目标物体初始位姿
        self._target_init_pose[env_ids, :7] = self._target.data.root_link_state_w[env_ids, :7].clone()
        self._target_init_pose[env_ids, :3] -= self.scene.env_origins[env_ids]
        self._target_quat_init[env_ids, :] = self._target.data.root_quat_w[env_ids, :].clone()
        
        # 重置 DP 动作缓存
        self._dp_action_sequence = None
        self._dp_action_idx = 0
        
        # 重置 RL 动作缓存
        self._rl_action[env_ids] = 0.0
        
        # 重置状态
        self._has_contacted[env_ids] = False
        self._successed[env_ids] = False
        
        # 更新统计
        self._episode_count += len(env_ids)
        
        # 打印统计信息
        if self._episodes > 0 and self._episodes % (self.num_envs * 10) == 0:
            success_rate = self._success_count / max(self._episode_count, 1) * 100.0
            print("\n" + "=" * 50)
            print(f"Episode {self._episodes}")
            print(f"  Overall Success Rate: {success_rate:.2f}%")
            print(f"  Success Count: {self._success_count}/{self._episode_count}")
            print(f"  Avg Residual Norm: {self.extras['avg_residual_norm']:.4f}")
            print("=" * 50 + "\n")
        
        self._episodes += len(env_ids)
    
    def _compute_intermediate_values(self):
        """计算中间值用于观测和奖励"""
        self._hand_index = self._finger_tip_index + [self._hand_base_link_index]
        (
            self.object_state,
            self.finger_thumb_state,
            self.finger_index_state,
            self.finger_middle_state,
            self.middle_point_state,
            self.hand_base_state,
            self.hand_state,
            self.dof_pos,
            self.dof_vel,
        ) = _compute_values(
            self.scene.env_origins,
            self._target.data.root_link_state_w.clone(),
            self._robot.data.body_link_state_w[:, self._hand_index].clone(),
            self._robot_index,
            self._robot.data.joint_pos,
            self._robot.data.joint_vel,
        )
    
    def _get_full_states(self):
        """获取完整状态用于 RL 策略"""
        # 包含 RL 上一步动作作为观测的一部分
        self._full_state = torch.cat(
            (
                # 机器人状态
                unscale(self.dof_pos, 
                       self._joint_limit_lower[:, self._robot_index], 
                       self._joint_limit_upper[:, self._robot_index]),
                # 物体状态
                self.object_state[:, :7],
                # 速度
                self.cfg.vel_obs_scale * self.dof_vel,
                self.cfg.vel_obs_scale * self.object_state[:, 7:],
                # 手指相对物体位置
                (self.hand_state[:, :, :3] - self.object_state[:, None, :3]).reshape(self.num_envs, -1),
                self.cfg.vel_obs_scale * self.hand_state[:, :, 3:].reshape(self.num_envs, -1),
                # RL 上一步动作
                self._rl_action,
            ),
            dim=-1,
        )
    
    def _maker_visualizer(self):
        """可视化调试信息"""
        if self._visualizer is not None:
            finger_tip_states = self._robot.data.body_link_state_w[:, self._finger_tip_index, :7]
            grasp_fingers_pos = (finger_tip_states[:, 0, :3] + finger_tip_states[:, 1, :3]) / 2
            target_state = self._target.data.root_link_state_w[:, :]
            
            marker_pos = torch.cat((
                finger_tip_states[0, :, :3],
                grasp_fingers_pos[0:1, :3],
                target_state[0:1, :3],
            ), dim=0)

            marker_rot = torch.cat((
                finger_tip_states[0, :, 3:7],
                torch.zeros((1, 4), device=self.device),
                target_state[0:1, 3:7],
            ), dim=0)

            self._visualizer.visualize(marker_pos, marker_rot)


# ===================== 辅助函数 =====================

@torch.jit.script
def _compute_values(
    env_origins: torch.Tensor,
    object_state: torch.Tensor,
    hand_state: torch.Tensor,
    robot_index: list[int],
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_envs = env_origins.shape[0]
    object_state[:, :3].sub_(env_origins)
    num_indexs = hand_state.shape[1]
    
    hand_state_offset = env_origins.repeat((1, num_indexs)).reshape(num_envs, num_indexs, 3)
    hand_state[:, :, :3].sub_(hand_state_offset)
    finger_thumb_state = hand_state[:, 0, :].clone()
    finger_index_state = hand_state[:, 1, :].clone()
    finger_middle_state = hand_state[:, 2, :].clone()
    hand_base_state = hand_state[:, 5, :].clone()
    middle_point_state = (finger_thumb_state + finger_index_state) / 2
    
    dof_pos = joint_pos[:, robot_index].clone()
    dof_vel = joint_vel[:, robot_index].clone()
    
    return (object_state, finger_thumb_state, finger_index_state, finger_middle_state, 
            middle_point_state, hand_base_state, hand_state, dof_pos, dof_vel)


@torch.jit.script
def unscale(x, lower, upper):
    return 2.0 * (x - lower) / (upper - lower) - 1.0


@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)

