# Task: DexGrasp
# Robot: Robot with two RealMan RM75-6F and a PsiBot G0-R
# Description: 
#   使用强化学习直接学习抓取策略
#   任务：抓取随机位置和旋转的物体，并抬起到目标高度
#   奖励：
#       - distance reward: 手指到物体的距离奖励
#       - pose reward: 手指抓取姿态奖励
#       - lift reward: 物体抬起高度奖励
#       - angle reward: 手指角度奖励
#       - action penalty: 动作惩罚（鼓励平滑控制）
#   观测：
#       - 状态观测：关节位置、速度、物体位姿、手指位置等
#       - 图像观测：base camera RGB 图像（可选）
#
# 关键差异（相比 IL/MP）：
#   1. 使用密集奖励（distance + pose + lift + angle）而非稀疏奖励
#   2. 直接学习策略，不依赖专家演示
#   3. 成功评估：只检查高度（待完善：应增加朝向检查）

""" Common Modules  """ 
from __future__ import annotations
import torch
import warnings
import os
import numpy as np
from datetime import datetime
from typing import Tuple

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

@configclass
class DexGraspEnvCfg(RLEnvCfg):
    """Configuration for RL environment."""

    # params
    max_episode_length = 256
    episode_length_s = 1.0 * max_episode_length / 60.0
    decimation = 2
    action_scale = 0.5
    action_space = 13
    state_space = 130
    robot_dim = 16
    observation_space = 130

    # other params from gym
    arm_hand_dof_speed_scale = 20.0
    vel_obs_scale = 0.2
    act_moving_average = 0.8
    lift_height_target = 0.3
    
    # 成功判断：朝向偏差阈值（sin²(θ/2)，0=完全一致，1=上下颠倒）
    # 0.1 约等于 37° 的偏差，0.05 约等于 26° 的偏差
    # 注意：RL 目前只检查高度，建议添加朝向检查以提高任务难度和实用性
    orientation_threshold: float = 0.1

    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=decimation,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 64, #new change
            max_velocity_iteration_count = 0,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            # gpu_max_rigid_patch_count = 4096 * 4096,   # 16M
            # gpu_collision_stack_size = 2**30,          # 1GB
            # gpu_found_lost_pairs_capacity = 137401003, # 137M
            gpu_max_rigid_patch_count = 2048 * 2048,     # 4M
            gpu_collision_stack_size = 2**29,            # 512MB  
            gpu_found_lost_pairs_capacity = 68600501,    # 68M
        ),
        render=RenderCfg(),
    )
    
    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/rl"

class DexGraspEnv(RLEnv):
    """Dexterous Grasp a Lego Block via a Multi-finger Hand."""

    cfg: DexGraspEnvCfg

    def __init__(self, cfg: DexGraspEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # ############### initiallize variables ###################
        self._arm_joint_num = 7
        self._hand_real_joint_num = 6
        self._episodes = 0

        # get instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["target"]
        self._visualizer = self.scene.visualizer

        # place holder for contact sensors which defined in the cfg
        self._contact_sensors = {}

        # arm joint index
        self._arm_joint_index = self._robot.find_joints(self._robot.actuators["arm2"].joint_names, preserve_order=True)[0]
        # hand joint index
        self._hand_joint_index = self._robot.find_joints(self._robot.actuators["hand2"].joint_names, preserve_order=True)[0][:6]
        self._hand_base_link_index = self._robot.find_bodies(["hand2_link_base"])[0][0]
        # hand real joint index
        self._hand_real_joint_index = self._hand_joint_index[:6]
        # finger tip link index
        self._finger_tip_index = self._robot.find_bodies([
            "hand2_link_1_4",
            "hand2_link_2_3",
            "hand2_link_3_3",
            "hand2_link_4_3",
            "hand2_link_5_3",
            ], preserve_order=True)[0]
        
        # robot setup
        # [note]: this index means real control index
        self._robot_index = self._arm_joint_index + self._hand_real_joint_index
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()
        self._curr_targets = self._robot.data.default_joint_pos.clone()
        self._prev_targets = self._robot.data.default_joint_pos.clone()
        
        # taregt object initial pose and lift height
        self._target_init_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self._target_quat_init = torch.zeros((self.num_envs, 4), device=self.device)  # 初始朝向（wxyz）
        self._target_lift_height = self.cfg.lift_height_target * torch.ones(self.num_envs, device=self.device)
        
        # camera setup
        self._camera_config = self.cfg.scene.robots_cfg['robot'].tiled_cameras['base_camera']
        self._img_shape = (self._camera_config.height, self._camera_config.width, 3)
        self._base_camera = self.scene.robots["robot"].tiled_cameras["base_camera"]
        
        # unit tensors
        self._x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self._y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self._z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        # variables for rl training
        self._pre_distance_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_pose_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_lift_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_angle_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_energy = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        
        # metrics for task
        self._contacted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._has_contacted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # 用于未接触移动检测
        self._successed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Print information of this environment
        print("Num envs: ", self.num_envs)
        print("Num bodies: ", self._robot.num_bodies)
        print("Num arm dofs: ", self._arm_joint_num)
        print("Num hand dofs: ", 6)
        print("hand_base_rigid_body_index: ", self._hand_base_link_index)
        
        self.extras = {'dist_reward': 0.0, 'pose_reward': 0.0, 'lift_reward': 0.0, 'angl_reward': 0.0, 'act_penalty': 0.0, 'success': 0.0}
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        '''        self.episodes += 1
        action range       : (-1,1)
        action index 0-6   : arm joint
        action index 7-12  : real hand joint (order: 1-1,2-1,3-1,4-1,5-1,1-2)
        action index 13-17 : fake hand joint (order: 1-3,2-2,3-2,4-2,5-2)
        '''
        # ============ control arm =============
        arm_targets = self._robot.data.joint_pos[:, self._arm_joint_index] + \
                      self.cfg.arm_hand_dof_speed_scale  * self.physics_dt * self.actions[:, :self._arm_joint_num]

        self._curr_targets[:, self._arm_joint_index] = saturate(
            arm_targets,
            self._joint_limit_lower[:, self._arm_joint_index],
            self._joint_limit_upper[:, self._arm_joint_index]
        )
        
        # ============ control hand ============
        self._curr_targets[:, self._hand_real_joint_index] = scale(
            self.actions[:, self._arm_joint_num:],
            self._joint_limit_lower[:, self._hand_real_joint_index],
            self._joint_limit_upper[:, self._hand_real_joint_index]
        )
        self._curr_targets[:, self._hand_real_joint_index] = (
            self.cfg.act_moving_average * self._curr_targets[:, self._hand_real_joint_index]
            + (1.0 - self.cfg.act_moving_average) * self._prev_targets[:, self._hand_real_joint_index]
        )
        self._curr_targets[:, self._hand_real_joint_index] = saturate(
            self._curr_targets[:, self._hand_real_joint_index],
            self._joint_limit_lower[:, self._hand_real_joint_index],
            self._joint_limit_upper[:, self._hand_real_joint_index]
        )
        
        self._prev_targets = self._curr_targets.clone()
        self._robot.set_joint_position_target(
            self._curr_targets[:, self._robot_index], joint_ids=self._robot_index
        ) 
        
    def step(self, action):
        # call super step first to apply action and sim step
        obs_buf, reward_buf, reset_terminated, reset_time_outs, extras = super().step(action)
        
        self._maker_visualizer()

        return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras
    
    def _get_observations(self) -> dict:
        # implement fingertip force sensors
        # self.fingertip_force_sensors = self.robot.root_physx_view.get_link_incoming_joint_force()[:, self._finger_tip_index]
        
        self._get_full_states()
        self._get_image_observations()
        observations = {
            "policy": self._full_state, 
            "critic":self._full_state, 
            "images": {
                "rgb": self._img_obs
            }
        }
        
        # save image
        # os.makedirs("images", exist_ok=True)
        # from PIL import Image
        # Image.fromarray(self._img_obs['rgb'][0].cpu().numpy()).save(f"images/{self._episodes}_rgb_{self.common_step_counter}.png")
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        distance_reward, pose_reward, angle_reward, lift_reward, action_penalty = _compute_rewards(
            self.finger_thumb_state,
            self.finger_index_state,
            self.middle_point_state,
            self._target_init_pose[:, :3],
            self.object_state[:, :3],
            self.actions[:, :self._arm_joint_num],
            self._curr_targets[:, self._hand_real_joint_index],
            self._prev_targets[:, self._hand_real_joint_index],
            self._z_unit_tensor,
            self._target_lift_height,
        )
        
        total_reward = (distance_reward + pose_reward + lift_reward + angle_reward  - self._pre_energy) - action_penalty
        
        self.extras['dist_reward'] += (distance_reward[0] - self._pre_distance_reward[0])  # type: ignore
        self.extras['pose_reward'] += (pose_reward[0] - self._pre_pose_reward[0])# type: ignore
        self.extras['lift_reward'] += (lift_reward[0].to('cpu').numpy() - self._pre_lift_reward[0].to('cpu').numpy())# type: ignore
        self.extras['angl_reward'] += (angle_reward[0] - self._pre_angle_reward[0])# type: ignore
        self.extras['act_penalty'] += action_penalty[0] # type: ignore

        # update pre reward
        self._pre_distance_reward = distance_reward
        self._pre_pose_reward = pose_reward
        self._pre_lift_reward = lift_reward
        self._pre_angle_reward = angle_reward
        self._pre_energy = distance_reward + pose_reward + lift_reward + angle_reward

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
        
        返回值 ∈ [0, 1]：
        - 0: 朝向完全一致
        - 1: 上下颠倒（180°偏差）
        
        Args:
            quat_init: 初始四元数 [N, 4] (wxyz)
            quat_current: 当前四元数 [N, 4] (wxyz)
        Returns:
            loss: 朝向偏差 [N]
        """
        # 四元数分量 (wxyz format)
        aw, ax, ay, az = quat_init.unbind(-1)
        bw, bx, by, bz = quat_current.unbind(-1)
        
        # 计算 conj(a)
        cw, cx, cy, cz = aw, -ax, -ay, -az
        
        # 计算相对四元数 Δq = conj(a) ⊗ b
        rw = cw * bw - cx * bx - cy * by - cz * bz
        rx = cw * bx + cx * bw + cy * bz - cz * by
        ry = cw * by - cx * bz + cy * bw + cz * bx
        rz = cw * bz + cx * by - cy * bx + cz * bw
        
        # 归一化（数值安全）
        norm = torch.sqrt(rw * rw + rx * rx + ry * ry + rz * rz) + 1e-8
        rx, ry = rx / norm, ry / norm
        
        # pitch+roll 误差：sin²(θ_pr/2) ∈ [0,1]
        loss = rx * rx + ry * ry
        return loss
    
    def _eval_success_with_orientation(self) -> torch.Tensor:
        """
        评估抓取是否成功（同时检查高度和朝向）
        
        成功条件：
        1. 物体被抬起到目标高度的 80% 以上
        2. 物体朝向偏离初始朝向不超过阈值
        
        Returns:
            bsuccessed: 成功标志 [N]
        """
        # 1. 检查抬起高度（达到目标高度的 80% 即可）
        height_lift = self._target.data.root_pos_w[:, 2] - self._target_init_pose[:, 2]
        height_check = height_lift >= (self.cfg.lift_height_target * 0.8)
        
        # 2. 检查朝向偏差
        current_quat = self._target.data.root_quat_w  # 当前朝向 (wxyz)
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        # 综合判断：高度 AND 朝向
        bsuccessed = height_check & orientation_check
        
        return bsuccessed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        
        time_out = (self.episode_length_buf >= self.max_episode_length).bool()
        resets = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 方式1：只检查高度（原始版本）
        # self._successed = (self._target.data.root_pos_w[:, 2] - self._target_init_pose[:, 2]) >= self.cfg.lift_height_target * 0.8
        
        # 方式2：检查高度和朝向（推荐，与 IL/MP 一致）
        self._successed = self._eval_success_with_orientation()
        
        self.extras['success'] = self._successed.float().mean().item() * 100.0
        
        return resets, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES # type: ignore
        
        # [note]: super()._reset_idx will apply_random
        super()._reset_idx(env_ids) # type: ignore
        
        for _ in range(30):
            self.sim.step(render=False) # new change
            self.scene.update(dt=self.physics_dt)
        
        ############ reset robot ################
        dof_pos = self._robot.data.default_joint_pos.clone()
        dof_vel = self._robot.data.default_joint_vel.clone()
        self._curr_targets[env_ids, :] = dof_pos[env_ids, :]
        self._prev_targets[env_ids, :] = dof_vel[env_ids, :]
        self._robot.set_joint_position_target(dof_pos, env_ids=env_ids) # type: ignore
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids) # type: ignore
        
        self._target_init_pose[env_ids, :7] = self._target.data.root_link_state_w[env_ids, :7].clone()
        self._target_init_pose[:, :3] -= self.scene.env_origins
        self._target_quat_init[env_ids, :] = self._target.data.root_quat_w[env_ids, :].clone()  # 保存初始朝向
        
        # 重置状态
        self._has_contacted[env_ids] = False
        
        # compute reward
        self._compute_intermediate_values()
        self._get_initial_reward()
        
        # print info
        if self._episodes > 0:
            reward_items = ['dist_reward', 'pose_reward', 'lift_reward', 'angl_reward', 'act_penalty']
            total_reward = sum([abs(self.extras[item]) for item in reward_items])
            if total_reward <=0:
                total_reward = 1.0
            print("\n")
            print("#" * 17, "Statistics", "#" * 17)
            print(f"success:        {self.extras['success']:.2f}%")
            print(f"dist_reward:    {self.extras['dist_reward']:.2f} ({(abs(self.extras['dist_reward']) / total_reward * 100):.2f}%)")
            print(f"angle_reward:   {self.extras['angl_reward']:.2f} ({(abs(self.extras['angl_reward']) / total_reward * 100):.2f}%)")
            print(f"pose_reward:    {self.extras['pose_reward']:.2f} ({(abs(self.extras['pose_reward']) / total_reward * 100):.2f}%)")
            print(f"lego_up_reward: {self.extras['lift_reward']:.2f} ({(abs(self.extras['lift_reward']) / total_reward * 100):.2f}%)")
            print(f"action_penalty: {self.extras['act_penalty']:.2f} ({(abs(self.extras['act_penalty']) / total_reward * 100):.2f}%)")
            print("#" * 15, "Statistics End", "#" * 15,"\n")
            
            for key in reward_items:
                self.extras[key] = 0.0
        
        self._episodes += self.num_envs
    
    def _compute_intermediate_values(self):
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
        self._full_state = torch.cat(
            (
                # robot state
                unscale(self.dof_pos, self._joint_limit_lower[:, self._robot_index], self._joint_limit_upper[:, self._robot_index]),
                # object state
                self.object_state[:, :7],
                # dof speed
                self.cfg.vel_obs_scale * self.dof_vel,
                self.cfg.vel_obs_scale * self.object_state[:, 7:],
                # goal
                # fingertips
                (self.hand_state[:, :, :3] - self.object_state[:, None, :3]).reshape(self.num_envs, -1),
                self.cfg.vel_obs_scale * self.hand_state[:, :, 3:].reshape(self.num_envs, -1),
                # actions
                self.actions,
            ),
            dim=-1,
        )
    
    def _get_image_observations(self):
        self._img_obs = self._base_camera.data.output["rgb"]
    
    def _get_initial_reward(self):
        distance_reward, pose_reward, angle_reward, lift_reward, _ = _compute_rewards(
            self.finger_thumb_state,
            self.finger_index_state,
            self.middle_point_state,
            self._target_init_pose[:, :3],
            self.object_state[:, :3],
            self.actions[:, :self._arm_joint_num],
            self._curr_targets[:, self._hand_real_joint_index],
            self._prev_targets[:, self._hand_real_joint_index],
            self._z_unit_tensor,
            self._target_lift_height,
        )
        
        self._pre_distance_reward = distance_reward
        self._pre_pose_reward = pose_reward
        self._pre_angle_reward = angle_reward
        self._pre_lift_reward = lift_reward
        self._pre_energy = distance_reward + pose_reward + angle_reward + lift_reward
    
    def _maker_visualizer(self):
        if self.cfg.enable_marker and self._visualizer is not None:
            finger_tip_states = self._robot.data.body_link_state_w[:,self._finger_tip_index,:7]
            grasp_fingers_pos = (finger_tip_states[:,0,:3] + finger_tip_states[:,1,:3]) / 2
            target_state = self._target.data.root_link_state_w[:,:]
            
            marker_pos = torch.cat((
                finger_tip_states[0, :, :3],
                grasp_fingers_pos[0:1,:3],
                target_state[0:1,:3],
                ), dim=0)

            marker_rot = torch.cat((
                finger_tip_states[0, :, 3:7],
                torch.zeros((1,4), device=self.device),
                target_state[0:1,3:7],
                ), dim=0)

            self._visualizer.visualize(marker_pos, marker_rot)

# helper functions
@torch.jit.script
def tolerance(x: torch.Tensor, y: torch.Tensor, r: float, margin: float = 0.0, 
              value_at_margin: float = 0.1) -> torch.Tensor:
    """Returns 1 when `x` falls inside the circle centered at `y` with radius `r`"""
    if margin < 0.0:
        raise ValueError('margin must be non-negative')

    # Calculate the Euclidean distance from each point in x to y
    distance = torch.norm(x - y, p=2, dim=-1)
    
    # Calculate in_bounds mask
    in_bounds = distance <= r
    
    # Handle the zero margin case
    if margin == 0.0:
        return torch.where(in_bounds, torch.ones_like(distance), torch.zeros_like(distance))
    
    # Calculate normalized distance for sigmoid
    d = (distance - r) / margin
    
    # Calculate sigmoid value for out-of-bounds points
    scale = torch.sqrt(-2.0 * torch.log(torch.tensor(value_at_margin, device=x.device)))
    sigmoid_value = torch.exp(-0.5 * (d * scale) ** 2)
    
    # Combine in_bounds and out_of_bounds values
    return torch.where(in_bounds, torch.ones_like(distance), sigmoid_value)

@torch.jit.script
def compute_angle_line_plane(p1: torch.Tensor, p2: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
    """Compute angle between a line (defined by two points) and a plane (defined by its normal)
    
    Args:
        p1: First point of the line, shape (batch, 3)
        p2: Second point of the line, shape (batch, 3) 
        plane_normal: Normal vector of the plane, shape (batch, 3)
    
    Returns:
        Angle between line and plane in radians, shape (batch)
    """
    # Compute the direction vector of the line
    line_direction = p2 - p1  # (batch, 3)
    
    # Normalize the line direction and the plane normal
    line_direction_norm = torch.norm(line_direction, dim=-1, keepdim=True)
    plane_normal_norm = torch.norm(plane_normal, dim=-1, keepdim=True)
    
    # Add small epsilon to avoid division by zero
    eps = torch.tensor(1e-8, device=p1.device)
    line_direction_normalized = line_direction / (line_direction_norm + eps)
    plane_normal_normalized = plane_normal / (plane_normal_norm + eps)
    
    # Compute the dot product between the line direction and the plane normal
    dot_product = torch.sum(line_direction_normalized * plane_normal_normalized, dim=-1)
    
    # Clamp the dot product to avoid numerical issues with acos
    dot_product_clamped = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
    
    # Compute the angle between the line direction and the plane normal
    angle_with_normal = torch.acos(dot_product_clamped)
    
    # Compute the angle between the line and the plane
    angle_line_plane = torch.tensor(torch.pi/2, device=p1.device) - angle_with_normal
    
    return angle_line_plane

@torch.jit.script
def _compute_rewards(
    finger_thumb_state: torch.Tensor,
    finger_index_state: torch.Tensor,
    middle_point_state: torch.Tensor,
    lego_init_pos: torch.Tensor,
    lego_pos: torch.Tensor,
    arm_actions: torch.Tensor,
    joint_pos_target: torch.Tensor,
    prev_joint_pos_target: torch.Tensor,
    z_unit_tensor: torch.Tensor,
    lift_height_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # define dist reward
    fingertip_pos = torch.stack([finger_thumb_state[:,:3], finger_index_state[:,:3]], dim=0)
    finger_dist = torch.norm(lego_pos.unsqueeze(0) - fingertip_pos, p=2, dim=-1).sum(dim=0)
    distance_reward = torch.exp(-5.0 * torch.clamp((finger_dist - 0.05), 0, None))
    
    # define pose reward
    pose_dist = tolerance(middle_point_state[:,:3], lego_pos, r=0.016, margin=0.01)
    pose_reward = pose_dist * 6.0

    # define angle reward
    angle_dist = compute_angle_line_plane(finger_thumb_state[:,:3], finger_index_state[:,:3], z_unit_tensor)
    angle_reward = torch.exp(-1.0 * torch.abs(angle_dist)) * 0.5
    
    # define lift reward
    lift_offset = torch.zeros_like(lego_pos)
    lift_offset[:, 2] = lift_height_target
    target_pos = lego_init_pos + lift_offset
    goal_dist = torch.norm(lego_pos - target_pos, p=2, dim=-1)
    lift_reward = pose_dist * 400.0 * torch.clamp((lift_height_target - goal_dist), -0.05, None)
    
    # define action penalty
    action_penalty = 0.001 * torch.sum(arm_actions.pow_(2), dim=-1)
    action_penalty.add_(0.001 * torch.sum(
        joint_pos_target.sub_(prev_joint_pos_target).pow_(2), 
        dim=-1
    ))
    
    return distance_reward, pose_reward, angle_reward, lift_reward, action_penalty

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
    num_indexs = hand_state.shape[1] # (num_envs, num_indexs, 3)
    
    hand_state_offset = env_origins.repeat((1, num_indexs)).reshape(num_envs, num_indexs, 3)
    hand_state[:, :, :3].sub_(hand_state_offset)
    finger_thumb_state  = hand_state[:,0, :].clone()
    finger_index_state  = hand_state[:,1, :].clone()
    finger_middle_state = hand_state[:,2, :].clone()
    hand_base_state     = hand_state[:,5, :].clone()
    middle_point_state  = (finger_thumb_state + finger_index_state) / 2
    
    # data for robot joint
    dof_pos = joint_pos[:, robot_index].clone()
    dof_vel = joint_vel[:, robot_index].clone()
    
    return object_state, finger_thumb_state, finger_index_state, finger_middle_state, middle_point_state, hand_base_state, hand_state, dof_pos, dof_vel

# scale data to [0,1]
@torch.jit.script
def norm(x, lower, upper):
    return (x-lower)/(upper-lower)

# scale data to [lower, upper]
@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)

@torch.jit.script
def unscale(x, lower, upper):
    return 2.0 * (x-lower) / (upper - lower) - 1.0

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * torch.pi, x_unit_tensor), quat_from_angle_axis(rand1 * torch.pi, y_unit_tensor)
    )

def to_torch(x: list, dtype: torch.dtype = torch.float, device: str = 'cuda:0', requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)