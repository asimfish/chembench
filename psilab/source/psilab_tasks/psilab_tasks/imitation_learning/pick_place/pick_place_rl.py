# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: 
# Date: 2025-05-27
# Vesion: 1.0

""" Common Modules  """ 
from __future__ import annotations
import torch
import warnings
from datetime import datetime
from typing import Tuple
import matplotlib

""" Isaac Sim Modules  """ 
import isaacsim.core.utils.torch as torch_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg, PhysxCfg, RenderCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate


""" Psi Lab Modules  """
from psilab import OUTPUT_DIR
from psilab.envs.rl_env import RLEnv 
from psilab.envs.rl_env_cfg import RLEnvCfg
from source.algo.utils.wandb_utils import WandbLog
from psilab.utils.timer_utils import Timer
from psilab.eval.grasp_rigid import eval_fail, eval_success
import source.algo.utils.tools as cp
from pathlib import Path
import os
import numpy as np
from pxr import UsdGeom, Usd
import omni.usd
from source.algo.utils.collect_data import CollectData
import yaml
import cv2
from typing import Tuple
from typing import List
from .rewards import StandbyReward, GraspReward, PositionReward, OrientationReward
@configclass
class DexPickPlaceEnvCfg(RLEnvCfg):
    """Configuration for RL environment."""

    # basic params
    print_infos = False
    decimation = 8
    action_chunk = 1
    # max_episode_length = 600 * decimation / action_chunk
    max_episode_length = 600
    episode_length_s = 1.0 * max_episode_length / 120.0
    action_scale = 0.5
    action_space = 13
    true_action_space = 13
    observation_space = 152
    state_space = 152
    table_height = 0.7535
    
    # robot control params
    arm_dof_speed_scale = 0.3
    hand_dof_speed_scale = arm_dof_speed_scale * 7
    vel_obs_scale = 0.2
    env_id_print_data = 0 # index of env to print status

    # task params
    grasp_offset = 0.15
    standby_distance_threshold = 0.1
    position_distance_threshold = 0.02
    orientation_distance_threshold = 0.02
    random_lift_targets = False
    base_lift_targets = [0.5, 0, 0.76]
    # base_lift_targets = [0.49,-0.012,0.822]
    # targets_range = [0.12, 0.155, 0]
    targets_range = [0.02, 0.02, 0]

    # reward params
    reward_func = [{'standby': False, 'grasp': True, 'position': True, 'orientation': True},
                    {'standby': False, 'grasp': True, 'position': True, 'orientation': True},
                    {'standby': False, 'grasp': True, 'position': True, 'orientation': True},
                    {'standby': True, 'grasp': False, 'position': True, 'orientation': True},
                    {'standby': True, 'grasp': False, 'position': False, 'orientation': False}] # ending state
    subtask_num = len(reward_func)


    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=decimation,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 16,
            max_velocity_iteration_count = 0,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            gpu_max_rigid_patch_count = 4096 * 4096,
            gpu_collision_stack_size = 1600000000,
            gpu_found_lost_pairs_capacity = 137401003,
            # gpu_total_aggregate_pairs_capacity = 4194304 * 4

        ),
        render=RenderCfg(),

    )

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/rl"

    # observation noise cfg
    position_obs_noise = False
    position_obs_noise_range = [0.01, 0.01, 0.01]
    position_obs_noise_type = "uniform" # uniform, gaussian
    
    # contact mode cfg
    contact_mode = False
    if contact_mode:
        action_space = 14

class DexPickPlaceEnv(RLEnv):
    """GraspLego RL environment."""

    cfg: DexPickPlaceEnvCfg

    def __init__(self, cfg: DexPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # ############### initiallize variables ###################
        with open(Path(__file__).resolve().parent / "config.yaml", "r") as f:
            self.yaml_cfg = yaml.safe_load(f)
        self._arm_joint_num = 7
        self._hand_real_joint_num = 6
        self._episodes = 0

        # get instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["target"]
        self._visualizer = self.scene.visualizer

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

        # arm joint index
        self._arm_joint_index = [self._robot.find_joints(joint_name)[0][0] for joint_name in self._robot.actuators["arm2"].joint_names]
        # hand joint index
        self._hand_joint_index = [self._robot.find_joints(joint_name)[0][0] for joint_name in self._robot.actuators["hand2"].joint_names]
        self._hand_base_link_index = self._robot.find_bodies(["hand2_link_base"])[0][0]
        # hand real joint index
        self._hand_real_joint_index = self._hand_joint_index[:6]
        # hand virtual joint index
        self._hand_virtual_joint_index = self._hand_joint_index[6:]
        # finger tip link index
        self._finger_tip_index = [
                self._robot.find_bodies(link_name)[0][0] 
                for link_name in [
                    "hand2_link_1_4",
                    "hand2_link_2_3",
                    "hand2_link_3_3",
                    "hand2_link_4_3",
                    "hand2_link_5_3",
                ]
            ]
        self._finger_joint_index = [
                self._robot.find_bodies(link_name)[0][0] 
                for link_name in [
                    "hand2_link_1_1",
                    "hand2_link_1_2",
                    "hand2_link_1_3",
                    "hand2_link_1_4",
                    "hand2_link_2_3",
                    "hand2_link_2_2",
                    "hand2_link_2_1",
                    
                ]
            ]
        # robot index
        # [note]: this index means real control index
        self._robot_index = self._arm_joint_index + self._hand_real_joint_index
        
        # joint limit
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()

        # unit tensors
        self._x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self._y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self._z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        # robot, object inital pose and goal
        self._curr_targets = self._robot.data.default_joint_pos.clone()
        self._prev_targets = self._robot.data.default_joint_pos.clone()
        self._target_init_pose = torch.zeros((self.num_envs, 7), device=self.device) # 物体最初始位姿
        self._target_subtask_init_pose = torch.zeros((self.num_envs, 7), device=self.device) # 物体在子任务中的初始位姿
        self._target_lift_pose = torch.zeros((self.num_envs, 3), device=self.device) # 物体在子任务中的目标位置
        self._target_orien_pose = torch.zeros((self.num_envs, 4), device=self.device) # 物体在子任务中的目标姿态
        self._final_target_pose = torch.zeros((self.num_envs, 3), device=self.device) # 物体最终目标位置
        self._hand_init_pose = torch.zeros((self.num_envs, 3), device=self.device) # 手最初始位姿
        self._hand_target_pose = torch.zeros((self.num_envs, 3), device=self.device) # 手在子任务中的目标位置
        self._grasp_point = torch.zeros((self.num_envs, 3), device=self.device) # 目标抓取位置
        self._init_highest_point_offset = torch.zeros((self.num_envs, 3), device=self.device)
        
        # object properties
        self._object_lwh = torch.zeros((self.num_envs, 3), device=self.device) # 物体 bounding box size
        self._staticFriction = torch.zeros((self.num_envs), device=self.device) # 物体表面静摩擦系数
        self._dynamicFriction = torch.zeros((self.num_envs), device=self.device) # 物体表面动摩擦系数
        self._restitution = torch.zeros((self.num_envs), device=self.device) # 物体表面恢复系数
        self._mass = torch.zeros((self.num_envs), device=self.device) # 物体质量

        # rl observation, action and rewards
        self._contact_states = torch.zeros((self.num_envs, 7, 3), device=self.device) # env, finger, force
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.true_action_space), device=self.device)
        self.prev_accelerate = torch.zeros((self.num_envs, self.cfg.true_action_space), device=self.device)
        self.curr_accelerate = torch.zeros((self.num_envs, self.cfg.true_action_space), device=self.device)
        self._pre_distance_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_pose_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_lift_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_angle_reward = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._pre_energy = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self._accumulated_lift_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # reward functions
        self.standby_reward = StandbyReward(self.num_envs, self.cfg.standby_distance_threshold)
        self.grasp_reward = GraspReward(self.num_envs, 
                                        distance_threshold=0.5, 
                                        pose_threshold=6.0, 
                                        angle_threshold=2.6)
        self.position_reward = PositionReward(self.num_envs, self.cfg.position_distance_threshold) 
        self.orientation_reward = OrientationReward(self.num_envs, self.cfg.orientation_distance_threshold)
        self.reward_func_active = {}
        for key, value in self.cfg.reward_func[0].items():
            self.reward_func_active[key] = torch.tensor(value, device=self.device).repeat(self.num_envs)

        # metrics for task
        self._contacted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._successed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # subtasks
        self.lift_targets_ = torch.zeros((self.num_envs, self.cfg.subtask_num, 3), device=self.device) # 各个子任务中的物体目标位置
        self._subtask_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._task_playing = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # collect data
        self.collect_data = CollectData(
            cfg=self.yaml_cfg,
            task_name = "Pick-Place",
            num_env = self.num_envs,
            sim_f = 60.0 / self.cfg.decimation,
            unchecking=self.yaml_cfg["play"]["unchecking"]
        )

        if "front" in self.scene.tiled_cameras:
            self.front_camera = self.scene.tiled_cameras["front"]
        
        # Print information of this environment
        print("Num envs: ", self.num_envs)
        print("Num bodies: ", self._robot.num_bodies)
        print("Num arm dofs: ", self._arm_joint_num)
        print("Num hand dofs: ", 6)
        print("hand_base_rigid_body_index: ", self._hand_base_link_index)
        
        self.extras = {'standby_reward': 0.0, 'dist_reward': 0.0, 'pose_reward': 0.0, 'lift_reward': 0.0, 'angl_reward': 0.0, 'orient_reward': 0.0, 'act_penalty': 0.0, 'success': 0.0}
        
        # initialize wandb
        if self.cfg.enable_wandb: 
            self._wandb = WandbLog()
            project = "PsiLab_v2.0_RL"
            task = "Pick-Place"
            objects = "Cola"
            cudas = "#07"
            name = task + "_" + objects + "_" + cudas + "_" + datetime.strftime(datetime.now(), '%m%d_%H%M%S')
            tags = []
            tags.append("complete grasp")
            tags.append("position noise")
            tags.append("Minor Scale Random")
            tags.append("jerk penalty10")
            tags.append("distance reward scale 1.3")
            tags.append("decimation8")
            tags.append("material random")
            parent_dir = Path(__file__).resolve().parent
            source_dir = Path(__file__).resolve().parent.parent.parent.parent
            self._wandb.init_wandb(project, name, tags)
            self._wandb.init_artifact("train_model", "model")
            self._wandb.upload_artifacts_from_path("train_model", str(parent_dir))
            self._wandb.init_artifact("rl", "model")
            self._wandb.upload_artifacts_from_path("rl", source_dir / "algo")

        # initialize Timer
        self._timer = Timer()

        # initiallize output count
        self._output_count = 0
        
        # initialiize data replay
        if self.yaml_cfg["play"]["replay"]:
            saved_data = np.load(self.yaml_cfg["play"]["saved_data_file"])
            replay_data_type = self.yaml_cfg["play"]["replay_data_type"]
            self.r_action = saved_data[replay_data_type].reshape(-1, self.yaml_cfg["play"]["record_buffer"][replay_data_type])
            self.r_obs = saved_data["real_obs"].reshape(-1, 19)
            self.r_curr_targets = saved_data["current_target"].reshape(-1, self.cfg.true_action_space)
            self.r_dof_pos = saved_data["dof_pos"].reshape(-1, self.cfg.true_action_space)
            self.r_v_targets = saved_data["dof_vel"].reshape(-1, self.cfg.true_action_space)
            self.r_contact = saved_data["contact"].reshape(-1, 1)
   
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions[..., :self.cfg.true_action_space].clone()
        
        # end effctor type
        if self.yaml_cfg["policy"]["ee_type"] == "gripper":
            self.actions[..., 9] = self.actions[..., 8] # assign index finger action to middle finger
            self.actions[..., 7] = 1 # lock thumb rotation 

        # prepare action samples for eval dp
        if self.yaml_cfg["play"]["eval_dp"]:
            self.actions = self.actions[:,:self.cfg.action_chunk,:].clone()
            self.dp_step = 0
            self.dp_mini_step = 0
            self.dp_actions = self.actions.clone()
            self.dp_actions[...,self._arm_joint_num:] = scale(
                self.dp_actions[...,self._arm_joint_num:], 
                self._joint_limit_lower[:, self._hand_real_joint_index], 
                self._joint_limit_upper[:, self._hand_real_joint_index],
                origin_lower = 0,
                origin_upper = 1)
            self.dp_interpolate_num = self.cfg.decimation / self.cfg.action_chunk
            if self.cfg.contact_mode:
                self.contact_action = actions[..., self.cfg.true_action_space:self.cfg.action_space].clone()
                for i in range(self.dp_actions.shape[1]):
                    for j in range(self.dp_actions.shape[0]):
                        if self.contact_action[j, i, 0] >= 0.9:
                            self.dp_actions[j, i, [8,12]] /= 1.1
            action = self.dp_actions[:,self.dp_step,:].clone()
            self.actions = action
            self.actions[:, :self._arm_joint_num] = (action[:, :self._arm_joint_num] - self._robot.data.joint_pos[:, self._arm_joint_index]) / self.physics_dt / self.cfg.arm_dof_speed_scale / self.dp_interpolate_num
            self.actions[:, self._arm_joint_num:] = (action[:, self._arm_joint_num:] - self._robot.data.joint_pos[:, self._hand_real_joint_index]) / self.physics_dt / self.cfg.hand_dof_speed_scale / self.dp_interpolate_num
        
        self.prev_accelerate = self.curr_accelerate.clone()
        self.curr_accelerate = self.actions - self.prev_actions 
        self.prev_actions = self.actions.clone()
        
        # replace action with data in replay mode
        if self.yaml_cfg["play"]["replay"]:
            if self.yaml_cfg["play"]["replay_data_type"] in ["dof_pos", "current_target"]:
                r_action = to_torch(self.r_action[self.episode_length_buf], device=self.device)
                if self.cfg.contact_mode and self.r_contact[self.episode_length_buf] > 0:
                    r_action[[8,12]] /= 1.05
                # else:
                #     r_action[[8,12]] *= 1.1
                r_action[[8,12]] = saturate(r_action[[8,12]], to_torch(0), to_torch(1))
                r_action[self._arm_joint_num:] = scale(r_action[self._arm_joint_num:],
                                     self._joint_limit_lower[0, self._hand_real_joint_index], 
                                     self._joint_limit_upper[0, self._hand_real_joint_index],
                                     origin_lower = 0,
                                     origin_upper = 1)
                self.actions[:, :self._arm_joint_num] = (r_action[:self._arm_joint_num] - self._robot.data.joint_pos[:, self._arm_joint_index]) / self.physics_dt / self.cfg.arm_dof_speed_scale / self.cfg.decimation
                self.actions[:, self._arm_joint_num:] = (r_action[self._arm_joint_num:] - self._robot.data.joint_pos[:, self._hand_real_joint_index]) / self.physics_dt / self.cfg.hand_dof_speed_scale / self.cfg.decimation
            elif self.yaml_cfg["play"]["replay_data_type"] in ["action"]:
                action = to_torch(self.r_action[self.episode_length_buf*self.cfg.decimation], device=self.device)
                self.actions[:,:] = action
                       
    def _apply_action(self) -> None:
        '''        self.episodes += 1
        action range       : (-1,1)
        action index 0-6   : arm joint
        action index 7-12  : real hand joint (order: 1-1,2-1,3-1,4-1,5-1,1-2)(拇指旋转，食指弯折，中指弯折，无名指弯折，小拇指弯折， 拇指弯折)
        action index 13-17 : fake hand joint (order: 2-2,3-2,4-2,5-2,1-3)
        '''
        # replace action with dp downsample action when eval dp
        if self.yaml_cfg["play"]["eval_dp"]:
            self.dp_mini_step += 1
            if self.dp_mini_step > self.dp_interpolate_num:
                self.dp_mini_step = 1
                self.dp_step += 1
            action = self.dp_actions[:,self.dp_step,:].clone()
            self.actions = action.clone()
            self.actions[:, :self._arm_joint_num] = (action[:, :self._arm_joint_num] - self._robot.data.joint_pos[:, self._arm_joint_index]) / self.physics_dt / self.cfg.arm_dof_speed_scale / self.dp_interpolate_num
            self.actions[:, self._arm_joint_num:] = (action[:, self._arm_joint_num:] - self._robot.data.joint_pos[:, self._hand_real_joint_index]) / self.physics_dt / self.cfg.hand_dof_speed_scale / self.dp_interpolate_num 

        # culculate target position for each dof
        self._curr_targets[:, self._arm_joint_index] = saturate(
            self._robot.data.joint_pos[:, self._arm_joint_index] + self.actions[:, :self._arm_joint_num] * self.physics_dt * self.cfg.arm_dof_speed_scale,
            self._joint_limit_lower[:, self._arm_joint_index],
            self._joint_limit_upper[:, self._arm_joint_index]
        )
        self._curr_targets[:, self._hand_real_joint_index] = saturate(
            self._robot.data.joint_pos[:, self._hand_real_joint_index] + self.actions[:, self._arm_joint_num:] * self.physics_dt * self.cfg.hand_dof_speed_scale,
            self._joint_limit_lower[:, self._hand_real_joint_index],
            self._joint_limit_upper[:, self._hand_real_joint_index]
        )  
        self._prev_targets = self._curr_targets.clone()
        # set target position and velocity
        self._robot.set_joint_position_target(
            self._curr_targets[:, self._robot_index], joint_ids=self._robot_index
        )
        dof_vel = (self._curr_targets[:, self._robot_index] - self._robot.data.joint_pos[:, self._robot_index]) / self.physics_dt
        self._robot.set_joint_velocity_target(
            dof_vel, 
            joint_ids=self._robot_index
        )
        # generate passive target for fake hand joints
        self._gen_passive_target()
        if self.yaml_cfg["play"]["use_passive_target"]:
            self._robot.set_joint_position_target(
                self.passive_target, joint_ids=self._hand_virtual_joint_index
            )

    def _gen_passive_target(self):
        self.passive_target = self._curr_targets.clone()
        self.passive_target = norm(self.passive_target[:, self._robot_index], 
                                    self._joint_limit_lower[:, self._robot_index],
                                    self._joint_limit_upper[:, self._robot_index])
        self.passive_target = self.passive_target[:,8:] # 2-1,3-1,4-1,5-1,1-2 -> 2-2,3-2,4-2,5-2,1-3
        self.passive_target = scale(self.passive_target,
                                         self._joint_limit_lower[:, self._hand_virtual_joint_index],
                                         self._joint_limit_upper[:, self._hand_virtual_joint_index],
                                         origin_lower=0,
                                         origin_upper=1)
           
    def step(self, action):
        # call super step first to apply action and sim step
        obs_buf, reward_buf, reset_terminated, reset_time_outs, extras = super().step(action)
        
        self._marker_visualizer()

        return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras

    def _get_observations(self) -> dict:
        # implement fingertip force sensors

        self._check_shift_subtask()
        
        self._get_full_observations()
        observations = {"policy": self._obs, "critic":self._obs}
        if self.yaml_cfg["play"]["eval_dp"]:
            observations = {"policy": self._dp_obs, "critic":self._dp_obs}

        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        # cuclulate reward terms
        standby_active = self.reward_func_active['standby']
        position_active = self.reward_func_active['position']
        grasp_active = self.reward_func_active['grasp']
        orientation_active = self.reward_func_active['orientation']
        self.standby_reward.update(self._contact_states, self.hand_base_state[:, :3], self._hand_target_pose, standby_active)
        standby_reward = self.standby_reward.get_reward()
        self.grasp_reward.update(self.hand_state[:, :7, :3], self._grasp_point, grasp_active)
        grasp_reward, distance_reward, pose_reward, angle_reward = self.grasp_reward.get_reward()
        self.position_reward.update(self._target_lift_pose,  self._target_subtask_init_pose[:, :3], self.object_state[:, :3], position_active, pose_reward)
        lift_reward = self.position_reward.get_reward()
        self.orientation_reward.update(self._target_orien_pose, self.object_state[:, 3:7], standby_reward + grasp_reward + lift_reward, orientation_active)
        orientation_reward = self.orientation_reward.get_reward()
        action_penalty = _compute_jerk_penalty(self.curr_accelerate, self.prev_accelerate, torch.abs(standby_reward + distance_reward + orientation_reward))
        
        # penalty dropping down objects
        task_failed_env_ids = self._is_task_failed(pose_reward, angle_reward)
        if len(task_failed_env_ids) > 0:
             action_penalty[task_failed_env_ids] = action_penalty[task_failed_env_ids] + 500 # penalty for task failed
        self._accumulated_lift_reward += (lift_reward - self._pre_lift_reward)
        total_reward = (standby_reward + distance_reward + pose_reward + lift_reward + angle_reward + orientation_reward - self._pre_energy) - action_penalty
        self.extras['standby_reward'] += (standby_reward.mean() - self._pre_standby_reward.mean())  # type: ignore
        self.extras['dist_reward'] += (distance_reward.mean() - self._pre_distance_reward.mean())  # type: ignore
        self.extras['pose_reward'] += (pose_reward.mean() - self._pre_pose_reward.mean())# type: ignore
        self.extras['lift_reward'] += (lift_reward.mean() - self._pre_lift_reward.mean())# type: ignore
        self.extras['angl_reward'] += (angle_reward.mean() - self._pre_angle_reward.mean())# type: ignore
        self.extras['orient_reward'] += (orientation_reward.mean() - self._pre_orientation_reward.mean())# type: ignore
        self.extras['act_penalty'] += action_penalty.mean() # type: ignore

        # update pre reward
        self._pre_standby_reward = standby_reward
        self._pre_distance_reward = distance_reward
        self._pre_pose_reward = pose_reward
        self._pre_lift_reward = lift_reward
        self._pre_angle_reward = angle_reward
        self._pre_orientation_reward = orientation_reward
        self._pre_energy = standby_reward + distance_reward + pose_reward + lift_reward + angle_reward + orientation_reward
        if self.cfg.print_infos:
            print(cp.blue(f"total_reward:{total_reward.mean().item():.4f},standby_reward:{standby_reward.mean().item():.4f},distance_reward:{distance_reward.mean().item():.4f},pose_reward:{pose_reward.mean().item():.4f},lift_reward:{lift_reward.mean().item():.4f},angle_reward:{angle_reward.mean().item():.4f},orientation_reward:{orientation_reward.mean().item():.4f},action_penalty:{action_penalty.mean().item():.4f}"))
        
        # collect data
        if self.collect_data.is_collect:
            episode_length_buf = int(self.episode_length_buf[0] - 1)
            self.collect_data.save_to_buffer("full_observation", episode_length_buf, self._obs.clone().cpu().numpy())
            self.collect_data.save_to_buffer("action", episode_length_buf, self.actions.clone().cpu().numpy())
            self.collect_data.save_to_buffer("current_target", episode_length_buf, self._curr_targets[:, self._robot_index].clone().cpu().numpy())
            dof_pos = self.dof_pos.clone()
            if self.yaml_cfg["play"]["for_real"]:
                dof_pos[:, 7:] = unscale(dof_pos[:, 7:], 
                                        self._joint_limit_lower[0, self._hand_real_joint_index], 
                                        self._joint_limit_upper[0, self._hand_real_joint_index],
                                        target_lower = 0,
                                        target_upper = 1)
            self.collect_data.save_to_buffer("dof_pos", episode_length_buf, dof_pos.clone().cpu().numpy())
            self.collect_data.save_to_buffer("dof_vel", episode_length_buf, self.dof_vel.clone().cpu().numpy())
            self.collect_data.save_to_buffer("passive_target", episode_length_buf, self.passive_target.clone().cpu().numpy())
            self.collect_data.save_to_buffer("passive_dof", episode_length_buf, self.passive_dof.clone().cpu().numpy())
            self.collect_data.save_to_buffer("reward", episode_length_buf, total_reward.unsqueeze(-1).clone().cpu().numpy())
            self.collect_data.save_to_buffer("subtask_index", episode_length_buf, self._subtask_index.unsqueeze(-1).clone().cpu().numpy())
            self.collect_data.save_to_buffer("real_obs", episode_length_buf, self._real_obs.clone().cpu().numpy())
            if self.yaml_cfg["policy"]["ee_type"] == "gripper":
                joint_is_contact = (self._contact_states != 0).any(dim=-1)  # shape: (num_envs, 7)
                thumb_is_contact = (joint_is_contact[:, 0:3]).any(dim=-1)  # shape: (num_envs,)
                index_is_contact = (joint_is_contact[:, 3:5]).any(dim=-1)  # shape: (num_envs,)
                middle_is_contact = (joint_is_contact[:, 5:7]).any(dim=-1)
                all_fingers_contact = (thumb_is_contact & (index_is_contact | middle_is_contact)).to(torch.bool)  # shape: (num_envs,)
            self.collect_data.save_to_buffer("contact", episode_length_buf, all_fingers_contact.unsqueeze(-1).clone().cpu().numpy())
            if hasattr(self, "front_camera") and self.yaml_cfg["play"]["is_collect_camera"]:
                if self.yaml_cfg["play"]["record_video"]["rgb"]:
                    img_obs = self.front_camera.data.output["rgb"]
                    frame = img_obs.clone().cpu().numpy()
                    frame = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frame])
                    self.collect_data.save_to_buffer("rgb", episode_length_buf, frame)
                if self.yaml_cfg["play"]["record_video"]["colored_depth"]:
                    depth_obs = self.front_camera.data.output["depth"]
                    frame = depth_obs.clone().cpu().numpy()
                    frame = (frame - frame.min(axis=(1,2), keepdims=True)) / \
                            (frame.max(axis=(1,2), keepdims=True) - frame.min(axis=(1,2), keepdims=True) + 1e-8)
                    cmap = matplotlib.colormaps.get_cmap('Spectral')
                    frame = frame.reshape(-1, 640, 640)
                    frame_colored = np.array([cmap(d)[:, :, :3] for d in frame])  # (B, H, W, 3)
                    depth_bgr = (frame_colored[..., ::-1] * 255).astype(np.uint8)
                    self.collect_data.save_to_buffer("colored_depth", episode_length_buf, depth_bgr)
                if self.yaml_cfg["play"]["record_video"]["depth"]:
                    depth_obs = self.front_camera.data.output["depth"]
                    frame = depth_obs.clone().cpu().numpy()
                    frame = np.array([cv2.convertScaleAbs(f, alpha=255.0 / f.max()) for f in frame])
                    frame = 255 - frame
                    self.collect_data.save_to_buffer("depth", episode_length_buf, frame)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.async_reset:
            bfailed, self._contacted = eval_fail(self._target,self._contact_sensors,self._contacted)
            # self._successed = eval_success(self._target, self._contact_sensors, self.cfg.lift_height_target)
            resets = self._successed & bfailed
        else:
            resets = torch.zeros(self.num_envs, device=self.device)
        
        # self._successed = (self._target.data.root_pos_w[:, 2] - self._target_init_pose[:, 2]) >= self.cfg.lift_height_target * 0.8
        self._successed = self._is_success(self.cfg.subtask_num - 1)
        self.extras['success'] = self._successed.float().mean().item() * 100.0
        
        return resets, time_out

    def _is_success(self, subtask_index: int) -> torch.Tensor:
        return self._subtask_index >= subtask_index

    def _is_subtask_success(self, target_pos: torch.Tensor, object_pos: torch.Tensor, ) -> torch.Tensor:
        
        grasp_success = self.grasp_reward.get_success()
        
        standby_success = self.standby_reward.get_success()
        
        position_success = self.position_reward.get_success()
        
        orientation_success = self.orientation_reward.get_success()
        
        return grasp_success & standby_success & position_success & orientation_success

    def _is_task_failed(self, pose_reward: torch.Tensor, angle_reward: torch.Tensor) -> torch.Tensor:
        task_failed_env_ids = (self.object_state[:, 2] - self._target_init_pose[:, 2] > 0.02) & ((pose_reward < 4.0) | (angle_reward < 2.6)) & (self._subtask_index > 0) & self.reward_func_active['grasp']
        task_failed_env_ids = torch.nonzero(task_failed_env_ids, as_tuple=False).squeeze(-1)
        return task_failed_env_ids

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if self.cfg.enable_output and self._data is not None:
            # get index of envs will save data
            # asynchronous reset
            if self.cfg.async_reset:
                env_save_list=[]
                for env_index in env_ids:
                    if not self.reset_time_outs[env_index]:
                        env_save_list.append(env_index)
                
            # synchronous reset
            else:
                env_save_list = []
                for i in range(self.scene.num_envs):
                    delta_z = self._target.data.root_com_pos_w[i,2] - self._target_init_pose[i,2]
                    if self._is_success()[i]:
                    # if abs(delta_z - self.cfg.lift_height_target)<0.1:
                        env_save_list.append(i)

            # save data 
            # print(env_save_list)
            if len(env_save_list)>0:
                print(env_save_list)
                # single env
                if self.scene.num_envs == 1:
                    # save_data(self._data,self.cfg)
                    pass
                # multi env
                elif self.scene.num_envs >1:
                    pass
                else:
                    raise Exception(f"Save Data Error as {self.scene.num_envs} is incorrect") 

                self._output_count += len(env_save_list)
                # record_time = self._timer.run_time() /60.0
                # record_rate = self._output_count / record_time
                #   
                # print(f"采集时长: {record_time} 分钟")
                # print(f"采集数据: {self._output_count} 条")
                # print(f"采集效率: {record_rate} 条/分钟")
            #
            record_time = self._timer.run_time() /60.0
            print(f"时长: {record_time} 分钟")
            record_rate = self._episodes / record_time
            print(f"效率: {record_rate} 条/分钟")
            print(f"成功条数/总条数: {self._output_count}/{self._episodes} ")

        # update episodes
        # TODO: change this part code for self._episodes += len(env_ids)
        if self.cfg.async_reset:
            self._episodes += 1
        else:
            self._episodes += self.num_envs
        
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES # type: ignore
        super()._reset_idx(env_ids) # type: ignore
        
        ############ reset robot ################
        dof_pos = self._robot.data.default_joint_pos.clone()
        dof_pos[..., 19] = self._joint_limit_upper[..., 19]
        dof_vel = self._robot.data.default_joint_vel.clone()
        self._curr_targets[env_ids, :] = dof_pos[env_ids, :]
        self._prev_targets[env_ids, :] = dof_pos[env_ids, :]
        # TODO: check api params
        self._robot.set_joint_position_target(dof_pos, env_ids=env_ids) # type: ignore
        self._robot.set_joint_velocity_target(dof_vel, env_ids=env_ids) # type: ignore
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids) # type: ignore
        
        ############ reset object ################
        # reset_position_noise = 0.00
        # lego_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device) * reset_position_noise
        # lego_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        # object_defaut_state = self._target.data.root_com_state_w[env_ids].clone()
        # object_defaut_state[:, 0:3].add_(lego_pos_noise)
        # # object_defaut_state[:, 0:3].add_(self.scene.env_origins[env_ids])
        # object_defaut_state[:, 3:7] = randomize_rotation(
        #     lego_rot_noise[:, 0], lego_rot_noise[:, 1], self._x_unit_tensor[env_ids], self._y_unit_tensor[env_ids]
        # )
        # object_defaut_state[:, 7:] = torch.zeros_like(self._target.data.root_com_state_w[env_ids, 7:])
        # self._target.write_root_link_pose_to_sim(object_defaut_state[:, :7], env_ids)    # type: ignore
        # self._target.write_root_com_velocity_to_sim(object_defaut_state[:, 7:], env_ids) # type: ignore
        
        for _ in range(30):
            self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids) # type: ignore
            self.sim.step(render=True)
            self.scene.update(dt=self.physics_dt)

        ############ last logs ################
        # wandb log
        if self.cfg.enable_wandb:
            self._wandb.set_data("subtask_index", self._subtask_index.float().mean().item())
            self._wandb.set_data("success", self._successed.float().mean().item() * 100.0)
            self._wandb.set_data("place_success", self._is_success(3).float().mean().item() * 100.0)
            self._wandb.set_data("grasp_success", self._is_success(2).float().mean().item() * 100.0)
            self._wandb.set_data("standby_reward", self.extras['standby_reward'])
            self._wandb.set_data("dist_reward", self.extras['dist_reward'])
            self._wandb.set_data("pose_reward", self.extras['pose_reward'])
            self._wandb.set_data("lift_reward", self.extras['lift_reward'])
            self._wandb.set_data("angl_reward", self.extras['angl_reward'])
            self._wandb.set_data("orient_reward", self.extras['orient_reward'])
            self._wandb.set_data("act_penalty", self.extras['act_penalty'])
            self._wandb.upload_all()
        
        # print info
        if self.cfg.env_id_print_data in env_ids and self.common_step_counter > 0:
            reward_items = ['standby_reward', 'dist_reward', 'pose_reward', 'lift_reward', 'angl_reward', 'orient_reward', 'act_penalty']
            total_reward = sum([abs(self.extras[item]) for item in reward_items])

            print("\n")
            print("#" * 17, " Statistics", "#" * 17)
            # print(f"env id:   {self.cfg.env_id_print_data}")
            print(f"subtask_index: {self._subtask_index.float().mean().item()}")
            print(f"success:  {self._successed.float().mean().item()*100.0:.2f}%")
            print(f"place_success:  {self._is_success(3).float().mean().item() * 100.0:.2f}%")
            print(f"grasp_success:  {self._is_success(2).float().mean().item() * 100.0:.2f}%")
            print(f"standby_reward:   {self.extras['standby_reward']:.2f} ({(abs(self.extras['standby_reward']) / total_reward * 100):.2f}%)")
            print(f"dist_reward:      {self.extras['dist_reward']:.2f} ({(abs(self.extras['dist_reward']) / total_reward * 100):.2f}%)")
            print(f"angle_reward:     {self.extras['angl_reward']:.2f} ({(abs(self.extras['angl_reward']) / total_reward * 100):.2f}%)")
            print(f"pose_reward:      {self.extras['pose_reward']:.2f} ({(abs(self.extras['pose_reward']) / total_reward * 100):.2f}%)")
            print(f"lego_up_reward:   {self.extras['lift_reward']:.2f} ({(abs(self.extras['lift_reward']) / total_reward * 100):.2f}%)")
            print(f"orient_reward:    {self.extras['orient_reward']:.2f} ({(abs(self.extras['orient_reward']) / total_reward * 100):.2f}%)")
            print(f"action_penalty:   {self.extras['act_penalty']:.2f} ({(abs(self.extras['act_penalty']) / total_reward * 100):.2f}%)")
            print(f"total_reward:     {sum([self.extras[item] for item in reward_items]):.2f}")
            print("#" * 15, "Statistics End", "#" * 15,"\n")
            
            self.extras = {'standby_reward': 0, 'dist_reward': 0, 'pose_reward': 0, 'lift_reward': 0, 'angl_reward': 0, 'orient_reward': 0, 'act_penalty': 0, 'success': 0}

        # reset object pose, properties, observations, reward terms
        self._target_init_pose[env_ids, :7] = self._target.data.root_com_state_w[env_ids, :7].clone()
        self._target_init_pose[env_ids, :3] -= self.scene.env_origins[env_ids]
        self._target_orien_pose[env_ids] = self._target_init_pose[env_ids, 3:7].clone()
        self._accumulated_lift_reward[env_ids] = torch.zeros(len(env_ids), device=self.device, dtype=torch.float)
        self._contact_states[env_ids] = torch.zeros((len(env_ids), 7, 3), device=self.device)

        # compute reward
        self._compute_intermediate_values()

        self._hand_target_pose = torch.zeros_like(self._hand_init_pose)
        self._reset_task_target(env_ids)

        self._init_highest_point_offset[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        delta_z =  ((self.object_state[env_ids, 2:3] - self.cfg.table_height) * self.cfg.grasp_offset).squeeze(1)
        self._init_highest_point_offset[env_ids] = init_heighest_offset_local(self.object_state[env_ids, 3:7], delta_z)
        self._grasp_point[env_ids] = torch.zeros((len(env_ids), 3), device=self.device)
        self._grasp_point[env_ids] = grasp_point(self.object_state[env_ids, :3], self.object_state[env_ids, 3:7], self._init_highest_point_offset[env_ids])

        self._get_initial_reward()
        
        self._object_lwh[env_ids] = self._bbox_LWH(env_ids)
        self._staticFriction[env_ids], self._dynamicFriction[env_ids], self._restitution[env_ids], self._mass[env_ids] = self._get_attr(env_ids)
        
        if self.collect_data.is_collect:
            self.collect_data.try_save_data_to_file()

        self._dp_obs = torch.zeros((self.num_envs, self.yaml_cfg["play"]["dp_obs_horizon"], self.yaml_cfg["play"]["record_buffer"]["real_obs"]), device=self.device)
        self._dp_obs_cold_start = 0
        
    def _reset_lift_targets(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)
        init_pos = self._target_init_pose[env_ids, :3]
        final_pos = torch.tensor(self.cfg.base_lift_targets, device=self.device).repeat(num_envs, 1)
        final_pos[:, 2] = init_pos[:, 2]
        end_pos = torch.zeros(num_envs, 3, device=self.device)
        if self.cfg.random_lift_targets:
            random_offsets = torch.rand(num_envs, 3, device=self.device) * 2 - 1  # [-1, 1] 范围
            random_offsets = random_offsets * torch.tensor(self.cfg.targets_range, device=self.device)
            final_pos = final_pos + random_offsets   
        mid_targets = (init_pos + final_pos) / 2
        mid_targets[:, 2] += 0.1
        self._final_target_pose[env_ids] = final_pos
        init_pos = init_pos.unsqueeze(1)
        mid_targets = mid_targets.unsqueeze(1)
        final_pos = final_pos.unsqueeze(1)
        end_pos = end_pos.unsqueeze(1)
        self.lift_targets_[env_ids] = torch.cat([init_pos, mid_targets, final_pos, final_pos, end_pos], dim=1)
        
    def _reset_task_target(self, env_ids: torch.Tensor):
        num_envs = len(env_ids)
        for key, value in self.cfg.reward_func[0].items():
            self.reward_func_active[key][env_ids] = torch.tensor(value, device=self.device).repeat(num_envs)
        self._subtask_index[env_ids] = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self._reset_lift_targets(env_ids)
        self._target_lift_pose[env_ids] = self.lift_targets_[env_ids,0]
        self._target_subtask_init_pose[env_ids] = self._target.data.root_com_state_w[env_ids,:7].clone()
        self._target_subtask_init_pose[env_ids, :3] -= self.scene.env_origins[env_ids]
        self._hand_init_pose[env_ids] = self.hand_base_state[env_ids, :3].clone()

    def _check_shift_subtask(self):
        """
        1.检查任务是否已经失败，若失败则重置任务
        2.检查当前子任务是否完成
        3.如果完成，则更新子任务，包括任务目标和reward fucntion
        """
        # reset task if failed
        task_failed_env_ids = self._is_task_failed(self._pre_pose_reward, self._pre_angle_reward)
        if len(task_failed_env_ids) > 0:
            self._reset_task_target(task_failed_env_ids)
            self._get_initial_reward()
        # shift subtask if finished
        self.subtask_finished = self._is_subtask_success(self._target_lift_pose[:, :3], self.object_state[:, :3])
        if torch.any(self.subtask_finished):
            self.subtask_finished = self.subtask_finished & self._task_playing
            task_to_next_mask = (self._subtask_index < self.cfg.subtask_num - 1) & self.subtask_finished
            # update subtask status
            self._subtask_index[task_to_next_mask] += 1
            self._task_playing = self._subtask_index < self.cfg.subtask_num - 1
            # update lift target and subtask init pose
            task_shift_mask = self._task_playing & task_to_next_mask
            if torch.any(task_shift_mask):
                env_indices = torch.where(task_shift_mask)[0]
                subtask_indices = self._subtask_index[env_indices]     
                self._target_lift_pose[task_shift_mask] = self.lift_targets_[env_indices, subtask_indices] 
                self._target_subtask_init_pose[task_shift_mask] = self._target.data.root_com_state_w[task_shift_mask, :7].clone()
                self._target_subtask_init_pose[task_shift_mask, :3] -= self.scene.env_origins[task_shift_mask]
            # update reward function active and hand target pose
            if torch.any(task_to_next_mask):
                indices = self._subtask_index[task_to_next_mask]   
                for key, _ in self.cfg.reward_func[0].items():
                    reward_active = torch.tensor([self.cfg.reward_func[idx][key] for idx in indices], device=self.device)
                    self.reward_func_active[key][task_to_next_mask] = reward_active
                
                standby_task_mask = self.reward_func_active['standby']
                self._hand_target_pose = torch.where(
                    standby_task_mask.unsqueeze(1).expand_as(self._hand_init_pose),
                    self._hand_init_pose,
                    torch.zeros_like(self._hand_init_pose)
                )
                # reset pre energy
                self._get_initial_reward()
                if self.cfg.print_infos :
                    print(cp.green(f"env 0 subtask shift {self._subtask_index[0] - 1} to {self._subtask_index[0]}"))
            
    def _compute_intermediate_values(self):
        self._hand_index = self._finger_joint_index + [self._hand_base_link_index]
        (
            self.object_state,
            self.finger_thumb_state,
            self.finger_index_state,
            self.finger_middle_state,
            self.middle_point_12,
            self.middle_point_13,
            self.hand_base_state,
            self.hand_state,
            self.dof_pos,
            self.dof_vel,
            self.passive_dof,
        ) = _compute_values(
            self.scene.env_origins,
            self._target.data.root_com_state_w.clone(),
            self._robot.data.body_link_state_w[:, self._hand_index].clone(),
            self._robot_index,
            self._hand_virtual_joint_index,
            self._robot.data.joint_pos,
            self._robot.data.joint_vel,
        )
        self._contact_states = torch.stack(
            (
                self._contact_sensors["hand2_link_1_1"].data.net_forces_w[:,0],
                self._contact_sensors["hand2_link_1_2"].data.net_forces_w[:,0],
                self._contact_sensors["hand2_link_1_3"].data.net_forces_w[:,0],
                self._contact_sensors["hand2_link_2_1"].data.net_forces_w[:,0],
                self._contact_sensors["hand2_link_2_2"].data.net_forces_w[:,0],
                self._contact_sensors["hand2_link_3_1"].data.net_forces_w[:,0],
                self._contact_sensors["hand2_link_3_2"].data.net_forces_w[:,0],
            ),
            dim=1,
        )
    
    def _get_full_observations(self):
        self._grasp_point = grasp_point(self.object_state[:, :3], self.object_state[:, 3:7], self._init_highest_point_offset)
        self.finger_normal_vec = quat_rotate_vector(self.hand_state[:,:3,3:7], torch.tensor([1., 0., 0.],device=self.device))
        self.fingertip_point =  self.object_state[:, :3].unsqueeze(1) - self.hand_state[:,:3,:3]
        cos_theta = (self.finger_normal_vec * self.fingertip_point).sum(-1) 
        reward_function_active_states = torch.cat([v.to(torch.float32).unsqueeze(1) for v in self.reward_func_active.values()], dim=-1)

        # add noise to object position observation
        noised_object_state = self.object_state[:, :7].clone()
        noised_grasp_ponit = self._grasp_point[:, :3]
        if self.cfg.position_obs_noise:
            noised_object_state[:, :3] = add_noise(noised_object_state[:, :3], [0, 1, 2], self.cfg.position_obs_noise_range, self.cfg.position_obs_noise_type)
            noise = noised_object_state[:, :3] - self.object_state[:, :3]
            noised_grasp_ponit = self._grasp_point[:, :3] + noise

        ### teacher version
        self._obs = torch.cat(
                (   
                    ### condition ###
                    # # 物体长宽高
                    # self._object_lwh,
                    # # 物体静摩擦系数
                    # self._staticFriction.reshape(self.num_envs, -1),
                    # # 物体动摩擦系数
                    # self._dynamicFriction.reshape(self.num_envs, -1),
                    # # 物体恢复系数
                    # self._restitution.reshape(self.num_envs, -1),

                    ### observation ###
                    # robot state
                    # 关节角度13个
                    unscale(self.dof_pos, self._joint_limit_lower[:, self._robot_index], self._joint_limit_upper[:, self._robot_index]),
                    # 关节速度
                    # self.cfg.vel_obs_scale * self.dof_vel,

                    # object state
                    # 物体位置、旋转7个
                    noised_object_state,
                    # 手位置 6*7=42个
                    self.hand_state[:, :, :7].reshape(self.num_envs, -1),
                    # 物体速度 7个
                    # self.cfg.vel_obs_scale * self.object_state[:, 7:],
                    # reward function active 4*1=4个
                    reward_function_active_states,
                    
                    # goal
                    # 物体目标位置 3个
                    self._target_lift_pose[:, :3],
                    # 物体目标旋转 4个
                    self._target_orien_pose[:, :4],
                    # 手目标位置 3个
                    self._hand_target_pose[:, :3],
                    
                    # fingertips
                    # 目标抓取位置
                    noised_grasp_ponit,
                    # 手与目标位置差 6*3=18个
                    (self.hand_state[:, :, :3] - noised_grasp_ponit[:, None, :3]).reshape(self.num_envs, -1),
                    # 手速度
                    # self.cfg.vel_obs_scale * self.hand_state[:, :, 3:].reshape(self.num_envs, -1),
                    # # 指尖法向量夹角cos值 3个
                    cos_theta,
                    # 接触力 3*3=9个
                    # self._contact_states.reshape(self.num_envs, -1),
                    # 指尖中点位置 2*3=6个
                    self.middle_point_12[:, :3],
                    self.middle_point_13[:, :3],

                    # actions
                    # 速度13个
                    self.actions,
                    # 加速度13个
                    self.curr_accelerate,
                ),
                dim=-1,
            )
        dof_pos = self.dof_pos.clone()
        dof_pos[:,7:] = unscale(dof_pos[:,7:], 
                                self._joint_limit_lower[:, self._hand_real_joint_index], 
                                self._joint_limit_upper[:, self._hand_real_joint_index],
                                target_lower = 0,
                                target_upper = 1)
        self._real_obs = torch.cat(
                (   
                    ### observation ###
                    # robot state
                    # 关节角度13个
                    dof_pos.reshape(self.num_envs, -1),

                    # object state
                    # 物体位置3个
                    noised_object_state[:, :3],

                    # goal
                    # 物体目标位置 3个
                    self._final_target_pose[:, :3],
                    
                ),
                dim=-1,
            )
        # replace real obs with replay buffer
        if self.yaml_cfg["play"]["replay"] and self.yaml_cfg["play"]["replay_data_type"] in ["real_obs"]:
            self._real_obs = to_torch(self.r_obs[self.episode_length_buf], device=self.device).unsqueeze(0)
        # build dp obs
        if self.yaml_cfg["play"]["dp_obs_horizon"] == 1:
            self._dp_obs = self._real_obs.unsqueeze(1).clone()
            self._dp_obs_cold_start = 1
        elif self.yaml_cfg["play"]["dp_obs_horizon"] > 1:
            self._dp_obs = self._real_obs.unsqueeze(1).repeat([1,8,1]).clone()
            # self._dp_obs = torch.roll(self._dp_obs, shifts=-1, dims=1)
            # self._dp_obs[:, -1, :] = self._real_obs.clone()
            self._dp_obs_cold_start += 1
            self._dp_obs_cold_start = min(self._dp_obs_cold_start, self.yaml_cfg["play"]["dp_obs_horizon"])
        else:
            raise ValueError("dp_obs_horizon can not smaller than 1")

    def _get_initial_reward(self):
        standby_active = self.reward_func_active['standby']
        position_active = self.reward_func_active['position']
        grasp_active = self.reward_func_active['grasp']
        orientation_active = self.reward_func_active['orientation']
        self.standby_reward.update(self._contact_states, self.hand_base_state[:, :3], self._hand_target_pose, standby_active)
        standby_reward = self.standby_reward.get_reward()
        self.grasp_reward.update(self.hand_state[:, :7, :3], self._grasp_point, grasp_active)
        grasp_reward, distance_reward, pose_reward, angle_reward = self.grasp_reward.get_reward()
        self.position_reward.update(self._target_lift_pose,  self._target_subtask_init_pose[:, :3], self.object_state[:, :3], position_active, pose_reward)
        lift_reward = self.position_reward.get_reward()
        self.orientation_reward.update(self._target_orien_pose, self.object_state[:, 3:7], standby_reward + grasp_reward + lift_reward, orientation_active)
        orientation_reward = self.orientation_reward.get_reward()
        self._pre_standby_reward = standby_reward
        self._pre_distance_reward = distance_reward
        self._pre_pose_reward = pose_reward
        self._pre_angle_reward = angle_reward
        self._pre_lift_reward = lift_reward
        self._pre_orientation_reward = orientation_reward
        self._pre_energy = standby_reward + distance_reward + pose_reward + angle_reward + lift_reward + orientation_reward
    
    def _marker_visualizer(self):
        if self.cfg.enable_marker and self._visualizer is not None:
            finger_jonit_link_0 = self._robot.data.body_link_state_w[:,self._finger_joint_index[0],:]
            finger_jonit_link_1 = self._robot.data.body_link_state_w[:,self._finger_joint_index[1],:]
            finger_jonit_link_2 = self._robot.data.body_link_state_w[:,self._finger_joint_index[2],:]
            finger_jonit_link_3 = self._robot.data.body_link_state_w[:,self._finger_joint_index[3],:]
            finger_jonit_link_4 = self._robot.data.body_link_state_w[:,self._finger_joint_index[4],:]
            finger_jonit_link_5 = self._robot.data.body_link_state_w[:,self._finger_joint_index[5],:]
            finger_jonit_link_6 = self._robot.data.body_link_state_w[:,self._finger_joint_index[6],:]
            target_state = self._target.data.root_com_state_w[:,:]
            grasp_point = self._grasp_point[:,:]
            # refresh visualize and marker
            marker_pos = torch.cat((
                self._final_target_pose[0:1,:3], # 目标位置
                finger_jonit_link_0[0:1,:3],
                finger_jonit_link_1[0:1,:3],
                finger_jonit_link_2[0:1,:3],
                finger_jonit_link_3[0:1,:3],
                finger_jonit_link_4[0:1,:3],
                finger_jonit_link_5[0:1,:3],
                finger_jonit_link_6[0:1,:3],
                target_state[0:1,:3],
                grasp_point[0:1,:3],
                # grasp_fingers_pos[0:1,:3]
                ),0)
            
            marker_rot = torch.cat((
                torch.tensor([[1., 0., 0., 0.]], device=self.device),
                finger_jonit_link_0[0:1,3:7],
                finger_jonit_link_1[0:1,3:7],
                finger_jonit_link_2[0:1,3:7],
                finger_jonit_link_3[0:1,3:7],
                finger_jonit_link_4[0:1,3:7],
                finger_jonit_link_5[0:1,3:7],
                finger_jonit_link_6[0:1,3:7],
                target_state[0:1,3:7],
                target_state[0:1,3:7],
                # torch.zeros((1,4),device=self.device)
                ),0)

            self._visualizer.visualize(marker_pos, marker_rot)
    
    def _get_attr(self, env_ids):
        def get_prim_attr(prim_path: str, attr_name: str):
            prim = prim_utils.get_prim_at_path(prim_path)
            attr_value = prim.GetAttribute(attr_name).Get()
            return attr_value
        prim_paths = [f"/World/envs/env_{i}/Target/physics_material" for i in env_ids]
        staticFriction = torch.tensor([get_prim_attr(p, "physics:staticFriction") for p in prim_paths], device="cuda", dtype=torch.float32)
        dynamicFriction = torch.tensor([get_prim_attr(p, "physics:dynamicFriction") for p in prim_paths], device="cuda", dtype=torch.float32)
        restitution = torch.tensor([get_prim_attr(p, "physics:restitution") for p in prim_paths], device="cuda", dtype=torch.float32)
        prim_paths = [f"/World/envs/env_{i}/Target/base_link" for i in env_ids]
        mass = torch.tensor([get_prim_attr(p, "physics:mass") for p in prim_paths], device="cuda", dtype=torch.float32) # auto-computed mass will get zero
        
        return staticFriction, dynamicFriction, restitution, mass
        
    def _bbox_LWH(self, env_ids):
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                    [UsdGeom.Tokens.default_],
                                     useExtentsHint=True)   # 只建一次
        
        def world_aabb_size(prim_path: str) -> float:
            bbox = bbox_cache.ComputeLocalBound(
                stage.GetPrimAtPath(prim_path)
            ).ComputeAlignedRange()
            return bbox.GetSize()   
    
        env_paths = [f"/World/envs/env_{i}/Target" for i in env_ids]
        sizes   = torch.tensor([world_aabb_size(p) for p in env_paths],
                                device="cuda", dtype=torch.float32)
        
        return sizes
    
def quat_to_euler_xyz(q, *, degrees=True):
    """
    四元数 [x, y, z, w]  →  (roll, pitch, yaw)  (XYZ 顺序)
    默认返回角度制；将 degrees=False 可改返回弧度。
    """
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    else:
        q = np.asarray(q, dtype=float)
    # ---- 单位化 ----
    q = q / np.linalg.norm(q)

    x, y, z, w = q
    # 旋转矩阵分量
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    # 根据 XYZ (roll-pitch-yaw) 推导出的解析式
    pitch = np.arcsin(-2 * (xz - wy))              # ∈ [-π/2, π/2]
    roll  = np.arctan2( 2 * (yz + wx), 1 - 2*(yy + zz))
    yaw   = np.arctan2( 2 * (xy + wz), 1 - 2*(xx + zz))

    if degrees:
        return np.degrees([roll, pitch, yaw])
    return roll, pitch, yaw


def euler_from_quat(q):
    """包装 API：输入四元数，直接打印欧拉角（度）。"""
    roll, pitch, yaw = quat_to_euler_xyz(q, degrees=True)
    return f"Euler XYZ (deg) → roll: {roll:.2f},  pitch: {pitch:.2f},  yaw: {yaw:.2f}"

# helper functions
@torch.jit.script
def add_noise(
    obs: torch.Tensor,
    dim: List[int],
    range_list: List[float],
    noise_type: str = "uniform"
) -> torch.Tensor:
    """
    通用加噪声函数。

    Args:
        obs (Tensor): 任意形状的观测张量，例如 (num_env, obs_dim)
        dim (List[int]): 需要加噪声的最后一维的索引列表，例如 [0, 1, 2]
        range_list (List[float]): 对应每个 dim 的噪声范围，例如 [0.01, 0.02, 0.01]
        noise_type (str): 'uniform' 或 'gaussian'

    Returns:
        Tensor: 加噪后的 obs
    """
    obs_noised = obs.clone()
    obs_dim = obs.size(-1)

    assert len(dim) == len(range_list), "dim 和 range_list 长度必须一致"
    assert obs_dim >= max(dim) + 1, "dim 超出 obs 的最后一维范围"

    for i in range(len(dim)):
        d = dim[i]
        r = range_list[i]
        if noise_type == "uniform":
            noise = (torch.rand_like(obs_noised[..., d]) * 2.0 - 1.0) * r
        elif noise_type == "gaussian":
            noise = torch.randn_like(obs_noised[..., d]) * r
        else:
            raise ValueError("noise_type 必须是 'uniform' 或 'gaussian'")
        obs_noised[..., d] += noise

    return obs_noised

@torch.jit.script
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数 (w,x,y,z) 转为旋转矩阵。
    参数
    ----
    q : Tensor[..., 4]

    返回
    ----
    R : Tensor[..., 3, 3]
    """
    eps: float = 1e-8
    # --- 归一化 ---
    norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    q = q / torch.clamp(norm, min=eps)

    # 拆分分量（TorchScript 不支持一次性“多变量解包”）
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # --- 预计算重复项 ---
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    two: float = 2.0  # 方便和论文公式对照

    # 每个元素都是 [...]-shaped Tensor
    r00 = 1.0 - two * (yy + zz)
    r01 = two * (xy - wz)
    r02 = two * (xz + wy)

    r10 = two * (xy + wz)
    r11 = 1.0 - two * (xx + zz)
    r12 = two * (yz - wx)

    r20 = two * (xz - wy)
    r21 = two * (yz + wx)
    r22 = 1.0 - two * (xx + yy)

    # --- 组装旋转矩阵 ---
    R = torch.stack((
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1)
        ), dim=-2)          # 最后两维变成 3×3
    return R

@torch.jit.script
def init_heighest_offset_local(
    quat0_wxyz: torch.Tensor,      # (..., 4)
    delta_z:       torch.Tensor     # (...,)  世界 z 高度 (实数)
) -> torch.Tensor:
    """
    返回: (..., 3)  ←  offset_local
    """
    # 旋转矩阵 R0
    R0 = quat_to_rotmat(quat0_wxyz)                # (...,3,3)

    # world 向量 [0, 0, h/2]                     # (...,)
    zeros    = torch.zeros_like(delta_z)
    delta    = torch.stack((zeros, zeros, delta_z), dim=-1)  # (...,3)

    # offset_local = R0ᵀ @ delta
    offset_local = (R0.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)
    return offset_local

@torch.jit.script
def grasp_point(
    com:           torch.Tensor,   # (..., 3)  当前质心
    quat_wxyz:     torch.Tensor,   # (..., 4)  当前四元数 (w,x,y,z)
    offset_local:  torch.Tensor    # (..., 3)  reset 时缓存
) -> torch.Tensor:
    """
    返回: (..., 3)  ←  a 点实时世界坐标
    """
    R = quat_to_rotmat(quat_wxyz)                         # (...,3,3)
    a_world = com + (R @ offset_local.unsqueeze(-1)).squeeze(-1)
    return a_world

@torch.jit.script
def quat_rotate_vector(q: torch.Tensor,          # [..., 4]
                       v: torch.Tensor           # [..., 3]
                      ) -> torch.Tensor:         # [..., 3]
    """
    用四元数 q 旋转任意向量 v。
    q 最后维度是 (w,x,y,z)；q 与 v 的批量维可广播。
    """
    R = quat_to_rotmat(q)                        # [..., 3, 3]
    v_expanded = v.unsqueeze(-1)                 # [..., 3, 1]
    v_rot = torch.matmul(R, v_expanded)          # [..., 3, 1]
    return v_rot.squeeze(-1)                     # [..., 3]

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
def _compute_jerk_penalty(
    acceleration: torch.Tensor,
    prev_accelerate: torch.Tensor,
    factor_reward: torch.Tensor,
) -> torch.Tensor:
    
    jerk_penalty = torch.sum((acceleration.sub(prev_accelerate)).pow(2), dim=-1)
    jerk_penalty = 0.001 * jerk_penalty * factor_reward
    return  jerk_penalty


@torch.jit.script
def _compute_values(
    env_origins: torch.Tensor,
    object_state: torch.Tensor,
    hand_state: torch.Tensor,
    robot_index: list[int],
    virtual_index: list[int],
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_envs = env_origins.shape[0]
    object_state[:, :3].sub_(env_origins)
    num_indexs = hand_state.shape[1] # (num_envs, num_indexs, 3)
    
    hand_state_offset = env_origins.repeat((1, num_indexs)).reshape(num_envs, num_indexs, 3)
    hand_state[:, :, :3].sub_(hand_state_offset)
    finger_thumb_state  = hand_state[:,0, :].clone()
    finger_index_state  = hand_state[:,1, :].clone()
    finger_middle_state = hand_state[:,2, :].clone()
    hand_base_state     = hand_state[:,5, :].clone()
    middle_point_12  = (finger_thumb_state + finger_index_state) / 2
    middle_point_13  = (finger_thumb_state + finger_middle_state) / 2
    
    # data for robot joint
    dof_pos = joint_pos[:, robot_index].clone()
    dof_vel = joint_vel[:, robot_index].clone()
    passive_dof = joint_pos[:, virtual_index].clone()
    
    return object_state, finger_thumb_state, finger_index_state, finger_middle_state, middle_point_12, middle_point_13, hand_base_state, hand_state, dof_pos, dof_vel, passive_dof

# scale data to [0,1]
@torch.jit.script
def norm(x, lower, upper):
    return (x-lower)/(upper-lower)

# scale data to [lower, upper]
@torch.jit.script
def scale(x, lower, upper, origin_lower: float=-1, origin_upper: float=1):
    return ((x - origin_lower) / (origin_upper - origin_lower) * (upper - lower) + lower)

@torch.jit.script
def unscale(x, lower, upper, target_lower: float=-1, target_upper: float=1):
    a = target_upper - target_lower
    b = target_lower
    return a * (x-lower) / (upper - lower) + b

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * torch.pi, x_unit_tensor), quat_from_angle_axis(rand1 * torch.pi, y_unit_tensor)
    )

def to_torch(x: list, dtype: torch.dtype = torch.float, device: str = 'cuda:0', requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)