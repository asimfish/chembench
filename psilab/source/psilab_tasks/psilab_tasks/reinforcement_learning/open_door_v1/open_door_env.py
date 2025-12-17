# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
from __future__ import annotations
import torch
from functools import wraps
import time
from datetime import datetime


""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.utils import configclass

""" Psi Lab Modules  """
from psilab import OUTPUT_DIR
from psilab.envs.rl_env import RLEnv 
from psilab.envs.rl_env_cfg import RLEnvCfg
from psilab.utils.wandb_utils import WandbLog
from psilab.utils.timer_utils import Timer
from psilab.utils.data_collect_utils import save_data
from psilab.utils.perf_utils import cost_time
from psilab.utils.math_utils import normalize_v1,unnormalize_v1,clamp,compute_angle_vector_plane

@configclass
class OpenDoorEnvCfg(RLEnvCfg):
    """Configuration for RL environment."""

    # params
    episode_length_s = 210 / 60.0
    decimation = 2
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # other params from gym
    arm_joint_velocity = 20.0
    hand_moving_average_weight = 0.8

    vel_obs_scale = 0.2
    env_id_print_data = 0 # index of env to print status
    lift_height_desired = 0.3 # target lift height target

    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt = 1 / 120, 
        render_interval=decimation,
        physx = PhysxCfg(
            solver_type = 1, # 0: pgs, 1: tgs
            max_position_iteration_count = 64,
            max_velocity_iteration_count = 0,
            bounce_threshold_velocity = 0.002,
            enable_ccd=True,
            gpu_max_rigid_patch_count = 4096 * 4096,
            gpu_collision_stack_size=2**30,
            gpu_found_lost_pairs_capacity = 137401003
            # gpu_total_aggregate_pairs_capacity=5196400

        ),
        render=RenderCfg(),

    )

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/rl"


class OpenDoorEnv(RLEnv):
    """GraspLego RL environment."""

    cfg: OpenDoorEnvCfg

    def __init__(self, cfg: OpenDoorEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # arm joint number used to compute
        self._arm_joint_num = 7

        # get instances in scene
        self._robot = self.scene.robots["robot"]
        self._door = self.scene.articulated_objects["door"]
        self._visualizer = self.scene.visualizer

        # door handle link index
        self._door_handle_index = self._door.find_bodies("handle")[0][0]
        self._door_joint_index = self._door.find_joints("joint_door")[0][0]

        # joint index: order is arm and hand
        arm_joint_index = self._robot.find_joints(self._robot.actuators["arm2"].joint_names,preserve_order=True)[0]
        hand_joint_index = self._robot.find_joints(self._robot.actuators["hand2"].joint_names,preserve_order=True)[0][:6]
        self._joint_index = arm_joint_index + hand_joint_index

        # hand base link index
        self._hand_base_link_index = self._robot.find_bodies(["hand2_link_base"])[0][0]

        # finger tip link index
        self._finger_tip_index = self._robot.find_bodies([
            "hand2_link_1_4",
            "hand2_link_2_3",
            "hand2_link_3_3",
            "hand2_link_4_3",
            "hand2_link_5_3",
        ],preserve_order=True)[0]

        # joint position target
        self._joint_pos_target = self._robot.data.joint_pos_target[:,self._joint_index]
        self._joint_pos_target_lasttime = self._robot.data.joint_pos_target[:,self._joint_index]

        # joint limit
        self._joint_pos_limit = self._robot.data.joint_limits[:,self._joint_index,:]

        # unit tensors which used to compute
        self._plane_norm_vector = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # angle desired to open
        self._door_joint_angle_target =  (30.0 * 3.14 /180) * torch.ones((self.num_envs),device=self.device)

        # reward lasttime
        self._reward_lasttime = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)

        # observations
        self._obs = torch.zeros((self.num_envs,self.cfg.observation_space), dtype=torch.float32, device=self.device)  # type: ignore

        # initialize wandb
        if self.cfg.enable_wandb: 
            self._wandb = WandbLog()
            project = "OpenDoor_v1_PPO"
            experiment_name = datetime.strftime(datetime.now(), '%m%d_%H%M%S')
            self._wandb.init_wandb(project,experiment_name)

        # initialize Timer
        self._timer = Timer()

        # success state of all envs
        self.success_state = torch.zeros((self.num_envs),dtype=torch.bool,device=self.device)
        # success rate(success_env_num / num_envs) which focus on current rate
        # note: this is different from policy eval result(self._episode_success_num/self._episode_num) which focus on all eval result 
        self.success_rate = 0 # success_env_num / num_envs
        # 
        self._step_func_cost_time = 0   # step function cost time, which used to compute sim fps 
        # extras info      
        self.extras = {
            'distance_reward': torch.zeros((self.num_envs),device=self.device), 
            'middle_point_dis_reward': torch.zeros((self.num_envs),device=self.device), 
            "gesture_reward": torch.zeros((self.num_envs),device=self.device),  
            "angle_reward": torch.zeros((self.num_envs),device=self.device),  
            'action_penalty': torch.zeros((self.num_envs),device=self.device), 
            "current_reward": torch.zeros((self.num_envs),device=self.device),
            "total_reward": torch.zeros((self.num_envs),device=self.device),
            "mean_reward": torch.zeros((1),device=self.device),
        }

    def _pre_physics_step(self, actions: torch.Tensor):
        # 
        self.actions = actions.clone()
      
    @cost_time
    def step(self, action):
        # call super step first to apply action and sim step
        obs_buf,reward_buf, reset_terminated, reset_time_outs, extras = super().step(action)
        
        # refresh marker only flag is true and visualizer is valid
        if self.cfg.enable_marker and self._visualizer is not None:

            # finger tip link state
            finger_tip_state = self._robot.data.body_link_state_w[0:1,self._finger_tip_index,:]
            # middle point position of thumb and index
            middle_point_pos = (finger_tip_state[:,0,:3] + finger_tip_state[:,1,:3]) / 2
            # door handle
            door_handle_state = self._door.data.body_link_state_w[0:1,self._door_handle_index,:]
            # refresh visualize and marker
            marker_pos = torch.cat((
                finger_tip_state[:,0,:3],
                finger_tip_state[:,1,:3],
                finger_tip_state[:,2,:3],
                finger_tip_state[:,3,:3],
                finger_tip_state[:,4,:3],
                door_handle_state[:,:3],
                middle_point_pos
                ),0)
            #
            marker_rot = torch.cat((
                finger_tip_state[:,0,3:7],
                finger_tip_state[:,1,3:7],
                finger_tip_state[:,2,3:7],
                finger_tip_state[:,3,3:7],
                finger_tip_state[:,4,3:7],
                door_handle_state[:,3:7],
                torch.zeros((1,4),device=self.device)
                ),0)
            # refresh marker position and orientation
            self._visualizer.visualize(
                marker_pos, 
                marker_rot)

        return obs_buf,reward_buf, reset_terminated, reset_time_outs, extras

    def _apply_action(self):
        # actions 范围 -1 到 1 
        # action index 0-6 : arm velocity (normed), order is same with self._arm_joint_index
        # action index 7-12 : hand real joint position target (normed), order is same with self._hand_real_joint_index

        # joint position current
        joint_pos = self._robot.data.joint_pos[:,self._joint_index]

        # store hand joint position target lasttime
        self._joint_pos_target_lasttime = self._joint_pos_target.clone()
        
        # compute arm joint position target
        self._joint_pos_target[:, :self._arm_joint_num] = \
            joint_pos[:, :self._arm_joint_num] \
            + self.cfg.arm_joint_velocity * self.physics_dt * self.actions[:, :self._arm_joint_num]

        # compute hand joint position target
        self._joint_pos_target[:, self._arm_joint_num:] = unnormalize_v1(
            self.actions[:, self._arm_joint_num:],
            self._joint_pos_limit[:,self._arm_joint_num:,0],
            self._joint_pos_limit[:,self._arm_joint_num:,1]
        )
        
        # change hand joint position target with weighted moving average method
        self._joint_pos_target[:, self._arm_joint_num:] = \
            self.cfg.hand_moving_average_weight * self._joint_pos_target[:, self._arm_joint_num:] \
            + (1.0 - self.cfg.hand_moving_average_weight) * self._joint_pos_target_lasttime[:, self._arm_joint_num:]

        # clamp joint position
        self._joint_pos_target = clamp(
            self._joint_pos_target, 
            self._joint_pos_limit[:,:,0],
            self._joint_pos_limit[:,:,1]
        )

        # set joint position target
        self._robot.set_joint_position_target(self._joint_pos_target,self._joint_index)
       
    def _get_observations(self) -> dict:
        
        # get state in global coordinate
        finger_tip_state = self._robot.data.body_link_state_w[:,self._finger_tip_index,:].clone()
        hand_base_state = self._robot.data.body_link_state_w[:,self._hand_base_link_index,:].clone()
        door_handle_state = self._door.data.body_link_state_w[:,self._door_handle_index,:].clone()
        # target_state =  self._target.data.root_link_state_w.clone()

        # compute state in local coordinate
        finger_tip_state[:,:,:3] -= self.scene.env_origins.unsqueeze(1).repeat(1,5,1)
        hand_base_state[:,:3] -= self.scene.env_origins
        door_handle_state[:,:3] -= self.scene.env_origins

        # compute delta position between finger tip and target
        finger_tip_state[:,:,:3] -= door_handle_state[:,:3].unsqueeze(1).repeat(1,5,1)
        finger_tip_state = finger_tip_state.flatten(1)

        # compute joint position norm
        joint_pos_norm = normalize_v1(
            self._robot.data.joint_pos[:,self._joint_index].clone(),
            self._joint_pos_limit[:,:,0],
            self._joint_pos_limit[:,:,1]
        )

        # compute joint velocity
        joint_vel = self.cfg.vel_obs_scale * self._robot.data.joint_vel[:,self._joint_index]

        self._obs = torch.cat(
            (
                joint_pos_norm,
                joint_vel,
                finger_tip_state,
                self.actions,
                hand_base_state,
                door_handle_state
            ),
            dim=1
        )

        return {"policy": self._obs, "critic":self._obs}

    def _get_rewards(self) -> torch.Tensor:
        
         # compute reward
        distance_reward,middle_point_dis_reward,gesture_reward,angle_reward = self._compute_rewards()

        # compute penalize actions
        delta_joint_pos_target = self._joint_pos_target - self._joint_pos_target_lasttime
        action_penalty = 0.001 * (torch.sum(self.actions[:,:self._arm_joint_num] ** 2, dim=-1) \
            + torch.sum((delta_joint_pos_target[:,self._arm_joint_num:]) ** 2, dim=-1))

        # compute current reward
        reward_current = distance_reward + middle_point_dis_reward + gesture_reward + angle_reward  - action_penalty
        
        # compute delta reward
        delta_reward = reward_current - self._reward_lasttime

        # update extra info
        self.extras['distance_reward'] = distance_reward # type: ignore
        self.extras['middle_point_dis_reward']  = middle_point_dis_reward # type: ignore
        self.extras['gesture_reward'] = gesture_reward # type: ignore
        self.extras['angle_reward'] = angle_reward # type: ignore
        self.extras['action_penalty'] = action_penalty # type: ignore
        self.extras['current_reward'] += delta_reward # type: ignore

        # store total reward last time
        self._reward_lasttime = distance_reward + middle_point_dis_reward + gesture_reward + angle_reward

        return delta_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        # 
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # async_reset
        if self.cfg.async_reset:
            door_joint_error = self._door_joint_angle_target - self._door.data.joint_pos[:,self._door_joint_index]
            resets = (torch.abs(door_joint_error)<0.05)
        # sync reset
        else:
            # never reset until time out
            resets = torch.zeros(self.num_envs,device=self.device)

        return resets, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # get env indexs
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # update eval result
        self._eval(env_ids)

        # update extra infos 
        for env_index in env_ids: # type: ignore
            # update total reward
            self.extras["total_reward"][env_index] = self.extras["current_reward"][env_index]
            # reset other reward
            self.extras["current_reward"][env_index] = 0 # type: ignore

        # update mean reward of all envs
        self.extras["mean_reward"] = self.extras["total_reward"].mean().to('cpu') # type: ignore

        # 
        if self.cfg.enable_output:
            self._save_data(env_ids)

        # 
        super()._reset_idx(env_ids) # type: ignore

        # log info after reset
        if self.cfg.enable_log:
            self._log_info(env_ids)

        # Update simulation time to ensure get latest state of onject in scene 
        self.scene.update(dt = 5 * self.physics_dt)
        
        # reset joint position target lasttime
        self._joint_pos_target[env_ids,:] = self._robot.data.joint_pos[env_ids,:][:,self._joint_index]

        # compute reward while penlize action is zero, as reward lasttime
        distance_reward,middle_point_dis_reward,gesture_reward,angle_reward = self._compute_rewards()
        self._reward_lasttime[env_ids] = distance_reward[env_ids] + middle_point_dis_reward[env_ids] + gesture_reward[env_ids] + angle_reward[env_ids]

    def _compute_rewards(self)-> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        
        # finger tip link position
        thumb_tip_link_pos = self._robot.data.body_link_state_w[:,self._finger_tip_index[0],:3]
        index_tip_link_pos = self._robot.data.body_link_state_w[:,self._finger_tip_index[1],:3]

        # target position
        # self._target.data.body_link_state_w[:,self._target_handle_index:].squeeze(1).clone()
        door_handle_pos = self._door.data.body_link_state_w[:,self._door_handle_index,:3]
        # door_joint_angle = self._door.data.joint_pos[:,self._door_joint_index]

        # the reward computed by total distacne between [thumb, index] finger tip and target
        distance_total = torch.norm(door_handle_pos - thumb_tip_link_pos, p=2, dim=-1) + torch.norm(door_handle_pos - index_tip_link_pos, p=2, dim=-1)
        distance_reward = 1.0 * torch.exp(- 5 * torch.clamp((distance_total - 0.02), 0, None)) # type: ignore

        # the reward computed by distacne between target and middle point of thumb and index finger tip
        middle_point_pos = (thumb_tip_link_pos + index_tip_link_pos) / 2
        middle_point_distance = torch.norm(door_handle_pos - middle_point_pos, p=2, dim=-1)
        middle_point_dis_reward = 6.0 * torch.exp(- 5 * middle_point_distance)

        # the reward computed by angle between vector which is from thumb finger tio to index finger tip, and XY Plane
        angle = compute_angle_vector_plane(index_tip_link_pos - thumb_tip_link_pos, self._plane_norm_vector)
        gesture_reward = distance_reward * torch.exp(-1.0 * torch.abs(angle)) * 0.5 # type: ignore

        # the reward computed by lifted height of target
        door_joint_error = self._door_joint_angle_target - self._door.data.joint_pos[:,self._door_joint_index]
        # angle_reward = 0.5 * middle_point_dis_reward * 0.166666667 * 400 * torch.clamp((0.3- door_joint_error), -0.05, None)
        angle_reward = 0.5 * middle_point_dis_reward * 0.166666667 * 400 * self._door.data.joint_pos[:,self._door_joint_index]

        #
        return distance_reward,middle_point_dis_reward,gesture_reward,angle_reward
    
    def _eval(self, env_ids: torch.Tensor | None):
        
        # get env indexs
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # compute angle error
        door_joint_error = self._door_joint_angle_target - self._door.data.joint_pos[:,self._door_joint_index]

        # compute success state of all envs
        succcess = (torch.abs(door_joint_error)<0.05)
        # update success state
        # for env_index in env_ids:
        self.success_state[env_ids] = succcess[env_ids]
            
        # get success number of reset envs
        success_num = torch.nonzero(self.success_state).shape[0]
        # compute current success rate
        self.success_rate = success_num / self.num_envs
        # compute success rate of all eval result
        self._episode_success_num+=len(torch.nonzero(succcess[env_ids]).squeeze(1).tolist())

    def _log_info(self, env_ids: torch.Tensor | None):

        # log extra info to screen
        if self.cfg.env_id_print_data in env_ids and self.common_step_counter > 0: # type: ignore
            reward_items = ['distance_reward', 'middle_point_dis_reward', 'gesture_reward', 'angle_reward','action_penalty']
            extras = {}
            for item in reward_items:
                extras[item] = self.extras[item].to('cpu').numpy() # type: ignore
                    
            total_reward = sum([abs(extras[item][self.cfg.env_id_print_data]) for item in reward_items])

            print("\n")
            print("#" * 17, " Statistics", "#" * 17)
            print(f"env id:   {self.cfg.env_id_print_data}")
            # print(f"episodes:   {self._episodes}")
            print(f"distance_reward:      {extras['distance_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['distance_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"middle_point_dis_reward:     {extras['middle_point_dis_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['middle_point_dis_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"gesture_reward:      {extras['gesture_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['gesture_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"angle_reward:   {extras['angle_reward'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['angle_reward'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
            print(f"action_penalty:   {extras['action_penalty'][self.cfg.env_id_print_data]:.2f} ({(abs(extras['action_penalty'][self.cfg.env_id_print_data]) / total_reward * 100):.2f}%)")
        
            print("#" * 15, "Statistics End", "#" * 15,"\n")

        # log policy evalutation result
        if self.cfg.enable_eval and self._episode_num>0:
            plocy_success_rate = float(self._episode_success_num) / float(self._episode_num)
            print(f"Policy成功率: {plocy_success_rate * 100.0} % ")
            print(f"成功次数/总次数: {self._episode_success_num}/{self._episode_num} ")  

        # update info to wandb
        if self.cfg.enable_wandb:  # type: ignore
            # self._wandb.step = self._timer.run_time()
            self._wandb.step = self.common_step_counter * self.num_envs
            # mean reward
            self._wandb.set_data("result/mean_reward",self.extras["mean_reward"]) # type: ignore
            self._wandb.upload("result/mean_reward")
            # success rate
            self._wandb.set_data("result/success_rate",self.success_rate) # type: ignore
            self._wandb.upload("result/success_rate")

    def _save_data(self, env_ids: torch.Tensor | None):
        # save data
        if self._data is not None: 
            env_to_save_indexs=[]
            for env_index in env_ids: # type: ignore
                if self.success_state[env_index]:
                    env_to_save_indexs.append(env_index)
            #
            if len(env_to_save_indexs)==0:
                return
            # save data
            self._data = save_data(self._data,self.cfg,self.scene,env_to_save_indexs,env_ids) # type: ignore
            # compute data collect result
            record_time = self._timer.run_time() /60.0
            record_rate = self._episode_success_num / record_time
            # log data collect result
            print(f"采集时长: {record_time} 分钟")
            print(f"采集数据: {self._episode_success_num} 条")
            print(f"采集效率: {record_rate} 条/分钟")
