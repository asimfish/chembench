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
from datetime import datetime
from scipy.spatial.transform import Rotation as R

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

""" Local Modules """
from .grasp_configs import get_grasp_config, GRASP_CONFIGS

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
    decimation = 4
    sample_step = 1

    # viewer config
    viewer = ViewerCfg(
        eye=(0.65,-0.2,1.1),
        lookat=(-15.0,0.1,0.3)
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

    # defualt ouput folder
    output_folder = OUTPUT_DIR + "/Mp_grasp/Beaker_003"
    # output_folder = OUTPUT_DIR + "/Mp_grasp/Beaker_003_Smooth"
    # lift desired height
    lift_height_desired = 0.3

class GraspBottleEnv(MPEnv):

    cfg: GraspBottleEnvCfg

    def __init__(self, cfg: GraspBottleEnvCfg, render_mode: str | None = None, **kwargs):

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

        # joint index: order is arm and hand
        self._arm_joint_index = self._robot.find_joints(self._robot.actuators["arm2"].joint_names,preserve_order=True)[0]
        self._hand_joint_index = self._robot.find_joints(self._robot.actuators["hand2"].joint_names,preserve_order=True)[0][:6]
        self._joint_index = self._arm_joint_index + self._hand_joint_index
        # eef link index
        self._eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]

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

        # 设置 RTX 渲染选项
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
        
        # ========== 启用 Interactive Path Tracing 模式 ==========
        # rendermode: 0 = RaytracedLighting, 1 = PathTracing, 2 = InteractivePathTracing (即 Realtime)
        # carb_settings_iface.set_int("/rtx/rendermode", 1)  # 1 = Path Tracing
        # # 或者使用字符串方式:
        # # carb_settings_iface.set_string("/rtx/rendermode", "PathTracing")
        
        # # Path Tracing 相关优化设置
        # carb_settings_iface.set_int("/rtx/pathtracing/spp", 1)  # Samples Per Pixel (每像素采样数)
        # carb_settings_iface.set_int("/rtx/pathtracing/totalSpp", 64)  # 累积采样数上限
        # carb_settings_iface.set_int("/rtx/pathtracing/maxBounces", 4)  # 最大光线反弹次数
        # carb_settings_iface.set_bool("/rtx/pathtracing/enabled", True)  # 确保启用
        
        # # 可选：启用 AI 降噪器以提高实时性能
        # carb_settings_iface.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)

    def create_trajectory(self,env_ids: torch.Tensor | None):
        
        env_len = env_ids.shape[0]
        #
        k1 = 0.5
        k2 = 0.1
        k1_step = int(k1 * self.max_episode_length)
        k2_step = int(k2 * self.max_episode_length)
        
        # ========== 抓取姿态配置 ==========
        # 从配置文件读取抓取参数（修改这里的物体名称即可切换不同物体的抓取配置）
        grasp_config = get_grasp_config("glass_beaker_100ml")
        grasp_euler_deg = grasp_config["euler_deg"]
        grasp_offset = grasp_config["offset"]
        
        # 欧拉角转四元数 (scipy 返回 xyzw，需要转换为 wxyz)
        quat_xyzw = R.from_euler('xyz', grasp_euler_deg, degrees=True).as_quat()
        eff_quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # 转换为 wxyz
        
        # 位置偏移 (相对于物体中心的偏移)
        eff_offset = torch.tensor(grasp_offset, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        eff_quat = torch.tensor(eff_quat_wxyz, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        
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

        # 拼接轨迹 50%移动手臂到抓取点 50%抬起手臂
        # 40%开合手，60%关闭手
        self._eef_pose_target[env_ids,:k1_step+k2_step,:] = eef_pose_target_1.unsqueeze(1).repeat(1,k1_step+k2_step,1)
        self._eef_pose_target[env_ids,k1_step+k2_step:,:] = eef_pose_target_2.unsqueeze(1).repeat(1,self.max_episode_length - k1_step - k2_step,1)
        self._hand_pos_target[env_ids,:k1_step,:] = hand_pos_target_1.unsqueeze(1).repeat(1,k1_step,1)
        self._hand_pos_target[env_ids,k1_step:,:] = hand_pos_target_2.unsqueeze(1).repeat(1,self.max_episode_length - k1_step,1)

        # 修改 eef 第一阶段轨迹
        delta_eef_pos = (1 / k1_step) * (eef_pose_target_1[:,:3] - self._robot.data.body_link_pos_w[env_ids,self._eef_link_index,:])
        delta_eef_quat = (1 / k1_step) * (eef_pose_target_1[:,3:7] - self._robot.data.body_link_quat_w[env_ids,self._eef_link_index,:])
        for i in range(int(k1_step * 0.3)):            
            self._eef_pose_target[env_ids,i,:3] = self._robot.data.body_link_pos_w[env_ids,self._eef_link_index,:3] + i * delta_eef_pos[:,:3]
            self._eef_pose_target[env_ids,i,3:7] = self._robot.data.body_link_quat_w[env_ids,self._eef_link_index,:] + i * delta_eef_quat[:,:]
        for i in range(int(k1_step * 0.3)):            
            self._eef_pose_target[env_ids,i,1] = self._robot.data.body_link_pos_w[env_ids,self._eef_link_index,1] + i * delta_eef_pos[:,1]
            self._eef_pose_target[env_ids,i,3:7] = self._robot.data.body_link_quat_w[env_ids,self._eef_link_index,:] + i * delta_eef_quat[:,:]

    def _minimum_jerk_interpolation(self, t: torch.Tensor) -> torch.Tensor:
        """
        最小加加速度轨迹插值函数 (Minimum Jerk Trajectory)
        s(t) = 10*t^3 - 15*t^4 + 6*t^5
        
        特性：
        - s(0) = 0, s(1) = 1
        - s'(0) = s'(1) = 0 (起点和终点速度为0)
        - s''(0) = s''(1) = 0 (起点和终点加速度为0)
        
        Args:
            t: 归一化时间 [0, 1]，shape: (N,) 或标量
        Returns:
            插值系数，shape: 同输入
        """
        t = torch.clamp(t, 0.0, 1.0)
        return 10 * t**3 - 15 * t**4 + 6 * t**5
    
    def _slerp_batch(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        批量球面线性插值 (SLERP) for quaternions (wxyz format)
        
        Args:
            q0: 起始四元数 [env_len, 4] (wxyz)
            q1: 目标四元数 [env_len, 4] (wxyz)
            t: 插值系数 [env_len] 或 [env_len, 1]，范围 [0, 1]
        Returns:
            插值后的四元数 [env_len, 4] (wxyz)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [env_len, 1]
        
        # 归一化四元数
        q0 = q0 / (torch.norm(q0, dim=1, keepdim=True) + 1e-8)
        q1 = q1 / (torch.norm(q1, dim=1, keepdim=True) + 1e-8)
        
        # 计算点积
        dot = torch.sum(q0 * q1, dim=1, keepdim=True)
        
        # 如果点积为负，反转一个四元数以取最短路径
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.abs(dot)
        
        # 当四元数非常接近时，使用线性插值避免数值问题
        linear_threshold = 0.9995
        
        # SLERP 插值
        theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
        sin_theta = torch.sin(theta)
        
        # 避免除零
        safe_sin_theta = torch.where(sin_theta.abs() < 1e-6, torch.ones_like(sin_theta), sin_theta)
        
        s0 = torch.sin((1.0 - t) * theta) / safe_sin_theta
        s1 = torch.sin(t * theta) / safe_sin_theta
        
        # 当接近时使用线性插值
        s0 = torch.where(dot > linear_threshold, 1.0 - t, s0)
        s1 = torch.where(dot > linear_threshold, t, s1)
        
        result = s0 * q0 + s1 * q1
        
        # 归一化结果
        return result / (torch.norm(result, dim=1, keepdim=True) + 1e-8)

    def create_trajectory_smooth(self, env_ids: torch.Tensor | None):
        """
        创建平滑的抓取轨迹
        
        使用最小加加速度轨迹(Minimum Jerk)实现平滑的位置过渡
        使用球面线性插值(SLERP)实现平滑的姿态过渡
        
        轨迹分为4个阶段：
        1. approach (20%): 从当前位置移动到预抓取位置（抓取点上方）
        2. descend (20%): 从预抓取位置下降到抓取位置
        3. grasp (20%): 保持位置，关闭手指
        4. lift (40%): 抬起物体到目标高度
        """
        env_len = env_ids.shape[0]
        total_steps = self.max_episode_length
        
        # ========== 阶段时间分配 ==========
        phase_ratios = {
            'approach': 0.20,  # 接近阶段
            'descend': 0.20,   # 下降阶段  
            'grasp': 0.10,     # 抓取阶段（手指闭合）
            'lift': 0.50       # 抬起阶段
        }
        
        approach_end = int(phase_ratios['approach'] * total_steps)
        descend_end = approach_end + int(phase_ratios['descend'] * total_steps)
        grasp_end = descend_end + int(phase_ratios['grasp'] * total_steps)
        lift_end = total_steps
        
        # ========== 抓取姿态配置 ==========
        grasp_config = get_grasp_config("glass_beaker_100ml")
        grasp_euler_deg = grasp_config["euler_deg"]
        grasp_offset = grasp_config["offset"]
        
        # 欧拉角转四元数 (scipy 返回 xyzw，需要转换为 wxyz)
        quat_xyzw = R.from_euler('xyz', grasp_euler_deg, degrees=True).as_quat()
        eff_quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        
        # ========== 计算关键位置 ==========
        # 目标物体相对于机器人基座的位置
        target_position = self._target.data.root_pos_w[env_ids, :] - self._robot.data.root_link_pos_w[env_ids, :]
        
        # 偏移和姿态
        eff_offset = torch.tensor(grasp_offset, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        eff_quat = torch.tensor(eff_quat_wxyz, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # 当前末端执行器位姿（相对于机器人基座）
        eef_pos_current = self._robot.data.body_link_pos_w[env_ids, self._eef_link_index, :] - self._robot.data.root_link_pos_w[env_ids, :]
        eef_quat_current = self._robot.data.body_link_quat_w[env_ids, self._eef_link_index, :]
        
        # 预抓取位置（抓取点上方）
        pre_grasp_height = 0.15  # 预抓取高度偏移
        pre_grasp_offset = torch.tensor([0, 0, pre_grasp_height], device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_pre_grasp = eff_offset + target_position + pre_grasp_offset
        
        # 抓取位置
        pos_grasp = eff_offset + target_position
        
        # 抬起位置
        lift_offset = torch.tensor([0, 0, self.cfg.lift_height_desired], device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_lift = pos_grasp + lift_offset
        
        # ========== 手指目标位置 ==========
        # 手指打开位置
        hand_pos_open = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index]
        # 手指闭合位置
        hand_pos_closed = self._joint_limit_lower[env_ids, :][:, self._hand_joint_index]
        hand_pos_closed[:, 0] = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index[0]]  # 拇指旋转取最大值
        
        # ========== 生成平滑轨迹 ==========
        for step in range(total_steps):
            if step < approach_end:
                # 阶段1: 接近 - 从当前位置移动到预抓取位置
                t_normalized = step / max(approach_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                
                # 位置插值
                pos_interp = eef_pos_current + t_smooth * (pos_pre_grasp - eef_pos_current)
                # 姿态插值
                t_batch = torch.full((env_len,), t_smooth.item(), device=self.device)
                quat_interp = self._slerp_batch(eef_quat_current, eff_quat, t_batch)
                # 手指保持打开
                hand_interp = hand_pos_open
                
            elif step < descend_end:
                # 阶段2: 下降 - 从预抓取位置下降到抓取位置
                t_normalized = (step - approach_end) / max(descend_end - approach_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                
                # 位置插值
                pos_interp = pos_pre_grasp + t_smooth * (pos_grasp - pos_pre_grasp)
                # 姿态保持目标姿态
                quat_interp = eff_quat
                # 手指保持打开
                hand_interp = hand_pos_open
                
            elif step < grasp_end:
                # 阶段3: 抓取 - 保持位置，平滑关闭手指
                t_normalized = (step - descend_end) / max(grasp_end - descend_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                
                # 位置保持在抓取位置
                pos_interp = pos_grasp
                quat_interp = eff_quat
                # 手指平滑闭合
                hand_interp = hand_pos_open + t_smooth * (hand_pos_closed - hand_pos_open)
                
            else:
                # 阶段4: 抬起 - 从抓取位置抬起到目标高度
                t_normalized = (step - grasp_end) / max(lift_end - grasp_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                
                # 位置插值
                pos_interp = pos_grasp + t_smooth * (pos_lift - pos_grasp)
                quat_interp = eff_quat
                # 手指保持闭合
                hand_interp = hand_pos_closed
            
            # 存储轨迹点
            self._eef_pose_target[env_ids, step, :3] = pos_interp
            self._eef_pose_target[env_ids, step, 3:7] = quat_interp
            self._hand_pos_target[env_ids, step, :] = hand_interp

    def step(self,actions):
        
        # set target
        eef_pose_target = torch.tensor([],device=self.device)
        hand_pos_target = torch.tensor([],device=self.device)
        # 
        for i in range(self.num_envs):
            eef_pose_target = torch.cat((eef_pose_target,self._eef_pose_target[i,self._episode_step[i],:].unsqueeze(0)), dim=0)
            hand_pos_target= torch.cat((hand_pos_target,self._hand_pos_target[i,self._episode_step[i],:].unsqueeze(0)), dim=0)

        self._robot.set_ik_command({"arm2":eef_pose_target})
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
        reset = (success | fail | time_out)
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
        # task evalutation
        bfailed,self._has_contacted = eval_fail(self._target,self._contact_sensors, self._has_contacted) # type: ignore
        # success eval
        bsuccessed = eval_success(self._target, self._contact_sensors,self._target_pos_init, self.cfg.lift_height_desired)
     
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
            # 记录保存前的时间戳，用于推断文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._data = save_data(
                data=self._data,
                cfg=self.cfg,
                scene=self.scene,
                env_indexs=success_ids,
                reset_env_indexs=env_ids.tolist(),
            )
            # 打印保存的文件路径
            if success_ids:
                saved_path = f"{self.cfg.output_folder}/{timestamp}_data.hdf5"
                print(f"[DATA] 已保存数据: {saved_path} (成功轨迹数: {len(success_ids)})")


        super()._reset_idx(env_ids)   

        # print logs
        if self.cfg.enable_log:
            self._log_info()
        

        # 
        # self.create_trajectory(env_ids)
        self.create_trajectory_smooth(env_ids)



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
