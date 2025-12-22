# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from __future__ import annotations
from typing import Any
from collections.abc import Sequence

""" Common Modules  """ 
import os
import torch
from datetime import datetime
from scipy.spatial.transform import Rotation as R

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg


""" Psi Lab Modules  """
from psilab.envs.mp_env import MPEnv 
from psilab.envs.mp_env_cfg import MPEnvCfg
from psilab.utils.timer_utils import Timer
from psilab.eval.grasp_rigid import eval_fail,eval_success
from psilab.utils.data_collect_utils import parse_data,save_data

""" Local Modules """
from ..config_loader import load_grasp_config, load_grasp_points, get_grasp_point_by_index, get_available_grasp_points

# ========== 任务配置（修改这里即可切换不同任务）==========
# TARGET_OBJECT_NAME = "mortar"  # 目标物体名称，如 "mortar", "glass_beaker_100ml" 等
TARGET_OBJECT_NAME = "glass_beaker_100ml"  # 目标物体名称，如 "mortar", "glass_beaker_100ml" 等
# TARGET_OBJECT_NAME = "glass_beaker_500ml"  # 目标物体名称，如 "mortar", "glass_beaker_100ml" 等
TASK_TYPE = "grasp"            # 任务类型：grasp, handover, pick_place, pour 等

# 数据根目录：统一存储到 chembench/data 下
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../data"))


@configclass
class GraspBottleEnvCfg(MPEnvCfg):
    """Configuration for Rl environment."""

    # fake params - These parameters are placeholders for episode length, action and observation spaces
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    episode_length_s = 4
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

    # ========== 物体抓取配置参数（从 object_config.json 加载）==========
    # 目标物体名称（使用文件顶部定义的 TARGET_OBJECT_NAME）
    target_object_name: str = TARGET_OBJECT_NAME
    
    # 抓取偏移 [x, y, z]（相对于物体中心）
    grasp_offset: list = None  # type: ignore
    
    # 抓取角度 [roll, pitch, yaw]（欧拉角，单位：度）
    grasp_euler_deg: list = None  # type: ignore
    
    # 抬起高度（单位：米）
    lift_height_desired: float = 0.25
    
    # 轨迹生成的时序参数
    phase_ratios: dict = None  # type: ignore
    
    # 手指闭合方式：True=平滑闭合，False=直接闭合（类似 create_trajectory）
    smooth_finger_close: bool = True

    # 是否启用轨迹平滑
    enable_trajectory_smooth: bool = False
    
    # 轨迹模式选择：
    # - "default": 使用原始 create_trajectory（不平滑）
    # - "smooth": 使用 create_trajectory_smooth（minimum jerk，有明显加减速）
    # - "constant_velocity": 使用恒速轨迹（推荐用于 Diffusion Policy 训练）
    trajectory_mode: str = "constant_velocity"
    
    # 成功判断：朝向偏差阈值（sin²(θ/2)，0=完全一致，1=上下颠倒）
    # 0.1 约等于 37° 的偏差，0.05 约等于 26° 的偏差
    orientation_threshold: float = 0.1
    
    # 目标成功次数：达到此数量后自动停止（设为 0 或 None 表示不限制）
    target_success_count: int = 50
    
    # 输出文件夹：chembench/data/motion_plan/{任务类型}/{物体名称}
    output_folder: str = None  # type: ignore
    
    ##是否输出每一步末端执行器的位置和旋转
    print_eef_pose: bool = False
    
    def __post_init__(self):
        """
        初始化后从 object_config.json 加载默认参数
        
        直接调用 config_loader 模块读取 JSON 配置文件
        
        支持两种模式：
        1. 单点模式（默认）：使用 grasp_offset 和 grasp_euler_deg
        2. 多点模式：从 grasp_points_N 周期性读取抓取点
        
        输出路径格式：chembench/data/motion_plan/{任务类型}/{物体名称}
        """
        # 从 JSON 加载基础抓取配置
        grasp_config = load_grasp_config(self.target_object_name)
        
        # 设置抓取偏移（如果未手动指定）
        if self.grasp_offset is None:
            self.grasp_offset = grasp_config["grasp_offset"]
        
        # 设置抓取角度（如果未手动指定）
        if self.grasp_euler_deg is None:
            self.grasp_euler_deg = grasp_config["grasp_euler_deg"]
        
        # 设置抬起高度（如果使用默认值）
        if self.lift_height_desired == 0.3:
            self.lift_height_desired = grasp_config["lift_height"]
        
        # 设置轨迹时序参数（如果未手动指定）
        if self.phase_ratios is None:
            timing = grasp_config["timing"]
            self.phase_ratios = {
                "approach": timing.get("approach_ratio", 0.4),
                "grasp": timing.get("grasp_ratio", 0.2),
                "lift": timing.get("lift_ratio", 0.4)
            }
        
        # 设置输出文件夹：chembench/data/motion_plan/{任务类型}/{物体名称}
        if self.output_folder is None:
            object_name = grasp_config.get("name_cn", self.target_object_name)
            self.output_folder = os.path.join(DATA_ROOT, "motion_plan", TASK_TYPE, object_name)

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
        self._target_quat_init = torch.zeros((self.num_envs,4),device=self.device)  # 初始朝向（wxyz）

        self.sim.set_camera_view([-0.7, -5.2, 1.3], [-1.2, -5.2, 1.1])
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
        
        # ========== 抓取姿态配置（从配置加载）==========
        grasp_euler_deg = self.cfg.grasp_euler_deg
        grasp_offset = self.cfg.grasp_offset
        
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
        # for i in range(int(k1_step * 0.3)):            
            # self._eef_pose_target[env_ids,i,:3] = self._robot.data.body_link_pos_w[env_ids,self._eef_link_index,:3] + i * delta_eef_pos[:,:3]
            # self._eef_pose_target[env_ids,i,3:7] = self._robot.data.body_link_quat_w[env_ids,self._eef_link_index,:] + i * delta_eef_quat[:,:]
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
    
    def _smooth_approach_interpolation(self, t: torch.Tensor) -> torch.Tensor:
        """
        更平滑的接近插值函数（末端减速更平缓）
        
        使用 7 阶多项式，在末端有更长的减速区间
        s(t) = 35*t^4 - 84*t^5 + 70*t^6 - 20*t^7
        
        特性：
        - s(0) = 0, s(1) = 1
        - s'(0) = s'(1) = 0 (起点和终点速度为0)
        - s''(0) = s''(1) = 0 (起点和终点加速度为0)
        - s'''(0) = s'''(1) = 0 (起点和终点加加速度为0)
        - 相比 minimum jerk，末端减速更加平缓
        
        Args:
            t: 归一化时间 [0, 1]，shape: (N,) 或标量
        Returns:
            插值系数，shape: 同输入
        """
        t = torch.clamp(t, 0.0, 1.0)
        return 35 * t**4 - 84 * t**5 + 70 * t**6 - 20 * t**7
    
    def _quasi_linear_interpolation(self, t: torch.Tensor, smooth_ratio: float = 0.05) -> torch.Tensor:
        """
        准线性插值函数 - 适合 Diffusion Policy 训练
        
        速度曲线近似梯形：
        - 开头 smooth_ratio 时间：平滑加速（使用半个余弦）
        - 中间 1-2*smooth_ratio 时间：恒定速度（线性）
        - 结尾 smooth_ratio 时间：平滑减速（使用半个余弦）
        
        关键特性：
        - 95% 时间保持恒定速度，数据分布均匀
        - 首尾有微小平滑，避免速度突变（对 IK 求解友好）
        - 速度变化很小，diffusion 模型容易拟合
        
        Args:
            t: 归一化时间 [0, 1]，shape: (N,) 或标量
            smooth_ratio: 首尾平滑区间占比，默认 0.05 (5%)
        Returns:
            插值系数，shape: 同输入
        """
        t = torch.clamp(t, 0.0, 1.0)
        sr = smooth_ratio
        
        # 梯形速度曲线的位移计算
        # 恒速段速度 v = 1 / (1 - sr)，这样总位移为 1
        # 加速段位移 = sr * v / 2 = sr / (2 * (1 - sr))
        # 匀速段位移 = (1 - 2*sr) * v = (1 - 2*sr) / (1 - sr)
        # 减速段位移 = sr * v / 2 = sr / (2 * (1 - sr))
        
        v_const = 1.0 / (1.0 - sr)  # 恒速段速度
        
        # 分段计算
        result = torch.zeros_like(t)
        
        # 加速段 [0, sr]：使用半个正弦实现平滑加速
        mask_accel = t < sr
        if mask_accel.any():
            t_accel = t[mask_accel] / sr  # 归一化到 [0, 1]
            # 位移 = 积分 v_const * sin(π*τ/2) dτ from 0 to t_accel
            # = sr * v_const * (1 - cos(π*t_accel/2)) * 2/π
            # 简化：使用 (1 - cos(π*t/2)) 形式，末端速度为 v_const
            s_accel = sr * v_const * (1.0 - torch.cos(t_accel * torch.pi / 2)) * 2.0 / torch.pi
            result[mask_accel] = s_accel
        
        # 匀速段 [sr, 1-sr]：线性插值
        mask_const = (t >= sr) & (t < 1 - sr)
        if mask_const.any():
            t_const = t[mask_const]
            # 加速段结束位置
            s_accel_end = sr * v_const * 2.0 / torch.pi
            # 匀速段位移
            s_const = s_accel_end + v_const * (t_const - sr)
            result[mask_const] = s_const
        
        # 减速段 [1-sr, 1]：使用半个余弦实现平滑减速
        mask_decel = t >= 1 - sr
        if mask_decel.any():
            t_decel = (t[mask_decel] - (1 - sr)) / sr  # 归一化到 [0, 1]
            # 减速段起始位置
            s_accel_end = sr * v_const * 2.0 / torch.pi
            s_const_end = s_accel_end + v_const * (1 - 2 * sr)
            # 减速段位移（使用 sin 实现减速）
            s_decel = s_const_end + sr * v_const * (torch.sin(t_decel * torch.pi / 2)) * 2.0 / torch.pi
            result[mask_decel] = s_decel
        
        # 归一化到 [0, 1]（由于数值计算，末端可能不精确为 1）
        result = result / (sr * v_const * 4.0 / torch.pi + v_const * (1 - 2 * sr))
        
        return torch.clamp(result, 0.0, 1.0)
    
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
        
        轨迹分为5个阶段（时序参数从配置文件加载）：
        1. approach: 从当前位置移动到预抓取位置（抓取点上方）
        2. descend: 从预抓取位置下降到抓取位置
        3. dwell: 在抓取位置稳定等待（确保IK收敛）
        4. grasp: 保持位置，关闭手指
        5. lift: 抬起物体到目标高度
        """
        env_len = env_ids.shape[0]
        total_steps = self.max_episode_length
        
        # ========== 阶段时间分配（从配置加载）==========
        # 从 JSON 配置的 timing 转换为5阶段分配
        cfg_ratios = self.cfg.phase_ratios
        
        # 将配置转换为5阶段（增加 dwell 稳定阶段）
        approach_total = cfg_ratios.get('approach', 0.4)
        grasp_total = cfg_ratios.get('grasp', 0.2)
        
        phase_ratios = {
            'approach': approach_total * 0.5,    # 接近预抓取位置
            'descend': approach_total * 0.5,     # 下降到抓取位置
            'dwell': approach_total * 0.00,       # 稳定等待（关键：让IK收敛）
            'grasp': grasp_total,                # 手指闭合
            'lift': cfg_ratios.get('lift', 0.4)  # 抬起阶段
        }
        
        approach_end = int(phase_ratios['approach'] * total_steps)
        descend_end = approach_end + int(phase_ratios['descend'] * total_steps)
        dwell_end = descend_end + int(phase_ratios['dwell'] * total_steps)
        grasp_end = dwell_end + int(phase_ratios['grasp'] * total_steps)
        lift_end = total_steps
        
        # ========== 抓取姿态配置（从配置加载）==========
        grasp_euler_deg = self.cfg.grasp_euler_deg
        grasp_offset = self.cfg.grasp_offset
        
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
        
        # 预抓取位置（抓取点上方 + y轴负方向偏移，从侧上方接近）

        # pre_grasp_height = 0.05   # z轴上方偏移 10cm
        # pre_grasp_y_offset = -0.10  # y轴负方向偏移 10cm（从侧面接近）
        # pre_grasp_x_offset = -0.02  # y轴负方向偏移 10cm（从侧面接近）

        pre_grasp_height = 0.1   # z轴上方偏移 10cm
        pre_grasp_y_offset = 0.00  # y轴负方向偏移 10cm（从侧面接近）
        pre_grasp_x_offset = 0.00  # y轴负方向偏移 10cm（从侧面接近）

        pre_grasp_offset = torch.tensor([pre_grasp_x_offset, pre_grasp_y_offset, pre_grasp_height], device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_pre_grasp = eff_offset + target_position + pre_grasp_offset
        
        # 抓取位置（精确位置）
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
                # 使用更平滑的插值函数，末端减速更平缓，避免撞到物体
                t_normalized = step / max(approach_end, 1)
                t_smooth = self._smooth_approach_interpolation(torch.tensor(t_normalized, device=self.device))
                # t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
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
                # t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                t_smooth = self._smooth_approach_interpolation(torch.tensor(t_normalized, device=self.device))
                # 位置插值
                pos_interp = pos_pre_grasp + t_smooth * (pos_grasp - pos_pre_grasp)
                # 姿态保持目标姿态
                quat_interp = eff_quat
                # 手指保持打开
                hand_interp = hand_pos_open
                
            elif step < dwell_end:
                # 阶段3: 稳定 - 在抓取位置保持不动，等待IK收敛
                # 位置精确保持在抓取位置（关键：提高抓取精度）
                pos_interp = pos_grasp
                quat_interp = eff_quat
                # 手指保持打开
                hand_interp = hand_pos_open
                
            elif step < grasp_end:
                # 阶段4: 抓取 - 保持位置，关闭手指
                # 位置保持在抓取位置
                pos_interp = pos_grasp
                quat_interp = eff_quat
                
                # 手指闭合方式
                if self.cfg.smooth_finger_close:
                    # 平滑闭合：使用最小加加速度插值
                    t_normalized = (step - dwell_end) / max(grasp_end - dwell_end, 1)
                    t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                    hand_interp = hand_pos_open + t_smooth * (hand_pos_closed - hand_pos_open)
                else:
                    # 直接闭合：整个 grasp 阶段都保持闭合状态
                    hand_interp = hand_pos_closed
                
            else:
                # 阶段5: 抬起 - 从抓取位置抬起到目标高度
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

    def create_trajectory_constant_velocity(self, env_ids: torch.Tensor | None):
        """
        创建恒速抓取轨迹 - 专为 Diffusion Policy 训练优化
        
        设计原则：
        1. 恒定速度运动 - 使用准线性插值，95%时间保持恒速
        2. 手指闭合后再抬升 - 确保抓取稳固
        3. 数据分布均匀 - 利于 diffusion 模型学习
        
        轨迹分为4个阶段（时间分配可配置）：
        1. approach: 从当前位置移动到预抓取位置（手指打开）
        2. descend: 从预抓取位置下降到抓取位置（手指打开）
        3. grasp_close: 保持在抓取位置，手指渐进闭合（关键阶段）
        4. lift: 手指完全闭合后，抬起物体
        
        时间分配：grasp 配置参数的 60% 分给 descend，40% 分给 grasp_close
        """
        env_len = env_ids.shape[0]
        total_steps = self.max_episode_length
        
        # ========== 阶段时间分配（4阶段）==========
        # 1. approach: 移动到预抓取位置
        # 2. descend: 下降到抓取位置
        # 3. grasp_close: 保持位置，手指闭合（关键：确保闭合完成再抬升）
        # 4. lift: 抬起物体
        cfg_ratios = self.cfg.phase_ratios
        approach_ratio = cfg_ratios.get('approach', 0.35)
        descend_ratio = cfg_ratios.get('grasp', 0.25) * 0.6  # grasp 的 60% 用于下降
        grasp_close_ratio = cfg_ratios.get('grasp', 0.25) * 0.4  # grasp 的 40% 用于手指闭合
        lift_ratio = cfg_ratios.get('lift', 0.25)
        
        # 归一化确保总和为1
        total_ratio = approach_ratio + descend_ratio + grasp_close_ratio + lift_ratio
        approach_ratio /= total_ratio
        descend_ratio /= total_ratio
        grasp_close_ratio /= total_ratio
        lift_ratio /= total_ratio
        
        approach_end = int(approach_ratio * total_steps)
        descend_end = approach_end + int(descend_ratio * total_steps)
        grasp_close_end = descend_end + int(grasp_close_ratio * total_steps)
        lift_end = total_steps
        
        # ========== 抓取姿态配置 ==========
        grasp_euler_deg = self.cfg.grasp_euler_deg
        grasp_offset = self.cfg.grasp_offset
        
        # 欧拉角转四元数 (scipy 返回 xyzw，需要转换为 wxyz)
        quat_xyzw = R.from_euler('xyz', grasp_euler_deg, degrees=True).as_quat()
        eff_quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        
        # ========== 计算关键位置 ==========
        target_position = self._target.data.root_pos_w[env_ids, :] - self._robot.data.root_link_pos_w[env_ids, :]
        eff_offset = torch.tensor(grasp_offset, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        eff_quat = torch.tensor(eff_quat_wxyz, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # 当前末端执行器位姿
        eef_pos_current = self._robot.data.body_link_pos_w[env_ids, self._eef_link_index, :] - self._robot.data.root_link_pos_w[env_ids, :]
        eef_quat_current = self._robot.data.body_link_quat_w[env_ids, self._eef_link_index, :]
        
        # 预抓取位置（从上方接近）
        pre_grasp_height = 0.1
        pre_grasp_offset = torch.tensor([0.0, 0.0, pre_grasp_height], device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_pre_grasp = eff_offset + target_position + pre_grasp_offset
        
        # 抓取位置
        pos_grasp = eff_offset + target_position
        
        # 抬起位置
        lift_offset = torch.tensor([0, 0, self.cfg.lift_height_desired], device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_lift = pos_grasp + lift_offset
        
        # ========== 手指位置 ==========
        hand_pos_open = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index]
        hand_pos_closed = self._joint_limit_lower[env_ids, :][:, self._hand_joint_index]
        hand_pos_closed[:, 0] = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index[0]]
        
        # ========== 生成轨迹（4阶段）==========
        for step in range(total_steps):
            
            # === 阶段1: 接近 - 移动到预抓取位置 ===
            if step < approach_end:
                t_normalized = step / max(approach_end, 1)
                t_interp = self._quasi_linear_interpolation(torch.tensor(t_normalized, device=self.device))
                
                pos_interp = eef_pos_current + t_interp * (pos_pre_grasp - eef_pos_current)
                
                # 姿态插值
                t_batch = torch.full((env_len,), t_interp.item(), device=self.device)
                quat_interp = self._slerp_batch(eef_quat_current, eff_quat, t_batch)
                
                # 手指保持打开
                hand_interp = hand_pos_open
                
            # === 阶段2: 下降 - 从预抓取位置下降到抓取位置 ===
            elif step < descend_end:
                t_normalized = (step - approach_end) / max(descend_end - approach_end, 1)
                t_interp = self._quasi_linear_interpolation(torch.tensor(t_normalized, device=self.device))
                
                pos_interp = pos_pre_grasp + t_interp * (pos_grasp - pos_pre_grasp)
                quat_interp = eff_quat
                
                # 手指保持打开
                hand_interp = hand_pos_open
                
            # === 阶段3: 手指闭合 - 保持在抓取位置，手指渐进闭合 ===
            elif step < grasp_close_end:
                # 位置保持在抓取位置
                pos_interp = pos_grasp
                quat_interp = eff_quat
                
                # 手指渐进闭合（使用准线性插值，速度均匀）
                t_finger = (step - descend_end) / max(grasp_close_end - descend_end, 1)
                t_finger_interp = self._quasi_linear_interpolation(
                    torch.tensor(t_finger, device=self.device), 
                    smooth_ratio=0.08  # 手指用稍大的平滑比例，更柔和
                )
                hand_interp = hand_pos_open + t_finger_interp * (hand_pos_closed - hand_pos_open)
                
            # === 阶段4: 抬起 - 手指已闭合，从抓取位置抬起 ===
            else:
                t_normalized = (step - grasp_close_end) / max(lift_end - grasp_close_end, 1)
                t_interp = self._quasi_linear_interpolation(torch.tensor(t_normalized, device=self.device))
                
                pos_interp = pos_grasp + t_interp * (pos_lift - pos_grasp)
                quat_interp = eff_quat
                
                # 手指保持完全闭合
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

        if self.cfg.print_eef_pose:
            # ========== 输出末端执行器位置和旋转 ==========
            # 获取当前末端执行器位姿（世界坐标系）
            eef_pos_w = self._robot.data.body_link_pos_w[:, self._eef_link_index, :]
            eef_quat_w = self._robot.data.body_link_quat_w[:, self._eef_link_index, :]
            # 获取物体位置
            object_pos_w = self._target.data.root_pos_w
            # 计算相对位置（EEF相对于物体）
            eef_pos_rel = eef_pos_w - object_pos_w
            # 打印第一个环境的信息（避免输出过多）
            pos_rel = eef_pos_rel[0].cpu().numpy()
            quat = eef_quat_w[0].cpu().numpy()  # wxyz 格式
            target_pos = eef_pose_target[0, :3].cpu().numpy()
            target_quat = eef_pose_target[0, 3:7].cpu().numpy()
            step = self._episode_step[0].item()
            # 四元数转欧拉角 (wxyz -> xyzw for scipy, then to degrees)
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]  # wxyz -> xyzw
            euler_deg = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
            target_quat_xyzw = [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
            target_euler_deg = R.from_quat(target_quat_xyzw).as_euler('xyz', degrees=True)
            print(f"[Step {step:3d}] EEF相对物体: [{pos_rel[0]:7.4f}, {pos_rel[1]:7.4f}, {pos_rel[2]:7.4f}] | "
                f"目标: [{target_pos[0]:7.4f}, {target_pos[1]:7.4f}, {target_pos[2]:7.4f}] | "
                f"角度(xyz): [{euler_deg[0]:7.2f}°, {euler_deg[1]:7.2f}°, {euler_deg[2]:7.2f}°]")

        # self._target_dof_vel = (self._curr_targets[:, self._robot_index]- self._robot.data.joint_pos[:, self._robot_index]) / self.cfg.sim.dt
        # self._robot.set_joint_position_target(self._curr_targets[:, self._arm_joint_index], joint_ids=self._arm_joint_index)
        # self._robot.set_joint_velocity_target(self._target_dof_vel, joint_ids=self._arm_joint_index)

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
        1. 物体被抬起到目标高度附近（±5cm）
        2. 物体与机器人保持接触
        3. 物体朝向偏离初始朝向不超过阈值
        """
        # 1. 检查抬起高度（只需高于目标高度即可）
        height_lift = self._target.data.root_pos_w[:, 2] - self._target_pos_init[:, 2]
        height_check = height_lift >= (self.cfg.lift_height_desired * 0.8)
        
        # 2. 检查接触状态
        contact_force_num = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        for sensor_name, contact_sensor in self._contact_sensors.items():
            forces = torch.sum(contact_sensor.data.net_forces_w, dim=[1, 2])
            contact_force_num = torch.where(forces > 0.0, contact_force_num + 1, contact_force_num)
        contacting = contact_force_num > 0
        
        # 3. 检查朝向偏差
        current_quat = self._target.data.root_quat_w  # 当前朝向 (wxyz)
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        # 综合判断：高度 AND 接触 AND 朝向
        bsuccessed = height_check & contacting & orientation_check
        
        return bsuccessed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # task evalutation
        bfailed, self._has_contacted = eval_fail(self._target, self._contact_sensors, self._has_contacted)  # type: ignore
        # success eval（使用新的带朝向检查的函数）
        bsuccessed = self._eval_success_with_orientation()
     
        # update success number
        self._episode_success_num += len(torch.nonzero(bsuccessed == True).squeeze(1).tolist())

        return bsuccessed, bfailed, time_out  # type: ignore
    
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
        

        # 根据配置选择轨迹生成模式
        trajectory_mode = getattr(self.cfg, 'trajectory_mode', 'smooth')
        
        if trajectory_mode == "constant_velocity":
            # 恒速轨迹 - 推荐用于 Diffusion Policy 训练
            self.create_trajectory_constant_velocity(env_ids)
        elif trajectory_mode == "smooth" or self.cfg.enable_trajectory_smooth:
            # 平滑轨迹 - 使用 minimum jerk，有明显加减速
            self.create_trajectory_smooth(env_ids)
        else:
            # 原始轨迹
            self.create_trajectory(env_ids)



        # reset variables
        self._episode_step[env_ids] = torch.zeros_like(self._episode_step[env_ids])
        self._target_pos_init[env_ids, :] = self._target.data.root_link_pos_w[env_ids, :].clone()
        self._target_quat_init[env_ids, :] = self._target.data.root_quat_w[env_ids, :].clone()  # 保存初始朝向
        self._has_contacted[env_ids] = torch.zeros_like(self._has_contacted[env_ids], device=self.device, dtype=torch.bool)  # type: ignore

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
            # 显示目标进度（如果设置了目标）
            if self.cfg.target_success_count and self.cfg.target_success_count > 0:
                info += f" | 目标: {self._episode_success_num}/{self.cfg.target_success_count}"
            print(info, end='\r')
        
        # 检查是否达到目标成功次数
        self._check_target_reached()
    
    def _check_target_reached(self):
        """检查是否达到目标成功次数，如果达到则停止程序"""
        if self.cfg.target_success_count and self.cfg.target_success_count > 0:
            if self._episode_success_num >= self.cfg.target_success_count:
                print(f"\n\n{'='*60}")
                print(f"🎉 已达到目标成功次数: {self._episode_success_num}/{self.cfg.target_success_count}")
                if self._episode_num > 0:
                    success_rate = float(self._episode_success_num) / float(self._episode_num) * 100
                    print(f"📊 最终成功率: {success_rate:.2f}%")
                if self.cfg.enable_output:
                    record_time = self._timer.run_time() / 60.0
                    print(f"⏱️  总耗时: {record_time:.2f} 分钟")
                    if record_time > 0:
                        print(f"📈 采集效率: {self._episode_success_num / record_time:.2f} 条/分钟")
                print(f"{'='*60}\n")
                # 退出程序
                import sys
                sys.exit(0)
