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
from ..config_loader import load_pick_place_config

# ========== 任务配置（修改这里即可切换不同任务）==========
TARGET_OBJECT_NAME = "erlenmeyer_flask_with_stopper"  # 目标物体名称
TASK_TYPE = "pick_place"       # 任务类型：pick_place

# 数据根目录：统一存储到 chembench/data 下
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../data"))


@configclass
class PickPlaceEnvCfg(MPEnvCfg):
    """Configuration for Pick and Place environment."""

    # fake params
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # 
    episode_length_s = 3  # 增加时间以容纳放置阶段
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
            solver_type = 1,
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
    sim.render.rendering_mode = "quality"
    sim.render.antialiasing_mode = "TAA"

    # ========== 抓取配置 ==========
    target_object_name: str = TARGET_OBJECT_NAME
    grasp_offset: list = None  # type: ignore
    grasp_euler_deg: list = None  # type: ignore
    
    # ========== Pick and Place 特有配置 ==========
    # 抬起高度（单位：米，中间点的高度）
    lift_height_desired: float = 0.15
    
    # 放置偏移 [x, y, z]（相对于计算出的放置位置的偏移）
    place_offset: list = [0, 0, 0.0]  # type: ignore
    
    # 放置角度（欧拉角，度）
    place_euler_deg: list = None  # type: ignore
    
    # 轨迹时序参数（7阶段）
    phase_ratios: dict = None  # type: ignore
    
    # 手指闭合方式
    smooth_finger_close: bool = True
    # 手指抓取模式：
    # - "all": 所有手指都闭合（默认）
    # - "pinch": 只有拇指和食指闭合（对应索引 0,1 和 5）
    # - "no_thumb": 其他手指闭合，大拇指不闭合
    finger_grasp_mode: str = "pinch"
    
    # 轨迹模式
    trajectory_mode: str = "smooth"
    
    # ========== 预抓取位置参数（侧上方接近） ==========
    # pre_grasp_height: float = 0.05   # z轴上方偏移（米）
    # pre_grasp_y_offset: float = 0.00  # y轴偏移（米）
    # pre_grasp_x_offset: float = 0.00  # x轴偏移（米）


    pre_grasp_height: float = 0.05
    pre_grasp_y_offset: float = -0.02
    pre_grasp_x_offset: float = -0.02


    # pre_grasp_height: float = 0.05
    # pre_grasp_y_offset: float = -0.02
    # pre_grasp_x_offset: float = -0.02

    # ========== 释放后中间点参数（避免撞物体） ==========
    post_release_height: float = 0.02   # z轴上方偏移（米）
    post_release_y_offset: float = -0.04  # y轴偏移（米）
    post_release_x_offset: float = -0.04  # x轴偏移（米）
    
    # 成功判断阈值
    orientation_threshold: float = 0.1
    place_position_threshold: float = 0.025  # 放置位置误差阈值（米）
    
    # 物体目标位置偏移（相对于初始位置）：[x_offset, y_offset]（单位：米）
    # target_xy_offset: list = [0.10, 0.08]  # type: ignore  # x方向0cm，y方向5cm
    target_xy_offset: list = [0.08, 0.08]  # type: ignore  # x方向0cm，y方向5cm
    
    # 目标成功次数
    target_success_count: int = 50
    
    # 输出配置
    output_folder: str = None  # type: ignore
    print_eef_pose: bool = False
    
    def __post_init__(self):
        """
        初始化后从 object_config.json 加载默认参数
        
        输出路径格式：chembench/data/motion_plan/pick_place/{物体名称}
        """
        # 从 JSON 加载 Pick and Place 配置
        pp_config = load_pick_place_config(self.target_object_name)
        
        # 设置抓取偏移（如果未手动指定）
        # if self.grasp_offset is None:
        self.grasp_offset = pp_config["grasp_offset"]
        
        # 设置抓取角度（如果未手动指定）
        # if self.grasp_euler_deg is None:
        self.grasp_euler_deg = pp_config["grasp_euler_deg"]
        
        # 设置抬起高度（从 pick_place 读取）
        # if self.lift_height_desired == 0.25:
        # self.lift_height_desired = pp_config.get("lift_height", 0.25)
        
        # 设置放置偏移
        # if self.place_offset == [0, 0, 0.02]:
        self.place_offset = pp_config.get("place_offset", [0, 0, 0.02])
        
        # 设置放置角度（如果未手动指定）
        # if self.place_euler_deg is None:
        self.place_euler_deg = pp_config.get("place_euler_deg", self.grasp_euler_deg)
        
        # 设置轨迹时序参数（从 pick_place.timing 读取）
        if self.phase_ratios is None:
            timing = pp_config.get("timing", {
                "approach": 0.15,
                "grasp": 0.10,
                "lift": 0.15,
                "transport": 0.15,
                "release": 0.10,
                "post_release": 0.10,  # 新增：释放后中间点
                "retreat": 0.25
            })
            # 确保包含 post_release 键（向后兼容旧的 JSON 配置）
            if 'post_release' not in timing:
                old_retreat = timing.get('retreat', 0.35)
                # post_release 占 retreat 的 40%，retreat 保留 60%
                timing['post_release'] = old_retreat * 0.4
                timing['retreat'] = old_retreat * 0.6
            self.phase_ratios = timing
        else:
            # 如果 phase_ratios 已经设置，也确保包含 post_release 键
            if 'post_release' not in self.phase_ratios:
                old_retreat = self.phase_ratios.get('retreat', 0.35)
                # post_release 占 retreat 的 40%，retreat 保留 60%
                self.phase_ratios['post_release'] = old_retreat * 0.4
                self.phase_ratios['retreat'] = old_retreat * 0.6
        
        # 设置输出文件夹
        if self.output_folder is None:
            object_name = pp_config.get("name_cn", self.target_object_name)
            self.output_folder = os.path.join(DATA_ROOT, "motion_plan", TASK_TYPE, object_name)

class PickPlaceEnv(MPEnv):

    cfg: PickPlaceEnvCfg

    def __init__(self, cfg: PickPlaceEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["bottle"]
        self._visualizer = self.scene.visualizer
        # self._target = self.scene.articulated_objects["bottle"]
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
        
        # ========== Pick and Place 关键参数 ==========
        # 将配置中的 target_xy_offset 转换为 tensor
        self._target_xy_offset = torch.tensor(self.cfg.target_xy_offset, dtype=torch.float32, device=self.device)

        self.sim.set_camera_view([-0.7, -5.2, 1.3], [-1.2, -5.2, 1.1])
        # 设置 RTX 渲染选项
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
        # 3. 可选：确保透明物体参与 Primary Ray Hit
        carb_settings_iface.set_bool("/rtx/hydra/segmentation/includeTransparent", True)

        # self.sim.render.rendering_mode = "quality"
        # self.sim.render.antialiasing_mode = "TAA"

    def _marker_visualizer(self):
        """
        可视化关键位置（Pick and Place）：
        1. 物体当前位置和旋转
        2. 中间抬高点的位置和旋转（物体应该被抬到的位置）
        3. 最终物体目标位置和旋转（物体应该被放置的位置）
        """
        if self.cfg.enable_marker and self._visualizer is not None:
            # 1. 物体当前位置和旋转
            target_current_pos = self._target.data.root_com_state_w[0:1, :3]
            target_current_pos[:,2] = target_current_pos[:, 2]  +  0.05
            target_current_quat = self._target.data.root_com_state_w[0:1, 3:7]
            
            # target_current_pos = self._target.data.root_pos_w[0:1, :] - self._robot.data.root_link_pos_w[0:1, :]
        
            # 2. 中间抬高点的位置和旋转
            # 计算中间抬高点（与轨迹生成逻辑一致）
            target_position_init = self._target_pos_init[0:1, :]  # 物体初始位置 
            target_xy_offset = self._target_xy_offset.unsqueeze(0)
            
            place_xy = target_position_init[:, :2] + target_xy_offset
            mid_xy = (target_position_init[:, :2] + place_xy) / 2.0
            # mid_xy = place_xy
            mid_z = target_position_init[:, 2] + self.cfg.lift_height_desired
            lift_pos = torch.cat([mid_xy, mid_z.unsqueeze(1)], dim=1)
            # 抬高点使用与物体相同的旋转
            lift_quat = target_current_quat.clone()
            
            # 3. 最终物体目标位置和旋转
            # 计算物体最终应该在的位置（不包含 place_offset，那是末端执行器的偏移）
            object_final_pos = torch.cat([place_xy, target_position_init[:, 2:3]], dim=1)
            object_final_pos[:,2] = 1.02
            # 最终位置使用与物体相同的旋转
            object_final_quat = target_current_quat.clone()
            
            # 拼接所有标记点
            marker_pos = torch.cat((
                target_current_pos,   # 物体当前位置
                lift_pos,             # 中间抬高点
                object_final_pos,     # 物体最终目标位置
            ), dim=0)
            
            marker_rot = torch.cat((
                target_current_quat,  # 物体当前旋转
                lift_quat,            # 中间抬高点旋转
                object_final_quat,    # 物体最终旋转
            ), dim=0)

            self._visualizer.visualize(marker_pos, marker_rot)

            
    def create_pick_place_trajectory(self, env_ids: torch.Tensor | None):
        """
        创建 Pick and Place 轨迹（8阶段，新增释放后中间点）
        
        放置位置计算逻辑：
        - 放置xy位置 = 物体初始xy + (0.01, 0.10)  # x+1cm, y+10cm
        - 中间抬高点xy = (物体初始xy + 放置xy) / 2  # 中点
        - 中间抬高点z = cfg.lift_height_desired  # 预设高度
        - 最终放置点 = 放置xy位置 + cfg.place_offset  # 加上放置偏移
        
        轨迹阶段（改进抓取流程，与 grasp_mp 一致）：
        1. approach: 从当前位置移动到预抓取位置（侧上方接近）
        2. descend: 从预抓取位置下降到抓取位置
        3. dwell: 在抓取位置稳定等待（让IK收敛）
        4. grasp: 手指闭合抓取物体
        5. lift: 抬起到中间抬高点
        6. transport: 从中间点运输到最终放置位置
        7. release: 松开手指
        8. post_release: 移动到释放后中间点（避免撞到物体）
        9. retreat: 撤回到最终安全位置
        """
        env_len = env_ids.shape[0]
        total_steps = self.max_episode_length
        
        # ========== 阶段时间分配 ==========
        # 将 approach 时间拆分为 approach (40%) + descend (50%) + dwell (10%)
        cfg_ratios = self.cfg.phase_ratios
        approach_total = cfg_ratios['approach']
        grasp_total = cfg_ratios['grasp']
        
        approach_end = int(approach_total * 0.4 * total_steps)  # 接近预抓取位置
        descend_end = approach_end + int(approach_total * 0.5 * total_steps)  # 下降到抓取位置
        dwell_end = descend_end + int(approach_total * 0.10 * total_steps)  # 稳定等待
        grasp_end = dwell_end + int(grasp_total * total_steps)
        lift_end = grasp_end + int(cfg_ratios['lift'] * total_steps)
        transport_end = lift_end + int(cfg_ratios['transport'] * total_steps)
        release_end = transport_end + int(cfg_ratios['release'] * total_steps)
        post_release_end = release_end + int(cfg_ratios['post_release'] * total_steps)  # 新增
        retreat_end = total_steps
        
        # ========== 抓取姿态配置 ==========
        grasp_euler_deg = self.cfg.grasp_euler_deg
        grasp_offset = self.cfg.grasp_offset
        
        quat_xyzw = R.from_euler('xyz', grasp_euler_deg, degrees=True).as_quat()
        eff_quat_grasp = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], 
                                       dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # ========== 放置姿态配置 ==========
        place_euler_deg = self.cfg.place_euler_deg
        place_quat_xyzw = R.from_euler('xyz', place_euler_deg, degrees=True).as_quat()
        eff_quat_place = torch.tensor([place_quat_xyzw[3], place_quat_xyzw[0], place_quat_xyzw[1], place_quat_xyzw[2]], 
                                       dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # ========== 计算关键位置 ==========
        # 物体初始位置（相对机器人基座）
        target_position_init = self._target.data.root_pos_w[env_ids, :] - self._robot.data.root_link_pos_w[env_ids, :]
        
        eff_offset = torch.tensor(grasp_offset, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # 当前末端执行器位姿
        eef_pos_current = self._robot.data.body_link_pos_w[env_ids, self._eef_link_index, :] - self._robot.data.root_link_pos_w[env_ids, :]
        eef_quat_current = self._robot.data.body_link_quat_w[env_ids, self._eef_link_index, :]
        
        # 0. 预抓取位置（抓取点上方 + 侧面偏移，从侧上方接近）
        pre_grasp_offset = torch.tensor([self.cfg.pre_grasp_x_offset, self.cfg.pre_grasp_y_offset, self.cfg.pre_grasp_height], 
                                        device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_pre_grasp = eff_offset + target_position_init + pre_grasp_offset
        
        # 1. 抓取位置
        pos_grasp = eff_offset + target_position_init
        
        # 2. 放置xy位置 = 物体初始xy + target_xy_offset
        target_xy_offset = self._target_xy_offset.unsqueeze(0).repeat(env_len, 1)
        place_xy = target_position_init[:, :2] + target_xy_offset
        
        # 物体的最终放置位置（物体中心应该在的位置）
        object_place_position = torch.cat([place_xy, target_position_init[:, 2:3]], dim=1)
        
        # 3. 中间抬高点 = xy中点 + (物体初始高度 + 抬高高度)
        mid_xy = (target_position_init[:, :2] + place_xy) / 2.0
        # mid_xy = place_xy
        mid_z = target_position_init[:, 2] + self.cfg.lift_height_desired  # 相对于物体初始高度
        pos_lift = torch.cat([mid_xy, mid_z.unsqueeze(1)], dim=1)
        pos_lift += eff_offset

        # 4. 末端执行器的最终放置位置 = 物体放置位置 + place_offset（类似抓取时的 grasp_offset）
        place_offset_tensor = torch.tensor(self.cfg.place_offset, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_place = object_place_position + place_offset_tensor  # 末端执行器位置
        
        # 5. 释放后中间点（避免直接上移撞到物体）
        post_release_offset = torch.tensor([self.cfg.post_release_x_offset, self.cfg.post_release_y_offset, self.cfg.post_release_height], 
                                           device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_post_release = pos_place + post_release_offset
        
        # 6. 最终撤回位置（释放后中间点上方再抬高5cm）
        retreat_offset = torch.tensor([0, 0, 0.12], device=self.device).unsqueeze(0).repeat(env_len, 1)
        pos_retreat = pos_post_release + retreat_offset
        
        # ========== 手指位置 ==========
        hand_pos_open = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index]
        hand_pos_closed = self._joint_limit_lower[env_ids, :][:, self._hand_joint_index]
        hand_pos_closed[:, 0] = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index[0]]
        
        if self.cfg.finger_grasp_mode == "pinch":
            hand_pos_closed[:, 2] = hand_pos_open[:, 2]
            hand_pos_closed[:, 3] = hand_pos_open[:, 3]
            hand_pos_closed[:, 4] = hand_pos_open[:, 4]
        
        if self.cfg.finger_grasp_mode == "no_thumb":
            hand_pos_closed[:, 5] = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index[1]]
        
        
        # ========== 生成轨迹 ==========
        for step in range(total_steps):
            if step < approach_end:
                # 阶段1: 接近 - 从当前位置移动到预抓取位置（侧上方）
                t_normalized = step / max(approach_end, 1)
                t_smooth = self._smooth_approach_interpolation(torch.tensor(t_normalized, device=self.device))
                pos_interp = eef_pos_current + t_smooth * (pos_pre_grasp - eef_pos_current)
                t_batch = torch.full((env_len,), t_smooth.item(), device=self.device)
                quat_interp = self._slerp_batch(eef_quat_current, eff_quat_grasp, t_batch)
                hand_interp = hand_pos_open
                
            elif step < descend_end:
                # 阶段2: 下降 - 从预抓取位置下降到抓取位置
                t_normalized = (step - approach_end) / max(descend_end - approach_end, 1)
                t_smooth = self._smooth_approach_interpolation(torch.tensor(t_normalized, device=self.device))
                pos_interp = pos_pre_grasp + t_smooth * (pos_grasp - pos_pre_grasp)
                quat_interp = eff_quat_grasp
                hand_interp = hand_pos_open
                
            elif step < dwell_end:
                # 阶段3: 稳定 - 在抓取位置保持不动，等待IK收敛
                # 位置精确保持在抓取位置（关键：提高抓取精度）
                pos_interp = pos_grasp
                quat_interp = eff_quat_grasp
                hand_interp = hand_pos_open
                
            elif step < grasp_end:
                # 阶段4: 抓取（手指闭合）
                pos_interp = pos_grasp
                quat_interp = eff_quat_grasp
                
                # 手指闭合方式
                if self.cfg.smooth_finger_close:
                    # 平滑闭合：使用最小加加速度插值
                    t_normalized = (step - dwell_end) / max(grasp_end - dwell_end, 1)
                    t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                    hand_interp = hand_pos_open + t_smooth * (hand_pos_closed - hand_pos_open)
                else:
                    # 直接闭合：整个 grasp 阶段都保持闭合状态
                    hand_interp = hand_pos_closed
                
            elif step < lift_end:
                # 阶段5: 抬起到中间抬高点
                t_normalized = (step - grasp_end) / max(lift_end - grasp_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                pos_interp = pos_grasp + t_smooth * (pos_lift - pos_grasp)
                # 姿态从抓取姿态渐变到放置姿态
                t_batch = torch.full((env_len,), t_smooth.item(), device=self.device)
                quat_interp = self._slerp_batch(eff_quat_grasp, eff_quat_place, t_batch)
                hand_interp = hand_pos_closed
                
            elif step < transport_end:
                # 阶段6: 从中间点运输到最终放置位置
                t_normalized = (step - lift_end) / max(transport_end - lift_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                pos_interp = pos_lift + t_smooth * (pos_place - pos_lift)
                quat_interp = eff_quat_place
                hand_interp = hand_pos_closed
                
            elif step < release_end:
                # 阶段7: 释放（手指打开）
                pos_interp = pos_place
                quat_interp = eff_quat_place
                t_normalized = (step - transport_end) / max(release_end - transport_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                hand_interp = hand_pos_closed + t_smooth * (hand_pos_open - hand_pos_closed)
                
            elif step < post_release_end:
                # 阶段8: 移动到释放后中间点（避免撞到物体）
                t_normalized = (step - release_end) / max(post_release_end - release_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                pos_interp = pos_place + t_smooth * (pos_post_release - pos_place)
                quat_interp = eff_quat_place
                hand_interp = hand_pos_open
                
            else:
                # 阶段9: 撤回到最终安全位置
                t_normalized = (step - post_release_end) / max(retreat_end - post_release_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                pos_interp = pos_post_release + t_smooth * (pos_retreat - pos_post_release)
                quat_interp = eff_quat_place
                hand_interp = hand_pos_open
            
            # 存储轨迹点
            self._eef_pose_target[env_ids, step, :3] = pos_interp
            self._eef_pose_target[env_ids, step, 3:7] = quat_interp
            self._hand_pos_target[env_ids, step, :] = hand_interp

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
            # print(f"[Step {step:3d}] EEF相对物体: [{pos_rel[0]:7.4f}, {pos_rel[1]:7.4f}, {pos_rel[2]:7.4f}] | "
            #     f"目标: [{target_pos[0]:7.4f}, {target_pos[1]:7.4f}, {target_pos[2]:7.4f}] | "
            #     f"角度(xyz): [{euler_deg[0]:7.2f}°, {euler_deg[1]:7.2f}°, {euler_deg[2]:7.2f}°]")

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
        
        self._marker_visualizer()
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


        ##是否实时查看mask图片
        # import matplotlib.pyplot as plt
        # head_camera_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][0,:,:,:].cpu().numpy()
        # # head_camera_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][0,:,:,:].cpu().numpy()
        # # head_camera_depth = self._robot.tiled_cameras["chest_camera"].data.output["depth"][0,:,:,:].cpu().numpy()
        # # head_camera_normal = self._robot.tiled_cameras["chest_camera"].data.output["normals"][0,:,:,:].cpu().numpy()
       
        # # plt.figure(figsize=(10, 10))
        # plt.imshow(head_camera_mask)
        # # plt.subplot(1, 3, 2)
        # # plt.imshow(head_camera_depth)
        # # plt.subplot(1, 3, 3)
        # # plt.imshow(head_camera_normal)
        # plt.show()
        # import time
        # time.sleep(0.1)


        # chest_camera_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"]
        
        # ========== 实时打印末端执行器信息 ==========
        if self._sim_step_counter % 10 == 0:  # 每10步打印一次，避免输出过多
            # 获取末端执行器位姿（世界坐标系）
            eef_pos_w = self._robot.data.body_link_pos_w[0, self._eef_link_index, :]
            eef_quat_w = self._robot.data.body_link_quat_w[0, self._eef_link_index, :]
            
            # 获取物体当前位置（世界坐标系）
            object_pos_w = self._target.data.root_pos_w[0, :]
            
            # 计算末端执行器相对物体当前位置的偏移
            eef_offset_current = eef_pos_w - object_pos_w
            
            # 计算物体的最终目标位置（世界坐标系）
            # 物体初始位置（世界坐标）
            target_pos_init_w = self._target_pos_init[0, :]
            # 目标xy偏移
            target_xy_offset = self._target_xy_offset
            # 最终目标xy位置
            place_xy = target_pos_init_w[:2] + target_xy_offset
            # 物体的最终目标位置（物体中心应该在的位置）
            object_place_position_w = torch.cat([place_xy, target_pos_init_w[2:3]], dim=0)
            
            # 计算末端执行器相对物体最终目标位置的偏移
            eef_offset_target = eef_pos_w - object_place_position_w
            
            # 计算物体当前位置到最终目标位置的距离
            object_to_target_distance = torch.norm(object_pos_w - object_place_position_w).item()
            
            # 四元数转欧拉角 (wxyz -> xyzw for scipy, then to degrees)
            quat_wxyz = eef_quat_w.cpu().numpy()
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]  # wxyz -> xyzw
            euler_deg = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
            
            # 获取当前episode步数
            current_step = self._episode_step[0].item()
            
            # print(f"\n[Step {current_step:3d} | Sim {self._sim_step_counter:5d}]")
            # print(f"  EEF→物体当前: [{eef_offset_current[0]:7.4f}, {eef_offset_current[1]:7.4f}, {eef_offset_current[2]:7.4f}]")
            # print(f"  EEF→放置目标: [{eef_offset_target[0]:7.4f}, {eef_offset_target[1]:7.4f}, {eef_offset_target[2]:7.4f}]")
            # print(f"  物体→目标距离: {object_to_target_distance:7.4f}m")
            # print(f"  EEF角度(xyz): [{euler_deg[0]:7.2f}°, {euler_deg[1]:7.2f}°, {euler_deg[2]:7.2f}°]")
        
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
        height_diff = current_z - self._target_pos_init[:, 2]
        
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

    def _eval_success_pick_place(self) -> torch.Tensor:
        """
        评估 Pick and Place 是否成功
        
        成功条件：
        1. 物体被放置在目标位置附近（xy平面误差小于阈值）
        2. 物体高度与初始高度差距不大于1cm（确保在桌面上）
        3. 物体朝向偏离不超过阈值
        4. 手指已松开（所有手指接触力为0）
        5. 末端执行器已抬高到物体初始高度 + 0.15米
        
        目标位置（物体中心）= 初始位置 + target_xy_offset
        """
        # 1. 计算目标放置位置（物体中心应该在的位置）
        initial_xy = self._target_pos_init[:, :2]
        target_xy_offset = self._target_xy_offset.unsqueeze(0).repeat(self.num_envs, 1)
        target_xy = initial_xy + target_xy_offset  # 物体中心的目标xy位置
        
        # 2. 检查物体当前位置（xy平面）
        current_xy = self._target.data.root_pos_w[:, :2]
        pos_error = torch.norm(current_xy - target_xy, dim=1)
        position_check = pos_error < self.cfg.place_position_threshold
        
        # 3. 检查物体高度（确保在桌面上，不是悬空或掉落）
        current_z = self._target.data.root_pos_w[:, 2]
        initial_z = self._target_pos_init[:, 2]
        height_diff = torch.abs(current_z - initial_z)
        height_check = height_diff < 0.01  # 高度差小于1cm
        
        # 4. 检查朝向
        current_quat = self._target.data.root_quat_w
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        # 5. 检查手指是否已松开（所有接触力为0）
        contact_force_num = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        for sensor_name, contact_sensor in self._contact_sensors.items():
            forces = torch.sum(contact_sensor.data.net_forces_w, dim=[1, 2])  # type: ignore
            force_magnitude = torch.abs(forces)  # 使用力的绝对值（大小）
            contact_force_num = torch.where(
                force_magnitude > 0.1,
                contact_force_num + 1,
                contact_force_num
            )
        
        # 手指已松开 = 接触力数量为0
        released = contact_force_num < 0.1  # type: ignore
        
        # 6. 检查末端执行器是否已抬高到安全高度
        eef_z = self._robot.data.body_link_pos_w[:, self._eef_link_index, 2]
        required_eef_height = self._target_pos_init[:, 2] + 0.08 # 物体初始高度 + 0.15米
        eef_height_check = eef_z >= required_eef_height
        
        # 综合判断（所有条件都要满足）
        bsuccessed = position_check & height_check & orientation_check & released & eef_height_check
        
        return bsuccessed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # task evalutation
        bfailed, self._has_contacted = eval_fail(self._target, self._contact_sensors, self._has_contacted)  # type: ignore
        
        # 新增：检测物体在未接触时被移动（被推动/碰撞）
        bfailed_moved = self._eval_fail_moved_without_contact()
        # bfailed = bfailed | bfailed_moved
        bfailed = bfailed_moved
        
        # success eval（使用 Pick and Place 的成功判断）
        bsuccessed = self._eval_success_pick_place()
     
        # update success number
        self._episode_success_num += len(torch.nonzero(bsuccessed == True).squeeze(1).tolist())

        return bsuccessed, bfailed, time_out  # type: ignore
    
    def _reset_idx(self, env_ids: torch.Tensor | None, success_ids:Sequence[int]|None=None):

        # get env indexs
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if success_ids is None:
            success_ids=[]
        
        # 截断 logic: 确保保存的数据不超过 target_success_count
        ids_to_save = success_ids
        if self.cfg.enable_output and self.cfg.target_success_count and self.cfg.target_success_count > 0:
            # 注意: _get_dones 已经更新了 _episode_success_num，包含了当前 batch
            current_total = self._episode_success_num
            batch_size = len(success_ids)
            prev_total = current_total - batch_size
            
            if prev_total < self.cfg.target_success_count:
                needed = self.cfg.target_success_count - prev_total
                if batch_size > needed:
                    ids_to_save = success_ids[:needed]
            else:
                ids_to_save = []

        # output data
        if self.cfg.enable_output:
            # 记录保存前的时间戳，用于推断文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._data = save_data(
                data=self._data,
                cfg=self.cfg,
                scene=self.scene,
                env_indexs=ids_to_save,
                reset_env_indexs=env_ids.tolist(),
            )
            # 打印保存的文件路径
            if ids_to_save:
                saved_path = f"{self.cfg.output_folder}/{timestamp}_data.hdf5"
                print(f"[DATA] 已保存数据: {saved_path} (成功轨迹数: {len(ids_to_save)})")


        super()._reset_idx(env_ids)   

        # print logs
        if self.cfg.enable_log:
            self._log_info()
        

        # 根据配置选择轨迹生成模式
        trajectory_mode = getattr(self.cfg, 'trajectory_mode', 'smooth')
        
        # Pick and Place 使用专用轨迹生成函数
        self.create_pick_place_trajectory(env_ids)



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
