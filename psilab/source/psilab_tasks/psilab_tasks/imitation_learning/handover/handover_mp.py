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
from ..config_loader import load_handover_config

# ========== 任务配置（修改这里即可切换不同任务）==========
TARGET_OBJECT_NAME = "clear_volumetric_flask_250ml"  # 目标物体名称
TASK_TYPE = "handover"       # 任务类型：handover

# 数据根目录：统一存储到 chembench/data 下
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../data"))


@configclass
class HandoverEnvCfg(MPEnvCfg):
    """Configuration for Handover environment (双手传递任务)."""

    # fake params
    action_scale = 0.5
    action_space = 26  # 双手：2个手臂 + 2个手 = 26
    observation_space = 260
    state_space = 260

    # 
    episode_length_s = 3.5 # 增加时间以容纳双手传递阶段
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

    # ========== 右手抓取配置 ==========
    target_object_name: str = TARGET_OBJECT_NAME
    right_grasp_offset: list = None  # type: ignore  # 右手抓取偏移
    right_grasp_euler_deg: list = None  # type: ignore  # 右手抓取角度
    
    # ========== 右手交接位置配置 ==========
    right_handover_offset: list = None  # type: ignore  # 右手在交接位置的偏移
    right_handover_euler_deg: list = None  # type: ignore  # 右手在交接位置的角度
    
    # ========== 左手接收配置 ==========
    left_grasp_offset: list = None  # type: ignore  # 左手抓取偏移
    left_grasp_euler_deg: list = None  # type: ignore  # 左手抓取角度
    
    # ========== 右手复位配置 ==========
    right_retreat_distance: float = 0.1  # 右手撤回距离（沿y轴负方向，单位：米）
    
    # ========== 右手预抓取位置参数（侧上方接近） ==========

    right_pre_grasp_height: float = 0.02
    right_pre_grasp_y_offset: float = -0.01
    right_pre_grasp_x_offset: float = -0.095

    ##量筒和容量瓶
    # right_pre_grasp_height: float = 0.02
    # right_pre_grasp_y_offset: float = -0.01
    # right_pre_grasp_x_offset: float = -0.04
    
    ##正常
    # right_pre_grasp_height: float = 0.02
    # right_pre_grasp_y_offset: float = -0.01
    # right_pre_grasp_x_offset: float = -0.04

    # right_pre_grasp_height: float = 0.05
    # right_pre_grasp_y_offset: float = -0.02
    # right_pre_grasp_x_offset: float = -0.02


   # ========== 左手预抓取位置参数 ==========    
    left_pre_grasp_height: float = 0.0
    left_pre_grasp_y_offset: float = 0.08
    left_pre_grasp_x_offset: float = -0.025
    
    ##正常
    # left_pre_grasp_height: float = 0.03   # z轴上方偏移（米）
    # left_pre_grasp_y_offset: float = 0.08  # y轴偏移（米）
    # left_pre_grasp_x_offset: float = -0.025  # x轴偏移（米）

    ##250ml棕色容量瓶
    # left_pre_grasp_height: float = 0.00   # z轴上方偏移（米）
    # left_pre_grasp_y_offset: float = 0.08  # y轴偏移（米）
    # left_pre_grasp_x_offset: float = -0.025  # x轴偏移（米）

   ##500ml量筒
    # left_pre_grasp_height: float = 0.02   # z轴上方偏移（米）
    # left_pre_grasp_y_offset: float = 0.07  # y轴偏移（米）
    # left_pre_grasp_x_offset: float = -0.016  # x轴偏移（米）



    # ========== 右手释放后中间点参数（避免撞物体） ==========
    right_post_release_height: float = 0.02
    right_post_release_y_offset: float = -0.04
    right_post_release_x_offset: float = -0.04
    ##正常
    # right_post_release_height: float = 0.02   # z轴上方偏移（米）
    # right_post_release_y_offset: float = -0.04  # y轴偏移（米）
    # right_post_release_x_offset: float = -0.04  # x轴偏移（米）

    ##量筒
    # right_post_release_height: float = 0.03
    # right_post_release_y_offset: float = -0.05
    # right_post_release_x_offset: float = -0.08
    
    # ========== Handover 特有配置 ==========
    # 传递位置 [x, y, z]（世界坐标系，会在代码中转换为相对机器人基座的坐标）
    # x: 前后（正值向前），y: 左右（正值向左），z: 上下（正值向上）
    # 默认值：世界坐标系中的绝对位置
    # handover_position: list = [-1.17, -5.05, 1.25]  # type: ignore

    # handover_position: list = [-1.17, -4.95, 1.15]  # type: ignore
    # tensor([[-1.6500, -5.0000, -0.0105]], device='cuda:0')
    ##相对机器人的偏移量
    handover_position: list = [0.48, 0.05, 1.1605]  # type: ignore

    # pos=(-1.15 , -5.2    , 1.2 ),
    # 抬起高度（单位：米，右手抬高到传递位置）
    lift_height_desired: float = 0.15
    
    # 轨迹时序参数
    phase_ratios: dict = None  # type: ignore
    
    # 手指闭合方式
    smooth_finger_close: bool = True
    # 手指抓取模式：
    # - "all": 所有手指都闭合（默认）
    # - "pinch": 只有拇指和食指闭合（对应索引 0,1 和 5）
    # - "no_thumb": 其他手指闭合，大拇指不闭合
    finger_grasp_mode: str = "all"
    
    # 轨迹模式
    trajectory_mode: str = "smooth"
    
    # 成功判断阈值
    orientation_threshold: float = 0.1
    handover_position_threshold: float = 0.03  # 传递位置误差阈值（米）
    
    # 目标成功次数
    target_success_count: int = 50
    
    # 输出配置
    output_folder: str = None  # type: ignore
    print_eef_pose: bool = False
    
    def __post_init__(self):
        """
        初始化后从 object_config.json 加载默认参数
        
        输出路径格式：chembench/data/motion_plan/handover/{物体名称}
        """
        # 从 JSON 加载 Handover 配置
        handover_config = load_handover_config(self.target_object_name)
        
        # 设置右手抓取偏移和角度（从handover配置中读取right_grasp_offset和right_grasp_euler_deg）
        if self.right_grasp_offset is None:
            self.right_grasp_offset = handover_config.get("right_grasp_offset", [0, 0, 0])
        if self.right_grasp_euler_deg is None:
            self.right_grasp_euler_deg = handover_config.get("right_grasp_euler_deg", [0, 0, 0])
        
        # 设置右手在交接位置的偏移和角度
        if self.right_handover_offset is None:
            self.right_handover_offset = handover_config.get("right_handover_offset", [0, 0, 0])
        if self.right_handover_euler_deg is None:
            self.right_handover_euler_deg = handover_config.get("right_handover_euler_deg", self.right_grasp_euler_deg)
        
        # 设置左手抓取偏移和角度
        if self.left_grasp_offset is None:
            self.left_grasp_offset = handover_config["left_grasp_offset"]
        if self.left_grasp_euler_deg is None:
            self.left_grasp_euler_deg = handover_config["left_grasp_euler_deg"]
        
        # 注意：handover_position 已在类定义中直接设置，不从 JSON 读取
        # 如果需要调整传递位置，直接修改上面的 handover_position 默认值
        
        # 设置抬起高度（可选：从配置文件覆盖）
        lift_height_json = handover_config.get("lift_height")
        if lift_height_json is not None:
            self.lift_height_desired = lift_height_json
        
        # 设置轨迹时序参数
        if self.phase_ratios is None:
            # self.phase_ratios = handover_config.get("timing", {
            #     "right_approach": 0.12,   # 右手接近（包含 approach + descend + dwell）
            #     "right_grasp": 0.10,      # 右手抓取
            #     "right_lift": 0.15,       # 右手抬起到传递位置
            #     "left_approach": 0.18,    # 左手接近（包含 approach + descend + dwell）
            #     "left_grasp": 0.10,       # 左手抓取
            #     "right_release": 0.08,    # 右手松开
            #     "right_post_release": 0.08,  # 右手移动到释放后中间点
            #     "right_retreat": 0.19     # 右手撤回（沿y轴负方向）
            # })

            self.phase_ratios = {
                "right_approach": 0.15,   # 右手接近（包含 approach + descend + dwell）
                "right_grasp": 0.10,      # 右手抓取
                "right_lift": 0.15,       # 右手抬起到传递位置
                "left_approach": 0.25,    # 左手接近（包含 approach + descend + dwell）
                "left_grasp": 0.10,       # 左手抓取
                "right_release": 0.1,     # 右手松开
                "right_retreat": 0.25     # 右手撤回（包含 post_release + retreat）
            }


        # 设置输出文件夹
        if self.output_folder is None:
            object_name = handover_config.get("name_cn", self.target_object_name)
            self.output_folder = os.path.join(DATA_ROOT, "motion_plan", TASK_TYPE, object_name)

class HandoverEnv(MPEnv):

    cfg: HandoverEnvCfg

    def __init__(self, cfg: HandoverEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        # instances in scene
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["bottle"]
        self._visualizer = self.scene.visualizer
        
        # ========== 双手contact sensors ==========
        # 右手 contact sensors
        self._contact_sensors_right = {}
        for key in ["hand2_link_base",
                    "hand2_link_1_1", "hand2_link_1_2", "hand2_link_1_3",
                    "hand2_link_2_1", "hand2_link_2_2",
                    "hand2_link_3_1", "hand2_link_3_2",
                    "hand2_link_4_1", "hand2_link_4_2",
                    "hand2_link_5_1", "hand2_link_5_2"]:
            self._contact_sensors_right[key] = self.scene.sensors[key]
        
        # 左手 contact sensors
        self._contact_sensors_left = {}
        for key in ["hand1_link_base",
                    "hand1_link_1_1", "hand1_link_1_2", "hand1_link_1_3",
                    "hand1_link_2_1", "hand1_link_2_2",
                    "hand1_link_3_1", "hand1_link_3_2",
                    "hand1_link_4_1", "hand1_link_4_2",
                    "hand1_link_5_1", "hand1_link_5_2"]:
            self._contact_sensors_left[key] = self.scene.sensors[key]

        # joint limit for compute later
        self._joint_limit_lower = self._robot.data.joint_limits[:,:,0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:,:,1].clone()

        # ========== 双手 joint indices ==========
        # 右手（arm2 + hand2）
        self._arm2_joint_index = self._robot.find_joints(self._robot.actuators["arm2"].joint_names, preserve_order=True)[0]
        self._hand2_joint_index = self._robot.find_joints(self._robot.actuators["hand2"].joint_names, preserve_order=True)[0][:6]
        
        # 左手（arm1 + hand1）
        self._arm1_joint_index = self._robot.find_joints(self._robot.actuators["arm1"].joint_names, preserve_order=True)[0]
        self._hand1_joint_index = self._robot.find_joints(self._robot.actuators["hand1"].joint_names, preserve_order=True)[0][:6]
        
        # eef link indices
        self._eef2_link_index = self._robot.find_bodies("arm2_link7")[0][0]  # 右手
        self._eef1_link_index = self._robot.find_bodies("arm1_link7")[0][0]  # 左手

        # total step number
        self._episode_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        
        # ========== 双手轨迹目标 ==========
        # 右手轨迹
        self._eef2_pose_target = torch.zeros((self.num_envs, self.max_episode_length, 7), device=self.device)
        self._hand2_pos_target = torch.zeros((self.num_envs, self.max_episode_length, len(self._hand2_joint_index)), device=self.device)
        
        # 左手轨迹
        self._eef1_pose_target = torch.zeros((self.num_envs, self.max_episode_length, 7), device=self.device)
        self._hand1_pos_target = torch.zeros((self.num_envs, self.max_episode_length, len(self._hand1_joint_index)), device=self.device)
        
        # initialize Timer
        self._timer = Timer()
        
        # variables
        self._has_contacted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_quat_init = torch.zeros((self.num_envs, 4), device=self.device)  # 初始朝向（wxyz）
        
        # ========== Handover 关键参数 ==========
        # 传递位置（从配置文件读取，世界坐标系，使用时会转换为相对机器人基座）
        self._handover_position = torch.tensor(self.cfg.handover_position, dtype=torch.float32, device=self.device)
        
        # 右手撤回距离（沿y轴负方向）
        self._right_retreat_distance = self.cfg.right_retreat_distance
        
        # ========== 阶段边界（用于显示当前阶段） ==========
        self._phase_boundaries = {}  # 将在 create_handover_trajectory 中设置

        self.sim.set_camera_view([-0.7, -5.2, 1.3], [-1.2, -5.2, 1.1])
        # 设置 RTX 渲染选项
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
        # 3. 可选：确保透明物体参与 Primary Ray Hit
        carb_settings_iface.set_bool("/rtx/hydra/segmentation/includeTransparent", True)

        # self.sim.render.rendering_mode = "quality"
        # self.sim.render.antialiasing_mode = "TAA"

        # 可选：设置分割的透明度阈值
        # carb_settings_iface.set_float("/rtx/hydra/segmentation/opacityThreshold", 1.0)


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
    def _marker_visualizer(self):
        """
        可视化 Handover 关键位置：
        1. 物体当前位置和旋转
        2. 传递位置（handover_position）
        """
        if self.cfg.enable_marker and self._visualizer is not None:
            # 1. 物体当前位置和旋转
            target_current_pos = self._target.data.root_com_state_w[0:1, :3]
            target_current_pos[:, 2] = target_current_pos[:, 2] + 0.05
            target_current_quat = self._target.data.root_com_state_w[0:1, 3:7]
            
            # 2. 传递位置（从配置文件读取）
            handover_pos = self._handover_position.unsqueeze(0)
            handover_pos = handover_pos + self._robot.data.root_link_pos_w[0:1, :]


            handover_quat = target_current_quat.clone()  # 使用物体当前旋转
            
            # 拼接所有标记点
            marker_pos = torch.cat((
                target_current_pos,   # 物体当前位置
                handover_pos,         # 传递位置
            ), dim=0)
            
            marker_rot = torch.cat((
                target_current_quat,  # 物体当前旋转
                handover_quat,        # 传递位置旋转
            ), dim=0)

            self._visualizer.visualize(marker_pos, marker_rot)

            
    def create_handover_trajectory(self, env_ids: torch.Tensor | None):
        """
        创建 Handover 双手传递轨迹（改进版，增加预抓取和释放后中间点）
        
        轨迹阶段：
        【右手阶段】
        1. r_approach: 右手移动到预抓取位置（物体上方）
        2. r_descend: 右手下降到抓取位置
        3. r_dwell: 右手稳定在抓取位置
        4. r_grasp: 右手手指闭合抓取物体
        5. r_lift: 右手抬起物体到传递位置
        
        【左手阶段】
        6. l_approach: 左手移动到预抓取位置（传递位置上方/侧方）
        7. l_descend: 左手下降到抓取位置
        8. l_dwell: 左手稳定在抓取位置
        9. l_grasp: 左手手指闭合抓取物体 ⭐ 关键阶段
        
        【交接阶段】（严格按顺序执行）
        10. r_release: 右手手指松开 ⚠️ 必须等待左手抓取完成（step >= l_grasp_end）
        11. r_post_release: 右手移动到释放后中间点（避免撞物体）
        12. r_retreat: 右手沿y轴负方向撤回
        
        ⚠️ 关键逻辑：左手必须先完成抓取，右手才能开始释放！
        - 阶段 1-5: 右手独立工作
        - 阶段 6-9: 左手独立工作，右手保持在交接位置并保持闭合
        - 阶段 10-12: 只有当 step >= l_grasp_end 时，右手才开始释放
        
        左手抓取后保持在传递位置不动
        """
        env_len = env_ids.shape[0]
        total_steps = self.max_episode_length
        
        # ========== 阶段时间分配 ==========
        cfg_ratios = self.cfg.phase_ratios
        
        # 右手阶段
        right_approach_ratio = cfg_ratios.get('right_approach', 0.12)
        right_grasp_ratio = cfg_ratios.get('right_grasp', 0.10)
        
        # 细分 right_approach 为 approach (40%) + descend (50%) + dwell (10%)
        r_approach_end = int(right_approach_ratio * 0.4 * total_steps)
        r_descend_end = r_approach_end + int(right_approach_ratio * 0.5 * total_steps)
        r_dwell_end = r_descend_end + int(right_approach_ratio * 0.1 * total_steps)
        r_grasp_end = r_dwell_end + int(right_grasp_ratio * total_steps)
        r_lift_end = r_grasp_end + int(cfg_ratios.get('right_lift', 0.15) * total_steps)
        
        # 左手阶段
        left_approach_ratio = cfg_ratios.get('left_approach', 0.18)
        left_grasp_ratio = cfg_ratios.get('left_grasp', 0.10)
        
        # ⭐ 左手在右手抬起阶段就开始移动（并行动作）
        # 左手 approach 阶段与右手 lift 阶段重叠
        l_approach_end = r_grasp_end + int(left_approach_ratio * 0.6 * total_steps)
        l_descend_end = l_approach_end + int(left_approach_ratio * 0.3 * total_steps)
        l_dwell_end = l_descend_end + int(left_approach_ratio * 0.1 * total_steps)
        l_grasp_end = l_dwell_end + int(left_grasp_ratio * total_steps)
        
        # 交接阶段
        right_retreat_ratio = cfg_ratios.get('right_retreat', 0.25)
        r_release_end = l_grasp_end + int(cfg_ratios.get('right_release', 0.08) * total_steps)
        
        # 细分 right_retreat 为 post_release (50%) + retreat (50%)
        r_post_release_end = r_release_end + int(right_retreat_ratio * 0.6 * total_steps)
        r_retreat_end = total_steps
        
        # ========== 保存阶段边界（用于显示） ==========
        self._phase_boundaries = {
            'r_approach': (0, r_approach_end),
            'r_descend': (r_approach_end, r_descend_end),
            'r_dwell': (r_descend_end, r_dwell_end),
            'r_grasp': (r_dwell_end, r_grasp_end),
            'r_lift': (r_grasp_end, r_lift_end),
            'l_approach': (r_grasp_end, l_approach_end),  # ⭐ 左手与右手 lift 并行
            'l_descend': (l_approach_end, l_descend_end),
            'l_dwell': (l_descend_end, l_dwell_end),
            'l_grasp': (l_dwell_end, l_grasp_end),
            'r_release': (l_grasp_end, r_release_end),
            'r_post_release': (r_release_end, r_post_release_end),
            'r_retreat': (r_post_release_end, r_retreat_end),
        }
        
        # ========== 右手抓取姿态配置 ==========
        right_grasp_offset = self.cfg.right_grasp_offset
        right_grasp_euler_deg = self.cfg.right_grasp_euler_deg
        
        quat_xyzw = R.from_euler('xyz', right_grasp_euler_deg, degrees=True).as_quat()
        eef2_quat_grasp = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], 
                                        dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # ========== 右手交接位置姿态配置 ==========
        right_handover_offset = self.cfg.right_handover_offset
        right_handover_euler_deg = self.cfg.right_handover_euler_deg
        
        quat_xyzw_handover = R.from_euler('xyz', right_handover_euler_deg, degrees=True).as_quat()
        eef2_quat_handover = torch.tensor([quat_xyzw_handover[3], quat_xyzw_handover[0], quat_xyzw_handover[1], quat_xyzw_handover[2]], 
                                          dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # ========== 左手抓取姿态配置 ==========
        left_grasp_offset = self.cfg.left_grasp_offset
        left_grasp_euler_deg = self.cfg.left_grasp_euler_deg
        
        quat_xyzw_left = R.from_euler('xyz', left_grasp_euler_deg, degrees=True).as_quat()
        eef1_quat_grasp = torch.tensor([quat_xyzw_left[3], quat_xyzw_left[0], quat_xyzw_left[1], quat_xyzw_left[2]], 
                                        dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # ========== 计算关键位置 ==========
        # 物体初始位置（相对机器人基座）
        target_position_init = self._target.data.root_pos_w[env_ids, :] - self._robot.data.root_link_pos_w[env_ids, :]
        
        # 右手抓取偏移
        right_eff_offset = torch.tensor(right_grasp_offset, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # 左手抓取偏移
        left_eff_offset = torch.tensor(left_grasp_offset, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        
        # 当前末端执行器位姿（右手和左手）
        eef2_pos_current = self._robot.data.body_link_pos_w[env_ids, self._eef2_link_index, :] - self._robot.data.root_link_pos_w[env_ids, :]
        eef2_quat_current = self._robot.data.body_link_quat_w[env_ids, self._eef2_link_index, :]
        
        eef1_pos_current = self._robot.data.body_link_pos_w[env_ids, self._eef1_link_index, :] - self._robot.data.root_link_pos_w[env_ids, :]
        eef1_quat_current = self._robot.data.body_link_quat_w[env_ids, self._eef1_link_index, :]
        
        # ========== 右手关键位置 ==========
        # 0. 右手预抓取位置（使用配置的偏移参数）
        right_pre_grasp_offset = torch.tensor(
            [self.cfg.right_pre_grasp_x_offset, self.cfg.right_pre_grasp_y_offset, self.cfg.right_pre_grasp_height], 
            device=self.device
        ).unsqueeze(0).repeat(env_len, 1)
        r_pos_pre_grasp = right_eff_offset + target_position_init + right_pre_grasp_offset
        
        # 1. 右手抓取位置
        r_pos_grasp = right_eff_offset + target_position_init
        
        # 2. 传递位置（从配置文件读取，世界坐标系 → 转换为相对机器人基座）
        handover_pos = self._handover_position.unsqueeze(0).repeat(env_len, 1)
        # handover_pos += self._robot.data.root_link_pos_w[env_ids, :]  # 转换为相对机器人的坐标
        # 右手在交接位置的偏移（类似 pick_place 的 place_offset）
        right_handover_offset_tensor = torch.tensor(right_handover_offset, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(env_len, 1)
        r_pos_handover = handover_pos + right_handover_offset_tensor
        
        # 调试输出关键位置
        # if env_ids[0].item() == 0:  # 只打印第一个环境
        #     print(f"\n{'='*80}")
        #     print(f"【轨迹关键位置】")
        #     print(f"{'='*80}")
        #     print(f"物体初始位置 (target_position_init): [{target_position_init[0, 0]:.4f}, {target_position_init[0, 1]:.4f}, {target_position_init[0, 2]:.4f}]")
        #     print(f"右手抓取偏移 (right_grasp_offset):   [{right_grasp_offset[0]:.4f}, {right_grasp_offset[1]:.4f}, {right_grasp_offset[2]:.4f}]")
        #     print(f"右手抓取位置 (r_pos_grasp):          [{r_pos_grasp[0, 0]:.4f}, {r_pos_grasp[0, 1]:.4f}, {r_pos_grasp[0, 2]:.4f}]")
        #     print(f"\n传递位置 (handover_pos):             [{handover_pos[0, 0]:.4f}, {handover_pos[0, 1]:.4f}, {handover_pos[0, 2]:.4f}]")
        #     print(f"右手交接偏移 (right_handover_offset): [{right_handover_offset[0]:.4f}, {right_handover_offset[1]:.4f}, {right_handover_offset[2]:.4f}]")
        #     print(f"右手交接位置 (r_pos_handover):       [{r_pos_handover[0, 0]:.4f}, {r_pos_handover[0, 1]:.4f}, {r_pos_handover[0, 2]:.4f}]")
        #     print(f"\n移动距离 (抓取→交接):")
        #     delta = r_pos_handover[0] - r_pos_grasp[0]
        #     print(f"  ΔX: {delta[0]:+.4f}m, ΔY: {delta[1]:+.4f}m, ΔZ: {delta[2]:+.4f}m")
        #     print(f"  总距离: {torch.norm(delta).item():.4f}m")
        #     print(f"{'='*80}\n")
        
        # 3. 右手撤回位置（沿y轴负方向移动10cm）
        retreat_offset = torch.tensor([-0.06, - 0.12, 0], device=self.device).unsqueeze(0).repeat(env_len, 1)
        r_pos_retreat = r_pos_handover + retreat_offset
        
        
        # right_post_release_y_offset: float = -0.05  # y轴偏移（米）
        # right_post_release_x_offset: float = -0.08  # x轴偏移（米）


        # 4. 右手释放后中间点（避免直接撤回时撞到物体）
        right_post_release_offset = torch.tensor([self.cfg.right_post_release_x_offset, self.cfg.right_post_release_y_offset, self.cfg.right_post_release_height], 
                                                  device=self.device).unsqueeze(0).repeat(env_len, 1)
        r_pos_post_release = r_pos_handover + right_post_release_offset
        
        # ========== 左手关键位置 ==========
        # 5. 左手预抓取位置（交接位置上方/侧方）
        left_pre_grasp_offset = torch.tensor([self.cfg.left_pre_grasp_x_offset, self.cfg.left_pre_grasp_y_offset, self.cfg.left_pre_grasp_height], 
                                             device=self.device).unsqueeze(0).repeat(env_len, 1)
        l_pos_pre_approach = handover_pos + left_eff_offset + left_pre_grasp_offset
        
        # 6. 左手接近传递位置（靠近物体旁边）
        l_pos_approach = handover_pos + left_eff_offset
        
        # ========== 手指位置 ==========
        # 右手手指
        hand2_pos_open = self._joint_limit_upper[env_ids, :][:, self._hand2_joint_index]
        hand2_pos_closed = self._joint_limit_lower[env_ids, :][:, self._hand2_joint_index]
        hand2_pos_closed[:, 0] = self._joint_limit_upper[env_ids, :][:, self._hand2_joint_index[0]]
        
        # 左手手指
        hand1_pos_open = self._joint_limit_upper[env_ids, :][:, self._hand1_joint_index]
        hand1_pos_closed = self._joint_limit_lower[env_ids, :][:, self._hand1_joint_index]
        hand1_pos_closed[:, 0] = self._joint_limit_upper[env_ids, :][:, self._hand1_joint_index[0]]
        
        # 根据手指抓取模式调整闭合状态
        if self.cfg.finger_grasp_mode == "pinch":
            # pinch 模式：只有拇指和食指闭合
            hand2_pos_closed[:, 2:5] = hand2_pos_open[:, 2:5]
            hand1_pos_closed[:, 2:5] = hand1_pos_open[:, 2:5]
        elif self.cfg.finger_grasp_mode == "no_thumb":
            # no_thumb 模式：大拇指保持打开状态（索引 5）
            hand2_pos_closed[:, 5] = self._joint_limit_upper[env_ids, :][:, self._hand2_joint_index[5]]
            hand1_pos_closed[:, 5] = self._joint_limit_upper[env_ids, :][:, self._hand1_joint_index[5]]
        
        # ========== 生成轨迹 ==========
        for step in range(total_steps):
            # ========== 右手轨迹 ==========
            if step < r_approach_end:
                # 阶段1: 右手接近 - 移动到预抓取位置
                t_normalized = step / max(r_approach_end, 1)
                t_smooth = self._smooth_approach_interpolation(torch.tensor(t_normalized, device=self.device))
                r_pos = eef2_pos_current + t_smooth * (r_pos_pre_grasp - eef2_pos_current)
                t_batch = torch.full((env_len,), t_smooth.item(), device=self.device)
                r_quat = self._slerp_batch(eef2_quat_current, eef2_quat_grasp, t_batch)
                r_hand = hand2_pos_open
                
            elif step < r_descend_end:
                # 阶段2: 右手下降到抓取位置
                t_normalized = (step - r_approach_end) / max(r_descend_end - r_approach_end, 1)
                t_smooth = self._smooth_approach_interpolation(torch.tensor(t_normalized, device=self.device))
                r_pos = r_pos_pre_grasp + t_smooth * (r_pos_grasp - r_pos_pre_grasp)
                r_quat = eef2_quat_grasp
                r_hand = hand2_pos_open
                
            elif step < r_dwell_end:
                # 阶段3: 右手稳定在抓取位置
                r_pos = r_pos_grasp
                r_quat = eef2_quat_grasp
                r_hand = hand2_pos_open
                
            elif step < r_grasp_end:
                # 阶段4: 右手抓取（手指闭合）
                r_pos = r_pos_grasp
                r_quat = eef2_quat_grasp
                if self.cfg.smooth_finger_close:
                    t_normalized = (step - r_dwell_end) / max(r_grasp_end - r_dwell_end, 1)
                    t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                    r_hand = hand2_pos_open + t_smooth * (hand2_pos_closed - hand2_pos_open)
                else:
                    r_hand = hand2_pos_closed
                    
            elif step < r_lift_end:
                # 阶段5: 右手抬起到传递位置（姿态从抓取姿态渐变到交接姿态）
                # 
                t_normalized = (step - r_grasp_end) / max(r_lift_end - r_grasp_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                r_pos = r_pos_grasp + t_smooth * (r_pos_handover - r_pos_grasp)
                # 姿态从抓取姿态渐变到交接姿态
                t_batch = torch.full((env_len,), t_smooth.item(), device=self.device)
                r_quat = self._slerp_batch(eef2_quat_grasp, eef2_quat_handover, t_batch)
                r_hand = hand2_pos_closed
                
            elif step < r_release_end:
                # 阶段6-9: 右手保持在传递位置（等待左手抓取）
                # 【关键逻辑】必须等待左手完成抓取后，右手才能释放
                r_pos = r_pos_handover
                r_quat = eef2_quat_handover  # 使用交接姿态
                
                if step < l_grasp_end:
                    # 左手还在抓取中，右手必须保持闭合状态（关键！）
                    r_hand = hand2_pos_closed
                else:
                    # 阶段10: 左手抓取完成后，右手才开始松开
                    # 此时 step >= l_grasp_end，确保左手已经完全抓住物体
                    t_normalized = (step - l_grasp_end) / max(r_release_end - l_grasp_end, 1)
                    t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                    r_hand = hand2_pos_closed + t_smooth * (hand2_pos_open - hand2_pos_closed)
                    
            elif step < r_post_release_end:
                # 阶段11: 右手移动到释放后中间点（避免撞到物体）
                t_normalized = (step - r_release_end) / max(r_post_release_end - r_release_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                r_pos = r_pos_handover + t_smooth * (r_pos_post_release - r_pos_handover)
                r_quat = eef2_quat_handover  # 保持交接姿态
                r_hand = hand2_pos_open
                    
            else:
                # 阶段12: 右手从释放后中间点撤回到最终位置
                t_normalized = (step - r_post_release_end) / max(r_retreat_end - r_post_release_end, 1)
                t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                r_pos = r_pos_post_release + t_smooth * (r_pos_retreat - r_pos_post_release)
                r_quat = eef2_quat_handover  # 保持姿态
                r_hand = hand2_pos_open
            
            # ========== 左手轨迹 ==========
            if step < r_grasp_end:
                # 阶段 1-4: 右手抓取阶段，左手保持初始位置（不动）
                l_pos = eef1_pos_current
                l_quat = eef1_quat_current
                l_hand = hand1_pos_open
                
            elif step < l_approach_end:
                # 阶段1: 左手接近预抓取位置（与右手 lift 阶段并行）
                t_normalized = (step - r_grasp_end) / max(l_approach_end - r_grasp_end, 1)
                t_smooth = self._smooth_approach_interpolation(torch.tensor(t_normalized, device=self.device))
                l_pos = eef1_pos_current + t_smooth * (l_pos_pre_approach - eef1_pos_current)
                t_batch = torch.full((env_len,), t_smooth.item(), device=self.device)
                l_quat = self._slerp_batch(eef1_quat_current, eef1_quat_grasp, t_batch)
                l_hand = hand1_pos_open
                
            elif step < l_descend_end:
                # 阶段2: 左手下降到抓取位置
                t_normalized = (step - l_approach_end) / max(l_descend_end - l_approach_end, 1)
                t_smooth = self._smooth_approach_interpolation(torch.tensor(t_normalized, device=self.device))
                l_pos = l_pos_pre_approach + t_smooth * (l_pos_approach - l_pos_pre_approach)
                l_quat = eef1_quat_grasp
                l_hand = hand1_pos_open
                
            elif step < l_dwell_end:
                # 阶段3: 左手稳定在抓取位置
                l_pos = l_pos_approach
                l_quat = eef1_quat_grasp
                l_hand = hand1_pos_open
                
            elif step < l_grasp_end:
                # 阶段4: 左手抓取（手指闭合）
                l_pos = l_pos_approach
                l_quat = eef1_quat_grasp
                if self.cfg.smooth_finger_close:
                    t_normalized = (step - l_dwell_end) / max(l_grasp_end - l_dwell_end, 1)
                    t_smooth = self._minimum_jerk_interpolation(torch.tensor(t_normalized, device=self.device))
                    l_hand = hand1_pos_open + t_smooth * (hand1_pos_closed - hand1_pos_open)
                else:
                    l_hand = hand1_pos_closed
                    
            else:
                # 阶段5: 左手保持抓取（不动）
                l_pos = l_pos_approach
                l_quat = eef1_quat_grasp
                l_hand = hand1_pos_closed
            
            # 存储轨迹点
            self._eef2_pose_target[env_ids, step, :3] = r_pos
            self._eef2_pose_target[env_ids, step, 3:7] = r_quat
            self._hand2_pos_target[env_ids, step, :] = r_hand
            
            self._eef1_pose_target[env_ids, step, :3] = l_pos
            self._eef1_pose_target[env_ids, step, 3:7] = l_quat
            self._hand1_pos_target[env_ids, step, :] = l_hand

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
    
    def _get_current_phase(self, step: int) -> str:
        """
        获取当前步数对应的阶段名称
        
        Args:
            step: 当前步数
            
        Returns:
            阶段名称（中文）
        """
        phase_names = {
            'r_approach': '右手接近',
            'r_descend': '右手下降',
            'r_dwell': '右手稳定',
            'r_grasp': '右手抓取',
            'r_lift': '右手抬起',
            'l_approach': '左手接近',
            'l_descend': '左手下降',
            'l_dwell': '左手稳定',
            'l_grasp': '左手抓取',
            'r_release': '右手松开',
            'r_post_release': '右手退出',
            'r_retreat': '右手撤回',
        }
        
        for phase_key, (start, end) in self._phase_boundaries.items():
            if start <= step < end:
                return phase_names.get(phase_key, phase_key)
        
        return '未知阶段'

    def step(self, actions):
        
        # ========== 设置双手目标 ==========
        eef2_pose_target = torch.tensor([], device=self.device)
        hand2_pos_target = torch.tensor([], device=self.device)
        eef1_pose_target = torch.tensor([], device=self.device)
        hand1_pos_target = torch.tensor([], device=self.device)
        
        # 从轨迹中提取当前步的目标
        for i in range(self.num_envs):
            step_idx = self._episode_step[i]
            # 右手
            eef2_pose_target = torch.cat((eef2_pose_target, self._eef2_pose_target[i, step_idx, :].unsqueeze(0)), dim=0)
            hand2_pos_target = torch.cat((hand2_pos_target, self._hand2_pos_target[i, step_idx, :].unsqueeze(0)), dim=0)
            # 左手
            eef1_pose_target = torch.cat((eef1_pose_target, self._eef1_pose_target[i, step_idx, :].unsqueeze(0)), dim=0)
            hand1_pos_target = torch.cat((hand1_pos_target, self._hand1_pos_target[i, step_idx, :].unsqueeze(0)), dim=0)

        # 设置双手IK命令
        self._robot.set_ik_command({
            "arm2": eef2_pose_target,  # 右手
            "arm1": eef1_pose_target   # 左手
        })
        
        # 设置双手手指位置
        self._robot.set_joint_position_target(hand2_pos_target, self._robot.actuators["hand2"].joint_indices[:6])  # type: ignore
        self._robot.set_joint_position_target(hand1_pos_target, self._robot.actuators["hand1"].joint_indices[:6])  # type: ignore

        if self.cfg.print_eef_pose:
            # ========== 输出末端执行器位置和旋转 ==========
            # 获取当前末端执行器位姿（世界坐标系）
            eef2_pos_w = self._robot.data.body_link_pos_w[:, self._eef2_link_index, :]
            eef2_quat_w = self._robot.data.body_link_quat_w[:, self._eef2_link_index, :]
            eef1_pos_w = self._robot.data.body_link_pos_w[:, self._eef1_link_index, :]
            eef1_quat_w = self._robot.data.body_link_quat_w[:, self._eef1_link_index, :]
            
            # 获取物体位置和交接位置
            object_pos_w = self._target.data.root_pos_w
            handover_pos_w = self._handover_position + self._robot.data.root_link_pos_w[:, :]
            
            # 计算各种偏移
            eef2_to_object = eef2_pos_w - object_pos_w  # 右手→物体
            eef1_to_handover = eef1_pos_w - handover_pos_w  # 左手→交接位置
            object_to_handover = object_pos_w - handover_pos_w  # 物体→交接位置
            
            # 打印第一个环境的信息（避免输出过多）
            r_to_obj = eef2_to_object[0].cpu().numpy()
            l_to_hand = eef1_to_handover[0].cpu().numpy()
            obj_to_hand = object_to_handover[0].cpu().numpy()
            
            # 获取双手的四元数和欧拉角
            r_quat = eef2_quat_w[0].cpu().numpy()  # wxyz 格式
            l_quat = eef1_quat_w[0].cpu().numpy()  # wxyz 格式
            
            step = self._episode_step[0].item()
            # 获取当前阶段
            phase = self._get_current_phase(step)
            
            # 四元数转欧拉角 (wxyz -> xyzw for scipy, then to degrees)
            r_quat_xyzw = [r_quat[1], r_quat[2], r_quat[3], r_quat[0]]
            r_euler_deg = R.from_quat(r_quat_xyzw).as_euler('xyz', degrees=True)
            l_quat_xyzw = [l_quat[1], l_quat[2], l_quat[3], l_quat[0]]
            l_euler_deg = R.from_quat(l_quat_xyzw).as_euler('xyz', degrees=True)
            
            # print(f"[Step {step:3d} | {phase}]")
            # print(f"  右手→物体:       [{r_to_obj[0]:+7.4f}, {r_to_obj[1]:+7.4f}, {r_to_obj[2]:+7.4f}]")
            # print(f"  左手→交接位置: [{l_to_hand[0]:+7.4f}, {l_to_hand[1]:+7.4f}, {l_to_hand[2]:+7.4f}]")
            # print(f"  物体→交接位置: [{obj_to_hand[0]:+7.4f}, {obj_to_hand[1]:+7.4f}, {obj_to_hand[2]:+7.4f}]")
            # print(f"  右手角度(xyz): [{r_euler_deg[0]:7.2f}°, {r_euler_deg[1]:7.2f}°, {r_euler_deg[2]:7.2f}°]")
            # print(f"  左手角度(xyz): [{l_euler_deg[0]:7.2f}°, {l_euler_deg[1]:7.2f}°, {l_euler_deg[2]:7.2f}°]")

        # sim step according to decimation
        for i in range(self.cfg.decimation):
            # sim step
            self.sim_step()
        
        # update episode step
        self._episode_step += 1
        
        self._episode_step = torch.clamp(
            self._episode_step,
            None,
            (self.max_episode_length - 1) * torch.ones_like(self._episode_step)
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


        
        # ========== 实时打印末端执行器信息 ==========
        if self._sim_step_counter % 10 == 0:  # 每10步打印一次，避免输出过多
            # 获取双手末端执行器位姿（世界坐标系）
            eef2_pos_w = self._robot.data.body_link_pos_w[0, self._eef2_link_index, :]
            eef2_quat_w = self._robot.data.body_link_quat_w[0, self._eef2_link_index, :]
            eef1_pos_w = self._robot.data.body_link_pos_w[0, self._eef1_link_index, :]
            eef1_quat_w = self._robot.data.body_link_quat_w[0, self._eef1_link_index, :]
            
            # 获取物体当前位置（世界坐标系）
            object_pos_w = self._target.data.root_pos_w[0, :]
            
            # 计算末端执行器相对物体当前位置的偏移
            eef2_offset_current = eef2_pos_w - object_pos_w  # 右手
            eef1_offset_current = eef1_pos_w - object_pos_w  # 左手
            
            # 计算传递位置
            handover_pos_w = self._handover_position + self._robot.data.root_link_pos_w[:, :]
            eef2_to_handover = torch.norm(eef2_pos_w - handover_pos_w).item()
            eef1_to_handover = torch.norm(eef1_pos_w - handover_pos_w).item()
            
            # 四元数转欧拉角 (wxyz -> xyzw for scipy, then to degrees)
            # 右手角度
            quat2_wxyz = eef2_quat_w.cpu().numpy()
            quat2_xyzw = [quat2_wxyz[1], quat2_wxyz[2], quat2_wxyz[3], quat2_wxyz[0]]
            euler2_deg = R.from_quat(quat2_xyzw).as_euler('xyz', degrees=True)
            
            # 左手角度
            quat1_wxyz = eef1_quat_w.cpu().numpy()
            quat1_xyzw = [quat1_wxyz[1], quat1_wxyz[2], quat1_wxyz[3], quat1_wxyz[0]]
            euler1_deg = R.from_quat(quat1_xyzw).as_euler('xyz', degrees=True)
            
            # 获取当前episode步数
            current_step = self._episode_step[0].item()
            
            # 获取当前阶段
            phase = self._get_current_phase(current_step)
            
            # 计算左手→交接位置的偏移（而不是左手→物体）
            eef1_to_handover_offset = eef1_pos_w - handover_pos_w  # 左手相对交接位置的偏移
            
            # 计算物体→交接位置的偏移
            object_to_handover_offset = object_pos_w - handover_pos_w  # 物体相对交接位置的偏移
            
            # print(f"\n[Step {current_step:3d} | Sim {self._sim_step_counter:5d} | {phase}]")
            # print(f"  右手→物体:       [{eef2_offset_current[0]:+7.4f}, {eef2_offset_current[1]:+7.4f}, {eef2_offset_current[2]:+7.4f}]")
            # print(f"  左手→交接位置: [{eef1_to_handover_offset[0]:+7.4f}, {eef1_to_handover_offset[1]:+7.4f}, {eef1_to_handover_offset[2]:+7.4f}]")
            # print(f"  物体→交接位置: [{object_to_handover_offset[0]:+7.4f}, {object_to_handover_offset[1]:+7.4f}, {object_to_handover_offset[2]:+7.4f}]")
            # print(f"  右手角度(xyz): [{euler2_deg[0]:7.2f}°, {euler2_deg[1]:7.2f}°, {euler2_deg[2]:7.2f}°]")
            # print(f"  左手角度(xyz): [{euler1_deg[0]:7.2f}°, {euler1_deg[1]:7.2f}°, {euler1_deg[2]:7.2f}°]")
        
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

    def _eval_success_handover(self) -> torch.Tensor:
        """
        评估 Handover 是否成功
        
        成功条件：
        1. 左手已抓取物体（左手手指有接触力）
        2. 右手已松开（右手手指接触力为0）
        3. 右手已撤回（沿y轴负方向移动）
        4. 物体朝向偏离不超过阈值
        5. 物体在交接位置附近（6cm内）
        6. 物体高度比初始高度至少高0.1m
        """
        # 1. 检查左手是否抓取物体
        contact_force_num_left = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        for sensor_name, contact_sensor in self._contact_sensors_left.items():
            forces = torch.sum(contact_sensor.data.net_forces_w, dim=[1, 2])  # type: ignore
            force_magnitude = torch.abs(forces)  # 使用力的绝对值（大小）
            contact_force_num_left = torch.where(
                force_magnitude > 0.1,  # 力的大小超过阈值（0.1N）
                contact_force_num_left + 1,
                contact_force_num_left
            )

        left_grasped = contact_force_num_left >= 2  # type: ignore  # 至少2个手指有接触力
        
        # 2. 检查右手是否松开
        contact_force_num_right = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        for sensor_name, contact_sensor in self._contact_sensors_right.items():
            forces = torch.sum(contact_sensor.data.net_forces_w, dim=[1, 2])  # type: ignore
            force_magnitude = torch.abs(forces)  # 使用力的绝对值（大小）
            contact_force_num_right = torch.where(
                force_magnitude > 0.1,  # 力的大小超过阈值（0.1N）
                contact_force_num_right + 1,
                contact_force_num_right
            )
        right_released = contact_force_num_right == 0  # type: ignore  # 右手完全无接触力
        
        # 3. 检查右手是否已撤回（y轴负方向）
        eef2_pos_w = self._robot.data.body_link_pos_w[:, self._eef2_link_index, :]
        handover_pos = self._handover_position.unsqueeze(0).repeat(self.num_envs, 1)  # [num_envs, 3]
        handover_pos += self._robot.data.root_link_pos_w[:, :]
        # 检查右手相对交接位置在y轴方向上是否已经向负方向移动了指定距离（5cm）
        y_distance = eef2_pos_w[:, 1] - handover_pos[:, 1]
        right_retreated = y_distance < -self._right_retreat_distance  # 向y负方向移动超过指定距离（0.05m）
        
        # 4. 检查物体朝向
        current_quat = self._target.data.root_quat_w
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        # 5. 检查物体是否在交接位置附近（6cm内）
        current_pos = self._target.data.root_pos_w  # [num_envs, 3]
        handover_pos = self._handover_position.unsqueeze(0).repeat(self.num_envs, 1)  # [num_envs, 3]
        handover_pos += self._robot.data.root_link_pos_w[:, :]
        distance = torch.norm(current_pos - handover_pos, dim=1)  # 计算3D欧氏距离
        position_check = distance < 0.06  # 在8cm范围内
        
        # 6. 检查物体高度是否比初始高度至少高0.1m
        current_height = current_pos[:, 2]  # z坐标
        initial_height = self._target_pos_init[:, 2]  # 初始z坐标
        height_diff = current_height - initial_height
        height_check = height_diff > 0.015  # 至少抬高3cm
        
        # 综合判断（所有条件都要满足）
        bsuccessed = left_grasped & right_released & right_retreated & orientation_check & position_check & height_check
        
        return bsuccessed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 任务评估（使用右手contact sensors检测初期失败）
        bfailed, self._has_contacted = eval_fail(self._target, self._contact_sensors_right, self._has_contacted)  # type: ignore
        
        # 新增：检测物体在未接触时被移动（被推动/碰撞）
        bfailed_moved = self._eval_fail_moved_without_contact()
        # bfailed = bfailed | bfailed_moved
        bfailed = bfailed_moved
        
        # success eval（使用 Handover 的成功判断）
        bsuccessed = self._eval_success_handover()
     
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
        
        # Handover 使用专用轨迹生成函数
        self.create_handover_trajectory(env_ids)

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
