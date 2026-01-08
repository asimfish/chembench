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

# 导入标准的点云加载和采样函数
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))
from extract_ground_truth_pointcloud import load_usd_mesh_as_trimesh, sample_pointcloud

# ========== 任务配置（修改这里即可切换不同任务）==========
# TARGET_OBJECT_NAME = "mortar"  # 目标物体名称，如 "mortar", "glass_beaker_100ml" 等
# glass_graduated_cylinder_500ml
# plastic_graduated_cylinder_500ml
TARGET_OBJECT_NAME = "glass_beaker_100ml"  # 目标物体名称
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
    episode_length_s = 2
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
            # rendering_mode="quality",
            # antialiasing_mode="TAA",
        ),

    )
    sim.render.rendering_mode = "quality"
    sim.render.antialiasing_mode = "TAA"

    # ========== 物体抓取配置参数（从 object_config.json 加载）==========
    # 目标物体名称（使用文件顶部定义的 TARGET_OBJECT_NAME）
    target_object_name: str = TARGET_OBJECT_NAME
    
    # 抓取偏移 [x, y, z]（相对于物体中心）
    grasp_offset: list = None  # type: ignore
    
    # 抓取角度 [roll, pitch, yaw]（欧拉角，单位：度）
    grasp_euler_deg: list = None  # type: ignore
    
    # 抬起高度（单位：米）
    lift_height_desired: float = 0.2
    
    # 轨迹生成的时序参数
    phase_ratios: dict = None  # type: ignore
    
    # 手指闭合方式：True=平滑闭合，False=直接闭合（类似 create_trajectory）
    smooth_finger_close: bool = True
    
    # 手指抓取模式：
    # - "all": 所有手指都闭合（默认）
    # - "pinch": 只有拇指和食指闭合（对应索引 0,1 和 5）
    # - "no_thumb":其他手指闭合，大拇指不闭合
    finger_grasp_mode: str = "no_thumb"
    
    # 预抓取位置偏移

    # 普通物体
    pre_grasp_height: float = 0.02
    pre_grasp_y_offset: float = -0.01
    pre_grasp_x_offset: float = -0.095
# [-0.02, -0.02, 0.05] 
    # 高的物体[-0.02, -0.02, 0.05]
    # pre_grasp_height: float = 0.02
    # pre_grasp_y_offset: float = -0.01
    # pre_grasp_x_offset: float = -0.095
 
    #250ml容量瓶 500ml容量瓶 透明试剂瓶大
    # pre_grasp_height = 0.03   # z轴上方偏移 10cm
    # pre_grasp_y_offset = 0.00  # y轴负方向偏移 10cm（从侧面接近）
    # pre_grasp_x_offset = -0.095  # y轴负方向偏移 10cm（从侧面接近）

    ##500 ml量筒 酒精灯
    # pre_grasp_height: float = 0.02      # z轴上方偏移（单位：米，默认8cm）
    # pre_grasp_y_offset: float = -0.01   # y轴偏移（单位：米，默认-2cm，从侧面接近）
    # pre_grasp_x_offset: float = -0.095   # x轴偏移（单位：米，默认-2cm）
    

    # 是否启用轨迹平滑
    enable_trajectory_smooth: bool = True
    
    # 轨迹模式选择：
    # - "default": 使用原始 create_trajectory（不平滑）
    # - "smooth": 使用 create_trajectory_smooth（minimum jerk，有明显加减速）
    # - "constant_velocity": 使用恒速轨迹（推荐用于 Diffusion Policy 训练）
    # trajectory_mode: str = "constant_velocity"
    trajectory_mode: str = "smooth"
    
    # 成功判断：朝向偏差阈值（sin²(θ/2)，0=完全一致，1=上下颠倒）
    # 0.1 约等于 37° 的偏差，0.05 约等于 26° 的偏差
    orientation_threshold: float = 0.05
    
    # 目标成功次数：达到此数量后自动停止（设为 0 或 None 表示不限制）
    target_success_count: int = 50
    
    # 输出文件夹：chembench/data/motion_plan/{任务类型}/{物体名称}
    output_folder: str = None  # type: ignore
    
    ##是否输出每一步末端执行器的位置和旋转
    print_eef_pose: bool = False
    
    # 是否启用阶段步数统计输出
    enable_phase_stats: bool = True
    
    # 是否加载真值点云（从USD采样）
    enable_pointcloud: bool = True
    
    # 材质切换配置
    enable_material_switch: bool = False  # 是否启用材质切换
    material_switch_step: int = 2  # 在哪一步切换材质
    target_material_path: str = "/World/Looks/M_Glass_Clear"  # 目标材质路径（示例）
    
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
        # if self.lift_height_desired == 0.3:
        #     self.lift_height_desired = grasp_config["lift_height"]
        

        # 设置轨迹时序参数（如果未手动指定）
        # if self.phase_ratios is None:
        #     timing = grasp_config["timing"]
        #     self.phase_ratios = {
        #         "approach": timing.get("approach_ratio", 0.4),
        #         "grasp": timing.get("grasp_ratio", 0.2),
        #         "lift": timing.get("lift_ratio", 0.4)
        #     }


        # 设置轨迹时序参数（如果未手动指定）
        if self.phase_ratios is None:
            # timing = grasp_config["timing"]
            self.phase_ratios = {
                "approach": 0.3,
                "grasp": 0.2,
                "lift":  0.5
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
        self._target = self.scene.rigid_objects["bottle"]
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
        
        # ========== 加载真值点云（从USD采样）==========
        if self.cfg.enable_pointcloud:
            self._load_ground_truth_pointcloud()
        else:
            print("⚠️  点云加载已禁用（enable_pointcloud=False）")
            self._base_pointcloud = None
        
        # ========== 阶段步数统计 ==========
        if self.cfg.enable_phase_stats:
            # 计算各阶段边界（与 create_trajectory_smooth 中的计算保持一致）
            total_steps = self.max_episode_length
            cfg_ratios = self.cfg.phase_ratios
            approach_total = cfg_ratios.get('approach', 0.4)
            grasp_total = cfg_ratios.get('grasp', 0.2)
            phase_ratios = {
                'approach': approach_total * 0.6,
                'descend': approach_total * 0.4,
                'dwell': approach_total * 0.10,
                'grasp': grasp_total,
                'lift': cfg_ratios.get('lift', 0.2)
            }
            self._phase_boundaries = {
                'approach_end': int(phase_ratios['approach'] * total_steps),
                'descend_end': int(phase_ratios['approach'] * total_steps) + int(phase_ratios['descend'] * total_steps),
                'dwell_end': int(phase_ratios['approach'] * total_steps) + int(phase_ratios['descend'] * total_steps) + int(phase_ratios['dwell'] * total_steps),
                'grasp_end': int(phase_ratios['approach'] * total_steps) + int(phase_ratios['descend'] * total_steps) + int(phase_ratios['dwell'] * total_steps) + int(phase_ratios['grasp'] * total_steps),
            }
            
            # 用于统计各阶段步数的列表（累积所有完成的episode）
            self._phase_steps_stats = {
                'approach': [],
                'descend': [],
                'dwell': [],
                'grasp': [],
                'lift': []
            }
            self._completed_episodes = 0  # 完成的 episode 总数

        self.sim.set_camera_view([-0.7, -5.2, 1.3], [-1.2, -5.2, 1.1])
        # 设置 RTX 渲染选项
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
        # 3. 可选：确保透明物体参与 Primary Ray Hit
        carb_settings_iface.set_bool("/rtx/hydra/segmentation/includeTransparent", True)
        
        # 材质切换标志（用于记录是否已经切换过）
        self._material_switched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
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

    def _load_ground_truth_pointcloud(self, num_points: int = 1024):
        """
        从USD文件加载并采样真值点云
        
        使用与 extract_ground_truth_pointcloud.py 完全相同的标准函数，确保：
        1. 正确处理USD transform层级
        2. 处理所有类型的n-gon（不只是三角形和四边形）
        3. 生成的点云与离线生成的ground truth完全一致
        
        Args:
            num_points: 采样点数（默认2048）
        """
        try:
            import numpy as np
            
            # 获取物体USD路径（基础路径指向 sim_ready，子目录有 solid_assets 和 solid_assets_new）
            usd_base_path = os.path.join(
                os.path.dirname(__file__), 
                "../../../../../../psilab/assets/usd/asset_collection/sim_ready"
            )
            
            # 根据 target_object_name 映射到USD文件
            # 映射关系来自 collect/objects_config.yaml
            object_usd_map = {
                # ========== solid_assets (旧版) ==========
                "glass_beaker_50ml": "glass_beaker_50ml/Beaker002.usd",
                "glass_beaker_100ml": "glass_beaker_100ml/Beaker003.usd",
                "glass_beaker_250ml": "glass_beaker_250ml/Beaker004.usd",
                "glass_beaker_500ml": "glass_beaker_500ml/Beaker005.usd",
                
                "brown_volumetric_flask_250ml": "brown_volumetric_flask_250ml/VolumetricFlask001.usd",
                "clear_volumetric_flask_250ml": "clear_volumetric_flask_250ml/VolumetricFlask002.usd",
                "clear_volumetric_flask_500ml": "clear_volumetric_flask_500ml/VolumetricFlask003.usd",
                "clear_volumetric_flask_1000ml": "clear_volumetric_flask_1000ml/VolumetricFlask004.usd",
                
                "mortar": "mortar/Mortar001.usd",
                
                "brown_reagent_bottle_small": "brown_reagent_bottle_small/ReagentBottle004.usd",
                "brown_reagent_bottle_large": "brown_reagent_bottle_large/ReagentBottle001.usd",
                "clear_reagent_bottle_small": "clear_reagent_bottle_small/ReagentBottle003.usd",
                "clear_reagent_bottle_large": "clear_reagent_bottle_large/ReagentBottle002.usd",
                
                "erlenmeyer_flask": "erlenmeyer_flask/ErlenmeyerFlask001.usd",
                
                "plastic_cylinder_100ml": "plastic_cylinder_100ml/GraduatedCylinder001.usd",
                "glass_cylinder_100ml": "glass_cylinder_100ml/GraduatedCylinder004.usd",
            }
            
            # ========== solid_assets_new (新版) ==========
            object_usd_map_new = {
                "plastic_graduated_cylinder_500ml": "solid_assets_new/plastic_graduated_cylinder_500ml/GraduatedCylinder002.usd",
                "glass_graduated_cylinder_500ml": "solid_assets_new/glass_graduated_cylinder_500ml/GraduatedCylinder003.usd",
                "erlenmeyer_flask_with_stopper": "solid_assets_new/erlenmeyer_flask_with_stopper/ErlenmeyerFlask002.usd",
                "funnel": "solid_assets_new/funnel/Funnel001.usd",
                "alcohol_lamp": "solid_assets_new/alcohol_lamp/AlcoholLamp001.usd",
            }
            
            # 合并两个映射
            object_usd_map.update(object_usd_map_new)
            
            usd_rel_path = object_usd_map.get(self.cfg.target_object_name)
            if usd_rel_path is None:
                print(f"⚠️ 未找到 {self.cfg.target_object_name} 的USD映射，跳过真值点云加载")
                self._base_pointcloud = None
                return
            
            # 如果路径包含 solid_assets_new，直接使用；否则添加 solid_assets 前缀
            if "solid_assets_new" in usd_rel_path:
                # 已经包含完整路径（solid_assets_new/...）
                full_usd_rel_path = usd_rel_path
            else:
                # 旧版资产，添加 solid_assets 前缀
                full_usd_rel_path = os.path.join("solid_assets", usd_rel_path)
            
            usd_path = os.path.abspath(os.path.join(usd_base_path, full_usd_rel_path))
            
            if not os.path.exists(usd_path):
                print(f"⚠️ USD文件不存在: {usd_path}，跳过真值点云加载")
                self._base_pointcloud = None
                return
            
            print(f"📦 加载真值点云: {usd_path}")
            
            # 使用标准函数加载USD mesh（确保正确处理transform层级）
            # 注意：加载单独的USD文件时，不指定root_prim_path（使用世界坐标系）
            # 这样采样的点云就是物体局部坐标系中的点，后续通过物体的世界位姿来变换
            mesh = load_usd_mesh_as_trimesh(
                usd_path=usd_path,
                root_prim_path=None,  # 不指定root，使用世界坐标系（即USD文件的局部坐标系）
                convert_to_meters=True,  # 假设USD单位需要转换为米
                time_code=None,
                process=False,
            )
            
            # 使用标准函数采样点云
            base_pc = sample_pointcloud(mesh, num_points=num_points, seed=0)
            
            # 转换为torch tensor并移到GPU
            self._base_pointcloud = torch.from_numpy(base_pc.astype(np.float32)).to(self.device)
            
            print(f"✅ 真值点云加载完成: {self._base_pointcloud.shape}")
            
        except Exception as e:
            print(f"⚠️ 加载真值点云时出错: {e}")
            import traceback
            traceback.print_exc()
            self._base_pointcloud = None
    
    def _transform_pointcloud(self, env_id: int) -> torch.Tensor | None:
        """
        将基础点云变换到当前物体姿态
        
        Args:
            env_id: 环境ID
            
        Returns:
            变换后的点云 (N, 3) 或 None
        """
        if self._base_pointcloud is None:
            return None
        
        # 获取物体当前位姿（世界坐标系）
        obj_pos_w = self._target.data.root_pos_w[env_id, :].clone()  # (3,)
        obj_quat_w = self._target.data.root_quat_w[env_id, :].clone()  # (4,) wxyz
        
        # 获取环境原点
        env_origin = self.scene.env_origins[env_id, :].clone()  # (3,)
        
        # 物体相对于环境原点的位置
        obj_pos_rel = obj_pos_w - env_origin  # (3,)
        
        # 将四元数转换为旋转矩阵
        # 四元数格式: wxyz
        w, x, y, z = obj_quat_w[0], obj_quat_w[1], obj_quat_w[2], obj_quat_w[3]
        
        # 构建旋转矩阵 (3x3)
        R = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
        ])  # (3, 3)
        
        # 应用变换: p' = R * p + t
        transformed_pc = torch.matmul(self._base_pointcloud, R.T) + obj_pos_rel.unsqueeze(0)  # (N, 3)
        
        # ⚠️ 修复：USD点云朝向与IsaacSim中的物体朝向相差180度
        # 对点云应用绕Z轴旋转180度的修正（相对于物体中心）
        # 旋转180度等价于: x' = -x, y' = -y, z' = z (相对于中心点)
        pc_center = obj_pos_rel  # 物体中心位置
        transformed_pc_centered = transformed_pc - pc_center.unsqueeze(0)  # 移到原点
        transformed_pc_centered[:, 0] = -transformed_pc_centered[:, 0]  # x取负
        transformed_pc_centered[:, 1] = -transformed_pc_centered[:, 1]  # y取负
        transformed_pc = transformed_pc_centered + pc_center.unsqueeze(0)  # 移回物体中心
        
        return transformed_pc
    
    def _switch_bottle_material(self, env_id: int):
        """
        切换指定环境中 bottle 的材质
        
        Args:
            env_id: 环境ID
        """
        try:
            from pxr import Usd, UsdShade, Sdf
            
            # 获取 USD stage
            stage = self.sim.stage
            
            # 构建 bottle 的材质路径
            # 格式: /World/envs/env_{env_id}/bottle/Beaker003/Visuals/Beaker003/M_Beaker003_001
            # bottle_prim_path = self._target.prim_paths[env_id]  # 获取 bottle 的基础路径
            
            # 查找材质绑定的 prim（通常在 Visuals 下）
            # 方法1: 直接指定完整路径（如果知道具体结构）
            # material_prim_path = f"{bottle_prim_path}/Beaker003/Visuals/Beaker003"

            material_prim_path = "/World/envs/env_0/Bottle/base/Visuals/Beaker003/M_Beaker003_001"

            
            # 获取 prim
            mesh_prim = stage.GetPrimAtPath(material_prim_path)
            
            # if not mesh_prim.IsValid():
            #     print(f"⚠️ 未找到材质绑定的 Prim: {material_prim_path}")
            #     # 尝试查找其他可能的路径
            #     print(f"📋 Bottle prim path: {bottle_prim_path}")
            #     # 遍历子 prim 查找
            #     bottle_prim = stage.GetPrimAtPath(bottle_prim_path)
            #     for child in bottle_prim.GetAllChildren():
            #         print(f"  - Child: {child.GetPath()}")
            #     return
            target_material_path = "/World/envs/env_0/Bottle/Physics/PhysicsMaterial"
            # 绑定新材质
            material_prim = stage.GetPrimAtPath(target_material_path)
            if not material_prim.IsValid():
                print(f"⚠️ 目标材质不存在: {target_material_path}")
                return
            
            # 创建材质绑定
            shade_material = UsdShade.Material(material_prim)
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(shade_material)
            
            print(f"✅ 环境 {env_id}: 已将材质切换为 {target_material_path}")
            
        except Exception as e:
            print(f"❌ 切换材质时出错 (env {env_id}): {e}")
            import traceback
            traceback.print_exc()


    def debug_find_fluid(self, env_id=0):
        from pxr import Usd, UsdShade, Sdf
            
        # 获取 USD stage
        stage = self.sim.stage
        # stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Bottle")
        print("Root exists:", root.IsValid(), root.GetPath())

        hits = []
        for p in root.Traverse():
            name = p.GetName().lower()
            t = p.GetTypeName()
            if "fluid" in name or "particle" in name or t.lower().startswith("physxparticle"):
                hits.append((str(p.GetPath()), t))
        for h in hits[:80]:
            print(h)

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
        # for i in range(int(k1_step * 0.3)):            
        #     self._eef_pose_target[env_ids,i,1] = self._robot.data.body_link_pos_w[env_ids,self._eef_link_index,1] + i * delta_eef_pos[:,1]
        #     self._eef_pose_target[env_ids,i,3:7] = self._robot.data.body_link_quat_w[env_ids,self._eef_link_index,:] + i * delta_eef_quat[:,:]

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
            'approach': approach_total * 0.6,    # 接近预抓取位置
            'descend': approach_total * 0.4,     # 下降到抓取位置
            'dwell': approach_total * 0.10,       # 稳定等待（关键：让IK收敛）
            'grasp': grasp_total,                # 手指闭合
            'lift': cfg_ratios.get('lift', 0.2)  # 抬起阶段
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
        
        # 预抓取位置（抓取点上方 + 侧面偏移，从侧上方接近）
        # 从配置参数读取偏移量
        pre_grasp_offset = torch.tensor(
            [self.cfg.pre_grasp_x_offset, self.cfg.pre_grasp_y_offset, self.cfg.pre_grasp_height], 
            device=self.device
        ).unsqueeze(0).repeat(env_len, 1)
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

        # hand_pos_closed[:, 1] = 0.8 * self._joint_limit_lower[env_ids, :][:, self._hand_joint_index[1]]
        # hand_pos_closed[:, 5] = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index[1]]

        hand_pos_closed[:, 0] = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index[0]]  # 拇指旋转取最大值
        
        # 根据抓取模式调整手指闭合
        if self.cfg.finger_grasp_mode == "pinch":
            # 只有拇指(0,1)和食指(5)闭合，其他手指保持打开
            # 索引: 0=拇指旋转, 1=拇指弯曲, 2=中指, 3=无名指, 4=小指, 5=食指
            hand_pos_closed[:, 2] = hand_pos_open[:, 2]  # 中指保持打开
            hand_pos_closed[:, 3] = hand_pos_open[:, 3]  # 无名指保持打开
            hand_pos_closed[:, 4] = hand_pos_open[:, 4]  # 小指保持打开
        


        if self.cfg.finger_grasp_mode == "no_thumb":
            hand_pos_closed[:, 5] = self._joint_limit_upper[env_ids, :][:, self._hand_joint_index[1]]
        
        
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
        descend_ratio = cfg_ratios.get('grasp', 0.25) * 0.7  # grasp 的 60% 用于下降
        grasp_close_ratio = cfg_ratios.get('grasp', 0.25) * 0.3  # grasp 的 40% 用于手指闭合
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
        
        # 根据抓取模式调整手指闭合
        if self.cfg.finger_grasp_mode == "pinch":
            # 只有拇指(0,1)和食指(5)闭合，其他手指保持打开
            hand_pos_closed[:, 2] = hand_pos_open[:, 2]  # 中指保持打开
            hand_pos_closed[:, 3] = hand_pos_open[:, 3]  # 无名指保持打开
            hand_pos_closed[:, 4] = hand_pos_open[:, 4]  # 小指保持打开
        
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

        # ========== 材质切换检查 ==========
        if self.cfg.enable_material_switch:
            for i in range(self.num_envs):
                # 检查是否到达指定步数且尚未切换过
                if self._episode_step[i] == self.cfg.material_switch_step and not self._material_switched[i]:
                    self._switch_bottle_material(i)
                    self._material_switched[i] = True

        # self.debug_find_fluid(0)

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
            
            # 判断当前阶段
            total_steps = self.max_episode_length
            cfg_ratios = self.cfg.phase_ratios
            approach_total = cfg_ratios.get('approach', 0.4)
            grasp_total = cfg_ratios.get('grasp', 0.2)
            phase_ratios = {
                'approach': approach_total * 0.6,
                'descend': approach_total * 0.4,
                'dwell': approach_total * 0.10,
                'grasp': grasp_total,
                'lift': cfg_ratios.get('lift', 0.2)
            }
            approach_end = int(phase_ratios['approach'] * total_steps)
            descend_end = approach_end + int(phase_ratios['descend'] * total_steps)
            dwell_end = descend_end + int(phase_ratios['dwell'] * total_steps)
            grasp_end = dwell_end + int(phase_ratios['grasp'] * total_steps)
            
            if step < approach_end:
                stage = "接近"
            elif step < descend_end:
                stage = "下降"
            elif step < dwell_end:
                stage = "稳定"
            elif step < grasp_end:
                stage = "抓取"
            else:
                stage = "抬起"
            
            # 四元数转欧拉角 (wxyz -> xyzw for scipy, then to degrees)
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]  # wxyz -> xyzw
            euler_deg = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
            target_quat_xyzw = [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
            target_euler_deg = R.from_quat(target_quat_xyzw).as_euler('xyz', degrees=True)
            print(f"[Step {step:3d}][{stage:>4s}] EEF相对物体: [{pos_rel[0]:7.4f}, {pos_rel[1]:7.4f}, {pos_rel[2]:7.4f}] | "
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
                scene = self.scene,
                env_obj = self,  # 传递环境对象
                pointcloud_transform_fn = self._transform_pointcloud if (self.cfg.enable_pointcloud and self._base_pointcloud is not None) else None  # 传递点云变换函数
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
        height_check = height_lift >= (self.cfg.lift_height_desired * 0.7 )
        
        # 2. 检查接触状态
        # contact_force_num = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        # for sensor_name, contact_sensor in self._contact_sensors.items():
        #     forces = torch.sum(contact_sensor.data.net_forces_w, dim=[1, 2])
        #     contact_force_num = torch.where(forces > 0.0, contact_force_num + 1, contact_force_num)
        # contacting = contact_force_num > 0
        
        # 3. 检查朝向偏差
        current_quat = self._target.data.root_quat_w  # 当前朝向 (wxyz)
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        # 综合判断：高度 AND 接触 AND 朝向
        bsuccessed = height_check  & orientation_check
        
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
        
        # success eval（使用新的带朝向检查的函数）
        bsuccessed = self._eval_success_with_orientation()
     
        # update success number (精确控制，不超过目标)
        new_success_count = len(torch.nonzero(bsuccessed == True).squeeze(1).tolist())
        
        if self.cfg.target_success_count and self.cfg.target_success_count > 0:
            # 计算还可以接受的成功数
            remaining = self.cfg.target_success_count - self._episode_success_num
            if remaining > 0:
                # 只增加不超过剩余目标的成功数
                actual_increment = min(new_success_count, remaining)
                self._episode_success_num += actual_increment
            # else: 已达到或超过目标，不再增加
        else:
            # 没有设置目标，正常增加
            self._episode_success_num += new_success_count
        
        # ========== 阶段步数统计 ==========
        if self.cfg.enable_phase_stats:
            # 检测哪些环境完成了（成功、失败或超时）
            done_envs = bsuccessed | bfailed | time_out
            done_indices = torch.nonzero(done_envs).squeeze(1)
            
            if len(done_indices) > 0:
                # 对于每个完成的环境，统计各阶段步数
                for env_idx in done_indices:
                    env_idx_int = env_idx.item()
                    current_step = self._episode_step[env_idx_int].item()
                    
                    # 判断当前步数到达了哪个阶段
                    boundaries = self._phase_boundaries
                    
                    # 计算各阶段实际使用的步数
                    if current_step <= boundaries['approach_end']:
                        # 只完成了 approach 阶段的一部分
                        self._phase_steps_stats['approach'].append(current_step)
                        self._phase_steps_stats['descend'].append(0)
                        self._phase_steps_stats['dwell'].append(0)
                        self._phase_steps_stats['grasp'].append(0)
                        self._phase_steps_stats['lift'].append(0)
                    elif current_step <= boundaries['descend_end']:
                        # 完成了 approach，部分完成 descend
                        self._phase_steps_stats['approach'].append(boundaries['approach_end'])
                        self._phase_steps_stats['descend'].append(current_step - boundaries['approach_end'])
                        self._phase_steps_stats['dwell'].append(0)
                        self._phase_steps_stats['grasp'].append(0)
                        self._phase_steps_stats['lift'].append(0)
                    elif current_step <= boundaries['dwell_end']:
                        # 完成了 approach 和 descend，部分完成 dwell
                        self._phase_steps_stats['approach'].append(boundaries['approach_end'])
                        self._phase_steps_stats['descend'].append(boundaries['descend_end'] - boundaries['approach_end'])
                        self._phase_steps_stats['dwell'].append(current_step - boundaries['descend_end'])
                        self._phase_steps_stats['grasp'].append(0)
                        self._phase_steps_stats['lift'].append(0)
                    elif current_step <= boundaries['grasp_end']:
                        # 完成了前三个阶段，部分完成 grasp
                        self._phase_steps_stats['approach'].append(boundaries['approach_end'])
                        self._phase_steps_stats['descend'].append(boundaries['descend_end'] - boundaries['approach_end'])
                        self._phase_steps_stats['dwell'].append(boundaries['dwell_end'] - boundaries['descend_end'])
                        self._phase_steps_stats['grasp'].append(current_step - boundaries['dwell_end'])
                        self._phase_steps_stats['lift'].append(0)
                    else:
                        # 完成了所有阶段（包括 lift）
                        self._phase_steps_stats['approach'].append(boundaries['approach_end'])
                        self._phase_steps_stats['descend'].append(boundaries['descend_end'] - boundaries['approach_end'])
                        self._phase_steps_stats['dwell'].append(boundaries['dwell_end'] - boundaries['descend_end'])
                        self._phase_steps_stats['grasp'].append(boundaries['grasp_end'] - boundaries['dwell_end'])
                        self._phase_steps_stats['lift'].append(current_step - boundaries['grasp_end'])
                    
                    self._completed_episodes += 1
                
                # 每完成 100 个 episode，输出一次统计信息
                if self._completed_episodes % 100 == 0:
                    self._print_phase_stats()

        return bsuccessed, bfailed, time_out  # type: ignore
    
    def _print_phase_stats(self):
        """输出各阶段步数的统计信息"""
        import numpy as np
        
        print(f"\n{'='*80}")
        print(f"阶段步数统计 (完成 {self._completed_episodes} 个 episodes)")
        print(f"{'='*80}")
        print(f"{'阶段':<12} | {'平均步数':>10} | {'最小步数':>10} | {'最大步数':>10} | {'标准差':>10}")
        print(f"{'-'*80}")
        
        for phase_name in ['approach', 'descend', 'dwell', 'grasp', 'lift']:
            steps = self._phase_steps_stats[phase_name]
            if len(steps) > 0:
                avg_steps = np.mean(steps)
                min_steps = np.min(steps)
                max_steps = np.max(steps)
                std_steps = np.std(steps)
                
                print(f"{phase_name:<12} | {avg_steps:>10.2f} | {min_steps:>10.0f} | {max_steps:>10.0f} | {std_steps:>10.2f}")
        
        print(f"{'='*80}\n")
    
    def _reset_idx(self, env_ids: torch.Tensor | None, success_ids:Sequence[int]|None=None):

        # get env indexs
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if success_ids is None:
            success_ids=[]
        
        # 截断 logic: 确保保存的数据恰好等于 target_success_count
        ids_to_save = success_ids
        prev_total = 0  # 初始化，避免后续使用时未定义
        
        if self.cfg.enable_output and self.cfg.target_success_count and self.cfg.target_success_count > 0:
            # 注意: _get_dones 已经更新了 _episode_success_num，包含了当前 batch
            current_total = self._episode_success_num
            batch_size = len(success_ids)
            prev_total = current_total - batch_size
            
            # 计算还需要多少条数据
            if prev_total < self.cfg.target_success_count:
                needed = self.cfg.target_success_count - prev_total
                if batch_size > needed:
                    # 当前批次超过需要的数量，只保存需要的部分
                    ids_to_save = success_ids[:needed]
                    print(f"\n[INFO] 截断保存：需要 {needed} 条，当前批次 {batch_size} 条，保存前 {needed} 条")
                elif batch_size < needed:
                    # 当前批次不够，全部保存
                    ids_to_save = success_ids
                else:
                    # 恰好等于需要的数量
                    ids_to_save = success_ids
            else:
                # 已经达到目标，不再保存
                ids_to_save = []
                print(f"\n[INFO] 已达到目标 {self.cfg.target_success_count} 条，跳过当前批次")

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
                saved_count = len(ids_to_save)
                # 计算已保存的总数
                if self.cfg.target_success_count and self.cfg.target_success_count > 0:
                    total_saved = min(prev_total + saved_count, self.cfg.target_success_count)
                    print(f"[DATA] 已保存数据: {saved_path} (本批次: {saved_count}条, 总计: {total_saved}/{self.cfg.target_success_count})")
                else:
                    print(f"[DATA] 已保存数据: {saved_path} (成功轨迹数: {saved_count})")


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
        self._material_switched[env_ids] = torch.zeros_like(self._material_switched[env_ids], device=self.device, dtype=torch.bool)  # type: ignore

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
                
                # 输出最终的阶段步数统计
                if self.cfg.enable_phase_stats and self._completed_episodes > 0:
                    self._print_final_phase_stats()
                
                # 退出程序
                import sys
                sys.exit(0)
    
    def _print_final_phase_stats(self):
        """输出最终的完整阶段步数统计"""
        import numpy as np
        
        print(f"\n{'='*80}")
        print(f"🎯 最终阶段步数统计汇总 (共 {self._completed_episodes} 个 episodes)")
        print(f"{'='*80}")
        print(f"{'阶段':<12} | {'平均步数':>10} | {'最小步数':>10} | {'最大步数':>10} | {'标准差':>10} | {'中位数':>10}")
        print(f"{'-'*80}")
        
        total_avg_steps = 0
        for phase_name in ['approach', 'descend', 'dwell', 'grasp', 'lift']:
            steps = self._phase_steps_stats[phase_name]
            if len(steps) > 0:
                avg_steps = np.mean(steps)
                min_steps = np.min(steps)
                max_steps = np.max(steps)
                std_steps = np.std(steps)
                median_steps = np.median(steps)
                
                total_avg_steps += avg_steps
                
                # 中文阶段名
                phase_names_zh = {
                    'approach': '接近',
                    'descend': '下降',
                    'dwell': '稳定',
                    'grasp': '抓取',
                    'lift': '抬起'
                }
                display_name = f"{phase_names_zh[phase_name]}({phase_name})"
                
                print(f"{display_name:<12} | {avg_steps:>10.2f} | {min_steps:>10.0f} | {max_steps:>10.0f} | {std_steps:>10.2f} | {median_steps:>10.1f}")
        
        print(f"{'-'*80}")
        print(f"{'总计':<12} | {total_avg_steps:>10.2f} |")
        print(f"{'='*80}")
        
        # 输出阶段占比
        print(f"\n📊 各阶段平均步数占比:")
        print(f"{'-'*80}")
        for phase_name in ['approach', 'descend', 'dwell', 'grasp', 'lift']:
            steps = self._phase_steps_stats[phase_name]
            if len(steps) > 0 and total_avg_steps > 0:
                avg_steps = np.mean(steps)
                percentage = (avg_steps / total_avg_steps) * 100
                
                phase_names_zh = {
                    'approach': '接近',
                    'descend': '下降',
                    'dwell': '稳定',
                    'grasp': '抓取',
                    'lift': '抬起'
                }
                display_name = f"{phase_names_zh[phase_name]}({phase_name})"
                
                bar_length = int(percentage / 2)  # 每个字符代表2%
                bar = '█' * bar_length
                print(f"{display_name:<12} | {bar:<50} {percentage:>6.2f}%")
        
        print(f"{'='*80}\n")
