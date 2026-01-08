"""
Grasp IL Evaluation - 抓取任务的模仿学习评估

本文件用于评估训练好的 Diffusion Policy 模型在抓取任务上的性能。

支持的观测模式（obs_mode）：
- rgb: 3通道 RGB 图像
- rgbm: 4通道 RGB + Mask
- nd: 4通道 Normal + Depth  
- rgbnd: 7通道 RGB + Normal + Depth
- state: 纯状态（无图像）
- rgb_masked: 3通道 RGB（mask 用于预处理，背景置黑，但输出还是 RGB 格式）
- rgb_masked_rgb: 6通道 RGB + RGB*Mask（原始RGB + 背景置黑RGB）
- point_cloud: 点云数据 (N, 6) - [x, y, z, r, g, b]

Mask 解耦实验（mask_mode）：
- real: 使用真实的 instance segmentation mask（默认）
- all_0: mask 通道全填 0（测试模型是否依赖 mask）
- all_1: mask 通道全填 1（测试模型是否依赖 mask）

关键参数说明：
1. orientation_threshold = 0.1 (MP 是 0.04)
   - IL 评估使用稍宽松的朝向标准，因为学习模型的精度通常低于 motion planning
   - 0.1 约等于 37° 偏差，0.04 约等于 23° 偏差
2. 不需要轨迹生成 - 直接使用策略模型输出关节位置
3. 支持多观测模式 - 用于消融实验和模型鲁棒性测试
4. 支持点云模式 - 用于 DP3 等基于点云的策略
"""

import copy
from typing import Literal, Any, Sequence
from dataclasses import MISSING

import torch
import hydra
from omegaconf import OmegaConf
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ... existing imports ...
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
# 这里引用我们刚创建的 image_utils
from psilab_tasks.imitation_learning.grasp.image_utils import process_batch_image_multimodal

# USD 点云提取相关导入
from pxr import Usd, UsdGeom


# -----------------------------
# USD 点云提取工具函数（从 extract_ground_truth_pointcloud.py 移植）
# -----------------------------
def apply_transform_points(points: np.ndarray, mat4: np.ndarray) -> np.ndarray:
    """points: (N,3), mat4: (4,4)"""
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    out = (mat4 @ pts_h.T).T
    return out[:, :3]


def quaternion_to_rotation_matrix_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """输入 q = [w, x, y, z]，返回 3x3 旋转矩阵"""
    import math
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = q
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 1e-12:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    else:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def pose_to_transform_matrix(pose: np.ndarray, quat_order: str = "wxyz") -> np.ndarray:
    """
    pose: [x,y,z,qw,qx,qy,qz] if quat_order='wxyz'
          [x,y,z,qx,qy,qz,qw] if quat_order='xyzw'
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    t = pose[:3]
    q = pose[3:]
    if quat_order.lower() == "wxyz":
        q_wxyz = q
    elif quat_order.lower() == "xyzw":
        q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    else:
        raise ValueError("quat_order must be 'wxyz' or 'xyzw'")
    R = quaternion_to_rotation_matrix_wxyz(q_wxyz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def triangulate_ngon(face_vertex_indices, face_vertex_counts, vertex_offset: int):
    """用 fan triangulation 将 n-gon 三角化"""
    faces = []
    idx = 0
    for c in face_vertex_counts:
        c = int(c)
        poly = face_vertex_indices[idx:idx + c]
        idx += c
        if c < 3:
            continue
        v0 = int(poly[0]) + vertex_offset
        for i in range(1, c - 1):
            faces.append([v0, int(poly[i]) + vertex_offset, int(poly[i + 1]) + vertex_offset])
    if len(faces) == 0:
        return None
    return np.asarray(faces, dtype=np.int64)


def load_usd_mesh_as_trimesh(
    usd_path: str,
    root_prim_path: str | None = None,
    time_code: float | None = None,
    convert_to_meters: bool = True,
):
    """从 USD 加载所有 Mesh，合并为一个 trimesh.Trimesh"""
    usd_path = str(usd_path)
    print(f"[USD点云] 加载 USD: {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"无法打开 USD: {usd_path}")

    if time_code is None:
        tc = Usd.TimeCode.Default()
    else:
        tc = Usd.TimeCode(float(time_code))

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if meters_per_unit is None:
        meters_per_unit = 1.0
    meters_per_unit = float(meters_per_unit)
    unit_scale = meters_per_unit if convert_to_meters else 1.0

    xcache = UsdGeom.XformCache(tc)

    if root_prim_path is not None:
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim or not root_prim.IsValid():
            raise ValueError(f"root_prim_path 不存在或无效：{root_prim_path}")
        root_to_world = np.array(xcache.GetLocalToWorldTransform(root_prim), dtype=np.float64)
        world_to_root = np.linalg.inv(root_to_world)
    else:
        world_to_root = np.eye(4, dtype=np.float64)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get(tc) if points_attr else None
        if points is None or len(points) == 0:
            continue

        verts = np.asarray(points, dtype=np.float64) * unit_scale
        mesh_to_world = np.array(xcache.GetLocalToWorldTransform(prim), dtype=np.float64)
        mesh_to_root = world_to_root @ mesh_to_world
        verts_root = apply_transform_points(verts, mesh_to_root)
        all_vertices.append(verts_root)

        fvi_attr = mesh.GetFaceVertexIndicesAttr()
        fvc_attr = mesh.GetFaceVertexCountsAttr()
        fvi = fvi_attr.Get(tc) if fvi_attr else None
        fvc = fvc_attr.Get(tc) if fvc_attr else None

        if fvi is not None and fvc is not None and len(fvi) > 0 and len(fvc) > 0:
            faces = triangulate_ngon(fvi, fvc, vertex_offset=vertex_offset)
            if faces is not None and len(faces) > 0:
                all_faces.append(faces)

        vertex_offset += verts_root.shape[0]

    if len(all_vertices) == 0:
        raise ValueError("USD 文件中没有找到有效的 UsdGeom.Mesh 点数据")

    vertices = np.vstack(all_vertices).astype(np.float64)
    faces = np.vstack(all_faces).astype(np.int64) if len(all_faces) > 0 else None

    print(f"[USD点云] 合并 mesh 完成：{vertices.shape[0]} 顶点，{0 if faces is None else faces.shape[0]} 三角面")

    if faces is None or faces.shape[0] == 0:
        pc = trimesh.points.PointCloud(vertices=vertices)
        return pc

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return tm


def sample_pointcloud_from_mesh(mesh, num_points: int, seed: int | None = None) -> np.ndarray:
    """从 Trimesh 表面采样点云"""
    if seed is not None:
        np.random.seed(int(seed))

    if isinstance(mesh, trimesh.Trimesh) and mesh.faces is not None and len(mesh.faces) > 0:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points.astype(np.float32)

    # PointCloud 或无面
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.shape[0] <= num_points:
        return verts
    idx = np.random.choice(verts.shape[0], num_points, replace=False)
    return verts[idx]


def load_diffusion_policy_from_checkpoint(checkpoint_path: str, device: str = 'cuda:0'):
    """
    从 checkpoint 加载 Diffusion Policy 模型
    支持两种格式：
    1. 标准 Diffusion Policy 格式（带 _target_）
    2. DP3 格式（不带 _target_，直接是配置字典）
    """
    import dill
    import sys
    
    # ⚠️ 重要：在导入任何模块之前，先添加 DP3 路径到 sys.path
    dp3_paths = [
        '/home/psibot/chembench/3D-Diffusion-Policy/3D-Diffusion-Policy',  # 实际的模块路径
        '/home/psibot/chembench/3D-Diffusion-Policy',
    ]
    for path in dp3_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # 注册 eval resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    print(f"Loading Diffusion Policy from: {checkpoint_path}")
    
    # 加载 checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location='cpu')
    
    cfg = payload['cfg']
    
    # 检查是哪种格式的 checkpoint
    if hasattr(cfg, '_target_'):
        # 标准 Diffusion Policy 格式
        print("  Detected: Standard Diffusion Policy checkpoint")
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
    else:
        # DP3 格式（或其他直接包含 state_dicts 的格式）
        print("  Detected: DP3 or similar checkpoint format")
        
        # 直接从配置创建模型
        policy_cfg = cfg.policy
        
        # 根据配置创建模型
        if hasattr(policy_cfg, '_target_'):
            print(f"  Creating policy from: {policy_cfg._target_}")
            try:
                # 方法1：使用 hydra.utils.instantiate（推荐，与官方 train.py 一致）
                # 这会自动实例化所有嵌套的 _target_，包括 noise_scheduler
                print("  Using hydra.utils.instantiate (official method)")
                # 注意：hydra 已在文件顶部（第35行）导入，不需要重复导入
                policy = hydra.utils.instantiate(cfg.policy)
                print("  DP3 policy created successfully via hydra.utils.instantiate")
                print(f"  Config: obs_as_global_cond={cfg.obs_as_global_cond}")
            except Exception as e:
                print(f"  Hydra instantiate failed: {e}, falling back to manual instantiation")
                import traceback
                traceback.print_exc()
                
                # 方法2：手动实例化（备用方案）
                try:
                    from diffusion_policy_3d.policy.dp3 import DP3
                    print("  Successfully imported DP3")
                    
                    # 实例化 noise scheduler
                    if hasattr(policy_cfg.noise_scheduler, '_target_'):
                        # 注意：hydra 已在文件顶部（第35行）导入，不需要重复导入
                        noise_scheduler = hydra.utils.instantiate(policy_cfg.noise_scheduler)
                        print(f"  Instantiated noise_scheduler via hydra: {type(noise_scheduler)}")
                    else:
                        # 如果没有 _target_，尝试手动创建
                        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
                        noise_scheduler = DDPMScheduler(**OmegaConf.to_container(cfg.policy.noise_scheduler, resolve=True))
                        print(f"  Instantiated noise_scheduler manually: {type(noise_scheduler)}")
                    
                    policy = DP3(
                        shape_meta=cfg.shape_meta,
                        noise_scheduler=noise_scheduler,
                        horizon=cfg.horizon,
                        n_action_steps=cfg.n_action_steps,
                        n_obs_steps=cfg.n_obs_steps,
                        num_inference_steps=policy_cfg.num_inference_steps,
                        obs_as_global_cond=cfg.obs_as_global_cond,
                        # DP3 特有参数
                        crop_shape=policy_cfg.crop_shape,
                        diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
                        down_dims=policy_cfg.down_dims,
                        kernel_size=policy_cfg.kernel_size,
                        n_groups=policy_cfg.n_groups,
                        use_pc_color=policy_cfg.use_pc_color,
                        encoder_output_dim=policy_cfg.encoder_output_dim if hasattr(policy_cfg, 'encoder_output_dim') else 64,
                        # FiLM 条件编码参数
                        condition_type=policy_cfg.condition_type if hasattr(policy_cfg, 'condition_type') else 'film',
                        use_down_condition=policy_cfg.use_down_condition if hasattr(policy_cfg, 'use_down_condition') else True,
                        use_mid_condition=policy_cfg.use_mid_condition if hasattr(policy_cfg, 'use_mid_condition') else True,
                        use_up_condition=policy_cfg.use_up_condition if hasattr(policy_cfg, 'use_up_condition') else True,
                        # PointNet 配置
                        pointnet_type=policy_cfg.pointnet_type if hasattr(policy_cfg, 'pointnet_type') else "pointnet",
                        pointcloud_encoder_cfg=policy_cfg.pointcloud_encoder_cfg if hasattr(policy_cfg, 'pointcloud_encoder_cfg') else None,
                    )
                    print("  DP3 policy created successfully via manual instantiation")
                except Exception as e2:
                    print(f"  Error in manual instantiation: {e2}")
                    import traceback
                    traceback.print_exc()
                    raise ImportError(
                        f"Failed to create DP3 model.\n"
                        f"Hydra error: {e}\n"
                        f"Manual error: {e2}\n"
                        f"Please ensure:\n"
                        f"1. 3D-Diffusion-Policy is at: /home/psibot/chembench/3D-Diffusion-Policy\n"
                        f"2. All required DP3 parameters are in the checkpoint config\n"
                        f"Current sys.path: {sys.path[:3]}"
                    )
        else:
            # 没有 _target_，直接尝试导入 DP3
            print("  No _target_ in policy config, assuming DP3 model")
            try:
                from diffusion_policy_3d.policy.dp3 import DP3
                from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
                print("  Successfully imported DP3")
                
                # 实例化 noise scheduler
                noise_scheduler = DDPMScheduler(**OmegaConf.to_container(cfg.policy.noise_scheduler, resolve=True))
                
                # 从 cfg 的顶层获取参数
                policy = DP3(
                    shape_meta=cfg.shape_meta,
                    noise_scheduler=noise_scheduler,
                    horizon=cfg.horizon,
                    n_action_steps=cfg.n_action_steps,
                    n_obs_steps=cfg.n_obs_steps,
                    num_inference_steps=policy_cfg.num_inference_steps if hasattr(policy_cfg, 'num_inference_steps') else 10,
                    obs_as_global_cond=cfg.obs_as_global_cond if hasattr(cfg, 'obs_as_global_cond') else True,
                )
                print("  DP3 policy created successfully")
            except Exception as e:
                raise ImportError(
                    f"Failed to create DP3 model: {e}\n"
                    f"Please ensure:\n"
                    f"1. 3D-Diffusion-Policy is at: /home/psibot/chembench/3D-Diffusion-Policy\n"
                    f"2. diffusion_policy_3d module exists in that directory\n"
                    f"Current sys.path: {sys.path[:3]}"
                )
        
        # 加载模型权重
        if 'state_dicts' in payload:
            if 'model' in payload['state_dicts']:
                policy.load_state_dict(payload['state_dicts']['model'])
                print("  Loaded model state_dict")
            elif 'ema' in payload['state_dicts']:
                policy.load_state_dict(payload['state_dicts']['ema'])
                print("  Loaded EMA model state_dict")
            elif 'ema_model' in payload['state_dicts']:
                policy.load_state_dict(payload['state_dicts']['ema_model'])
                print("  Loaded EMA model state_dict")
            else:
                # 尝试第一个可用的 state_dict
                first_key = list(payload['state_dicts'].keys())[0]
                policy.load_state_dict(payload['state_dicts'][first_key])
                print(f"  Loaded {first_key} state_dict")
        else:
            print("  Warning: No state_dicts found in checkpoint!")
    
    policy.eval()
    policy.to(device)
    
    print(f"  Model loaded successfully on {device}")
    
    return policy


def visualize_pointcloud_debug(point_cloud_tensor, title="Point Cloud", save_path=None, show=True):
    """
    调试用：可视化点云数据（3D scatter plot）
    
    Args:
        point_cloud_tensor: torch.Tensor, shape [B, N, 6] or [N, 6]
                           6 通道: [x, y, z, r, g, b]
        title: 图表标题
        save_path: 保存路径（可选）
        show: 是否显示图形
    
    示例:
        visualize_pointcloud_debug(point_cloud_data[0], "Current Observation")
    """
    # 转换为 numpy
    if isinstance(point_cloud_tensor, torch.Tensor):
        pc = point_cloud_tensor.detach().cpu().numpy()
    else:
        pc = point_cloud_tensor
    
    # 如果是批量数据，只可视化第一个
    if pc.ndim == 3:
        pc = pc[0]  # [N, 6]
    
    # 提取 XYZ 和 RGB
    xyz = pc[:, :3]  # [N, 3]
    rgb = pc[:, 3:6] if pc.shape[1] >= 6 else None  # [N, 3]
    
    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 确定颜色
    if rgb is not None and np.any(rgb != 0):
        # 如果 RGB 不全是零，使用 RGB 着色
        colors = rgb
        # 归一化到 [0, 1]（如果需要）
        if colors.max() > 1.0:
            colors = colors / 255.0
    else:
        # 使用 Z 值着色（训练数据 RGB 是零）
        colors = xyz[:, 2]
    
    # 绘制点云
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                        c=colors, s=5, alpha=0.6, cmap='viridis')
    
    # 设置标签和标题
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{title}\nPoints: {len(xyz)}')
    
    # 添加颜色条（如果使用标量着色）
    if rgb is None or np.all(rgb == 0):
        plt.colorbar(scatter, ax=ax, label='Z value')
    
    # 设置相等的坐标轴比例
    max_range = np.array([xyz[:, 0].max()-xyz[:, 0].min(),
                         xyz[:, 1].max()-xyz[:, 1].min(),
                         xyz[:, 2].max()-xyz[:, 2].min()]).max() / 2.0
    
    mid_x = (xyz[:, 0].max()+xyz[:, 0].min()) * 0.5
    mid_y = (xyz[:, 1].max()+xyz[:, 1].min()) * 0.5
    mid_z = (xyz[:, 2].max()+xyz[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 添加坐标系信息
    info_text = f"Range: X[{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]\n"
    info_text += f"       Y[{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]\n"
    info_text += f"       Z[{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
             verticalalignment='top', fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"点云图已保存到: {save_path}")
    
    # 显示
    if show:
        plt.show()
    else:
        plt.close()


@configclass
class GraspBottleEnvCfg(ILEnvCfg):
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
    # 0.1 约等于 37° 的偏差，0.05 约等于 26° 的偏差
    # 注意：IL 评估使用稍宽松的标准（MP 用 0.04），因为学习模型精度通常低于 motion planning
    orientation_threshold: float = 0.05
    
    # 观测模式配置
    # 可选: "rgb", "rgbm", "nd", "rgbnd", "state", "rgb_masked", "rgb_masked_rgb", "point_cloud"
    # 注意: rgb_masked 生成的数据与 rgb 兼容（都是3通道），可以用 rgb 模型测试
    # rgb_masked_rgb 是6通道（原始RGB + 背景置黑RGB），需要专门的6通道模型
    # point_cloud 是点云数据 (N, 6) - [x, y, z, 0, 0, 0]，用于 DP3 等模型（注意：RGB通道用零填充）
    obs_mode: Literal["rgb", "rgbm", "nd", "rgbnd", "state", "rgb_masked", "rgb_masked_rgb", "point_cloud"] = "point_cloud"
    
    # 点云配置（仅当 obs_mode="point_cloud" 时使用）
    num_points: int = 2048  # 点云采样点数
    # 注意：use_point_color 已废弃，训练数据固定使用 [x,y,z,0,0,0] 格式
    # RGB 通道用零填充以保持与训练数据一致
    use_point_color: bool = True  # 固定为 True（但实际颜色是0），保持接口兼容性
    point_cloud_camera: str = "third_camera"  # 使用哪个相机的点云: "third_camera", "chest_camera", "head_camera"
    point_cloud_sampling: Literal["random", "fps"] = "fps"  # 点云采样方式: "random"(快速) 或 "fps"(最远点采样,与训练一致)
    
    # Ground Truth 点云配置（从 USD 提取高质量点云）
    use_ground_truth_pointcloud: bool = True  # 是否使用从 USD 提取的 GT 点云（替代相机深度图重建）
    ground_truth_usd_path: str = "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_volumetric_flask_500ml/VolumetricFlask003.usd"  # USD 文件路径（例如："/path/to/object.usd"）
    
    ground_truth_root_prim: str = ""  # USD root prim 路径（例如："/World/Object"），不填则用世界系
    ground_truth_convert_to_meters: bool = True  # 是否使用 USD metersPerUnit 转换单位
    ground_truth_quat_order: Literal["wxyz", "xyzw"] = "wxyz"  # 四元数顺序（Isaac Lab 使用 wxyz）
    ground_truth_seed: int = 42  # GT 点云采样随机种子
    
    # Mask 解耦实验配置
    # "real": 使用真实的 mask（默认）
    # "all_0": mask 通道填充全0（测试模型是否依赖 mask）
    # "all_1": mask 通道填充全1（测试模型是否依赖 mask）
    mask_mode: str = "real"
    
    # # 调试可视化配置
    # debug_visualize_pointcloud: bool = False  # 是否可视化点云数据（调试用）
    # debug_vis_interval: int = 50  # 可视化间隔（每N步可视化一次）
    # debug_vis_save_path: str = ""  # 保存路径（为空则只显示不保存）

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

        # load policy
        self.base_policy = load_diffusion_policy_from_checkpoint(self.cfg.checkpoint,self.device)
        
        # 获取 n_obs_steps 从模型配置
        self.n_obs_steps = self.base_policy.n_obs_steps if hasattr(self.base_policy, 'n_obs_steps') else 2
        print(f"[GraspBottleEnv] Policy requires n_obs_steps = {self.n_obs_steps}")
        
        # 初始化观察历史缓冲区（用于存储多步观察）
        self._obs_history = None  # 会在 reset 时初始化
        
        # 初始化 Action 队列（解决 DP3 输出步数 > decimation 的问题）
        # DP3 一次预测 n_action_steps 步，但每次环境只执行 decimation 步
        # 因此需要维护一个队列来存储未执行的 action
        self._action_queue = None  # [B, remaining_steps, action_dim]，会在 reset 时初始化
        self._action_queue_ptr = None  # 当前队列指针 [B]，指示下一个要执行的 action 索引
        
        # 加载 Ground Truth 点云（如果启用）
        self._gt_pointcloud_base = None  # 物体局部坐标系下的点云 (N, 3)
        if self.cfg.obs_mode == "point_cloud" and self.cfg.use_ground_truth_pointcloud:
            if not self.cfg.ground_truth_usd_path:
                raise ValueError("use_ground_truth_pointcloud=True 但未指定 ground_truth_usd_path")
            
            print(f"\n{'='*50}")
            print(f"[GT点云] 从 USD 加载物体点云")
            print(f"  USD 路径: {self.cfg.ground_truth_usd_path}")
            print(f"  Root prim: {self.cfg.ground_truth_root_prim or '世界坐标系'}")
            print(f"  采样点数: {self.cfg.num_points}")
            print(f"  采样方式: {self.cfg.point_cloud_sampling}")
            print(f"{'='*50}\n")
            
            try:
                # 加载 USD mesh
                mesh = load_usd_mesh_as_trimesh(
                    usd_path=self.cfg.ground_truth_usd_path,
                    root_prim_path=self.cfg.ground_truth_root_prim if self.cfg.ground_truth_root_prim else None,
                    time_code=None,
                    convert_to_meters=self.cfg.ground_truth_convert_to_meters,
                )
                
                # 采样点云（物体局部坐标系）
                base_pc = sample_pointcloud_from_mesh(
                    mesh, 
                    num_points=self.cfg.num_points, 
                    seed=self.cfg.ground_truth_seed
                )
                
                # 转换为 torch tensor 并移到 GPU
                self._gt_pointcloud_base = torch.from_numpy(base_pc).to(self.device, dtype=torch.float32)
                
                print(f"[GT点云] ✓ 加载成功：{self._gt_pointcloud_base.shape}")
                print(f"  点云范围: x=[{self._gt_pointcloud_base[:, 0].min():.3f}, {self._gt_pointcloud_base[:, 0].max():.3f}], "
                      f"y=[{self._gt_pointcloud_base[:, 1].min():.3f}, {self._gt_pointcloud_base[:, 1].max():.3f}], "
                      f"z=[{self._gt_pointcloud_base[:, 2].min():.3f}, {self._gt_pointcloud_base[:, 2].max():.3f}]\n")
            except Exception as e:
                import traceback
                print(f"[GT点云] ✗ 加载失败：{e}")
                traceback.print_exc()
                raise
        
        # initialize Timer
        self._timer = Timer()
        # variables used to store contact flag
        self._has_contacted = torch.zeros(self.num_envs,device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs,3),device=self.device)
        self._target_quat_init = torch.zeros((self.num_envs,4),device=self.device)  # 初始朝向（wxyz）
        
        # 记录成功时的步数列表
        self._success_steps_list: list[int] = []
        self.sim.set_camera_view([-0.7, -5.2, 1.3], [-1.2, -5.2, 1.1])
        # 设置 RTX 渲染选项: Fractional Cutout Opacity
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)

    def _furthest_point_sampling(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        最远点采样（FPS）- PyTorch 实现
        
        与训练数据生成时的采样方式一致，确保点云均匀分布在物体表面
        
        Args:
            points: [N, D] - N 个点，D 维特征（通常是 6: xyz + rgb）
            num_samples: 目标采样点数
        
        Returns:
            sampled_points: [K, D] - 采样后的点
        """
        device = points.device
        N, D = points.shape
        
        if N == 0:
            # 空点云，返回零填充
            return torch.zeros(num_samples, D, device=device)
        
        if N <= num_samples:
            # 如果点数不足，重复采样
            indices = torch.randint(0, N, (num_samples,), device=device)
            return points[indices]
        
        # 初始化
        sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
        distances = torch.ones(N, device=device) * 1e10
        
        # 随机选择第一个点
        farthest = torch.randint(0, N, (1,), device=device).item()
        
        for i in range(num_samples):
            sampled_indices[i] = farthest
            centroid = points[farthest, :3]  # 只用 xyz 计算距离（不考虑 RGB）
            
            # 计算到当前点的欧氏距离
            dist = torch.sum((points[:, :3] - centroid) ** 2, dim=-1)
            
            # 更新每个点到已采样点集的最小距离
            mask = dist < distances
            distances[mask] = dist[mask]
            
            # 选择距离最远的点作为下一个采样点
            farthest = torch.argmax(distances).item()
        
        return points[sampled_indices]

    def _process_point_cloud(self, camera_name: str) -> torch.Tensor:
        """
        生成点云数据
        
        支持两种模式：
        1. 从相机深度图重建点云（默认，与 zarr_utils.py 的处理方式保持一致）
        2. 使用从 USD 提取的 Ground Truth 点云（更高质量）
        
        Args:
            camera_name: 相机名称 ("chest_camera", "head_camera", "third_camera")（仅在相机模式下使用）
        
        Returns:
            点云数据 [B, N, 6]
            - 6通道: [x, y, z, 0, 0, 0] - 与训练数据保持一致（RGB用零填充）
        """
        import torch.nn.functional as F
        
        # ==================== Ground Truth 点云模式 ====================
        if self.cfg.use_ground_truth_pointcloud and self._gt_pointcloud_base is not None:
            """
            使用从 USD 提取的高质量点云，根据当前物体 pose 变换到世界坐标系
            
            优势：
            - 几何精度高（直接从 mesh 采样）
            - 无深度图噪声
            - 无遮挡问题
            """
            B = self._target.data.root_state_w.shape[0]
            device = self.device
            
            # 获取物体当前 pose（世界坐标系）
            target_pos = self._target.data.root_pos_w  # [B, 3]
            target_quat = self._target.data.root_quat_w  # [B, 4] wxyz
            
            # 将点云从物体局部坐标系变换到世界坐标系
            transformed_points = []
            for b in range(B):
                # 构造变换矩阵
                pose = torch.cat([target_pos[b], target_quat[b]]).cpu().numpy()  # [7] - [x,y,z,w,x,y,z]
                T = pose_to_transform_matrix(pose, quat_order=self.cfg.ground_truth_quat_order)
                
                # 应用变换
                points_local = self._gt_pointcloud_base.cpu().numpy()  # [N, 3]
                points_world = apply_transform_points(points_local, T)  # [N, 3]
                points_world_torch = torch.from_numpy(points_world).to(device, dtype=torch.float32)
                
                # ⚠️ 修复：USD点云朝向与IsaacSim中的物体朝向相差180度
                # 对点云应用绕Z轴旋转180度的修正（相对于物体中心）
                # 旋转180度等价于: x' = -x, y' = -y, z' = z (相对于中心点)
                pc_center = target_pos[b]  # 物体中心位置（世界坐标系）
                points_centered = points_world_torch - pc_center.unsqueeze(0)  # 移到物体中心
                points_centered[:, 0] = -points_centered[:, 0]  # x取负
                points_centered[:, 1] = -points_centered[:, 1]  # y取负
                points_corrected = points_centered + pc_center.unsqueeze(0)  # 移回世界坐标
                
                transformed_points.append(points_corrected)
            
            point_cloud_xyz = torch.stack(transformed_points, dim=0)  # [B, N, 3]
            
            # 减去环境原点，转换为相对坐标（与训练数据保持一致）
            point_cloud_xyz = point_cloud_xyz - self.scene.env_origins.unsqueeze(1)  # [B, N, 3]
            
            # 添加零填充的 RGB 通道（与训练数据格式一致）
            zeros_rgb = torch.zeros((B, self.cfg.num_points, 3), device=device, dtype=torch.float32)

            # point_cloud = point_cloud_xyz
            point_cloud = torch.cat([point_cloud_xyz, zeros_rgb], dim=-1)  # [B, N, 6]
            
            return point_cloud
        
        # ==================== 相机深度图重建点云模式（默认）====================
        
        camera = self._robot.tiled_cameras[camera_name]
        
        # 获取深度图 [B, H, W, 1] - 原始分辨率 (480, 640)
        depth_original = camera.data.output["depth"][:, :, :, 0]  # [B, H, W]
        
        # ⚠️ 重要：Resize 到 224x224（与训练数据一致）
        # 训练时在 zarr_utils.py 中将深度图 resize 到了 224x224
        target_size = 224
        depth = F.interpolate(
            depth_original.unsqueeze(1),  # [B, 1, H, W]
            size=(target_size, target_size),
            mode='nearest'
        ).squeeze(1)  # [B, 224, 224]


        # 获取 mask（用于过滤物体区域，与训练数据一致）
        if "instance_segmentation_fast" in camera.data.output:
            mask_original = camera.data.output["instance_segmentation_fast"][:, :, :, 0]  # [B, H, W]
            # Resize mask 到 224x224
            mask = F.interpolate(
                mask_original.unsqueeze(1).float(),  # [B, 1, H, W]
                size=(target_size, target_size),
                mode='nearest'
            ).squeeze(1)  # [B, 224, 224]
        else:
            # 如果没有 mask，创建全1的 mask（保留所有点）
            mask = torch.ones_like(depth)
        
        # ⚠️ 重要：不要归一化深度！
        # 训练数据使用的是真实深度值（单位：米），不是归一化的 [0,1]
        # Isaac Lab 的 depth 输出就是真实深度（米），直接使用即可
        
        # 计算有效点的 mask
        valid_mask = depth > 0
        object_mask = (mask > 0) & valid_mask

        
        # ⚠️ 重要：获取缩放后的相机内参（与训练数据一致）
        # 训练时在 zarr_utils.py 中将内参从 640x480 缩放到了 224x224
        intrinsics = camera.data.intrinsic_matrices[0]  # [3, 3] - 原始内参
        fx_original = intrinsics[0, 0]
        fy_original = intrinsics[1, 1]
        cx_original = intrinsics[0, 2]
        cy_original = intrinsics[1, 2]
        
        # 获取原始分辨率
        original_height, original_width = depth_original.shape[1:]  # 480, 640
        
        # 缩放内参到 224x224（与训练数据一致）
        scale_x = target_size / original_width
        scale_y = target_size / original_height
        fx = fx_original * scale_x
        fy = fy_original * scale_y
        cx = cx_original * scale_x
        cy = cy_original * scale_y
        
        B, H, W = depth.shape  # 现在是 [B, 224, 224]
        device = depth.device
        
        ##clip depth
        depth = torch.clamp(depth, 0.2, 1.8)

        # 生成像素坐标网格（224x224）
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 反投影到相机坐标系（与训练数据保持一致：直接使用相机坐标系）
        # 先对所有像素进行反投影，然后再过滤
        # X_cam = (u - cx) * depth / fx
        # Y_cam = (v - cy) * depth / fy  
        # Z_cam = depth
        x_cam = (u - cx) * depth / fx  # [B, H, W]
        y_cam = (v - cy) * depth / fy  # [B, H, W]
        z_cam = depth  # [B, H, W]
        

        # 组合成点云 [B, H, W, 3] - 保持在相机坐标系
        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, H, W, 3]
        
        # ⚠️ 重要：与训练数据保持一致，不进行坐标系转换！
        # 训练时 camera_pointcloud_utils.py 中的 get_pointcloud() 没有传入 position/orientation
        # 因此训练数据是在相机坐标系下的，推理时也必须使用相机坐标系
        
        # 将点云 reshape 为 [B, H*W, 3]
        points_cam_flat = points_cam.reshape(B, -1, 3)  # [B, H*W, 3]
        
        # 过滤无效点：depth > 0 AND mask > 0（与训练数据一致）
        # 注意：object_mask 已经在前面计算过了
        valid_mask = object_mask.reshape(B, -1)  # [B, H*W]
        
        # ⚠️ 重要：RGB 用零填充，与训练数据保持一致！
        # 训练数据中 RGB 通道全是 0，所以推理时也必须用 0
        zeros_rgb = torch.zeros((B, H*W, 3), device=device, dtype=torch.float32)
        point_cloud = torch.cat([points_cam_flat, zeros_rgb], dim=-1)  # [B, H*W, 6] - [x,y,z,0,0,0]
        
        # 采样固定数量的点
        # 支持两种采样方式：random（快速）或 fps（与训练一致）
        sampled_points = []
        for b in range(B):
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
            
            if len(valid_indices) == 0:
                # 如果没有有效点，返回零点云
                print(f"  ⚠️  警告：batch {b} 没有有效点云，返回零点云")
                sampled_points.append(torch.zeros(self.cfg.num_points, 6, device=device))
            else:
                # 提取有效点
                valid_points = point_cloud[b, valid_indices]  # [N_valid, 6]
                
                if self.cfg.point_cloud_sampling == "fps":
                    # FPS 采样（与训练数据一致，点分布均匀）
                    sampled_pc = self._furthest_point_sampling(valid_points, self.cfg.num_points)
                    sampled_points.append(sampled_pc)
                else:  # "random"
                    # 随机采样（快速，但点分布可能不均匀）
                    if len(valid_indices) < self.cfg.num_points:
                        # 如果有效点不足，重复采样
                        sampled_indices = torch.randint(0, len(valid_indices), (self.cfg.num_points,), device=device)
                    else:
                        # 随机采样
                        perm = torch.randperm(len(valid_indices), device=device)
                        sampled_indices = perm[:self.cfg.num_points]
                    sampled_points.append(valid_points[sampled_indices])
        
        sampled_point_cloud = torch.stack(sampled_points, dim=0)  # [B, N, 6]
        
        return sampled_point_cloud

    def step(self,actions):
        
        # get obs for policy
        eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        eef_state = self._robot.data.body_link_state_w[:,eef_link_index,:7].clone()
        eef_state[:,:3] -= self._robot.data.root_state_w[:,:3]
        
        # process image
        
        # 获取基础 RGB 图像
        chest_rgb = self._robot.tiled_cameras["chest_camera"].data.output["rgb"][:, :, :, :]
        head_rgb = self._robot.tiled_cameras["head_camera"].data.output["rgb"][:, :, :, :]
        third_rgb = self._robot.tiled_cameras["third_camera"].data.output["rgb"][:, :, :, :]  # 新增：第三人称相机
        
        # 初始化可选通道
        chest_mask, head_mask, third_mask = None, None, None
        chest_depth, head_depth, third_depth = None, None, None
        chest_normal, head_normal, third_normal = None, None, None

        # 根据 obs_mode 获取所需的额外通道
        # 1. Mask
        if self.cfg.obs_mode in ["rgbm", "rgb_masked", "rgb_masked_rgb"]:
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
                # 解耦实验：mask 通道填充全 1（会被归一化为 1.0）
                chest_mask = torch.ones_like(chest_rgb[:, :, :, 0])
                head_mask = torch.ones_like(head_rgb[:, :, :, 0])
                third_mask = torch.ones_like(third_rgb[:, :, :, 0])
        # 2. Depth
        if self.cfg.obs_mode in ["nd", "rgbnd"]:
            if "depth" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_depth = self._robot.tiled_cameras["chest_camera"].data.output["depth"][:, :, :, 0]
            if "depth" in self._robot.tiled_cameras["head_camera"].data.output:
                head_depth = self._robot.tiled_cameras["head_camera"].data.output["depth"][:, :, :, 0]
            if "depth" in self._robot.tiled_cameras["third_camera"].data.output:
                third_depth = self._robot.tiled_cameras["third_camera"].data.output["depth"][:, :, :, 0]

        # 3. Normal
        if self.cfg.obs_mode in ["nd", "rgbnd"]:
            if "normals" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_normal = self._robot.tiled_cameras["chest_camera"].data.output["normals"][:, :, :, :3] # 取前3通道
            if "normals" in self._robot.tiled_cameras["head_camera"].data.output:
                head_normal = self._robot.tiled_cameras["head_camera"].data.output["normals"][:, :, :, :3]
            if "normals" in self._robot.tiled_cameras["third_camera"].data.output:
                third_normal = self._robot.tiled_cameras["third_camera"].data.output["normals"][:, :, :, :3]

        # 统一处理图像
        chest_camera_img = None
        head_camera_img = None
        third_camera_img = None
        point_cloud_data = None

        if self.cfg.obs_mode == 'point_cloud':
            # 点云模式：生成点云数据
            point_cloud_data = self._process_point_cloud(self.cfg.point_cloud_camera)
        elif self.cfg.obs_mode != 'state':
            # 图像模式：处理图像数据
            # 胸部相机
            chest_camera_img = process_batch_image_multimodal(
                rgb=chest_rgb, 
                mask=chest_mask, 
                depth=chest_depth, 
                normal=chest_normal, 
                obs_mode=self.cfg.obs_mode
            )
            
            # 头部相机
            head_camera_img = process_batch_image_multimodal(
                rgb=head_rgb, 
                mask=head_mask, 
                depth=head_depth, 
                normal=head_normal, 
                obs_mode=self.cfg.obs_mode
            )
            
            # 第三人称相机
            third_camera_img = process_batch_image_multimodal(
                rgb=third_rgb, 
                mask=third_mask, 
                depth=third_depth, 
                normal=third_normal, 
                obs_mode=self.cfg.obs_mode
            )

        #  目标物体位姿（相对于环境原点，与训练数据保持一致）
        target_pose = self._target.data.root_state_w[:,:7].clone()
        target_pose[:,:3] -= self.scene.env_origins
        
        # create obs dict
        # 根据 obs_mode 选择 key 名称
        obs_key_map = {
            "rgb": "rgb",
            "rgbm": "rgb", # 兼容旧代码，RGBM 也用 rgb后缀 ? 
                           # NO, check timm_obs_encoder.py. 
                           # timm_obs_encoder 区分 rgb, rgbm, nd, rgbnd 类型
                           # 但是 obs_key_map 在这里是指 obs dict 的 key。
                           # 通常训练时 key 是 chest_camera_rgb 等。
                           # 我们需要确认 checkpoint 期望的 key 是什么。
                           # 根据 timm_obs_encoder，它会读取 shape_meta 中的 key。
                           # 通常为了兼容性，我们可能还是用 'chest_camera_rgb' 作为 key，
                           # 但是其 shape 会根据 obs_mode 变化。
                           # 如果训练时的 shape_meta 定义了 type='rgbm'，
                           # 那么 obs_encoder 会去读对应的 key。
        }
        
        # 假设所有模式都使用 xxx_camera_rgb 作为 key，但内容通道数不同
        # 这是一个常见的 trick，避免修改大量的 yaml 配置
            # 目标物体位姿


        ##state based
        # if self.cfg.obs_mode == 'state':
        #     current_obs = {
        #         # 'target_pose': target_pose.unsqueeze(1),
        #         # 'third_person_camera_rgb': third_person_camera_rgb.unsqueeze(1),
        #         'arm2_pos': self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
        #         # 'arm2_vel': self._robot.data.joint_vel[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
        #         'hand2_pos': self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
        #         # 'hand2_vel': self._robot.data.joint_vel[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
        #         'arm2_eef_pos': eef_state[:,:3].unsqueeze(1),
        #         'arm2_eef_quat': eef_state[:,3:7].unsqueeze(1),
        #         'target_pose': target_pose.unsqueeze(1),
        #     }
        #     # policy model step
        #     with torch.no_grad():
        #         base_act_seq = self.base_policy.predict_action(current_obs)['action']    
        #     # sim step according to decimation
        #     for i in range(self.cfg.decimation):
        #         #
        #         self._action = base_act_seq[:,i,:]
        #         # sim step
        #         self.sim_step()
                
        #     return super().step(actions)


        # data shape is Batch_size,1,data_shape...
        if self.cfg.obs_mode == 'state':
            current_obs = {
                # 'target_pose': target_pose.unsqueeze(1),
                'arm2_pos': self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
                'hand2_pos': self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
                'arm2_eef_pos': eef_state[:,:3].unsqueeze(1),
                'arm2_eef_quat': eef_state[:,3:7].unsqueeze(1),
                'target_pose': target_pose.unsqueeze(1),
            }
            # 对于非点云模式，暂时不维护历史（TODO: 如果模型需要，可以添加）
            policy_obs = current_obs
        elif self.cfg.obs_mode == 'point_cloud':
            # 点云模式：使用 DP3 期望的 key 名称
            # 根据配置文件，key 应该是 'point_cloud' 和 'agent_pos'
            # agent_pos = arm2_pos(7) + hand2_pos(6) = 13
            arm2_pos = self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices]
            hand2_pos = self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]] # type: ignore
            agent_pos = torch.cat([arm2_pos, hand2_pos], dim=-1)  # [B, 13]
            
            current_obs = {
                'point_cloud': point_cloud_data.unsqueeze(1),  # [B, 1, N, 6]
                'agent_pos': agent_pos.unsqueeze(1),  # [B, 1, 13]
            }
            
            # save_path=None
            # visualize_pointcloud_debug(
            #     point_cloud_data,  # [B, N, 6]
            #     title=f"Point Cloud",
            #     save_path=None,
            #     show=(save_path is None)  # 如果保存则不显示，否则显示
            # )
            # # ====== 调试：可视化点云数据 ======
            # 在配置中设置 debug_visualize_pointcloud=True 来启用
            # if self.cfg.debug_visualize_pointcloud:
            #     if not hasattr(self, '_vis_counter'):
            #         self._vis_counter = 0
                
            #     self._vis_counter += 1
            #     # 第一次或每隔 debug_vis_interval 步可视化
            #     if self._vis_counter == 1 or self._vis_counter % self.cfg.debug_vis_interval == 0:
            #         print(f"\n[DEBUG] Visualizing point cloud at step {self._vis_counter}")
                    
            #         # 确定保存路径
            #         save_path = None
            #         if self.cfg.debug_vis_save_path:
            #             save_path = f"{self.cfg.debug_vis_save_path}_step_{self._vis_counter}.png"
                    
            #         visualize_pointcloud_debug(
            #             point_cloud_data,  # [B, N, 6]
            #             title=f"Point Cloud - {self.cfg.point_cloud_camera} - Step {self._vis_counter}",
            #             save_path=save_path,
            #             show=(save_path is None)  # 如果保存则不显示，否则显示
            #         )
            # # ====================================
            
            # 维护观察历史：需要 n_obs_steps 个历史观察
            if self._obs_history is None:
                # 第一次调用：用当前观察重复填充历史
                self._obs_history = {
                    'point_cloud': current_obs['point_cloud'].repeat(1, self.n_obs_steps, 1, 1),  # [B, n_obs_steps, N, 6]
                    'agent_pos': current_obs['agent_pos'].repeat(1, self.n_obs_steps, 1),  # [B, n_obs_steps, 13]
                }

                print(f"[GraspBottleEnv] Initialized obs_history with shape: point_cloud={self._obs_history['point_cloud'].shape}, agent_pos={self._obs_history['agent_pos'].shape}")
            else:
                # 更新历史：移除最旧的观察，添加新观察
                self._obs_history = {
                    'point_cloud': torch.cat([self._obs_history['point_cloud'][:, 1:], current_obs['point_cloud']], dim=1),
                    'agent_pos': torch.cat([self._obs_history['agent_pos'][:, 1:], current_obs['agent_pos']], dim=1),
                }
            
            # 使用完整的观察历史作为输入
            policy_obs = self._obs_history
        else:
            current_obs = {
                'chest_camera_rgb': chest_camera_img.unsqueeze(1),
                'head_camera_rgb': head_camera_img.unsqueeze(1),
                'third_camera_rgb': third_camera_img.unsqueeze(1),  # 新增：第三人称相机
                'arm2_pos': self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
                # 'arm2_vel': self._robot.data.joint_vel[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
                'hand2_pos': self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
                # 'hand2_vel': self._robot.data.joint_vel[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
                'arm2_eef_pos': eef_state[:,:3].unsqueeze(1),
                'arm2_eef_quat': eef_state[:,3:7].unsqueeze(1),
                'target_pose': target_pose.unsqueeze(1),
            }
            # 对于非点云模式，暂时不维护历史（TODO: 如果模型需要，可以添加）
            policy_obs = current_obs
        
        # 如果是 nd 模式，可能 obs key 是 chest_camera_normals ?
        # 让我们查看 MultiModalImageDataset 的 _get_data_keys
        # obs_mode == "nd": keys.extend(['chest_camera_normals', ...])
        # 这意味着训练时 nd 模式下 key 变了。
        # 如果我们在这里只返回 'chest_camera_rgb'，那么 policy.predict_action 会报错吗？
        # policy.predict_action 内部会调用 obs_encoder.forward(obs_dict)。
        # obs_encoder 会根据 shape_meta 中的 key 去 obs_dict 找数据。
        
        # 因此，我们需要根据 obs_mode 正确设置 key。
        
        if self.cfg.obs_mode == "nd":
             # nd 模式下，训练数据可能是分开的 normals 和 depth?
             # 不，MultiModalImageDataset _sample_to_data 中：
             # chest_frames = self._process_nd_batch(...)
             # obs_data = {'chest_camera_rgb': chest_frames ...} 
             # 等等！注意看 _sample_to_data 的最后：
             # obs_data = {'chest_camera_rgb': chest_frames, ...}
             # 似乎 MultiModalImageDataset 把处理后的多通道数据都放到了 'chest_camera_rgb' 这个 key 下？
             # 让我们仔细检查提供的 MultiModalImageDataset 代码片段。
             
             # 代码片段中：
             # if self.obs_mode == "rgb": chest_frames = ...
             # elif self.obs_mode == "rgbm": chest_frames = ...
             # ...
             # obs_data = {
             #    'chest_camera_rgb': chest_frames,
             #    'head_camera_rgb': head_frames,
             #    ...
             # }
             
             # 没错！MultiModalImageDataset 确实把所有模式的图像数据都赋给了 'chest_camera_rgb' 这个 key。
             # 这意味着无论 obs_mode 是什么，policy 都期望在 'chest_camera_rgb' 中找到数据。
             # 只是这个数据的 shape (通道数) 会不同。
             
             pass # current_obs 已经使用了 'chest_camera_rgb'，所以不需要修改 key。

        # ==================== Action 队列管理（解决 DP3 输出步数 > decimation 问题）====================
        # DP3 一次预测 n_action_steps（例如 16）步，但每次环境只执行 decimation（例如 4）步
        # 策略：维护一个 action 队列，只有当队列不足时才调用 predict_action()
        
        # B = self._robot.data.joint_pos.shape[0]  # batch size（环境数量）
        
        # # 检查是否需要重新预测 action
        # need_prediction = False
        # if self._action_queue is None or self._action_queue_ptr is None:
        #     # 第一次调用，需要初始化
        #     need_prediction = True
        #     print(f"[Action Queue] 首次初始化，需要预测 action")
        # else:
        #     # 检查每个环境的 action 队列是否不足
        #     # 如果队列剩余 action < decimation，则需要重新预测
        #     remaining_actions = self._action_queue.shape[1] - self._action_queue_ptr
        #     need_prediction = (remaining_actions < self.cfg.decimation).any()
        #     if need_prediction:
        #         print(f"[Action Queue] 队列不足，剩余步数={remaining_actions.tolist()}, decimation={self.cfg.decimation}")
        
        # if need_prediction:
        #     # 调用策略模型预测 action
        #     print(f"[Action Queue] 调用 predict_action(), 输入 obs keys: {policy_obs.keys()}")
        #     with torch.no_grad():
        #         base_act_seq = self.base_policy.predict_action(policy_obs)['action']  # [B, n_action_steps, action_dim]
            
        #     print(f"[Action Queue] 预测完成: shape={base_act_seq.shape}, device={base_act_seq.device}")
            
        #     # 更新 action 队列
        #     self._action_queue = base_act_seq
        #     self._action_queue_ptr = torch.zeros(B, dtype=torch.long, device=self.device)
            
        #     print(f"[Action Queue] 队列已更新: queue_shape={self._action_queue.shape}, "
        #           f"ptr_shape={self._action_queue_ptr.shape}, ptr={self._action_queue_ptr.tolist()}")
        
        # # 安全检查：确保队列和指针已初始化
        # if self._action_queue is None or self._action_queue_ptr is None:
        #     raise RuntimeError(
        #         f"[Action Queue] 致命错误：队列未初始化！\n"
        #         f"  _action_queue = {self._action_queue}\n"
        #         f"  _action_queue_ptr = {self._action_queue_ptr}\n"
        #         f"  need_prediction = {need_prediction}\n"
        #         f"这不应该发生，请检查 predict_action() 是否正常执行。"
        #     )
        
        # # 从队列中提取当前需要执行的 action（decimation 步）
        # for i in range(self.cfg.decimation):
        #     # 对于每个环境，从队列中取出对应的 action
        #     batch_indices = torch.arange(B, device=self.device)
        #     action_indices = self._action_queue_ptr[batch_indices]
            
        #     # 提取 action: [B, action_dim]
        #     self._action = self._action_queue[batch_indices, action_indices, :]
            
        #     # 更新队列指针
        #     self._action_queue_ptr += 1
            
        #     # sim step
        #     self.sim_step()



        with torch.no_grad():
            base_act_seq = self.base_policy.predict_action(policy_obs)['action']  # [B, n_action_steps, action_dim]

        # sim step according to decimation - 只执行前 decimation 步
        for i in range(self.cfg.decimation):
            self._action = base_act_seq[:, i, :]
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



        # import matplotlib.pyplot as plt
        # head_camera_mask = self._robot.tiled_cameras["head_camera"].data.output["instance_segmentation_fast"][0,:,:,:].cpu().numpy()
        # # # head_camera_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][0,:,:,:].cpu().numpy()
        # # # head_camera_depth = self._robot.tiled_cameras["chest_camera"].data.output["depth"][0,:,:,:].cpu().numpy()
        # # # head_camera_normal = self._robot.tiled_cameras["chest_camera"].data.output["normals"][0,:,:,:].cpu().numpy()
       
        # # # plt.figure(figsize=(10, 10))
        # plt.imshow(head_camera_mask)
        # # # plt.subplot(1, 3, 2)
        # # # plt.imshow(head_camera_depth)
        # # # plt.subplot(1, 3, 3)
        # # # plt.imshow(head_camera_normal)
        # plt.show()
        # import time
        # time.sleep(0.1)



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
        
        # 重置观察历史缓冲区
        # 初始化为 None，会在第一次调用时用当前观察填充
        self._obs_history = None
        
        # 重置 action 队列
        # 初始化为 None，会在第一次调用 step() 时触发预测
        self._action_queue = None
        self._action_queue_ptr = None

    def _quat_orientation_loss(self, quat_init: torch.Tensor, quat_current: torch.Tensor) -> torch.Tensor:
        """
        计算两个四元数之间的 pitch+roll 朝向偏差
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

    def _eval_fail_moved_without_contact(self) -> torch.Tensor:
        """
        评估物体在未接触时是否被移动（被推动/碰撞）
        
        失败条件：物体未被接触，但 z 方向位移超过 2cm
        
        Returns:
            bfailed: 失败标志 [N]
        """
        # 检查物体是否未被接触
        not_contacted = ~self._has_contacted
        
        # 计算 z 方向位移（使用 object_state 需要先计算）
        current_z = self._target.data.root_pos_w[:, 2]
        height_diff = current_z - self._target_pos_init[:, 2]
        
        # 物体未被接触但 z 方向移动超过 2cm，判定为失败
        moved_too_much = height_diff > 0.02
        
        bfailed = not_contacted & moved_too_much
        
        return bfailed

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
        height_lift = self._target.data.root_pos_w[:, 2] - self._target_pos_init[:, 2]
        height_check = height_lift >= (self.cfg.lift_height_desired * 0.7 )
        
        # 2. 检查朝向偏差
        current_quat = self._target.data.root_quat_w  # 当前朝向 (wxyz)
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        # 综合判断：高度 AND 朝向
        bsuccessed = height_check & orientation_check
        
        return bsuccessed

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # task evalutation
        bfailed,self._has_contacted = eval_fail(self._target,self._contact_sensors, self._has_contacted) # type: ignore
        
        # 新增：检测物体在未接触时被移动（被推动/碰撞）
        bfailed_moved = self._eval_fail_moved_without_contact()
        bfailed = bfailed_moved
        
        # success eval（使用带朝向检查的函数）
        bsuccessed = self._eval_success_with_orientation()
        
        # 记录成功环境的步数
        success_indices = torch.nonzero(bsuccessed == True).squeeze(1).tolist()
        if isinstance(success_indices, int):
            success_indices = [success_indices]
        for idx in success_indices:
            # episode_length_buf 记录的是当前 episode 的步数
            success_step = self.episode_length_buf[idx].item()
            self._success_steps_list.append(int(success_step))
        
        # update success number
        self._episode_success_num += len(success_indices)

        return bsuccessed, bfailed, time_out # type: ignore
    

    def _reset_idx(self, env_ids: torch.Tensor | None, success_ids:Sequence[int]|None=None):

        # get env indexs
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

        
        # if self.cfg.enable_log:
        self._log_info()
        
        # 重置 action 队列（当有环境重置时，清空整个队列，让下次 step() 重新预测）
        # 注意：这是简化策略，更精细的做法是只重置对应环境的队列，但实现复杂且收益有限
        if self._action_queue is not None and len(env_ids) > 0:
            self._action_queue = None
            self._action_queue_ptr = None
        
        # variables used to store contact flag
        self._has_contacted[env_ids] = torch.zeros_like(self._has_contacted[env_ids],device=self.device, dtype=torch.bool) # type: ignore
        self._target_pos_init[env_ids,:]=self._target.data.root_link_pos_w[env_ids,:].clone()
        self._target_quat_init[env_ids,:]=self._target.data.root_quat_w[env_ids,:].clone()  # 保存初始朝向
    
    def _log_info(self):
        # log policy evalutation result
        if self.cfg.enable_eval and self._episode_num > 0:
            # 设置日志打印间隔
            # 单环境或少量环境：每个 episode 都打印
            # 多环境：每 num_envs 个 episode 打印一次（约等于每轮并行完成后打印）
            if self.num_envs <= 4:
                log_interval = 1  # 单环境或少量环境，每次都打印
            else:
                log_interval = self.num_envs  # 多环境，每轮打印一次
            
            # 只在达到打印间隔时输出日志
            if self._episode_num % log_interval == 0 or self._episode_num >= self.cfg.max_episode:
                policy_success_rate = float(self._episode_success_num) / float(self._episode_num)
                
                # 计算测试时间和效率
                test_time_sec = self._timer.run_time()
                test_time_min = test_time_sec / 60.0
                test_rate = self._episode_num / test_time_min if test_time_min > 0 else 0
                
                print(f"\n{'='*50}")
                print(f"[Episode {self._episode_num}/{self.cfg.max_episode}] 评估统计")
                print(f"  Policy成功率: {policy_success_rate * 100.0:.2f}%")
                print(f"  成功次数/总次数: {self._episode_success_num}/{self._episode_num}")
                
                # 输出成功步数统计
                if len(self._success_steps_list) > 0:
                    avg_success_steps = sum(self._success_steps_list) / len(self._success_steps_list)
                    min_steps = min(self._success_steps_list)
                    max_steps = max(self._success_steps_list)
                    print(f"  成功步数: 平均={avg_success_steps:.1f}, 最小={min_steps}, 最大={max_steps}")
                    # 显示最近的成功步数（最多显示最近10个）
                    recent_steps = self._success_steps_list[-10:] if len(self._success_steps_list) > 10 else self._success_steps_list
                    print(f"  最近成功步数: {recent_steps}")
                
                print(f"  测试时间: {test_time_min:.2f} 分钟 ({test_time_sec:.1f} 秒)")
                print(f"  测试效率: {test_rate:.2f} episode/分钟")
                
                if self.cfg.enable_output:
                    # compute data collect result
                    record_rate = self._episode_success_num / test_time_min if test_time_min > 0 else 0
                    print(f"  采集效率: {record_rate:.2f} 条/分钟")
                print(f"{'='*50}\n")
