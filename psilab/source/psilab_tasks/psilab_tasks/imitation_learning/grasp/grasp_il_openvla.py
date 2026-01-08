"""
Grasp IL Evaluation with OpenVLA - 使用 OpenVLA 模型的抓取任务评估

基于 run_libero_eval.py 的测试方式，为抓取任务提供 OpenVLA 模型评估。

Usage:
    python grasp_il_openvla.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_config <CONFIG_PATH> \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>

支持的观测模式（通过 task_config 配置）：
- rgb: 3通道 RGB 图像
- rgbm: 4通道 RGB + Mask
- nd: 4通道 Normal + Depth
- rgbnd: 7通道 RGB + Normal + Depth
- state: 纯状态（无图像）

关键参数说明：
1. orientation_threshold = 0.05 (约26°偏差)
   - OpenVLA 可能需要稍宽松的朝向标准
2. 直接使用 OpenVLA 输出的 action
3. 支持多观测模式的图像输入
"""

import os
import sys
from dataclasses import dataclass, MISSING
from pathlib import Path
from typing import Literal, Any, Sequence, Optional, Union
import copy

import torch
import draccus
import numpy as np
import tqdm

# Isaac Lab imports
from isaaclab.sim import SimulationCfg, PhysxCfg, RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

# Local imports
from psilab import OUTPUT_DIR
from psilab.scene import SceneCfg
from psilab.envs.il_env import ILEnv
from psilab.envs.il_env_cfg import ILEnvCfg
from psilab.utils.timer_utils import Timer
from psilab.eval.grasp_rigid import eval_fail
from psilab.utils.data_collect_utils import parse_data, save_data
from psilab_tasks.imitation_learning.grasp.image_utils import process_batch_image_multimodal

# OpenVLA imports (will be imported in load function)
import wandb


def load_openvla_model(cfg):
    """
    加载 OpenVLA 模型
    
    参考 robot_utils.py 中的 get_model() 函数
    """
    print(f"Loading OpenVLA model from: {cfg.pretrained_checkpoint}")
    
    if cfg.model_family == "openvla":
        # 导入 OpenVLA 相关模块
        # 添加 OpenVLA 路径
        openvla_path = '/home/psibot/chembench/openvla'
        if openvla_path not in sys.path:
            sys.path.insert(0, openvla_path)
        
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
        
        # 加载模型
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16 if not (cfg.load_in_8bit or cfg.load_in_4bit) else torch.float16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # 加载 processor
        processor = AutoProcessor.from_pretrained(
            cfg.pretrained_checkpoint,
            trust_remote_code=True,
        )
        
        # 移到 GPU（如果不是 8bit/4bit）
        if not (cfg.load_in_8bit or cfg.load_in_4bit):
            model = model.to(cfg.device)
        
        model.eval()
        
        print(f"  OpenVLA model loaded successfully on {cfg.device}")
        return model, processor
    else:
        raise ValueError(f"Unsupported model_family: {cfg.model_family}")


def get_openvla_action(
    model,
    processor,
    observation: dict,
    task_description: str,
    unnorm_key: str,
    center_crop: bool = True,
):
    """
    使用 OpenVLA 模型预测 action
    
    Args:
        model: OpenVLA 模型
        processor: OpenVLA processor
        observation: 观测字典，包含 'full_image' (numpy array, H x W x C)
        task_description: 任务描述文本
        unnorm_key: action unnormalization key
        center_crop: 是否使用 center crop
    
    Returns:
        action: numpy array, shape [action_dim]
    """
    # 获取图像
    image = observation["full_image"]  # [H, W, C], numpy array
    
    # 构建 prompt
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    
    # 预处理图像和文本
    inputs = processor(prompt, image).to(model.device, dtype=model.dtype)
    
    # 预测 action
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    
    # 转换为 numpy
    action = action.cpu().numpy()
    
    return action


@dataclass
class OpenVLAEvalConfig:
    """OpenVLA 评估配置"""
    # fmt: off
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                                    # 模型类型（目前只支持 openvla）
    pretrained_checkpoint: Union[str, Path] = MISSING                # 预训练模型路径（必需）
    load_in_8bit: bool = False                                       # 是否使用 8-bit 量化
    load_in_4bit: bool = False                                       # 是否使用 4-bit 量化
    center_crop: bool = True                                         # 是否使用 center crop（如果训练时用了 random crop）
    
    #################################################################################################################
    # Task environment parameters
    #################################################################################################################
    task_config: Union[str, Path] = MISSING                          # 任务配置文件路径（必需）
    task_description: str = "grasp the bottle"                       # 任务描述（用于 prompt）
    unnorm_key: str = "psilab_grasp"                                 # Action unnormalization key（需要在模型中预先定义）
    
    num_envs: int = 1                                                # 并行环境数量
    max_episode: int = 50                                            # 最大测试 episode 数
    
    #################################################################################################################
    # Observation mode
    #################################################################################################################
    obs_mode: Literal["rgb", "rgbm", "nd", "rgbnd", "state"] = "rgb"  # 观测模式
    mask_mode: str = "real"                                           # Mask 模式（real/all_0/all_1）
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    device: str = "cuda:0"                                           # 设备
    run_id_note: Optional[str] = None                                # 额外的运行 ID 标记
    local_log_dir: str = "./experiments/logs"                        # 本地日志目录
    
    use_wandb: bool = False                                          # 是否使用 W&B 记录
    wandb_project: str = "openvla-grasp"                             # W&B 项目名
    wandb_entity: str = "YOUR_WANDB_ENTITY"                          # W&B entity
    
    seed: int = 42                                                   # 随机种子
    headless: bool = True                                            # 是否使用 headless 模式
    
    # fmt: on


@configclass
class GraspBottleEnvCfg(ILEnvCfg):
    """抓取环境配置"""
    
    # fake params
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130
    
    episode_length_s = 2
    decimation = 4
    sample_step = 1
    
    # viewer config
    viewer = ViewerCfg(
        eye=(1.2, 0.0, 1.2),
        lookat=(-15.0, 0.0, 0.3)
    )
    
    # simulation config
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=32,
            max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.002,
            enable_ccd=True,
            gpu_found_lost_pairs_capacity=137401003
        ),
        render=RenderCfg(
            enable_translucency=True,
        ),
    )
    
    # scene config
    scene: SceneCfg = MISSING  # type: ignore
    
    # output folder
    output_folder = OUTPUT_DIR + "/il_openvla"
    
    # lift desired height
    lift_height_desired = 0.2
    
    # 成功判断：朝向偏差阈值
    orientation_threshold: float = 0.05
    
    # 观测模式配置
    obs_mode: Literal["rgb", "rgbm", "nd", "rgbnd", "state"] = "rgb"
    
    # Mask 解耦实验配置
    mask_mode: str = "real"


class GraspBottleEnvOpenVLA(ILEnv):
    """使用 OpenVLA 的抓取环境"""
    
    cfg: GraspBottleEnvCfg
    
    def __init__(
        self,
        cfg: GraspBottleEnvCfg,
        model,
        processor,
        openvla_cfg: OpenVLAEvalConfig,
        render_mode: str | None = None,
        **kwargs
    ):
        # 禁用差分 IK 控制器
        cfg.scene.robots_cfg["robot"].diff_ik_controllers = None  # type: ignore
        
        super().__init__(cfg, render_mode, **kwargs)
        
        # 保存模型和配置
        self.model = model
        self.processor = processor
        self.openvla_cfg = openvla_cfg
        
        # 场景实例
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["bottle"]
        
        # 初始化接触传感器
        self._contact_sensors = {}
        for key in [
            "hand2_link_base", "hand2_link_1_1", "hand2_link_1_2", "hand2_link_1_3",
            "hand2_link_2_1", "hand2_link_2_2", "hand2_link_3_1", "hand2_link_3_2",
            "hand2_link_4_1", "hand2_link_4_2", "hand2_link_5_1", "hand2_link_5_2"
        ]:
            self._contact_sensors[key] = self.scene.sensors[key]
        
        # 关节限位
        self._joint_limit_lower = self._robot.data.joint_limits[:, :, 0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:, :, 1].clone()
        
        # Timer
        self._timer = Timer()
        
        # 状态变量
        self._has_contacted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # type: ignore
        self._target_pos_init = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_quat_init = torch.zeros((self.num_envs, 4), device=self.device)
        self._success_steps_list: list[int] = []
        
        # 设置相机视角
        self.sim.set_camera_view([-0.7, -5.2, 1.3], [-1.2, -5.2, 1.1])
        
        # 设置 RTX 渲染选项
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
        
        print(f"[GraspBottleEnvOpenVLA] Initialized with obs_mode={cfg.obs_mode}")
    
    def _process_observation_for_openvla(self) -> dict:
        """
        处理观测数据，生成 OpenVLA 期望的输入格式
        
        Returns:
            observation: 包含 'full_image' 的字典，full_image 是 numpy array [H, W, C]
        """
        # 获取基础 RGB 图像（使用主相机，比如 chest_camera）
        # 根据 obs_mode 选择合适的相机和处理方式
        if self.cfg.obs_mode == "state":
            raise ValueError("OpenVLA requires image input, but obs_mode is 'state'")
        
        # 使用胸部相机作为主相机
        chest_rgb = self._robot.tiled_cameras["chest_camera"].data.output["rgb"][:, :, :, :]
        
        # 初始化可选通道
        chest_mask = None
        chest_depth = None
        chest_normal = None
        
        # 根据 obs_mode 获取额外通道
        if self.cfg.obs_mode in ["rgbm"]:
            if self.cfg.mask_mode == "real":
                if "instance_segmentation_fast" in self._robot.tiled_cameras["chest_camera"].data.output:
                    chest_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
            elif self.cfg.mask_mode == "all_0":
                chest_mask = torch.zeros_like(chest_rgb[:, :, :, 0])
            elif self.cfg.mask_mode == "all_1":
                chest_mask = torch.ones_like(chest_rgb[:, :, :, 0])
        
        if self.cfg.obs_mode in ["nd", "rgbnd"]:
            if "depth" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_depth = self._robot.tiled_cameras["chest_camera"].data.output["depth"][:, :, :, 0]
            if "normals" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_normal = self._robot.tiled_cameras["chest_camera"].data.output["normals"][:, :, :, :3]
        
        # 处理图像
        chest_camera_img = process_batch_image_multimodal(
            rgb=chest_rgb,
            mask=chest_mask,
            depth=chest_depth,
            normal=chest_normal,
            obs_mode=self.cfg.obs_mode
        )  # [B, C, H, W]
        
        # OpenVLA 期望输入: [H, W, C]
        # 取第一个 batch（假设单环境或批量处理第一个）
        image = chest_camera_img[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        
        # 构建观测字典
        observation = {
            "full_image": image,  # numpy array [H, W, C]
        }
        
        return observation
    
    def step(self, actions):
        """环境步进"""
        
        # 获取观测
        observation = self._process_observation_for_openvla()
        
        # 使用 OpenVLA 预测 action
        with torch.no_grad():
            action_np = get_openvla_action(
                model=self.model,
                processor=self.processor,
                observation=observation,
                task_description=self.openvla_cfg.task_description,
                unnorm_key=self.openvla_cfg.unnorm_key,
                center_crop=self.openvla_cfg.center_crop,
            )  # [action_dim]
        
        # 转换为 torch tensor
        action = torch.from_numpy(action_np).float().to(self.device).unsqueeze(0)  # [1, action_dim]
        
        # 执行 action（根据 decimation 重复执行）
        for i in range(self.cfg.decimation):
            self._action = action
            self.sim_step()
        
        return super().step(actions)
    
    def sim_step(self):
        """仿真步进"""
        
        # 设置目标关节位置
        self._robot.set_joint_position_target(
            self._action[:, :7],
            self._robot.actuators["arm2"].joint_indices
        )  # type: ignore
        self._robot.set_joint_position_target(
            self._action[:, 7:],
            self._robot.actuators["hand2"].joint_indices[:6]
        )  # type: ignore
        
        # 写入仿真
        self._robot.write_data_to_sim()
        super().sim_step()
        
        # 解析数据（如果需要保存）
        if self.cfg.enable_output and self._sim_step_counter % self.cfg.sample_step == 0:
            parse_data(
                sim_time=self._sim_step_counter * self.cfg.sim.dt,
                data=self._data,
                scene=self.scene
            )
        
        # 获取 dones
        success, fail, time_out = self._get_dones()
        reset = success | fail | time_out
        reset_ids = torch.nonzero(reset == True).squeeze()
        reset_ids = reset_ids.unsqueeze(0) if reset_ids.size() == torch.Size([]) else reset_ids
        
        success_ids = torch.nonzero(success == True).squeeze().tolist()
        success_ids = [success_ids] if type(success_ids) == int else success_ids
        
        # 重置环境
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids, success_ids)  # type: ignore
            self.scene.write_data_to_sim()
            self.sim.forward()
            
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
    
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """重置环境"""
        super().reset()
    
    def _quat_orientation_loss(self, quat_init: torch.Tensor, quat_current: torch.Tensor) -> torch.Tensor:
        """计算四元数朝向偏差"""
        aw, ax, ay, az = quat_init.unbind(-1)
        bw, bx, by, bz = quat_current.unbind(-1)
        
        cw, cx, cy, cz = aw, -ax, -ay, -az
        
        rw = cw * bw - cx * bx - cy * by - cz * bz
        rx = cw * bx + cx * bw + cy * bz - cz * by
        ry = cw * by - cx * bz + cy * bw + cz * bx
        rz = cw * bz + cx * by - cy * bx + cz * bw
        
        norm = torch.sqrt(rw * rw + rx * rx + ry * ry + rz * rz) + 1e-8
        rx, ry = rx / norm, ry / norm
        
        loss = rx * rx + ry * ry
        return loss
    
    def _eval_fail_moved_without_contact(self) -> torch.Tensor:
        """评估物体在未接触时是否被移动"""
        not_contacted = ~self._has_contacted
        current_z = self._target.data.root_pos_w[:, 2]
        height_diff = current_z - self._target_pos_init[:, 2]
        moved_too_much = height_diff > 0.02
        bfailed = not_contacted & moved_too_much
        return bfailed
    
    def _eval_success_with_orientation(self) -> torch.Tensor:
        """评估抓取是否成功（检查高度和朝向）"""
        height_lift = self._target.data.root_pos_w[:, 2] - self._target_pos_init[:, 2]
        height_check = height_lift >= (self.cfg.lift_height_desired * 0.6)
        
        current_quat = self._target.data.root_quat_w
        orientation_loss = self._quat_orientation_loss(self._target_quat_init, current_quat)
        orientation_check = orientation_loss < self.cfg.orientation_threshold
        
        bsuccessed = height_check & orientation_check
        return bsuccessed
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取完成状态"""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        bfailed, self._has_contacted = eval_fail(
            self._target,
            self._contact_sensors,
            self._has_contacted
        )  # type: ignore
        
        bfailed_moved = self._eval_fail_moved_without_contact()
        bfailed = bfailed_moved
        
        bsuccessed = self._eval_success_with_orientation()
        
        # 记录成功步数
        success_indices = torch.nonzero(bsuccessed == True).squeeze(1).tolist()
        if isinstance(success_indices, int):
            success_indices = [success_indices]
        for idx in success_indices:
            success_step = self.episode_length_buf[idx].item()
            self._success_steps_list.append(int(success_step))
        
        self._episode_success_num += len(success_indices)
        
        return bsuccessed, bfailed, time_out  # type: ignore
    
    def _reset_idx(self, env_ids: torch.Tensor | None, success_ids: Sequence[int] | None = None):
        """重置指定环境"""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        
        if success_ids is None:
            success_ids = []
        
        # 保存数据
        if self.cfg.enable_output:
            self._data = save_data(
                data=self._data,
                cfg=self.cfg,
                scene=self.scene,
                env_indexs=success_ids,
                reset_env_indexs=env_ids.tolist(),
            )
        
        super()._reset_idx(env_ids)
        
        self._log_info()
        
        # 重置状态
        self._has_contacted[env_ids] = torch.zeros_like(
            self._has_contacted[env_ids],
            device=self.device,
            dtype=torch.bool
        )  # type: ignore
        self._target_pos_init[env_ids, :] = self._target.data.root_link_pos_w[env_ids, :].clone()
        self._target_quat_init[env_ids, :] = self._target.data.root_quat_w[env_ids, :].clone()
    
    def _log_info(self):
        """打印日志信息"""
        if self.cfg.enable_eval and self._episode_num > 0:
            if self.num_envs <= 4:
                log_interval = 1
            else:
                log_interval = self.num_envs
            
            if self._episode_num % log_interval == 0 or self._episode_num >= self.cfg.max_episode:
                policy_success_rate = float(self._episode_success_num) / float(self._episode_num)
                
                test_time_sec = self._timer.run_time()
                test_time_min = test_time_sec / 60.0
                test_rate = self._episode_num / test_time_min if test_time_min > 0 else 0
                
                print(f"\n{'=' * 50}")
                print(f"[Episode {self._episode_num}/{self.cfg.max_episode}] OpenVLA 评估统计")
                print(f"  成功率: {policy_success_rate * 100.0:.2f}%")
                print(f"  成功次数/总次数: {self._episode_success_num}/{self._episode_num}")
                
                if len(self._success_steps_list) > 0:
                    avg_success_steps = sum(self._success_steps_list) / len(self._success_steps_list)
                    min_steps = min(self._success_steps_list)
                    max_steps = max(self._success_steps_list)
                    print(f"  成功步数: 平均={avg_success_steps:.1f}, 最小={min_steps}, 最大={max_steps}")
                    recent_steps = self._success_steps_list[-10:] if len(self._success_steps_list) > 10 else self._success_steps_list
                    print(f"  最近成功步数: {recent_steps}")
                
                print(f"  测试时间: {test_time_min:.2f} 分钟 ({test_time_sec:.1f} 秒)")
                print(f"  测试效率: {test_rate:.2f} episode/分钟")
                print(f"{'=' * 50}\n")


@draccus.wrap()
def eval_openvla_grasp(cfg: OpenVLAEvalConfig) -> None:
    """OpenVLA 抓取任务评估主函数"""
    
    # 验证参数
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert cfg.task_config is not None, "cfg.task_config must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting center_crop==True because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    
    # 设置随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # 加载 OpenVLA 模型
    model, processor = load_openvla_model(cfg)
    
    # 检查 unnorm_key 是否存在
    if cfg.model_family == "openvla":
        if hasattr(model, 'norm_stats'):
            if cfg.unnorm_key not in model.norm_stats:
                print(f"⚠️  Warning: unnorm_key '{cfg.unnorm_key}' not found in model.norm_stats")
                print(f"   Available keys: {list(model.norm_stats.keys())}")
                print(f"   Using first available key as fallback")
                if len(model.norm_stats) > 0:
                    cfg.unnorm_key = list(model.norm_stats.keys())[0]
                    print(f"   Fallback unnorm_key: {cfg.unnorm_key}")
    
    # 初始化日志
    from datetime import datetime
    DATE_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"EVAL-grasp-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to: {local_log_filepath}")
    
    # 初始化 W&B
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config=cfg.__dict__,
        )
    
    # 加载任务配置（这里需要实现配置加载逻辑）
    # TODO: 从 task_config 路径加载 YAML 配置
    print(f"\n⚠️  Note: Task config loading from '{cfg.task_config}' needs to be implemented")
    print(f"   For now, using default configuration")
    
    # 创建环境配置
    env_cfg = GraspBottleEnvCfg()
    env_cfg.obs_mode = cfg.obs_mode
    env_cfg.mask_mode = cfg.mask_mode
    env_cfg.max_episode = cfg.max_episode
    env_cfg.enable_eval = True
    env_cfg.enable_output = False  # 默认不保存数据
    
    # 创建环境
    print("\nCreating environment...")
    # TODO: 需要加载 scene 配置
    # env_cfg.scene = load_scene_config(cfg.task_config)
    
    print(f"\n⚠️  Environment creation requires scene configuration")
    print(f"   Please implement scene loading logic or provide scene config")
    
    # 示例：如何使用环境（需要实际的 scene 配置）
    # env = GraspBottleEnvOpenVLA(
    #     cfg=env_cfg,
    #     model=model,
    #     processor=processor,
    #     openvla_cfg=cfg,
    #     render_mode=None if cfg.headless else "human",
    # )
    
    # # 运行评估
    # print("\nStarting evaluation...")
    # for episode_idx in tqdm.tqdm(range(cfg.max_episode)):
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         obs, reward, done, info = env.step(None)
    #
    #     # 记录结果
    #     if cfg.use_wandb:
    #         wandb.log({
    #             "episode": episode_idx,
    #             "success": info.get("success", 0),
    #         })
    
    # 关闭日志
    log_file.close()
    
    print(f"\nEvaluation complete! Results saved to: {local_log_filepath}")
    
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    eval_openvla_grasp()

