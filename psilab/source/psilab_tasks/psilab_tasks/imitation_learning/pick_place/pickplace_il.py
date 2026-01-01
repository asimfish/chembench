import copy
from typing import Literal, Any, Sequence
from dataclasses import MISSING

import torch
import hydra
from omegaconf import OmegaConf

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

def load_diffusion_policy_from_checkpoint(checkpoint_path: str, device: str = 'cuda:0'):
    """
    从 checkpoint 加载 Diffusion Policy 模型
    """
    import dill
    
    # 注册 eval resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    
    print(f"Loading Diffusion Policy from: {checkpoint_path}")
    
    # 加载 checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location='cpu')
    
    cfg = payload['cfg']
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
    
    policy.eval()
    policy.to(device)
    
    return policy

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
    lift_height_desired = 0.25
    
    # 成功判断：朝向偏差阈值（sin²(θ/2)，0=完全一致，1=上下颠倒）
    # 0.1 约等于 37° 的偏差，0.05 约等于 26° 的偏差
    orientation_threshold: float = 0.1
    
    # 观测模式配置
    # 可选: "rgb", "rgbm", "nd", "rgbnd", "state"
    obs_mode: Literal["rgb", "rgbm", "nd", "rgbnd", "state"] = "nd"
    
    # Mask 解耦实验配置
    # "real": 使用真实的 mask（默认）
    # "all_0": mask 通道填充全0（测试模型是否依赖 mask）
    # "all_1": mask 通道填充全1（测试模型是否依赖 mask）
    mask_mode: str = "real"

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

    def step(self,actions):
        
        # get obs for policy
        eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        eef_state = self._robot.data.body_link_state_w[:,eef_link_index,:7].clone()
        eef_state[:,:3] -= self._robot.data.root_state_w[:,:3]
        
        # process image
        
        # 获取基础 RGB 图像
        chest_rgb = self._robot.tiled_cameras["chest_camera"].data.output["rgb"][:, :, :, :]
        head_rgb = self._robot.tiled_cameras["head_camera"].data.output["rgb"][:, :, :, :]
        
        # 初始化可选通道
        chest_mask, head_mask = None, None
        chest_depth, head_depth = None, None
        chest_normal, head_normal = None, None

        # 根据 obs_mode 获取所需的额外通道
        # 1. Mask
        if self.cfg.obs_mode == "rgbm":
            if self.cfg.mask_mode == "real":
                if "instance_segmentation_fast" in self._robot.tiled_cameras["chest_camera"].data.output:
                    chest_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
                if "instance_segmentation_fast" in self._robot.tiled_cameras["head_camera"].data.output:
                    head_mask = self._robot.tiled_cameras["head_camera"].data.output["instance_segmentation_fast"][:, :, :, 0]
            elif self.cfg.mask_mode == "all_0":
                chest_mask = torch.zeros_like(chest_rgb[:, :, :, 0])
                head_mask = torch.zeros_like(head_rgb[:, :, :, 0])
            elif self.cfg.mask_mode == "all_1":
                # 解耦实验：mask 通道填充全 1（会被归一化为 1.0）
                chest_mask = torch.ones_like(chest_rgb[:, :, :, 0])
                head_mask = torch.ones_like(head_rgb[:, :, :, 0])
        
        # 2. Depth
        if self.cfg.obs_mode in ["nd", "rgbnd"]:
            if "depth" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_depth = self._robot.tiled_cameras["chest_camera"].data.output["depth"][:, :, :, 0]
            if "depth" in self._robot.tiled_cameras["head_camera"].data.output:
                head_depth = self._robot.tiled_cameras["head_camera"].data.output["depth"][:, :, :, 0]

        # 3. Normal
        if self.cfg.obs_mode in ["nd", "rgbnd"]:
            if "normals" in self._robot.tiled_cameras["chest_camera"].data.output:
                chest_normal = self._robot.tiled_cameras["chest_camera"].data.output["normals"][:, :, :, :3] # 取前3通道
            if "normals" in self._robot.tiled_cameras["head_camera"].data.output:
                head_normal = self._robot.tiled_cameras["head_camera"].data.output["normals"][:, :, :, :3]

        # 统一处理图像
        chest_camera_img = None
        head_camera_img = None

        if self.cfg.obs_mode != 'state':
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
                'target_pose': target_pose.unsqueeze(1),
                'arm2_pos': self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
                'hand2_pos': self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
                'arm2_eef_pos': eef_state[:,:3].unsqueeze(1),
                'arm2_eef_quat': eef_state[:,3:7].unsqueeze(1),
                # 'target_pose': target_pose.unsqueeze(1),
            }
        else:
            current_obs = {
                'chest_camera_rgb': chest_camera_img.unsqueeze(1),
                'head_camera_rgb': head_camera_img.unsqueeze(1),
                # 'third_person_camera_rgb': third_person_camera_rgb.unsqueeze(1),
                'arm2_pos': self._robot.data.joint_pos[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
                # 'arm2_vel': self._robot.data.joint_vel[:,self._robot.actuators["arm2"].joint_indices].unsqueeze(1),
                'hand2_pos': self._robot.data.joint_pos[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
                # 'hand2_vel': self._robot.data.joint_vel[:,self._robot.actuators["hand2"].joint_indices[:6]].unsqueeze(1), # type: ignore
                'arm2_eef_pos': eef_state[:,:3].unsqueeze(1),
                'arm2_eef_quat': eef_state[:,3:7].unsqueeze(1),
                'target_pose': target_pose.unsqueeze(1),
            }
        
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

        # policy model step
        with torch.no_grad():
            base_act_seq = self.base_policy.predict_action(current_obs)['action']    
        # sim step according to decimation
        for i in range(self.cfg.decimation):
            #
            self._action = base_act_seq[:,i,:]
            # sim step
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
        """
        # 1. 检查抬起高度（只需高于目标高度即可）
        height_lift = self._target.data.root_pos_w[:, 2] - self._target_pos_init[:, 2]
        height_check = height_lift >= (self.cfg.lift_height_desired * 0.8 )
        
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
