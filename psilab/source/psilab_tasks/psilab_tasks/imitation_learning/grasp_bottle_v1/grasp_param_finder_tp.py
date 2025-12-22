# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-12-13
# Description: 遥操作查找抓取参数工具
# 
# 功能说明:
# 1. 通过遥操作移动手臂到目标抓取位置
# 2. 按 Enter 键锁定手臂位置并记录末端执行器位姿
# 3. 闭合手指进行抓取测试
# 4. 按 S 解除手臂锁定，检验能否抓起物体
# 5. 按 Y 确认抓取成功，输出配置参数
# 6. 按 N 放弃本次尝试，继续调整
# 7. 按 R 重置场景

""" Python Modules  """ 
from __future__ import annotations
from dataclasses import MISSING
from typing import Any
from collections.abc import Sequence

""" Common Modules  """ 
import torch
import numpy as np
import carb
import omni
from scipy.spatial.transform import Rotation as R

""" Isaac Lab Modules  """ 
from isaaclab.sim import SimulationCfg,PhysxCfg,RenderCfg
from isaaclab.utils import configclass
from isaaclab.envs.common import ViewerCfg

""" Psi Lab Modules  """
from psilab.envs.tp_env import TPEnv 
from psilab.envs.tp_env_cfg import TPEnvCfg
from psilab.utils.timer_utils import Timer
from psilab.devices.configs.psi_glove_cfg import PSIGLOVE_PSI_DC_02_CFG
from psilab import OUTPUT_DIR
from psilab.utils.math_utils import unnormalize_v2


@configclass
class GraspParamFinderEnvCfg(TPEnvCfg):
    """抓取参数查找器配置"""

    episode_length_s = 1 * 210 / 60.0
    decimation = 1
    action_scale = 0.5
    action_space = 13
    observation_space = 130
    state_space = 130

    # device
    device_type = "psi-glove"
    device_cfg = PSIGLOVE_PSI_DC_02_CFG

    viewer = ViewerCfg(
        eye=(2.0, 0.0, 1.2),
        lookat=(-15.0, 0.0, 0.3)
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1/120, 
        render_interval=1,
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

    sample_step = 1
    
    # scene config
    scene = MISSING # type: ignore
    
    output_folder = OUTPUT_DIR + "/grasp_param_finder"

    # 手指闭合阈值 (平均值低于此值时认为手指闭合)
    hand_close_threshold = 0.3
    
    # 目标物体名称 (用于输出配置)
    target_object_name = "glass_beaker_100ml"


class GraspParamFinderEnv(TPEnv):
    """遥操作查找抓取参数的环境"""

    cfg: GraspParamFinderEnvCfg

    def __init__(self, cfg: GraspParamFinderEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 场景对象
        self._robot = self.scene.robots["robot"]
        self._target = self.scene.rigid_objects["bottle"]

        # 启动设备
        self._device.start()  # type: ignore

        # 末端执行器索引
        self._arm2_eef_link_index = self._robot.find_bodies("arm2_link7")[0][0]
        
        # 关节索引
        self._arm2_joint_index = self._robot.actuators["arm2"].joint_indices
        self._hand2_joint_index = self._robot.actuators["hand2"].joint_indices[:6]

        # 关节限制
        self._joint_limit_lower = self._robot.data.joint_limits[:, :, 0].clone()
        self._joint_limit_upper = self._robot.data.joint_limits[:, :, 1].clone()

        # 末端执行器期望位姿
        self._arm2_eef_pose_desired = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)
        self._arm2_eef_pose_init = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)

        # 状态变量
        self._arm_locked = False           # 手臂是否锁定
        self._waiting_for_confirm = False  # 是否等待用户确认
        self._locked_eef_pose = None       # 锁定时的末端执行器位姿
        self._locked_target_pos = None     # 锁定时的物体位置
        
        # 注册键盘处理
        self._register_keyboard_handler()
        
        # 打印使用说明
        self._print_instructions()

        # 设置 RTX 渲染选项: Fractional Cutout Opacity
        import carb
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)


    def _print_instructions(self):
        """打印使用说明"""
        print("\n" + "=" * 60)
        print("抓取参数查找器 - 使用说明")
        print("=" * 60)
        print("1. 四指握拳开启控制模式")
        print("2. 遥操作移动手臂到目标抓取位置")
        print("3. 按 [Enter] 锁定手臂位置")
        print("4. 闭合手指进行抓取测试")
        print("5. 按 [S] 解除手臂锁定 → 检验能否抓起物体")
        print("6. 按 [Y] 确认抓取成功 → 输出配置参数")
        print("7. 按 [N] 放弃本次尝试 → 继续调整")
        print("8. 按 [R] 或 手势(食指+小拇指伸直) 重置场景")
        print("=" * 60)
        print(f"目标物体: {self.cfg.target_object_name}")
        print("=" * 60 + "\n")

    def _register_keyboard_handler(self):
        """注册键盘回调"""
        appwindow = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        input_interface.subscribe_to_keyboard_events(keyboard, self._keyboard_event_handler)

    def _keyboard_event_handler(self, event, *args, **kwargs):
        """处理键盘事件"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Enter: 锁定手臂位置
            if event.input == carb.input.KeyboardInput.ENTER:
                if not self._arm_locked and self._device.bControl:
                    self._lock_arm_position()
            
            # S: 解除手臂锁定，检验抓取效果
            elif event.input == carb.input.KeyboardInput.S:
                if self._arm_locked:
                    self._unlock_arm_for_test()
            
            # Y: 确认抓取成功
            elif event.input == carb.input.KeyboardInput.Y:
                if self._waiting_for_confirm:
                    self._confirm_grasp_success()
            
            # N: 放弃本次尝试
            elif event.input == carb.input.KeyboardInput.N:
                if self._waiting_for_confirm:
                    self._cancel_grasp()
            
            # R: 重置场景
            elif event.input == carb.input.KeyboardInput.R:
                self._reset_finder()
        
        return True

    def _lock_arm_position(self):
        """锁定手臂位置并记录末端执行器位姿"""
        self._arm_locked = True
        
        # 获取当前末端执行器的世界位姿
        eef_pos_w = self._robot.data.body_link_pos_w[0, self._arm2_eef_link_index, :].clone()
        eef_quat_w = self._robot.data.body_link_quat_w[0, self._arm2_eef_link_index, :].clone()
        
        # 获取当前物体位置
        # target_position = self._target.data.root_pos_w[env_ids,:]-self._robot.data.root_link_pos_w[env_ids,:]

        
        target_pos_w = self._target.data.root_pos_w[0, :].clone()
        
        # 保存锁定时的位姿
        self._locked_eef_pose = torch.cat([eef_pos_w, eef_quat_w])
        self._locked_target_pos = target_pos_w
        
        # 计算相对偏移
        offset = eef_pos_w - target_pos_w
        
        # 四元数转欧拉角
        quat_wxyz = eef_quat_w.cpu().numpy()
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        euler_deg = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
        
        print("\n" + "=" * 60)
        print("手臂位置已锁定！")
        print("=" * 60)
        print(f"末端执行器世界位置: [{eef_pos_w[0]:.4f}, {eef_pos_w[1]:.4f}, {eef_pos_w[2]:.4f}]")
        print(f"末端执行器世界旋转 (wxyz): [{quat_wxyz[0]:.4f}, {quat_wxyz[1]:.4f}, {quat_wxyz[2]:.4f}, {quat_wxyz[3]:.4f}]")
        print(f"末端执行器欧拉角 (xyz, deg): [{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}]")
        print("-" * 60)
        print(f"物体位置: [{target_pos_w[0]:.4f}, {target_pos_w[1]:.4f}, {target_pos_w[2]:.4f}]")
        print(f"相对偏移: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
        print("=" * 60)
        print("现在可以闭合手指进行抓取测试...")
        print("按 [S] 解除手臂锁定检验抓取效果")
        print("按 [Y] 确认成功，按 [N] 放弃")
        print("=" * 60 + "\n")
        
        self._waiting_for_confirm = True

    def _unlock_arm_for_test(self):
        """解除手臂锁定，用于检验抓取效果"""
        self._arm_locked = False
        
        # 更新初始位姿为当前位姿，这样解锁后手臂位置不会突变
        self._arm2_eef_pose_init = self._robot.data.body_link_state_w[:, self._arm2_eef_link_index, :7].clone()
        self._arm2_eef_pose_init[:, :3] -= self._robot.data.root_state_w[:, :3]
        
        # 重置设备的追踪器初始位姿，使当前位置成为新的起点
        # 设置为默认值后，下一次更新时会自动将当前位姿设为初始位姿
        self._device._tracker_pose_init[self._device.cfg.tracker_serial_right] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device.cfg.device
        )
        
        print("\n" + "=" * 60)
        print("手臂已解锁！")
        print("=" * 60)
        print("现在可以移动手臂检验抓取效果")
        print("如果物体跟随手臂移动，说明抓取成功")
        print("-" * 60)
        print("按 [Y] 确认抓取成功 → 输出配置参数")
        print("按 [N] 放弃本次尝试 → 重新调整")
        print("按 [Enter] 重新锁定手臂位置")
        print("=" * 60 + "\n")

    def _confirm_grasp_success(self):
        """确认抓取成功，输出配置参数"""
        if self._locked_eef_pose is None or self._locked_target_pos is None:
            print("[ERROR] 没有锁定的位姿数据")
            return
        
        # 计算相对偏移
        eef_pos = self._locked_eef_pose[:3]
        eef_quat = self._locked_eef_pose[3:]
        offset = eef_pos - self._locked_target_pos
        
        # 四元数转欧拉角
        quat_wxyz = eef_quat.cpu().numpy()
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        euler_deg = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
        
        # 输出可直接使用的配置
        print("\n" + "=" * 60)
        print("抓取成功！以下是可以添加到 grasp_configs.py 的配置：")
        print("=" * 60)
        print(f'''
    "{self.cfg.target_object_name}": {{
        "offset": [{offset[0].item():.4f}, {offset[1].item():.4f}, {offset[2].item():.4f}],
        "euler_deg": [{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}],
        "description": "{self.cfg.target_object_name} 抓取配置"
    }},
''')
        print("=" * 60)
        print("按 [R] 重置场景继续测试其他物体")
        print("=" * 60 + "\n")
        
        self._waiting_for_confirm = False

    def _cancel_grasp(self):
        """放弃本次尝试"""
        print("\n[INFO] 已放弃本次尝试，解锁手臂位置")
        print("[INFO] 继续调整位置，按 [Enter] 再次锁定\n")
        self._arm_locked = False
        self._waiting_for_confirm = False
        self._locked_eef_pose = None
        self._locked_target_pos = None

    def _reset_finder(self):
        """重置查找器状态"""
        print("\n[INFO] 重置场景...\n")
        self._arm_locked = False
        self._waiting_for_confirm = False
        self._locked_eef_pose = None
        self._locked_target_pos = None
        self._device.reset()
        self.reset()

    def step(self, actions):
        self.sim_step()
        return super().step(actions)
        
    def sim_step(self):
        # 手势控制：四指握拳开启控制模式
        if not self._device.bControl and self._device._hand_right_pos_norm[1]<0.1 and self._device._hand_right_pos_norm[2]<0.1 and self._device._hand_right_pos_norm[3]<0.1 and self._device._hand_right_pos_norm[4]<0.1:
            self._device.bControl = True
        
        # 设备控制
        if self._device.bControl:
            # 如果手臂未锁定，继续遥操控制
            if not self._arm_locked:
                delta_right_pos = self._device._delta_tracker_right_pos
                target_right_pos = self._arm2_eef_pose_init[0, 0:3] + delta_right_pos
                
                self._arm2_eef_pose_desired[0, 0:3] = target_right_pos
                self._arm2_eef_pose_desired[0, 3:] = self._device._wrist_right_quat
                
                self.scene.robots["robot"].set_ik_command({
                    "arm2": self._arm2_eef_pose_desired,
                })
                self._robot.ik_step()
            
            # 简化的手指控制：根据平均闭合程度决定张开或合拢
            hand_norm = self._device._hand_right_pos_norm
            # 计算四指平均值（食指、中指、无名指、小指）
            avg_finger_close = (hand_norm[1] + hand_norm[2] + hand_norm[3] + hand_norm[4]) / 4.0
            
            if avg_finger_close < self.cfg.hand_close_threshold:
                # 手指闭合 → 抓取姿态
                hand_pos_target = self._joint_limit_lower[:, self._hand2_joint_index].clone()
                hand_pos_target[:, 0] = self._joint_limit_upper[:, self._hand2_joint_index[0]]  # 拇指旋转取最大值
            else:
                # 手指张开 → 打开姿态
                hand_pos_target = self._joint_limit_upper[:, self._hand2_joint_index].clone()
            
            self._robot.set_joint_position_target(hand_pos_target, self._hand2_joint_index)
        
        # 手势复位：中指和无名指弯曲，食指和小拇指伸直
        if not self._device.bReset:
            hand_norm = self._device._hand_right_pos_norm
            if hand_norm[1] > 0.7 and hand_norm[2] < 0.1 and hand_norm[3] < 0.1 and hand_norm[4] > 0.7:
                self._device.bReset = True
                self._reset_finder()
        
        # 物理仿真步进
        super().sim_step()
        self._sim_step_counter += 1

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, extras = super().reset()
        
        # 等待场景稳定
        for i in range(50):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        
        # 初始化末端执行器初始位姿
        self._arm2_eef_pose_init = self._robot.data.body_link_state_w[:, self._arm2_eef_link_index, :7].clone()
        self._arm2_eef_pose_init[:, :3] -= self._robot.data.root_state_w[:, :3]
        
        self._print_instructions()
        
        return obs, extras

    def _reset_idx(self, env_ids: torch.Tensor | None, success_ids: Sequence[int] | None = None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)

