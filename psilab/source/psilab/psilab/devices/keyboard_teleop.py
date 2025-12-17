# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-01-22
# Vesion: 1.0

import carb
import omni
import torch



from .teleop_base import TeleOpBase


# Todo: 完善键盘操作逻辑和数据结构，增加配置文件以适配多种机型
class KeyboardTeleOp(TeleOpBase):
    """
    键盘遥操作控制机器人运动，
    包括手臂运动控制和末端执行机构控制， 将一个手臂和一个末端执行机构作为为1组考虑
    多组手臂+执行机构时考虑用Tab按键进行切换
    """
    def __init__(self, robot_cfg: RobotCfg, device:str):

        super().__init__()

        # get variables from robot cfg
        self.ik_name: list[str] = list(robot_cfg.ik_cfg.keys())
        self.eef_name: list[str] = robot_cfg.eff_name
        # index of arm and eef controlled
        self.index = 0
        # As robot is comprised of arm(ik) and effector(joint),
        # output of teleoperation device is dict
        # key : ik/effort name
        # value: command tensor
        # tips:
        # command for ik is delta position(xyz, meter) and delta eular angle(roll,pitch,yaw, rad)
        # command for eef is joint position(rad)
        self.output = {}
        self.output_default = {}
        # ik output
        for ik_name in self.ik_name:
            self.output[ik_name] = torch.zeros(6,device=device)
            self.output_default[ik_name] = self.output[ik_name].clone()
        # eef output
        for eef_name in self.eef_name:
            self.output[eef_name] = torch.zeros(len(robot_cfg.eff_joint_name[eef_name]),device=device)
            self.output_default[eef_name] = self.output[eef_name].clone()

        # 注册键盘句柄
        self.register_keyboard_handler()
    
    def reset(self):
        self.bReset = False
        self.bRecording = False
        self.output = {key:self.output_default[key].clone() for key in list(self.output_default.keys())}


    def set_default_output(self, value:dict[str,torch.Tensor]):
        self.output_default = value
        

    def register_keyboard_handler(self):
        """
        Sets up the keyboard callback functionality with omniverse
        """
        appwindow = omni.appwindow.get_default_app_window() # type: ignore
        input_interface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, self.keyboard_event_handler)

    def keyboard_event_handler(self, event, *args, **kwargs):
        """
        控制按键如下：
        U/I按键：末端执行机构沿自身坐标系X轴正方向前进/后退
        J/K按键：末端执行机构沿自身坐标系Y轴正方向前进/后退
        N/M按键：末端执行机构沿自身坐标系Z轴正方向前进/后退
        7/8按键：末端执行机构沿自身坐标系X轴顺时针/逆时针旋转
        4/5按键：末端执行机构沿自身坐标系Y轴顺时针/逆时针旋转
        1/2按键：末端执行机构沿自身坐标系Z轴顺时针/逆时针旋转
        9/6按键：末端执行机构Open/Closed
        """
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            or event.type == carb.input.KeyboardEventType.KEY_REPEAT
        ):
            # Tab 按键切换 控制手臂
            if event.input == carb.input.KeyboardInput.TAB:
                self.index+=1
                if self.index >= len(self.ik_name):
                    self.index=0

            # Handle special cases
            if event.input == carb.input.KeyboardInput.Z:
                self.bReset = True

            # delta position on X Axis, -0.01 meter
            if event.input == carb.input.KeyboardInput.U:
                self.output[self.ik_name[self.index]][0] = 0.01
            # delta position on X Axis, 0.01 meter
            if event.input == carb.input.KeyboardInput.I:
                self.output[self.ik_name[self.index]][0] = -0.01
            # delta position on Y Axis, -0.01 meter
            if event.input == carb.input.KeyboardInput.J:
                self.output[self.ik_name[self.index]][1] = 0.01
            # delta position on Y Axis, 0.01 meter
            if event.input == carb.input.KeyboardInput.K:
                self.output[self.ik_name[self.index]][1] = -0.01
            # delta position on Z Axis, -0.01 meter
            if event.input == carb.input.KeyboardInput.N:
                self.output[self.ik_name[self.index]][2] = 0.01
            # delta position on Z Axis, 0.01 meter
            if event.input == carb.input.KeyboardInput.M:
                self.output[self.ik_name[self.index]][2] = -0.01

            # delta angle in body frame, roll, -0.1 rad 
            if event.input == carb.input.KeyboardInput.NUMPAD_7:
                self.output[self.ik_name[self.index]][3] = 0.1
            
            # delta angle in body frame, roll, 0.1 rad 
            if event.input == carb.input.KeyboardInput.NUMPAD_8:
                self.output[self.ik_name[self.index]][3] = -0.1
            
            # delta angle in body frame, pitch, -0.1 rad 
            if event.input == carb.input.KeyboardInput.NUMPAD_4:
                self.output[self.ik_name[self.index]][4] = 0.1

            # delta angle in body frame, pitch, 0.1 rad 
            if event.input == carb.input.KeyboardInput.NUMPAD_5:
                self.output[self.ik_name[self.index]][4] = -0.1

            # delta angle in body frame, yaw, -0.1 rad 
            if event.input == carb.input.KeyboardInput.NUMPAD_1:
                self.output[self.ik_name[self.index]][5] = 0.1
                
            # delta angle in body frame, yaw, 0.1 rad 
            if event.input == carb.input.KeyboardInput.NUMPAD_2:
                self.output[self.ik_name[self.index]][5] = -0.1

            # hand grasp, open
            if event.input == carb.input.KeyboardInput.NUMPAD_9:
                for i in range(1,11):
                    self.output[self.eef_name[self.index]][i] = 0.1

            if event.input == carb.input.KeyboardInput.NUMPAD_6:
                for i in range(1,11):
                    self.output[self.eef_name[self.index]][i] = 3.1
                # self.output[self.hand_name[self.arm_index]][1,:] = 2.0

        
        # If we release a key, clear the active action and keypress
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.output[self.ik_name[self.index]]=self.output_default[self.ik_name[self.index]].clone()
            
        
        # Callback always needs to return True
        return True


        
