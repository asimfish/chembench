# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
import torch

""" IsaacLab Modules  """ 
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor

""" PsiLab Modules  """ 
from psilab.assets.robot.robot_base import RobotBase

def eval_success(robot: RobotBase, grasp_target: RigidObject, place_target: RigidObject, relationship:str,contact_sensors: dict[str,ContactSensor]) -> bool:

    # 成功条件 = 成功条件1 and 成功条件2
    # 成功条件1: grasp target 与 place target 关系
    # 成功条件2: robot双手与target的接触力数量 == 0
    # 成功条件3: grasp target 速度为零

    b_relation = False
    # if relationship == "on":

    # elif relationship == "in":
    # 计算目标接触
    target_contact_force_num =0
    force_matrix_w = contact_sensors["grasp_target"].data.force_matrix_w[0,:,:] # type: ignore
    for index in range(force_matrix_w.size()[0]):
        if not force_matrix_w[0,index,:].equal(torch.tensor([0.0,0.0,0.0],device="cuda:0")):
            target_contact_force_num+=1

    # 计算物体速度
    vel_linear = torch.norm(grasp_target.data.root_vel_w[0,:3], p=2, dim=-1)
    vel_angular = torch.norm(grasp_target.data.root_vel_w[0,3:], p=2, dim=-1) 
    if vel_linear < 0.05:
        b_static = True
    else:
        b_static = False
    # 计算接触力数量
    contact_force_num =0
    contact_sensors_hand : dict[str,ContactSensor]= {}
    contact_sensors_hand["left_hand"] = contact_sensors["left_hand"]
    contact_sensors_hand["right_hand"] = contact_sensors["right_hand"]

    for sensor_name,contact_sensor in contact_sensors_hand.items():
        net_forces_w = contact_sensor.data.net_forces_w[0,:,:] # type: ignore
        for index in range(net_forces_w.size()[0]):
            if not net_forces_w[index].equal(torch.tensor([0.0,0.0,0.0],device="cuda:0")):
                contact_force_num+=1
        pass
    # print(height_cur - height_init)
    if target_contact_force_num>0 and b_static and contact_force_num==0:
        return True

    return False

def eval_fail(robot: RobotBase, grasp_target: RigidObject, place_target: RigidObject, relationship:str,contact_sensors: dict[str,ContactSensor],has_grasped:bool) -> tuple[bool,bool]:

    # 失败条件 = 失败条件1 or 失败条件2
    # 失败条件1：has_grasped and target静止 and target与robot没有接触点 and target 与 place target 没有接触点

    # if not has_grasped:
    #     return (False,False)
    
    # 计算物体速度
    vel_linear = torch.norm(grasp_target.data.root_vel_w[0,:3], p=2, dim=-1)
    vel_angular = torch.norm(grasp_target.data.root_vel_w[0,3:], p=2, dim=-1) 
    if vel_linear < 0.05:
        b_static = True
    else:
        b_static = False

   # 计算目标接触
    target_contact_force_num =0
    force_matrix_w = contact_sensors["grasp_target"].data.force_matrix_w[0,:,:] # type: ignore
    for index in range(force_matrix_w.size()[0]):
        if not force_matrix_w[0,index,:].equal(torch.tensor([0.0,0.0,0.0],device="cuda:0")):
            target_contact_force_num+=1

    # 计算接触力数量
    contact_force_num =0
    contact_sensors_hand : dict[str,ContactSensor]= {}
    contact_sensors_hand["left_hand"] = contact_sensors["left_hand"]
    contact_sensors_hand["right_hand"] = contact_sensors["right_hand"]

    for sensor_name,contact_sensor in contact_sensors_hand.items():
        net_forces_w = contact_sensor.data.net_forces_w[0,:,:] # type: ignore
        force_matrix_w = contact_sensor.data.force_matrix_w[0,:,0,:] # type: ignore

        for index in range(net_forces_w.size()[0]):
            # if not force_matrix_w[index,:].equal(torch.tensor([0.0,0.0,0.0],device=robot.cfg.device)):
            #     contact_force_num+=1
            if not net_forces_w[index].equal(torch.tensor([0.0,0.0,0.0],device=device="cuda:0")):
                contact_force_num+=1
        pass


    print(f"Has Grasped: {has_grasped}, Static: {b_static}, contact_num_1: {target_contact_force_num}, contact_num_1: {contact_force_num}")
    if has_grasped and b_static and target_contact_force_num==0 and contact_force_num==0:

        return (True,has_grasped)
    
    if contact_force_num==0:
        return  (False,has_grasped)
    else:
        return  (False,True)