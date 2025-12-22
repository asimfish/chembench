# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Common Modules  """ 
import torch

""" IsaacLab Modules  """ 
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor


def eval_success(target: RigidObject, contact_sensors: dict[str,ContactSensor], target_pos_init:torch.Tensor,lift_height_desired:float) -> torch.Tensor:
    """The evaluate of whether the grasp is successful. """

    # compute the height lifted of target
    height_lift = target.data.root_pos_w[:,2] - target_pos_init[:,2]
    # print(height_lift)
    # get force number between target and robot in contact sensor
    num_envs= target.data.default_root_state.shape[0]
    contact_force_num = torch.zeros(num_envs, dtype=torch.int8,device=target.device)
    for sensor_name,contact_sensor in contact_sensors.items():
        forces = torch.sum(contact_sensor.data.net_forces_w, dim=[1,2]) # type: ignore
        contact_force_num = torch.where(
            forces>0.0,
            contact_force_num+1,
            contact_force_num
        )
    
    # we thougt target and robot is contacting while force number is positive
    contacting = contact_force_num>0 # type: ignore

    # grasp is success while target is lifted to desired height and is contacting with robot
    bsuccessed = (abs(height_lift-lift_height_desired)<=0.05) & contacting
    #
    return bsuccessed

def eval_success_only_height(target: RigidObject,  target_pos_init:torch.Tensor, lift_height_desired:float) -> torch.Tensor:
    """The evaluate of whether the grasp is successful only according lift height. """

    # compute the height lifted of target
    height_lifted = target.data.root_pos_w[:,2] - target_pos_init[:,2]

    # grasp is success while target is lifted to desired height and is contacting with robot
    bsuccessed = (abs(height_lifted-lift_height_desired)<0.05)
    #
    return bsuccessed

def eval_fail(target: RigidObject, contact_sensors: dict[str,ContactSensor], has_contacted:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    """The evaluate of whether the grasp is failed. """


    # get velocity on Z-axis
    velocity_z = torch.round(target.data.root_state_w[:,9], decimals = 2)
    angle_velocity = torch.round(target.data.root_state_w[:,10:], decimals = 2)
    # we thougt target is falling down while velocity on Z-axis is greater than 0.2m/s or angle velocity is greater than 5 deg/s
    # bfalling = (velocity_z<=-0.1) | (angle_velocity[:,0]>=2)| (angle_velocity[:,1]>=2)| (angle_velocity[:,2]>=2)
    # bfalling = (velocity_z<=-0.1) | (angle_velocity[:,0]>=2)| (angle_velocity[:,1]>=2)| (angle_velocity[:,2]>=2)
    bfalling = ((velocity_z<=1) | (angle_velocity[:,0]>=10)| (angle_velocity[:,1]>=10)| (angle_velocity[:,2]>=10)) & 0
    # print(f'angle_velocity: {angle_velocity}')
    # get force number between target and robot in contact sensor
    contact_force_num = torch.zeros(has_contacted.shape, dtype=torch.int8,device=has_contacted.device)
    for sensor_name,contact_sensor in contact_sensors.items():
        forces = torch.sum(contact_sensor.data.net_forces_w, dim=[1,2]) # type: ignore
        contact_force_num = torch.where(
            forces>0.0,
            contact_force_num+1,
            contact_force_num
        )
    
    # we thougt target and robot is contacting while force number is positive
    contacting = contact_force_num>0 # type: ignore

    # grasp is failed while target is falling down from robot
    # method 1:
    # bfailed = bfalling & has_contacted & (~contacting)
    # method 2:
    bfailed = bfalling & (~contacting)

    # update contacted flags
    has_contacted = has_contacted | contacting # type: ignore
    
    return bfailed,has_contacted