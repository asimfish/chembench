# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-19
# Vesion: 1.0


""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates random demo")
parser.add_argument("--robot_name",type=str,default=None,help="The name of the robot to analyze.")
parser.add_argument("--log_path", type=str, default=None, help="Path to the log file to store the results in.")


""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)

""" Common Modules  """ 
import numpy
import torch
import math
import os
import matplotlib.pyplot as plt
import time

""" IsaacLab Modules  """ 
import isaaclab.sim as sim_utils
from isaaclab.sim import (
    SimulationContext,SimulationCfg,PhysxCfg,RenderCfg,
    RigidBodyPropertiesCfg,RigidBodyMaterialCfg)
from isaaclab.envs.common import ViewerCfg
from isaaclab.assets import AssetBaseCfg,RigidObjectCfg,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCameraCfg

""" Psi Lab Modules  """ 
from psilab import PSILAB_USD_ASSET_DIR
from psilab.scene.sence import Scene,SceneCfg
from psilab.assets.light import DomeLightCfg
from psilab.assets.robot import RobotBaseCfg
from robot_settings import PSILAB_PATH,ROBOT_CONFIG

def create_robot_cfg(robot_name:str) -> RobotBaseCfg:

    robot_cfg:RobotBaseCfg = RobotBaseCfg(
        prim_path = ROBOT_CONFIG[robot_name]["prim_path"],
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_CONFIG[robot_name]["spawn"]["usd_path"],
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=255,
            ),
        ),
        init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0,0.0,0.0,0.0),
            joint_pos=ROBOT_CONFIG[robot_name]["init_state"]["joint_pos"],
            
        ),
        actuators={}
        
    )
    
    # replace actuator
    actuators = ROBOT_CONFIG[robot_name]["actuators"]
    for actuator_name,actuator_cfg in actuators.items():
        robot_cfg.actuators[actuator_name] = ImplicitActuatorCfg(
            joint_names_expr=actuator_cfg["joint_names_expr"],
            stiffness=actuator_cfg["stiffness"],
            damping=actuator_cfg["damping"],
            # effort_limit_sim=actuator_cfg["effort_limit_sim"], 
        )

    return robot_cfg

# viewer config
VIEWER_CFG = ViewerCfg(
    eye=(3.75,0.0,1.2),
    lookat=(-15.0,0.0,0.3)
)

# simulation  config
SIM_CFG : SimulationCfg = SimulationCfg(
    dt = 1 / 120, 
    render_interval=1,
    enable_scene_query_support=True,
    physx = PhysxCfg(
        solver_type = 1, # 0: pgs, 1: tgs
        max_position_iteration_count = 64,
        max_velocity_iteration_count = 0,
        bounce_threshold_velocity = 0.002,
        gpu_max_rigid_patch_count = 4096 * 4096,
        gpu_collision_stack_size=2**30,
        gpu_found_lost_pairs_capacity = 137401003
    ),
    render=RenderCfg(),
)

# scene config
SCENE_CFG = SceneCfg(
        
        num_envs = 1, 
        env_spacing=2.5, 
        replicate_physics=True,

        # global light
        global_light_cfg = DomeLightCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(
                intensity=100.0, 
                color=(0.75, 0.75, 0.75)
            )
        ),

        # static object
        static_objects_cfg = {
            "ground" : AssetBaseCfg(
                prim_path="/World/envs/env_[0-9]+/Ground", 
                spawn=sim_utils.UsdFileCfg(
                    usd_path=PSILAB_USD_ASSET_DIR + "/envs/grid/default_environment.usd"
                ),
                init_state = RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), 
                    rot= (1.0, 0.0, 0.0, 0.0)
                )
            ),
        },
        
    )

# change robot config if specified robot name
if args_cli.robot_name:
    SCENE_CFG.robots_cfg = {"robot": create_robot_cfg(args_cli.robot_name)}
else:
    print(f"No robot name specified. ")
    exit(1)

# create a simulation context to control the simulator
if SimulationContext.instance() is None:
    sim: SimulationContext = SimulationContext(SIM_CFG)
else:
    raise RuntimeError("Simulation context already exists. Cannot create a new one.")

# create scene
scene = Scene(SCENE_CFG)
#
sim.reset()
sim.set_camera_view(eye=VIEWER_CFG.eye, target=VIEWER_CFG.lookat)
# 
robot = scene.robots["robot"]

#
sim_time = robot.num_joints

#
joint_num_total = robot.num_joints

# get all joint limits
joint_pos_lower_limit = robot.data.joint_limits[0,:,0]
joint_pos_upper_limit = robot.data.joint_limits[0,:,1]
joint_vel_limit = robot.data.joint_velocity_limits[0,:]

# Sine wave trajectory parameters
period = ROBOT_CONFIG[args_cli.robot_name]["period"]
freq = 1.0 / period
amplitude_pos = 0.5 *(joint_pos_upper_limit - joint_pos_lower_limit)
amplitude_vel = joint_vel_limit / (2.0 * math.pi * freq)
amplitude = ROBOT_CONFIG[args_cli.robot_name]["amplitude_scale"] *  torch.min(amplitude_pos, amplitude_vel)

# get sim time
sim_time = torch.arange(0, ROBOT_CONFIG[args_cli.robot_name]["sim_time"], SIM_CFG.dt, device=scene.device)

# get joint initial position and velocity
joint_pos_init = 0.5 *(joint_pos_lower_limit + joint_pos_upper_limit)
joint_vel_init = 2.0 * math.pi * freq * amplitude

# get joint trajectory for all joints
joint_pos_target = joint_pos_init.unsqueeze(0).repeat(sim_time.shape[0],1) \
    + amplitude.unsqueeze(0).repeat(sim_time.shape[0],1) * \
        torch.sin(2.0 * math.pi * freq * sim_time.unsqueeze(1).repeat(1,joint_num_total))

# joint velocity target
joint_vel_target = amplitude.unsqueeze(0).repeat(sim_time.shape[0],1) * \
    2.0 * math.pi * freq * torch.cos(2.0 * math.pi * freq * sim_time.unsqueeze(1).repeat(1,joint_num_total))

# get active joint names and index
active_joint_names = robot.joint_names
# remove excluded joints from joint names if exists
if "excluded_joints" in ROBOT_CONFIG[args_cli.robot_name].keys():
    excluded_joint_names = ROBOT_CONFIG[args_cli.robot_name]["excluded_joints"]
    for name in excluded_joint_names:
        active_joint_names.remove(name)
active_joint_index = robot.find_joints(active_joint_names,preserve_order=True)[0]

# remove mimic joints from active joint
active_joint_names_without_mimic = active_joint_names.copy()
if "mimic_joints" in ROBOT_CONFIG[args_cli.robot_name].keys():
    mimic_joint_names = ROBOT_CONFIG[args_cli.robot_name]["mimic_joints"]
    for name in mimic_joint_names:
        active_joint_names_without_mimic.remove(name)
active_joint_names_without_mimic_index = robot.find_joints(active_joint_names_without_mimic,preserve_order=True)[0]
#
active_joint_num = len(active_joint_names)

# initiallize active joint
# robot.data.default_joint_pos[0,active_joint_index] = joint_pos_init[active_joint_index]
# robot.data.default_joint_vel[0,active_joint_index] = joint_vel_init[active_joint_index]
robot.write_joint_state_to_sim(
    position=joint_pos_init[active_joint_index].unsqueeze(0),
    velocity=joint_vel_init[active_joint_index].unsqueeze(0),
    joint_ids=active_joint_index
)
# scene.write_data_to_sim()
# scene.reset()

# robot.write_joint_state_to_sim(
#     position=joint_pos_init[active_joint_index].unsqueeze(0),
#     velocity=joint_vel_init[active_joint_index].unsqueeze(0),
#     joint_ids=active_joint_index
# )
# # robot.write_joint_position_to_sim(joint_pos_init[active_joint_index],joint_ids=active_joint_index)
# # robot.write_joint_velocity_to_sim(joint_vel_init[active_joint_index],joint_ids=active_joint_index)
robot.set_joint_position_target(joint_pos_init[active_joint_index],joint_ids=active_joint_index)
robot.set_joint_velocity_target(joint_vel_init[active_joint_index],joint_ids=active_joint_index)
robot.write_data_to_sim()
# 

# update 2 step, otherwise will get wrong state for velocity
scene.update(5 * scene.physics_dt)

# initiallize other data buffer
joint_pos = torch.zeros_like(joint_pos_target)
joint_vel = torch.zeros_like(joint_pos_target)

# test loop
for i in range(sim_time.shape[0]):
    # get state of active joints
    joint_pos[i,active_joint_index] = robot.data.joint_pos[0,active_joint_index].clone()
    joint_vel[i,active_joint_index] = robot.data.joint_vel[0,active_joint_index].clone()
    # robot.data.joint_vel_target[0,active_joint_index]
    # only set target for active joints
    robot.set_joint_position_target(joint_pos_target[i,active_joint_index],joint_ids=active_joint_index)
    robot.set_joint_velocity_target(joint_vel_target[i,active_joint_index],joint_ids=active_joint_index)
    robot.write_data_to_sim()
    #
    sim.step()
    scene.update(scene.physics_dt)


# get log dir
if args_cli.log_path:
    log_dir = os.path.dirname(args_cli.log_path)
else:
    print(f"No Log Path specified. Cannot save the results. ")
    exit(1)
    
# create dir
log_dir = os.path.join(log_dir, args_cli.robot_name)
os.makedirs(log_dir, exist_ok=True)


# plot result
sim_time = sim_time.cpu().tolist()
sub_plot_num = math.ceil(math.sqrt(active_joint_num))
# position
plt.figure(dpi=300,figsize=(24,14))
for i in range(active_joint_num):
    joint_index_temp = active_joint_index[i]
    joint_pos_target_temp = joint_pos_target[:,joint_index_temp].cpu().tolist()
    joint_pos_temp = joint_pos[:,joint_index_temp].cpu().tolist()
    plt.subplot(sub_plot_num, sub_plot_num, i+1)
    plt.plot(sim_time,joint_pos_target_temp,'r-')
    plt.plot(sim_time,joint_pos_temp,'b-')
    plt.title(f"{active_joint_names[i]}")
    plt.legend(['Target','Actual'],loc='upper right')
plt.suptitle("Joint Position")
plt.subplots_adjust(hspace=0.4,wspace=0.4)
# pass
plt.savefig(f'{log_dir}/Joint_Position.jpg')
plt.clf()

# velocity
plt.figure(dpi=300,figsize=(24,14))
for i in range(active_joint_num):
    joint_index_temp = active_joint_index[i]
    joint_vel_target_temp = joint_vel_target[:,joint_index_temp].cpu().tolist()
    joint_vel_temp = joint_vel[:,joint_index_temp].cpu().tolist()
    plt.subplot(sub_plot_num, sub_plot_num, i+1)
    plt.plot(sim_time,joint_vel_target_temp,'r-')
    plt.plot(sim_time,joint_vel_temp,'b-')
    plt.title(f"{active_joint_names[i]}")
    plt.legend(['Target','Actual'],loc='upper right')
plt.suptitle("Joint Velocity")
plt.subplots_adjust(hspace=0.4,wspace=0.4)
# pass
plt.savefig(f'{log_dir}/Joint_Velocity.jpg')
plt.clf()

# log result
pos_error = joint_pos[:,active_joint_index] - joint_pos_target[:,active_joint_index]
vel_error = joint_vel[:,active_joint_index] - joint_vel_target[:,active_joint_index]

pos_error_mean = torch.mean(torch.abs(pos_error),dim=0).cpu().tolist()
vel_error_mean = torch.mean(torch.abs(vel_error),dim=0).cpu().tolist()
pos_error_rmse = torch.sqrt(torch.mean(pos_error**2,dim=0)).cpu().tolist()
vel_error_rmse = torch.sqrt(torch.mean(vel_error**2,dim=0)).cpu().tolist()



# write result to log file
if args_cli.log_path is not None:
    with open(f'{log_dir}/Result.log', "w") as f:
        result = f"Robot : {args_cli.robot_name}\n"
        for actuator_name in robot.actuators.keys():
            # skip excluded actuators
            if "excluded_actuators" in ROBOT_CONFIG[args_cli.robot_name].keys():
                if actuator_name in ROBOT_CONFIG[args_cli.robot_name]["excluded_actuators"]:
                    continue
            #
            for joint_name in robot.actuators[actuator_name].joint_names:
                # 
                index = active_joint_names.index(joint_name)
                #
                result += f"    {joint_name}"
                result += f"   -pos_error_mean: {pos_error_mean[index]}"
                result += f"   -vel_error_mean: {vel_error_mean[index]}"
                result += f"   -pos_error_rmse: {pos_error_rmse[index]}"
                result += f"   -vel_error_rmse: {vel_error_rmse[index]}"
                result += "\n"
        f.write(result)
    f.close()