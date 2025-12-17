# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Arguments parse """
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates lego grasp task demo from gym.")

parser.add_argument("--hdf5_file_folder", type=str, default="", help="Name of the task.")
parser.add_argument("--enable_state", action="store_true", default=False, help="Whether to all state of scene or not.")


""" Must First Start APP, or import omni.isaac.lab.sim as sim_utils will be error."""
from isaaclab.app import AppLauncher

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# store args befor create app as it will pop some arg from args_cli
enable_cameras = args_cli.enable_cameras

# launch omniverse app
app_launcher = AppLauncher(args_cli)


""" Common Modules  """ 
import os
import sys
import math
import torch
import time
import json
import h5py
import re
import numpy
import matplotlib.pyplot as plt
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from datetime import datetime

""" Isaac Lab Modules  """ 
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg,load_cfg_from_registry
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.envs.common import ViewerCfg

import isaaclab_tasks  # noqa: F401
from psilab.envs.tp_env import TPEnv

""" Psi RL Modules  """ 
# import psilab.tasks # noqa: F401
import psilab_tasks
from psilab.scene import Scene,SceneCfg

from psilab.utils.gym_utils import make

from psilab.utils.config_utils import scene_cfg
from psilab_tasks.utils import parse_scene_cfg
# from psilab_tasks.replay.replay_env import ReplayEnv

# get hdf5 files in folder
file_list = os.listdir(args_cli.hdf5_file_folder)
hdf5_files = []
for file in file_list:
    if file.split('.')[-1] == 'hdf5':
        hdf5_files.append(os.path.join(args_cli.hdf5_file_folder,file))

# get task and scene
if len(hdf5_files)>0:
    h5_file = h5py.File(hdf5_files[0], 'r') # type: ignore
    task_id = h5_file["/task"][0].decode('utf-8')# type: ignore
        # print(aa)
    scene_id = h5_file["/scene"][0].decode('utf-8')# type: ignore


# parse argumanets for isaac lab rl env config
env_cfg= parse_env_cfg(
    task_id, 
    device=args_cli.device,
    num_envs=1,
    # use_fabric=not args_cli.disable_fabric
    )

# parse argumanets for psi lab scene config
env_cfg.scene = parse_scene_cfg(
    task_id, 
    False,
    scene_id,
    None,
    1,
)

# clear camera configs in scene while "enable_cameras" flag is True
if enable_cameras is False:
    env_cfg.scene.cameras_cfg ={}
    env_cfg.scene.tiled_cameras_cfg = {}
    for robot_cfg in env_cfg.scene.robots_cfg.values():
        robot_cfg.cameras = {} # type: ignore
        robot_cfg.tiled_cameras = {} # type: ignore

# 
if "IL" or "MP" in task_id:
    env_cfg.max_step =0 # type: ignore
    env_cfg.checkpoint = "/home/admin01/Work/02-PsiLab/psi-lab-v2/logs/diffusion_policy/Grasp_Bottle_v1/epoch=0450-train_loss=0.020.ckpt" # type: ignore
    env_cfg.max_episode = 0 # type: ignore
    
# create env
# env = gym.make(args_cli.task, cfg=env_cfg)
# create env
env = make(task_id, cfg=env_cfg)
# reset env before loop
env.reset()
sim = env.sim
scene : Scene = env.scene

sim.set_camera_view(eye=(2.75,0.0,1.2), target=(-15.0,0.0,0.3))


#    
for hdf5_file in hdf5_files:
    print(hdf5_file)
    # read hdf5 file
    file = h5py.File(hdf5_file, 'r')
    # detect is multi env
    is_multi_env = re.match(r"env_[0-9]+", "".join(list(file.keys()))) is not None
    # get all env data
    env_data_list : list[h5py.Group] = []
    if is_multi_env:
        for key_name in file.keys():
            if re.match(r"env_[0-9]+", key_name) and isinstance(file[key_name],h5py.Group):
                env_data_list.append(file[key_name]) # type: ignore
    else:
        env_data_list.append(file)
    # replay all data
    for env_data in env_data_list:
        # replay all state
        if args_cli.enable_state:
            for step in range(env_data["sim_time"].len()): # type: ignore
                # robot
                for robot_name in list(env_data["robots"].keys()): # type: ignore
                    # joint
                    for joint_group_name in env_data["robots/"+robot_name+"/extra/joint_name"]:
                        if joint_group_name=="all":
                            continue
                        # 
                        joint_pos = torch.tensor(env_data["robots/"+robot_name+"/"+joint_group_name + "_pos"][:][step],device="cuda:0").unsqueeze(0)# type: ignore
                        joint_vel = torch.tensor(env_data["robots/"+robot_name+"/"+joint_group_name + "_vel"][:][step],device="cuda:0").unsqueeze(0)# type: ignore
                        joint_indexs = env_data["robots/"+robot_name+"/extra/joint_index/"+joint_group_name][:].tolist() # type: ignore
                        scene.robots[robot_name].write_joint_state_to_sim(joint_pos,joint_vel,joint_indexs)
                    # get camera keys 
                    key_list = list(env_data["robots/"+robot_name].keys()) # type: ignore
                    image_key_list = []
                    for key in key_list:
                        if key.split(".")[-1] in ["rgb"]:
                            image_key_list.append(key)
                    # get image to show
                    if len(image_key_list)>0:        
                        num = len(image_key_list)
                        shape_base = env_data["robots/"+robot_name+"/"+image_key_list[0]][:][step].shape # type: ignore
                        # shape = shape_base
                        shape = (shape_base[0], shape_base[1]* num,shape_base[2]) 
                        image = numpy.empty(shape, dtype = numpy.uint8)
                        # splice image
                        for i in range(num):
                            camera_image = env_data["robots/"+robot_name+"/"+image_key_list[i]][:][step] # type: ignore
                            image[:,i*shape_base[1]:(i+1)*shape_base[1],:] = camera_image

                
                        
                        plt.clf()
                        plt.imshow(image)
                        plt.show(block=False)
                        plt.pause(0.0001)  # 暂停一段时间，不然画的太快会卡住显示不出来
                        plt.ioff() 
                # static object
                # rigid object
                for object_name in list(env_data["rigid_objects"].keys()): # type: ignore
                    state = torch.cat((
                        torch.tensor(env_data["rigid_objects/"+object_name][:][step],device="cuda:0"),# type: ignore
                        torch.zeros(6,device="cuda:0")),0).unsqueeze(0)
                    # state[0,:3] -= env_origin[0,:]
                    env.scene.rigid_objects[object_name].write_root_state_to_sim(state)  

                #env step
                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(dt=env.physics_dt)
                #
                time.sleep(0.1)
        # only replay joint target
        else:
            dt = env_data["sim_time"][1] - env_data["sim_time"][0] # type: ignore
            time_last_control=0 # type: ignore
            step = 0
            # 
            # reset rigid object pose
            for object_name in list(env_data["rigid_objects"].keys()): # type: ignore
                state = torch.cat((
                    torch.tensor(env_data["rigid_objects/"+object_name][:][step],device="cuda:0"),# type: ignore
                    torch.zeros(6,device="cuda:0")),0).unsqueeze(0)
                # state[0,:3] -= env_origin[0,:]
                env.scene.rigid_objects[object_name].write_root_state_to_sim(state) 
                sim.step(render=True)
                scene.update(dt=env.physics_dt) 
            # loop
            while True:
                # update step
                time_current = time.time()
                if time_current - time_last_control >= 0.1:
                    step+=1
                    if step >= (env_data["sim_time"].len()-1): # type: ignore
                        
                        break

                    # robot update
                    for robot_name in list(env_data["robots"].keys()): # type: ignore
                        # joint
                        for joint_group_name in env_data["robots/"+robot_name+"/extra/joint_name"]:
                            if joint_group_name=="all":
                                continue
                            # 
                            joint_pos_target = torch.tensor(env_data["robots/"+robot_name+"/"+joint_group_name + "_pos_target"][:][step],device="cuda:0").unsqueeze(0)# type: ignore
                            joint_indexs = env_data["robots/"+robot_name+"/extra/joint_index/"+joint_group_name][:].tolist() # type: ignore
                            scene.robots[robot_name].data.joint_pos_target[0,joint_indexs] = joint_pos_target
                            scene.robots[robot_name].write_data_to_sim()
                    #
                    scene.write_data_to_sim()
                    # 
                    time_last_control = time.time()
                #
                sim.step(render=True)
                #
            # reset scene
            scene.reset()

        