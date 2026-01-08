# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

""" Python Modules  """ 
from collections.abc import Sequence

""" Common Modules  """ 
import time
from datetime import datetime
import h5py
import json
import torch
import os
import numpy
import copy
""" IsaacLab Modules  """ 
from isaaclab.utils.configclass import class_to_dict

""" Psilab Modules  """ 
from psilab.scene import Scene
from psilab.envs.rl_env_cfg import RLEnvCfg
from psilab.utils.hdf5_utils import write_data_to_hdf5,convert_tensor_to_cpu

def create_data_buffer(scene:Scene) -> dict :

    #
    data_buffer = {}  

    if scene.num_envs == 1:
        data_buffer = _create_data_buffer(scene)
    else:
        for i in range(0,scene.num_envs):
           data_buffer[f"env_{i}"] = _create_data_buffer(scene)

    return data_buffer

def parse_data(sim_time:float, data: dict, scene:Scene, env_obj=None, pointcloud_transform_fn=None):

    if scene.num_envs == 1:
        _parse_data(sim_time,data,scene,device=scene.device,env_id=0,env_obj=env_obj,pointcloud_transform_fn=pointcloud_transform_fn)
    else:
        for i in range(0,scene.num_envs):
            _parse_data(sim_time,data[f"env_{i}"],scene,i,env_obj=env_obj,pointcloud_transform_fn=pointcloud_transform_fn)

def save_data(data: dict, cfg:RLEnvCfg, scene:Scene, env_indexs:Sequence[int]|None=None, reset_env_indexs:Sequence[int]|None=None):
    """
    Save all data of robot and objects in scene as hdf5 file, and save scene config as json file
    """ 
    #
    if env_indexs is None:
        env_indexs = range(cfg.scene.num_envs) # type: ignore
    
    #
    if reset_env_indexs is None:
        reset_env_indexs = range(cfg.scene.num_envs) # type: ignore

    # create folder if not exist
    if not os.path.exists(cfg.output_folder): # type: ignore
        os.makedirs(cfg.output_folder) # type: ignore

    # copy data dict
    data_temp = {}
    if scene.num_envs == 1:
        # move
        data_temp = data
        # add task id and scene config 
        data_temp["task"] = [cfg.task_id]
        data_temp["scene"] = [cfg.scene_id]
    else:
        if cfg.async_reset:
            #
            for env_index in env_indexs: # type: ignore
                # move
                data_temp[f"env_{env_index}"] = data[f"env_{env_index}"]
                # add task id and scene config 
                data_temp[f"env_{env_index}"]["task"] = [cfg.task_id]
                data_temp[f"env_{env_index}"]["scene"] = [cfg.scene_id]
        else:
            # 
            for env_index in env_indexs: # type: ignore
                # move
                data_temp[f"env_{env_index}"] = data[f"env_{env_index}"]
            # add task id and scene config 
            data_temp["task"] = [cfg.task_id]
            data_temp["scene"] = [cfg.scene_id]
    
    # reset data dict
    if scene.num_envs == 1:
        # reset
        data = _create_data_buffer(scene)
    else:
        for env_index in reset_env_indexs: # type: ignore
            # reset
            data[f"env_{env_index}"] = _create_data_buffer(scene)

    # don't save as no env to save data
    if not env_indexs:
        return data

    if scene.num_envs > 1 and cfg.async_reset:
        # 
        # create hdf5 file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i in range(len(env_indexs)):
            # create hdf5 file
            filename = f"/{timestamp}_{i}_data.hdf5"
            h5_file = h5py.File(cfg.output_folder+filename, 'w') # type: ignore
            
            # gpu->cpu
            convert_tensor_to_cpu(data_temp[f"env_{env_indexs[i]}"])
            # writr dict to hdf5
            write_data_to_hdf5(data_temp[f"env_{env_indexs[i]}"],h5_file,"/")
            # 
            h5_file.close()
    else:
        # create hdf5 file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/{timestamp}_data.hdf5"
        h5_file = h5py.File(cfg.output_folder+filename, 'w') # type: ignore
        
        # gpu->cpu
        convert_tensor_to_cpu(data_temp)
        # writr dict to hdf5
        write_data_to_hdf5(data_temp,h5_file,"/")
        # 
        h5_file.close()

    # clear
    torch.cuda.empty_cache()
    
    return data

def _create_data_buffer(scene:Scene) -> dict :
    # 
    data = {}
    # add simulator time
    data["sim_time"] = []
    # add env orgin position
    data["env_orgin"] = []
    # add lights
    data["lights"] ={}
    for light_name,light in scene.lights.items():
        # light dict
        data["lights"][light_name] = {}
        # add color
        data["lights"][light_name]["color"]=[]
        # add intensity
        data["lights"][light_name]["intensity"]=[]
    # add robots
    data["robots"] = {}
    for robot_name,robot in scene.robots.items():
        # robot dict
        data["robots"][robot_name] = {}
        # add pose 
        data["robots"][robot_name]["pose"] = []
        # add actuators
        for actuator_name in robot.actuators.keys():
            data["robots"][robot_name][actuator_name+"_pos"] = []
            data["robots"][robot_name][actuator_name+"_pos_target"] = []
            data["robots"][robot_name][actuator_name+"_vel"] = []
            data["robots"][robot_name][actuator_name+"_vel_target"] = []
        # add eef state according to ik controllers
        for eef_name in robot.eef_links.keys():
            data["robots"][robot_name][eef_name+"_eef_pose"] = []
            data["robots"][robot_name][eef_name+"_vel"] = []
        # add cameras 
        for camera_name,camera in robot.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                data["robots"][robot_name][camera_name+ "." + data_type] = []
        # add tiled cameras 
        for camera_name,camera in robot.tiled_cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                data["robots"][robot_name][camera_name+ "." + data_type] = []
        # add contact sensors
        for contact_name in robot.cameras.keys():
            data["robots"][robot_name][contact_name] = []
        # extra info
        data["robots"][robot_name]["extra"] = {
            "joint_name" : {},
            "joint_index" : {},
            "cameras" : {},
            "tiled_cameras" : {}
        }
        # extra info: all joint
        data["robots"][robot_name]["extra"]["joint_name"]["all"] = robot.joint_names
        data["robots"][robot_name]["extra"]["joint_index"]["all"] = robot.find_joints(robot.joint_names)[0]
        for actuator_name,actuator in robot.actuators.items():
            data["robots"][robot_name]["extra"]["joint_name"][actuator_name] = actuator.joint_names
            data["robots"][robot_name]["extra"]["joint_index"][actuator_name] = actuator.joint_indices
        # extra info: cameras
        for camera_name,camera in robot.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                data["robots"][robot_name]["extra"]["cameras"][camera_name+ "." + data_type] = {}
        # extra info: tiled cameras
        for camera_name,camera in robot.tiled_cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                data["robots"][robot_name]["extra"]["tiled_cameras"][camera_name+ "." + data_type] = {}
    # add rigid object
    data["rigid_objects"] = {}
    for object_name in scene.rigid_objects.keys():
        data["rigid_objects"][object_name]=[]
    # add articulated objects
    data["articulated_objects"] = {}
    for object_name,object in scene.articulated_objects.items():
        # articulated object dict
        data["articulated_objects"][object_name] = {}
        # add pose 
        data["articulated_objects"][object_name]["pose"] = []
        # add actuators
        for actuator_name in object.actuators.keys():
            data["articulated_objects"][object_name][actuator_name+"_pos"] = []
    # add deformable object
    data["deformable_objects"] = {}
    for object_name in scene.deformable_objects.keys():
        data["deformable_objects"][object_name]=[]
    # add cameras
    data["cameras"] = {}
    for camera_name,camera in scene.cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            data["cameras"][camera_name+ "." + data_type] = []
    # add tiled cameras
    for camera_name,camera in scene.tiled_cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            data["cameras"][camera_name+ "." + data_type] = []
    #
    # extra info
    data["extra"] = {
        "cameras" : {},
        "tiled_cameras" : {},
    }   
    # extra info:camera
    for camera_name,camera in scene.cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            #
            data["extra"]["cameras"][camera_name+ "." + data_type]={}        
    # extra info:tiled camera
    for camera_name,camera in scene.tiled_cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            #
            data["extra"]["tiled_cameras"][camera_name+ "." + data_type]={}
    # 
    return data

def _parse_data(sim_time:float, data: dict, scene:Scene, env_id:int = 0, device:str = "cpu", env_obj=None, pointcloud_transform_fn=None):

    # time stamps
    data["sim_time"].append(sim_time)
    # 
    # data["env_origins"].append(sim_time)
    # lights: global light and local lights
    for light_name, light in scene.lights.items():
        # add color
        data["lights"][light_name]["color"].append(light.color)
        # add intensity
        data["lights"][light_name]["intensity"].append(light.intensity)

    # robots
    for robot_name,robot in scene.robots.items():
        # add pose
        pose = robot.data.root_state_w[env_id,:7].clone()
        pose[:3] -=  scene.env_origins[env_id,:]
        data["robots"][robot_name]["pose"].append(pose.to(device))
        # add actuators
        for actuator_name,actuator in robot.actuators.items():
            data["robots"][robot_name][actuator_name+"_pos"].append(robot.data.joint_pos[env_id,actuator.joint_indices].to(device))
            data["robots"][robot_name][actuator_name+"_pos_target"].append(robot.data.joint_pos_target[env_id,actuator.joint_indices].to(device))
            data["robots"][robot_name][actuator_name+"_vel"].append(robot.data.joint_vel[env_id,actuator.joint_indices].to(device))
            data["robots"][robot_name][actuator_name+"_vel_target"].append(robot.data.joint_vel_target[env_id,actuator.joint_indices].to(device))
        # add eef state according to ik controllers
        for eef_name,eef_index in robot.eef_links.items():
            # transform eef position from world coordinate to robot coordinate
            eef_state = robot.data.body_link_state_w[env_id,eef_index,:7].to(device).clone()
            eef_state[:3] -= robot.data.root_state_w[env_id,:3].to(device)
            data["robots"][robot_name][eef_name+"_eef_pose"].append(eef_state)
        # add cameras
        for camera_name,camera in robot.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                image = _get_image_safe(camera.data.output[data_type][env_id,:,:,:],device)
                data["robots"][robot_name][camera_name+ "." + data_type].append(image)
        # add tiled cameras
        for camera_name,camera in robot.tiled_cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                # normals image only data of 1th, 2th and 3th channels is useful
                # the data of 4th channel is meaningless data
                if data_type == "normals": 
                    image = _get_image_safe(camera.data.output[data_type][env_id,:,:,:3],device)
                else:
                    image = _get_image_safe(camera.data.output[data_type][env_id,:,:,:],device)
                data["robots"][robot_name][camera_name+ "." + data_type].append(image)
        # extra info: cameras
        for camera_name,camera in robot.cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                # semantic egmentation
                if data_type=="semantic_segmentation":       
                    #
                    for color_str, semantic_data in camera.data.info["semantic_segmentation"]["idToLabels"].items(): # type: ignore
                        for semantic_key,semantic_value in semantic_data.items():
                            if semantic_value in scene.rigid_objects.keys():
                                # convert color_str:(int, int, int, int) to [int,int,int,int]
                                color = [numpy.uint8(value_temp) for value_temp in color_str[2:-1].split(", ")]
                                # RGBA
                                if semantic_value not in data["robots"][robot_name]["extra"]["cameras"][camera_name+ "." + data_type].keys():
                                    data["robots"][robot_name]["extra"]["cameras"][camera_name+ "." + data_type][semantic_value]=color
        # extra info: tiled cameras  
        for camera_name,camera in robot.tiled_cameras.items():
            # multi-type
            for data_type in camera.cfg.data_types:
                # semantic egmentation
                if data_type=="semantic_segmentation":
                    #
                    for color_str, semantic_data in camera.data.info["semantic_segmentation"]["idToLabels"].items(): # type: ignore
                        for semantic_key,semantic_value in semantic_data.items():
                            if semantic_value in scene.rigid_objects.keys():
                                # convert color_str:(int, int, int, int) to [int,int,int,int]
                                color = [numpy.uint8(value_temp) for value_temp in color_str[2:-1].split(", ")]
                                # RGBA
                                if semantic_value not in data["robots"][robot_name]["extra"]["tiled_cameras"][camera_name+ "." + data_type].keys():
                                    data["robots"][robot_name]["extra"]["tiled_cameras"][camera_name+ "." + data_type][semantic_value]=color

        # # add contact sensors
        # for contact_name,contact in robot.cameras.items():
        #     self._data["robots"][robot_name][contact_name].append()
                    
    # add rigid object
    for object_name,object in scene.rigid_objects.items():
        pose = object.data.root_state_w[env_id,:7].clone()
        pose[:3] -=  scene.env_origins[env_id,:]
        data["rigid_objects"][object_name].append(pose.to(device))
   
    # add articulated object
    for object_name,object in scene.articulated_objects.items():
        # add pose
        pose = object.data.root_state_w[env_id,:7].clone()
        pose[:3] -=  scene.env_origins[env_id,:]
        data["articulated_objects"][object_name]["pose"].append(pose.to(device))
        # add actuators
        for actuator_name,actuator in object.actuators.items():
            data["articulated_objects"][object_name][actuator_name+"_pos"].append(object.data.joint_pos[env_id,actuator.joint_indices].to(device))

    # ========== 添加真值点云（从USD采样并变换）==========
    if pointcloud_transform_fn is not None:
        # 如果数据字典中还没有 ground_truth_pointcloud key，创建它
        if "ground_truth_pointcloud" not in data:
            data["ground_truth_pointcloud"] = []
        
        # 调用变换函数获取当前帧的真值点云
        try:
            transformed_pc = pointcloud_transform_fn(env_id)  # (N, 3) torch.Tensor
            if transformed_pc is not None:
                # 转换为numpy并保存
                pc_numpy = transformed_pc.cpu().numpy()
                data["ground_truth_pointcloud"].append(pc_numpy)
        except Exception as e:
            print(f"⚠️ 真值点云变换失败 (env_id={env_id}): {e}")

    # add deformable object
    # for object_name in self.scene.deformable_objects.items():
    #     self._data["deformable_objects"][object_name].append(object.data.root_link_state_w[env_id,:7])
    # add cameras
    for camera_name,camera in scene.cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            image = _get_image_safe(camera.data.output[data_type][env_id,:,:,:],device)
            data["cameras"][camera_name+ "." + data_type].append(image)
    # add tiled cameras
    for camera_name,camera in scene.tiled_cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            # normals image only data of 1th, 2th and 3th channels is useful
            # the data of 4th channel is meaningless data
            if data_type == "normals": 
                image = _get_image_safe(camera.data.output[data_type][env_id,:,:,:3],device)
            else:
                image = _get_image_safe(camera.data.output[data_type][env_id,:,:,:],device)
            data["cameras"][camera_name+ "." + data_type].append(image)
    #
    # extra info: cameras
    for camera_name,camera in scene.cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            # semantic egmentation
            if data_type=="semantic_segmentation":       
                #
                for color_str, semantic_data in camera.data.info["semantic_segmentation"]["idToLabels"].items(): # type: ignore
                    for semantic_key,semantic_value in semantic_data.items():
                        if semantic_value in scene.rigid_objects.keys():
                            # convert color_str:(int, int, int, int) to [int,int,int,int]
                            color = [numpy.uint8(value_temp) for value_temp in color_str[2:-1].split(", ")]
                            # RGBA
                            if semantic_value not in data["extra"]["cameras"][camera_name+ "." + data_type].keys():
                                data["extra"]["cameras"][camera_name+ "." + data_type][semantic_value]=color
    # extra info: tiled cameras  
    for camera_name,camera in scene.tiled_cameras.items():
        # multi-type
        for data_type in camera.cfg.data_types:
            # semantic egmentation
            if data_type=="semantic_segmentation":
                #
                for color_str, semantic_data in camera.data.info["semantic_segmentation"]["idToLabels"].items(): # type: ignore
                    for semantic_key,semantic_value in semantic_data.items():
                        if semantic_value in scene.rigid_objects.keys():
                            # convert color_str:(int, int, int, int) to [int,int,int,int]
                            color = [numpy.uint8(value_temp) for value_temp in color_str[2:-1].split(", ")]
                            # RGBA
                            if semantic_value not in data["extra"]["tiled_cameras"][camera_name+ "." + data_type].keys():
                                data["extra"]["tiled_cameras"][camera_name+ "." + data_type][semantic_value]=color

def _get_image_safe(image:torch.Tensor, device:str = "cpu")->torch.Tensor:
    image = image.to(device)
    if device != "cpu":
        image = image.clone()

    return image


