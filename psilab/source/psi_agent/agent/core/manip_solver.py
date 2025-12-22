# -*- coding: utf-8 -*-
import sys 
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
import re
import cv2
import json
import copy
import pickle
import logging
import requests
import numpy as np
from dataclasses import dataclass

from omniagent.base_utils.transforms import calculate_rotation_matrix, rotate_around_axis
from omniagent.base_utils.object import OmniObject, transform_coordinates_3d
from omniagent.base_utils.data_utils import pose_difference
from omniagent.base_utils.fix_rotation import rotate_180_along_axis

from scipy.spatial.transform import Rotation as R, Slerp

def interpolate_rotation_matrices(rot_matrix1, rot_matrix2, num_interpolations):
    # Convert the rotation matrices to rotation objects
    rot1 = R.from_matrix(rot_matrix1)
    rot2 = R.from_matrix(rot_matrix2)
    
    # Convert the rotation objects to quaternions
    quat1 = rot1.as_quat()
    quat2 = rot2.as_quat()
    
    # Define the times of the known rotations
    times = [0, 1]
    
    # Create the Slerp object
    slerp = Slerp(times, R.from_quat([quat1, quat2]))
    
    # Define the times of the interpolations
    interp_times = np.linspace(0, 1, num_interpolations)
    
    # Perform the interpolation
    interp_rots = slerp(interp_times)
    
    # Convert the interpolated rotations to matrices
    interp_matrices = interp_rots.as_matrix()
    
    return interp_matrices




def format_object(obj, distance, type='active'):
    if obj is None:
        return None
    xyz, direction = obj.xyz, obj.direction
    
    direction = direction / np.linalg.norm(direction) * distance
    type = type.lower()
    if type=='active':
        xyz_start = xyz
        xyz_end = xyz_start + direction
    elif type=='passive' or type=='plane':
        xyz_end = xyz
        xyz_start = xyz_end - direction


    

    part2obj = np.eye(4)
    part2obj[:3, 3] = xyz_start
    obj.obj2part = np.linalg.inv(part2obj)


    obj_info = {
        'pose': obj.obj_pose,
        'length': obj.obj_length,
        'xyz_start': xyz_start,
        'xyz_end': xyz_end,
        'obj2part': obj.obj2part
    }
    return obj_info

def obj2world(obj_info):
    obj_pose = obj_info['pose']
    obj_length = obj_info['length']
    obj2part = obj_info['obj2part']
    xyz_start = obj_info['xyz_start']
    xyz_end = obj_info['xyz_end']


    arrow_in_obj = np.array([xyz_start, xyz_end]).transpose(1,0)
    arrow_in_world = transform_coordinates_3d(arrow_in_obj, obj_pose).transpose(1,0)

    xyz_start_world, xyz_end_world = arrow_in_world
    direction_world = xyz_end_world - xyz_start_world
    direction_world = direction_world / np.linalg.norm(direction_world)

    obj_info_world = {
        'pose': obj_pose,
        'length': obj_length,
        'obj2part': obj2part,
        'xyz_start': xyz_start_world,
        'xyz_end': xyz_end_world,
        'direction': direction_world,

    }
    return obj_info_world



# maobo 
def get_aligned_pose(active_obj, passive_obj, distance=0.06, N=1):
    try:
        active_object = format_object(active_obj, type='active', distance=distance)
        passive_object = format_object(passive_obj, type='passive', distance=distance)
    except:
        import ipdb;ipdb.set_trace()
        print('error')



    active_obj_world = obj2world(active_object)
    current_obj_pose = active_obj_world['pose']
    if passive_object is None:
        return current_obj_pose[np.newaxis, ...]
    
    passive_obj_world = obj2world(passive_object)
    
    R = calculate_rotation_matrix(active_obj_world['direction'], passive_obj_world['direction'])
    T = passive_obj_world['xyz_start'] - R @ active_obj_world['xyz_start']
    transform_matrix = np.eye(4)
    transform_matrix[:3,:3] = R
    transform_matrix[:3,3] = T
    target_obj_pose = transform_matrix @ current_obj_pose

    poses = []
    for angle in [i * 360 / N for i in range(N)]:
        pose_rotated = rotate_around_axis(
            target_obj_pose, 
            passive_obj_world['xyz_start'], 
            passive_obj_world['direction'], 
            angle)
        poses.append(pose_rotated)
    return np.stack(poses)



# dataclass stage
@dataclass
class ActionStage:
    """Class for keeping track of an item in inventory."""
    active_obj_ID: str
    passive_obj_ID: str
    pose_to_active_obj: np.ndarray
    pose_to_passive_obj: np.ndarray
    transform_world: np.ndarray
    gripper_action: str
    motion_type: str
    
    def __init__(self, 
                 active_obj_ID, 
                 passive_obj_ID, 
                 pose_to_active_obj=np.eye(4),
                 pose_to_passive_obj=np.eye(4), 
                 transform_world=np.eye(4), 
                 motion_type='Simple',
                 gripper_action=None, 
                 ):
        self.active_obj_ID = active_obj_ID
        self.passive_obj_ID = passive_obj_ID
        self.pose_to_active_obj = pose_to_active_obj
        self.pose_to_passive_obj = pose_to_passive_obj
        self.transform_world = transform_world
        self.motion_type = motion_type
        self.gripper_action = gripper_action
        


def grasp(grasp_pose, active_obj_ID, passive_obj_ID, *args, **kwargs):
    transform_world = np.eye(4)


    stages = []
    # moveTo pregrasp pose
    pre_pose = np.array([
        [1., 0, 0, 0],
        [0, 1., 0, 0], 
        [0, 0, 1., -0.08],
        [0, 0, 0, 1]])
    pre_grasp_pose = grasp_pose @ pre_pose   
    gripper_action = None
    motion_type = 'AvoidObs'
    stages.append( (active_obj_ID, passive_obj_ID, pre_grasp_pose, gripper_action, transform_world, motion_type))
    # moveTo grasp pose
    gripper_action = None
    motion_type = 'Simple'
    stages.append((active_obj_ID, passive_obj_ID, grasp_pose, gripper_action, transform_world, motion_type))
    # grasp
    gripper_action = 'close'
    motion_type = 'Simple'
    stages.append((active_obj_ID, passive_obj_ID, grasp_pose, gripper_action, transform_world, motion_type))
    # pick-up
    gripper_action = None
    motion_type = 'Simple'
    transform_up = np.eye(4)
    transform_up[2,3] = 0.18   
    stages.append((active_obj_ID, passive_obj_ID, grasp_pose, gripper_action, transform_up, motion_type))
    
    return stages


def place(target_pose, active_obj_ID, passive_obj_ID, *args, **kwargs):
    gripper_action = None

    stages = []

    # moveTo pre-place position
    target_pose_canonical = target_pose
    motion_type = 'Simple'
    transform_up = np.eye(4)
    transform_up[2,3] = 0.08    
    stages.append((active_obj_ID, passive_obj_ID, target_pose_canonical, gripper_action, transform_up, motion_type))
    
    # place
    stages.append((active_obj_ID, passive_obj_ID, target_pose_canonical, gripper_action, np.eye(4), 'Simple'))
    
    # open_gripper
    stages.append((active_obj_ID, passive_obj_ID, None, 'open', np.eye(4), 'Simple'))
    
    # reverse to pre-place position to avvoid collision when reseting
    # transform_up_new = np.eye(4)
    # transform_up_new[2,3] = 0.1
    # stages.append((active_obj_ID, passive_obj_ID, target_pose_canonical, None, transform_up_new, motion_type))
    
    return stages


def pour(target_pose, current_pose, obj2part, active_obj_ID, passive_obj_ID, **kwargs):
    target_part_pose = target_pose @ np.linalg.inv(obj2part)
    current_part_pose = current_pose @ np.linalg.inv(obj2part)

    gripper_action = None
    transform_up = np.eye(4)
    transform_up[2,3] = 0.05    # 8cm above the target pose


    stages = []

    # moveTo pre-pour position
    pre_pour_part_pose = np.copy(target_part_pose)
    pre_pour_part_pose[:3, :3] = current_part_pose[:3, :3]
    pre_pour_pose = pre_pour_part_pose @ obj2part
    motion_type = 'AvoidObs'
    stages.append((active_obj_ID, passive_obj_ID, pre_pour_pose, gripper_action, transform_up, motion_type))


    # moveTo pre-pour position
    # motion_type = 'Simple'
    # transform_up = transform_up.copy()
    # transform_up[2,3] = 0.01
    # stages.append((active_obj_ID, passive_obj_ID, pre_pour_pose, gripper_action, transform_up, motion_type))
    
    # # Pouring
    # motion_type = 'Simple'
    # rotations = interpolate_rotation_matrices(current_part_pose[:3,:3], target_part_pose[:3,:3], 4)
    # for i, rotation in enumerate(rotations[1:3]):
    #     target_part_pose_step = np.copy(target_part_pose)
    #     target_part_pose_step[:3,:3] = rotation
    #     target_pose_step = target_part_pose_step @ obj2part
    #     stages.append((active_obj_ID, passive_obj_ID, target_pose_step, gripper_action, transform_up, motion_type))
    # Pouring
    motion_type = 'Trajectory'
    rotations = interpolate_rotation_matrices(current_part_pose[:3,:3], target_part_pose[:3,:3], 600)[:400]
    target_part_pose_list = np.tile(target_part_pose, (len(rotations), 1, 1))
    target_part_pose_list[:, :3, :3] = rotations
    target_pose_list = target_part_pose_list @ obj2part[np.newaxis, ...]
    stages.append((active_obj_ID, passive_obj_ID, target_pose_list, gripper_action, transform_up, motion_type))
        
    # # reverse
    # motion_type = 'AvoidObs'
    # stages.append((active_obj_ID, passive_obj_ID, current_pose, gripper_action, np.eye(4), motion_type))

    # # open_gripper
    # gripper_action = 'open'
    # stages.append((active_obj_ID, passive_obj_ID, None, gripper_action, np.eye(4), motion_type))
    
    return stages



def rotate(target_pose, active_obj_ID, passive_obj_ID, vector_direction, *args, **kwargs):
    pass


def pull(gripper_pose, active_obj_ID, passive_obj_ID, vector_direction, *args, **kwargs):
    stages = []

    norm_direction = vector_direction / np.linalg.norm(vector_direction)
    displacement = norm_direction * 0.15
    
    # pull
    target_pull_pose = gripper_pose.copy()
    target_pull_pose[:3, 3] += displacement
    motion_type = 'Simple'      # TODO Pose指定方向后，要用力控模式
    gripper_action = None
    stages.append((active_obj_ID, passive_obj_ID, target_pull_pose, gripper_action, np.eye(4), motion_type))
    
    # open_gripper
    motion_type = 'Simple'
    gripper_action = 'open'
    stages.append((active_obj_ID, passive_obj_ID, None, gripper_action, np.eye(4), motion_type))
    return stages




def push(target_pose, active_obj_ID, passive_obj_ID, vector_direction, *args, **kwargs):
    stages = []

    # moveTo pre-push position
    pre_push_transform = np.array([
        [1., 0, 0, 0],
        [0, 1., 0, 0], 
        [0, 0, 1., -0.04],
        [0, 0, 0, 1]])
    pre_push_pose = target_pose @ pre_push_transform  
    motion_type = 'AvoidObs'
    gripper_action = None
    stages.append((active_obj_ID, passive_obj_ID, pre_push_pose, gripper_action, np.eye(4), motion_type))
    
    # close_gripper
    motion_type = 'Simple'
    gripper_action = 'close'
    stages.append((active_obj_ID, passive_obj_ID, pre_push_pose, gripper_action, np.eye(4), motion_type))
    
    # push
    push_transform = np.array([
        [1., 0, 0, 0],
        [0, 1., 0, 0], 
        [0, 0, 1., 0.08],
        [0, 0, 0, 1]])
    push_pose = target_pose @ push_transform   
    motion_type = 'Simple'      # TODO Pose指定方向后，要用力控模式
    stages.append((active_obj_ID, passive_obj_ID, push_pose, gripper_action, np.eye(4), motion_type))
    return stages


def load_task_solution(task_dir, registered_objects):
    # import ipdb;ipdb.set_trace()

    task_info = json.load(open('%s/task_info.json'%task_dir, 'rb'))
    stages = task_info['stages']
    
    if os.path.exists('%s/observation.pkl'%task_dir):
        observation = pickle.load(open('%s/observation.pkl'%task_dir, 'rb'))
    else:
        observation = None
    
    objects = {
        'gripper': OmniObject('gripper')
    }
    # obj_gripper.elements = {'active': {'grasp': {'xyz': np.array([0,0,0]), 'direction': np.array([0,0,1])}}}
    
    for stage in stages:
        for obj_id in [stage['active_object_id'], stage['passive_object_id']]:
            if obj_id in objects:
                continue
            
            
            if obj_id=='fix_pose':
                obj = OmniObject('fix_pose')
                for _info in task_info['objects']:
                    if _info['object_id']=='fix_pose':
                        break
                obj.set_pose(np.array(_info['fix_pose']), np.array([0.001, 0.001, 0.001]))
                obj.elements = {
                    'active': {},
                    'passive': {
                        'place': {
                            'xyz': np.array([0,0,0]),
                            'direction': np.array(_info['fix_direction'])
                        }
                    }
                }
                # obj.set_pose

                
            else:
                
                if obj_id in registered_objects['simple_place_object']['object_list']:
                    obj_dir = registered_objects['simple_place_object']['obj_dir']
                elif obj_id in registered_objects['insert_place_object']['object_list']:
                    obj_dir = registered_objects['insert_place_object']['obj_dir']
                elif obj_id in registered_objects:
                    obj_dir = registered_objects[obj_id]['obj_dir']
                else:
                    obj_dir = '%s/%s'%(task_dir, obj_id)
                obj = OmniObject.from_obj_dir(obj_dir)
                
            objects[obj_id] = obj
    return stages, objects, observation


articulate_actions = ['push', 'pull', 'rotate']


# robot_gripper_2_grasp_gripper = np.array([
#     [0., 0., 1.],
#     [0., 1., 0.],
#     [-1., 0., 0.]])  

def generate_action_stages(objects, stages, observation, robot):
    gripper2obj = None
    
    action_stages = []
    for stage in stages:
        action = stage['action']
        active_obj = objects[stage['active_object_id']]
        passive_obj = objects[stage['passive_object_id']]
       
        passive_element = passive_obj.elements['passive'][action]
        
        
        if action=='grasp':
            gripper_pose_canonical_list = np.array(passive_element['grasp_pose'])
            
  
            gripper_pose_canonical_list[:,:3,:3] = gripper_pose_canonical_list[:,:3,:3] @ robot.robot_gripper_2_grasp_gripper[np.newaxis, ...]
            
            current_gripper_pose = robot.get_ee_pose('gripper')
            target_gripper_pose_pass_ik = []
            for grasp_gripper_canonical in gripper_pose_canonical_list:
                target_gripper_pose = passive_obj.obj_pose @ grasp_gripper_canonical 
                for gripper_pose in [target_gripper_pose, rotate_180_along_axis(target_gripper_pose, 'z')]:
                    ik_success, _ = robot.solve_ik(gripper_pose, ee_type='gripper', type='Simple')
                    if ik_success:
                        target_gripper_pose_pass_ik.append(gripper_pose)
            # import ipdb;ipdb.set_trace()
            assert(len(target_gripper_pose_pass_ik)>0), 'No grasp_gripper_pose can pass IK'
            closest_pose, closest_angle_diff = target_gripper_pose_pass_ik[0], np.inf
            for target_gripper_pose in target_gripper_pose_pass_ik:
                _, angle_diff = pose_difference(current_gripper_pose, target_gripper_pose)
                if angle_diff < closest_angle_diff:
                    closest_pose, closest_angle_diff = target_gripper_pose, angle_diff
                
            # else:
            #     import ipdb;ipdb.set_trace()
            #     gripper_pose_canonical = np.linalg.inv(passive_obj.obj_pose) @ active_obj.obj_pose
            closest_pose_canonical = np.linalg.inv(passive_obj.obj_pose) @ closest_pose
            substages = globals()[action](closest_pose_canonical, stage['active_object_id'], stage['passive_object_id'])
            gripper2obj = closest_pose_canonical
        elif action=='pull' or action=='rotate':
            gripper_pose_canonical = gripper2obj
            substages = globals()[action](gripper_pose_canonical, stage['active_object_id'], stage['passive_object_id'], passive_element['direction'])
        else:   
            # interaction between two rigid objects
            active_element = active_obj.elements['active'][action]
            obj_pose = active_obj.obj_pose
            anchor_pose = passive_obj.obj_pose
            # import ipdb;ipdb.set_trace()
            current_obj_pose_canonical = np.linalg.inv(anchor_pose) @ obj_pose
            active_obj.xyz, active_obj.direction = active_element['xyz'], active_element['direction']
            passive_obj.xyz, passive_obj.direction = passive_element['xyz'], passive_element['direction']
            
            
            if active_obj.name=='gripper':
                gripper2obj = np.eye(4)
            # 解IK （curobo-collision-detection)
            target_obj_pose_pass_ik = []
            for target_obj_pose in get_aligned_pose(active_obj, passive_obj, N=18):
                target_gripper_pose = target_obj_pose @ gripper2obj#[np.newaxis, ...]
                ik_success, _ = robot.solve_ik(target_gripper_pose, ee_type='gripper', type='AvoidObs')
                print(target_gripper_pose[:3,3])
                print(ik_success)
                # ik_success = True
                if ik_success:
                    target_obj_pose_pass_ik.append(target_obj_pose)
            # import ipdb;ipdb.set_trace()
            assert(len(target_obj_pose_pass_ik)>0), 'No target_obj_pose can pass IK'
            _gripper_pose = np.stack(target_obj_pose_pass_ik) @ gripper2obj[np.newaxis, ...]
            closest_pose, closest_angle_diff = _gripper_pose[0], np.inf
            for i in range(_gripper_pose.shape[0]):
                _, angle_diff = pose_difference(current_gripper_pose, _gripper_pose[i])
                if angle_diff < closest_angle_diff:
                    closest_pose, closest_angle_diff = _gripper_pose[i], angle_diff
            cloest_target_obj_pose =  closest_pose @ np.linalg.inv(gripper2obj)
            target_obj_pose_canonical = np.linalg.inv(anchor_pose) @ cloest_target_obj_pose     # TODO 暂时只取一个可行解，后面要结合grasp pose做整条trajectory的joint solve
            
            

            part2obj = np.eye(4)
            part2obj[:3, 3] = active_obj.xyz
            obj2part = np.linalg.inv(part2obj)
            
            # import ipdb;ipdb.set_trace()
            # from omniagent.servers.view_render.api import ObjRenderAPI
            # render_api = ObjRenderAPI(observation, active_obj)
            # cv2.imwrite('1.png', render_api.render_obj_at_scene([target_obj_pose_pass_ik[0]], observation['image'])[0])

            substages = globals()[action](
                target_pose = target_obj_pose_canonical,
                current_pose = current_obj_pose_canonical,
                obj2part = obj2part,
                active_obj_ID = stage['active_object_id'], 
                passive_obj_ID = stage['passive_object_id'],
                vector_direction = passive_element['direction']
            )

        action_stages.append(substages)
        
        
    return action_stages


def solve_target_gripper_pose(stage, objects):
    active_obj_ID, passive_obj_ID, target_pose_canonical, gripper_action, transform_world, motion_type = stage
    
    anchor_pose = objects[passive_obj_ID].obj_pose
    
    
    if motion_type=='Trajectory':
        assert len(target_pose_canonical.shape)==3, 'The target_pose should be a list of poses'
        target_pose = anchor_pose[np.newaxis, ...] @ target_pose_canonical
        target_pose = transform_world[np.newaxis, ...] @ target_pose
    else:
        target_pose = anchor_pose @ target_pose_canonical
        target_pose = transform_world @ target_pose
    assert 'gripper' in objects, 'The gripper should be the first one in the object list'
    current_gripper_pose = objects['gripper'].obj_pose
    
    if active_obj_ID=='gripper':
        target_gripper_pose = target_pose
    else:
        current_obj_pose = objects[active_obj_ID].obj_pose
        gripper2obj = np.linalg.inv(current_obj_pose) @ current_gripper_pose
        if len(target_pose.shape)==3:
            gripper2obj = gripper2obj[np.newaxis, ...]
        
        target_obj_pose = target_pose
        target_gripper_pose = target_obj_pose @ gripper2obj

    return target_gripper_pose





