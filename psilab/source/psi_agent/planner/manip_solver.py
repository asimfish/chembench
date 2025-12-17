# -*- coding: utf-8 -*-
import os
import sys 
import copy
import time
import trimesh
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
import numpy as np
from dataclasses import dataclass

from base_utils.transforms import calculate_rotation_matrix, rotate_around_axis
from base_utils.object import OmniObject, transform_coordinates_3d
from base_utils.data_utils import pose_difference, vector_difference, pose_difference_batch, vector_difference_batch
from base_utils.fix_rotation import rotate_180_along_axis, translate_along_axis, rotation_matrix_to_quaternion


from .action import build_stage






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




def get_aligned_pose(active_obj, passive_obj, distance=0.01, N=1):
    try:
        active_object = format_object(active_obj, type='active', distance=distance)
        passive_object = format_object(passive_obj, type='passive', distance=distance)
    except:
        print('error')



    active_obj_world = obj2world(active_object)
    current_obj_pose = active_obj_world['pose']
    if passive_object is None:
        return current_obj_pose[np.newaxis, ...]
    
    passive_obj_world = obj2world(passive_object)
    
    R = calculate_rotation_matrix(active_obj_world['direction'], passive_obj_world['direction'])
    T = passive_obj_world['xyz_end'] - R @ active_obj_world['xyz_start']
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



def load_task_solution(task_info):
    # task_info = json.load(open('%s/task_info.json'%task_dir, 'rb'))
    stages = task_info['stages']
    

    objects = {
        'gripper': OmniObject('gripper')
    }

    for obj_info in task_info['objects']:
        obj_id = obj_info['object_id']
        obj_dir = obj_info["data_info_dir"]
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
        else:
            obj = OmniObject.from_obj_dir(obj_dir, obj_info=obj_info)    
            objects[obj_id] = obj

            if hasattr(obj, 'part_ids'):
                if hasattr(obj, 'part_joint_limits') and obj.part_joint_limits is not None:
                    obj_parts_joint_limits = obj.part_joint_limits
                for part_id in obj.part_ids:
                    id = obj_id + '/%s'%part_id
                    objects[id] = copy.deepcopy(obj)
                    objects[id].name = id
                    objects[id].part_joint_limit = obj_parts_joint_limits[part_id]
                if len(obj.part_ids):
                    del objects[obj_id]
    return stages, objects





def parse_stage(stage, objects):
    action = stage['action']
    if action in ['reset']:
        return action, 'gripper', 'gripper', None, None, None, None
    active_obj_id = stage['active']['object_id']
    if 'part_id' in stage['active']:
        active_obj_id += '/%s'%stage['active']['part_id']

    passive_obj_id = stage['passive']['object_id']
    if 'part_id' in stage['passive']:
        passive_obj_id += '/%s'%stage['passive']['part_id']
    
    active_obj = objects[active_obj_id]
    passive_obj = objects[passive_obj_id]

    single_obj = action in ['pull', 'rotate', 'slide', 'shave', 'brush', 'wipe']

    def _load_element(obj, type):
        if action in ['pick', 'hook']:
            action_mapped = 'grasp'
        else:
            action_mapped = action
        if action_mapped=='grasp' and type=='active':
            return None, None
        elif obj.name=='gripper':
            element = obj.elements[type][action_mapped]
            return element, 'default'
        primitive = stage[type]['primitive'] if stage[type]['primitive'] is not None else 'default'
        if primitive != 'default' or (action_mapped=='grasp' and type=='passive'):
            if action_mapped not in obj.elements[type]:
                print('No %s element for %s'%(action_mapped, obj.name))
                return None, None
            element = obj.elements[type][action_mapped][primitive]
        else:
            element = []
            print(obj.part_joint_limits)
            for primitive in obj.elements[type][action_mapped]:
                _element = obj.elements[type][action_mapped][primitive]
                if isinstance(_element, list):
                    element += _element
                else:
                    element.append(_element)
        return element, primitive


    passive_element, passive_primitive = _load_element(passive_obj, 'passive')
    if not single_obj:
        active_element, active_primitive = _load_element(active_obj, 'active')
    else:
        active_element, active_primitive = passive_element, passive_primitive
    return action, active_obj_id, passive_obj_id, active_element, passive_element, active_primitive, passive_primitive



def select_obj(objects, stages, robot):
    gripper2obj = None
    extra_params = stages[0].get('extra_params', {})
    arm = extra_params.get('arm', 'right')
    current_gripper_pose = robot.get_ee_pose('gripper', arm=arm)
    

    ''' 初筛抓取pose，得到 grasp_poses_canonical, grasp_poses '''
    grasp_stage_id = None

    if stages[0]['action'] in ['push', 'reset']:
        gripper2obj = current_gripper_pose
    elif stages[0]['action'] in ['pick', 'grasp', 'hook']:
        action = stages[0]['action']

        ''' 筛掉无IK解的grasp pose '''
        grasp_stage_id = 0
        grasp_stage = parse_stage(stages[0], objects)
        _, _, passive_obj_id, _, passive_element, _, _ = grasp_stage
        grasp_obj_id = passive_obj_id
        grasp_poses_canonical = passive_element['grasp_pose'].copy()
        
        grasp_poses_canonical[:,:3,:3] = grasp_poses_canonical[:,:3,:3] @ robot.robot_gripper_2_grasp_gripper[np.newaxis, ...]
        
        grasp_poses_canonical_flip = rotate_180_along_axis(grasp_poses_canonical, 'z')
        grasp_poses_canonical = np.concatenate([grasp_poses_canonical, grasp_poses_canonical_flip], axis=0)

        grasp_poses = objects[passive_obj_id].obj_pose[np.newaxis, ...] @ grasp_poses_canonical 
        # filter with IK-checking

        ik_success, jacobian_score = robot.solve_ik(grasp_poses, ee_type='gripper', arm=arm, type='Simple')
        grasp_poses_canonical, grasp_poses = grasp_poses_canonical[ik_success], grasp_poses[ik_success]
        
        print('%s, %s, Filtered grasp pose with isaac-sim IK: %d/%d'%(action, passive_obj_id, grasp_poses.shape[0], ik_success.shape[0]))
        # ik_success, _ = robot.solve_ik(grasp_poses, ee_type='gripper', type='AvoidObs')
        # grasp_poses_canonical, grasp_poses = grasp_poses_canonical[ik_success], grasp_poses[ik_success]
        # print('Filtered grasp pose with curobo IK: %d/%d'%(grasp_poses.shape[0], ik_success.shape[0]))
        if len(grasp_poses)==0:
            print(action, 'No grasp_gripper_pose can pass IK')
            return []


    ''' 基于有IK解的grasp pose分数，选择最优的passive primitive element，同时选出最优的一个grasp pose'''
    if grasp_stage_id is not None:
        next_stage_id = grasp_stage_id + 1
        if next_stage_id<len(stages):
            action, active_obj_id, passive_obj_id, active_elements, passive_elements, active_primitive, passive_primitive = parse_stage(stages[next_stage_id], objects)
            
            single_obj = active_obj_id==passive_obj_id
            
            
            active_obj = objects[active_obj_id]
            passive_obj = objects[passive_obj_id]
            passive_element = passive_elements[np.random.choice(len(passive_elements))]


            if action=='place':     # TODO A2D暂时这样搞，Franka要取消
                # import ipdb; ipdb.set_trace()
                obj_pose = active_obj.obj_pose
                mesh = trimesh.load(active_obj.info['mesh_file'])
                mesh.apply_scale(0.001)
                mesh.apply_transform(obj_pose)
                pts, _ =  trimesh.sample.sample_surface(mesh, 200) # 表面采样
                xyz = np.array([np.mean(pts[:, 0]), np.mean(pts[:, 1]), np.percentile(pts[:, 2], 1)])
           


                direction = np.array([0, 0, -1])
                xyz_canonical = (np.linalg.inv(obj_pose) @ np.array([*xyz, 1]))[:3]
                direction_canonical = (np.linalg.inv(obj_pose) @ np.array([*direction, 0]))[:3]
                active_elements = [{'xyz': xyz_canonical, 'direction': direction_canonical}]

            # import ipdb; ipdb.set_trace()
            t0 = time.time()
            element_ik_score = []
            grasp_pose_ik_score = []
            for active_element in active_elements:
                # interaction between two rigid objects
                obj_pose = active_obj.obj_pose
                anchor_pose = passive_obj.obj_pose
                current_obj_pose_canonical = np.linalg.inv(anchor_pose) @ obj_pose
                
                N_align = 12
                if not single_obj:
                    active_obj.xyz, active_obj.direction = active_element['xyz'], active_element['direction']
                    passive_obj.xyz, passive_obj.direction = passive_element['xyz'], passive_element['direction']
                    target_obj_poses = get_aligned_pose(active_obj, passive_obj, N=N_align)
                else:   # 物体自身移动
                    transform = np.eye(4)
                    transform[:3, 3] = active_element['xyz']
                    target_obj_poses = (obj_pose @ transform)[np.newaxis, ...]
                    N_align = 1
                
                N_obj_pose = target_obj_poses.shape[0]
                N_grasp_pose = grasp_poses_canonical.shape[0]
                target_gripper_poses = (target_obj_poses[:, np.newaxis, ...] @ grasp_poses_canonical[np.newaxis, ...]).reshape(-1, 4, 4)

                ik_success, _ = robot.solve_ik(target_gripper_poses, ee_type='gripper', type='Simple', arm=arm)
                element_ik_score.append(np.max(ik_success.reshape(N_obj_pose, N_grasp_pose).sum(axis=1)))

                grasp_pose_ik = ik_success.reshape(N_obj_pose, N_grasp_pose)
                grasp_pose_ik_score.append(np.sum(grasp_pose_ik, axis=0))
            
            print(time.time() - t0)
            # import ipdb; ipdb.set_trace()
            best_element_id = np.argmax(element_ik_score)
            best_active_element = active_elements[best_element_id]

            if not single_obj:

                active_obj.elements['active'][action] = {active_primitive: best_active_element}
            
            grasp_ik_score = grasp_pose_ik_score[best_element_id]
            
            # import ipdb;ipdb.set_trace()
            best_grasp_poses = grasp_poses[grasp_ik_score>=np.median(grasp_ik_score)/2]
            if best_grasp_poses.shape[0] > 1:   # further select the best grasp pose with the smallest pose difference
                cost_pos, cost_rot = pose_difference_batch(best_grasp_poses, current_gripper_pose)
                cost_forward_vec = vector_difference_batch(best_grasp_poses, current_gripper_pose)
                total_cost = cost_pos*0.3 + cost_rot*1.0 + cost_forward_vec*2.0

                # TODO 如果前面换成curobo IK了，这里的筛选可以disable掉
                idx_sorted = np.argsort(total_cost)
                if len(idx_sorted)>10:
                    idx_sorted = idx_sorted[:10]
                ik_success, _ = robot.solve_ik(best_grasp_poses[idx_sorted], ee_type='gripper', type='AvoidObs', arm=arm)
                if ik_success.sum()==0:
                    # import ipdb; ipdb.set_trace()
                    print(action, 'No best_grasp_pose can pass curobo IK')
                    return []
                print(action, 'Filtered grasp pose with curobo IK: %d/%d'%(ik_success.sum(), ik_success.shape[0]))
                best_grasp_pose = best_grasp_poses[idx_sorted][ik_success][0]   


                # min_cost_index = np.argmin(total_cost)
                # best_grasp_pose = best_grasp_poses[min_cost_index]
            else:
                best_grasp_pose = best_grasp_poses[0]
            best_grasp_pose_canonical = np.linalg.inv(objects[grasp_obj_id].obj_pose) @ best_grasp_pose
            gripper2obj = best_grasp_pose_canonical
        else:
            gripper2obj = grasp_poses_canonical[0]
    return gripper2obj


def split_grasp_stages(stages):
    split_stages = []
    i = 0
    while i<len(stages):
        if stages[i]['action'] in ['pick', 'grasp', 'hook']:
            if (i+1)<len(stages) and stages[i+1]['action'] not in ['pick', 'grasp', 'hook']:
                split_stages.append([stages[i], stages[i+1]])
                i += 2
            else:
                split_stages.append([stages[i]])
                i += 1
        else:
            split_stages.append([stages[i]])
            i += 1
    return split_stages



def generate_action_stages(objects, all_stages, robot):
    split_stages = split_grasp_stages(all_stages)
    
    current_gripper_pose = robot.get_ee_pose('gripper')
    action_stages = []
    for stages in split_stages:
        gripper2obj = select_obj(objects, stages, robot)
        if gripper2obj is None or len(gripper2obj)==0:
            print('No gripper2obj pose can pass IK')
            gripper2obj = select_obj(objects, stages, robot)
            return []
        for stage in stages:
            print("what is stage what is stage what is stage ++++++++++++++++++++++")
            print(stage)
            extra_params = stage.get('extra_params', {})
            arm = extra_params.get('arm', 'right')
            action, active_obj_id, passive_obj_id, active_elements, passive_elements, active_primitive, passive_primitive = parse_stage(stage, objects)
            active_obj = objects[active_obj_id]
            passive_obj = objects[passive_obj_id]
            
            single_obj = active_obj_id==passive_obj_id

            substages = None
            if action in ['reset']:
                substages = True
            elif action in ['pick', 'grasp', 'hook']:
                substages = build_stage(action)(active_obj_id, passive_obj_id, active_elements, passive_elements, gripper2obj, extra_params=stage.get('extra_params', None))
            elif action in ['slide', 'shave', 'brush', 'wipe', 'pull', 'rotate', 'pull', 'move', 'reset']:       # grasp + 物体自身运动
                passive_element = passive_elements[np.random.choice(len(passive_elements))]
                substages = build_stage(action)(
                    active_obj_id = active_obj_id,
                    passive_obj_id=passive_obj_id, 
                    passive_element=passive_element
                    )
            else:
                passive_element = passive_elements[np.random.choice(len(passive_elements))]
                # active_element = active_elements[np.random.choice(len(active_elements))] if isinstance(active_elements, list) else active_elements
                if not isinstance(active_elements, list):
                    active_elements = [active_elements]
                
                
                for active_element in active_elements:
                    # interaction between two rigid objects
                    obj_pose = active_obj.obj_pose
                    anchor_pose = passive_obj.obj_pose
                    current_obj_pose_canonical = np.linalg.inv(anchor_pose) @ obj_pose
                    active_obj.xyz, active_obj.direction = active_element['xyz'], active_element['direction']
                    passive_obj.xyz, passive_obj.direction = passive_element['xyz'], passive_element['direction']
                    if active_obj.name=='gripper':
                        gripper2obj = np.eye(4)
                    target_obj_poses = get_aligned_pose(active_obj, passive_obj, N=18)
                    target_gripper_poses = target_obj_poses @ gripper2obj[np.newaxis, ...]
                    ik_success, _ = robot.solve_ik(target_gripper_poses, ee_type='gripper', type='Simple', arm=arm)
                    target_gripper_poses_pass_ik = target_gripper_poses[ik_success]

                    if len(target_gripper_poses_pass_ik)==0:
                        print(action, ': No target_obj_pose can pass isaac IK')
                        continue
                        # return []
                    
                    # gripper2world = active_obj.obj_pose @ gripper2obj
                    cost_pos, cost_rot = pose_difference_batch(target_gripper_poses_pass_ik, current_gripper_pose)
                    cost_forward_vec = vector_difference_batch(target_gripper_poses_pass_ik, current_gripper_pose)
                    total_cost = cost_pos*0.3 + cost_rot*1.0 + cost_forward_vec*2.0

                    # TODO 如果前面换成curobo IK了，这里的筛选可以disable掉
                    idx_sorted = np.argsort(total_cost)
                    if len(idx_sorted)>10:
                        idx_sorted = idx_sorted[:10]
                    ik_success, _ = robot.solve_ik(target_gripper_poses_pass_ik[idx_sorted], ee_type='gripper', type='AvoidObs', arm=arm)
                    if ik_success.sum()==0:
                        print(action, ': No target pose can pass curobo IK.')
                        continue
                        return []
                    best_target_gripper_pose = target_gripper_poses_pass_ik[idx_sorted][ik_success][0]
                    best_target_obj_pose = best_target_gripper_pose @ np.linalg.inv(gripper2obj)
                    target_obj_pose_canonical = np.linalg.inv(anchor_pose) @ best_target_obj_pose     # TODO 暂时只取一个可行解，后面要结合grasp pose做整条trajectory的joint solve
                        
                    part2obj = np.eye(4)
                    part2obj[:3, 3] = active_obj.xyz
                    obj2part = np.linalg.inv(part2obj)

                    substages = build_stage(action)(
                        active_obj_id = active_obj_id, 
                        passive_obj_id = passive_obj_id,
                        target_pose = target_obj_pose_canonical,
                        current_pose = current_obj_pose_canonical,
                        obj2part = obj2part,
                        vector_direction = passive_element['direction'],
                        passive_element=passive_element
                    )
                    break
                
            if substages is None:
                print(action, ': No target_obj_pose can pass IK')
                return []
            action_stages.append((action, substages))
            
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





