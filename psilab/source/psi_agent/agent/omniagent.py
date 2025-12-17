# -*- coding: utf-8 -*-
import json
import time
import glob
import pickle
import numpy as np

from planner import Planner, OmniObject

from .base import BaseAgent
# from graspapi import GraspApi
from robot import Robot
import os

from scipy.spatial.transform import Rotation
from planner.manip_solver import load_task_solution, generate_action_stages, split_grasp_stages
from base_utils.data_utils import pose_difference
from base_utils.fix_rotation import rotation_matrix_to_quaternion, rotate_along_axis, translate_along_axis


class Agent(BaseAgent):
    def __init__(self, robot: Robot, planner: Planner):
        super().__init__(robot)
        self.planner = planner
        self.attached_obj_id = None

    def start_recording(self, task_name, camera_prim_list, fps):
        self.robot.client.start_recording(
            task_name=task_name,
            fps=fps,
            data_keys={
                "camera": {
                    "camera_prim_list": camera_prim_list,
                    "render_depth": False,
                    "render_semantic": False,
                },
                "pose": ["/World/Raise_A2/gripper_center"],
                "joint_position": True,
                "gripper": True,
            },
        )

    def generate_layout(self, task_file):
        self.task_file = task_file
        with open(task_file, "r") as f:
            task_info = json.load(f)
        

        # add mass for stable manipulation
        for stage in task_info['stages']:
            if stage['action'] in ['place', 'insert', 'pour']:
                obj_id = stage['passive']['object_id']
                for i in range(len(task_info['objects'])):
                    if task_info['objects'][i]['object_id'] == obj_id:
                        task_info['objects'][i]['mass'] = 10
                        break
 
        self.articulated_objs = []
        for object_info in task_info["objects"]:
            is_articulated = object_info.get('is_articulated', False)
            if is_articulated:
                self.articulated_objs.append(object_info['object_id'])
            object_info['material'] = 'general'
            self.add_object(object_info)
        time.sleep(2)

            
        self.arm = task_info["arm"]

        ''' For A2D: Fix camera rotaton to look at target object '''
        task_related_objs = []
        for stage in task_info['stages']:
            for type in ['active', 'passive']:
                obj_id = stage[type]['object_id']
                if obj_id == 'gripper' or obj_id in task_related_objs:
                    continue
                task_related_objs.append(obj_id)
        
        target_lookat_point = []
        for obj in task_info['objects']:
            if obj['object_id'] not in task_related_objs:
                continue
            target_lookat_point.append(obj['position'])
        target_lookat_point = np.mean(np.stack(target_lookat_point), axis=0)
        self.robot.client.SetTargetPoint(target_lookat_point.tolist())

        material_infos = []
        for key in task_info['object_with_material']:
            material_infos += task_info['object_with_material'][key]
        if len(material_infos):
            self.robot.client.SetMaterial(material_infos)
            time.sleep(0.3)

        light_infos = []
        for key in task_info['lights']:
            light_infos += task_info['lights'][key]
        if len(light_infos):
            self.robot.client.SetLight(light_infos)
            time.sleep(0.3)

        
    def update_objects(self, objects, arm='right'):
        # update gripper pose
        objects['gripper'].obj_pose = self.robot.get_ee_pose(ee_type='gripper', id=arm)

        # update object pose
        for obj_id in objects:
            if obj_id=='gripper':
                continue
                        
            # TODO(unify part_name and obj_name)
            if '/' in obj_id:
                obj_name = obj_id.split('/')[0]
                part_name = obj_id.split('/')[1]

                object_joint_state = self.robot.client.get_object_joint('/World/Objects/%s'%obj_name)
                for joint_name, joint_position, joint_velocity in zip(object_joint_state.joint_names, object_joint_state.joint_positions, object_joint_state.joint_velocities):
                    if joint_name[-1] == part_name[-1]:
                        objects[obj_id].joint_position = joint_position
                        objects[obj_id].joint_velocity = joint_velocity

            objects[obj_id].obj_pose = self.robot.get_prim_world_pose('/World/Objects/%s'%obj_id)
            if 'simple_place' in objects[obj_id].info and objects[obj_id].info['simple_place']:
                down_direction_world = (np.linalg.inv(objects[obj_id].obj_pose) @ np.array([0,0,-1,1]))[:3]
                down_direction_world = down_direction_world / np.linalg.norm(down_direction_world) * 0.08
                objects[obj_id].elements['active']['place']['direction'] = down_direction_world

        return objects
    

    def check_task_file(self, task_file):
        with open(task_file, "r") as f:
            task_info = json.load(f)

        
        objs_dir = {}
        objs_interaction = {}
        for obj_info in task_info["objects"]:
            obj_id = obj_info["object_id"]
            objs_dir[obj_id] = obj_info["data_info_dir"]
            if "interaction" in obj_info:
                objs_interaction[obj_id] = obj_info["interaction"]
            else:
                objs_interaction[obj_id] = json.load(open(obj_info["data_info_dir"]+'/interaction.json'))['interaction']


        for stage in task_info['stages']:
            active_obj_id = stage['active']['object_id']
            passive_obj_id = stage['passive']['object_id']
            if active_obj_id != 'gripper':
                if active_obj_id not in objs_dir:
                    print('Active obj not in objs_dir: %s'%active_obj_id)
                    return False
            if passive_obj_id != 'gripper':
                if passive_obj_id not in objs_dir:
                    print('Passive obj not in objs_dir: %s'%passive_obj_id)
                    return False
            data_root = os.path.dirname(os.path.dirname(__file__))+"/assets"
            print("777777777777777777777777777777777777777777777777777")
            print(data_root)
            if stage['action'] in ['grasp', 'pick']:
                passive_obj_id = stage['passive']['object_id']
                obj_dir = objs_dir[passive_obj_id]
                primitive = stage['passive']['primitive']
                if primitive is None:
                    file = 'grasp_pose/grasp_pose.pkl'
                else:
                    file = objs_interaction[passive_obj_id]['passive']['grasp'][primitive]
                    if isinstance(file, list):
                        file = file[0]
                print("88888888888888888888888888888888888888888888")
                
                grasp_file = os.path.join(data_root, obj_dir, file)
                print(grasp_file)
                if not os.path.exists(grasp_file):
                    print('-- Grasp file not exist: %s'%grasp_file)
                    return False

                _data = pickle.load(open(grasp_file, 'rb'))
                if len(_data['grasp_pose'])==0:
                    print('-- Grasp file empty: %s'%grasp_file)
                    return False
        return True


    def run(self, task_folder, camera_list, use_recording, fps=10):
        tasks = glob.glob(task_folder + "/*.json")
        for task_file in tasks:

            if not self.check_task_file(task_file):
                print("Task file bad: %s"%task_file)
                continue
            print("Start Task:", task_file)
            self.reset()
            self.generate_layout(task_file)

            self.robot.open_gripper(id='right')
            self.robot.open_gripper(id='left')

            self.robot.reset_pose = {
                'right': self.robot.get_ee_pose(ee_type='gripper', id='right'),
                'left': self.robot.get_ee_pose(ee_type='gripper', id='left'),
            }
            print('Reset pose:', self.robot.reset_pose)


            task_info = json.load(open(task_file, 'rb')) 
            stages, objects = load_task_solution(task_info)   
            objects = self.update_objects(objects)
            split_stages = split_grasp_stages(stages)

            if use_recording:
                self.start_recording(task_name="[%s]" % (os.path.basename(task_file).split(".")[0]), 
                                    camera_prim_list=camera_list,fps=fps)  # TODO 录制判断
                
            stage_id = -1
            substages = None
            for _stages in split_stages:
                extra_params = _stages[0].get('extra_params', {})
                arm = extra_params.get('arm', 'right')
                # generate action stages and slove ik to filter pose
                action_stages = generate_action_stages(objects, _stages, self.robot)
                if not len(action_stages):
                    success = False
                    print('No action stage generated.')
                    break

                # Execution
                success = True
                
                print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
                print(_stages)
                print(action_stages)
                for action, substages in action_stages:
                    stage_id += 1
                    print('>>>>  Stage [%d]  <<<<'%(stage_id+1))
                    if action in ['reset']:
                        init_pose = self.robot.reset_pose[arm]
                        curr_pose = self.robot.get_ee_pose(ee_type='gripper', id=arm)
                        interp_pose = init_pose.copy()
                        interp_pose[:3,3] = curr_pose[:3,3] + (init_pose[:3,3] - curr_pose[:3,3]) * 0.25
                        success = self.robot.move_pose(self.robot.reset_pose[arm], type='AvoidObs', arm=arm, block=True)
                        continue    
                    if action in ['grasp', 'pick']:
                        obj_id = substages.passive_obj_id
                        if  obj_id.split('/')[0] not in self.articulated_objs:
                            self.robot.target_object = substages.passive_obj_id
                        
                    while len(substages):
                        # get next step actionddd
                        objects = self.update_objects(objects, arm=arm)
                        target_gripper_pose, motion_type, gripper_action, arm = substages.get_action(objects)
                        arm = extra_params.get('arm', 'right')
                        self.robot.client.set_frame_state(action, substages.step_id, self.attached_obj_id is not None)
                    
                        
                        if False:
                            _pose = translate_along_axis(target_gripper_pose, -0.4, 'y', use_local=False)
                            _pose = translate_along_axis(_pose, 0.3, 'z', use_local=False)
                            _pose = translate_along_axis(_pose, 0.1, 'x', use_local=False)
                            # _pose = rotate_along_axis(_pose, 30, 'z', use_local=True)
                            _pose = rotate_along_axis(_pose, -90, 'x', use_local=True)
                            
                        # if True:
                            zz = _pose.copy()
                            zz = translate_along_axis(zz, 0.1, 'z', use_local=True)
                           
                            zz = rotate_along_axis(zz, 90, 'z', use_local=True)
                            
                            zz = translate_along_axis(zz, -0.1, 'x', use_local=False)
                            zz = translate_along_axis(zz, -0.03, 'z', use_local=True)
                            _pose = zz

                            self.robot.move_pose(_pose, 'Simple', arm="left", block=True)
                        # import ipdb;ipdb.set_trace()
                            
                        
                    
                        # execution action
                        if target_gripper_pose is not None:
                            print("{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}} zoudaole zheli ")
                            self.robot.move_pose(target_gripper_pose, motion_type, arm=arm, block=True)


                            
                            if False:
                                gripper_pose = []

                                
                                gripper2part = np.linalg.inv(objects['bag'].obj_pose) @ self.robot.get_ee_pose()
                                robot_gripper_2_grasp_gripper = np.eye(4)
                                robot_gripper_2_grasp_gripper[:3,:3] = self.robot.robot_gripper_2_grasp_gripper
                                gripper_grasp2part = gripper2part @ np.linalg.inv(robot_gripper_2_grasp_gripper)
                                gripper_pose.append(gripper_grasp2part)
                                gripper_pose = np.stack(gripper_pose)
                                pkl_data = {'grasp_pose': gripper_pose, 'width': np.ones(gripper_pose.shape[0])*0.03}

                                file = 'assets/objects/selfscan/handbag/selfScan_handbag_001/grasp_pose/hook_left.pkl'
                                origin_data = pickle.load(open(file, 'rb'))
                                pickle.dump(pkl_data, open(file, 'wb'))
        

                        self.robot.client.set_frame_state(action, substages.step_id, self.attached_obj_id is not None)
                        self.robot.set_gripper_action(gripper_action, arm=arm)
                        if gripper_action=='open':
                            time.sleep(1)
                        

                        
                        self.robot.client.set_frame_state(action, substages.step_id, self.attached_obj_id is not None)
                        
                        # check sub-stage completion
                        objects['gripper'].obj_pose = self.robot.get_ee_pose(ee_type='gripper', id=arm)
                        objects = self.update_objects(objects, arm=arm)

                        # import ipdb;ipdb.set_trace()
                        success = substages.check_completion(objects)
                        self.robot.client.set_frame_state(action, substages.step_id, self.attached_obj_id is not None)
                        if success==False:
                            # import ipdb;ipdb.set_trace()
                            print('Failed at sub-stage %d'%substages.step_id)
                            break

                        # attach grasped object to gripper           # TODO avoid articulated objects
                        if arm=='right':
                            if gripper_action=='close': # TODO  确定是grasp才行！！
                                self.attached_obj_id = substages.passive_obj_id
                            elif gripper_action=='open':
                                self.attached_obj_id = None
                        self.robot.client.set_frame_state(action, substages.step_id, self.attached_obj_id is not None)

                    if success==False:
                        break
                if success==False:
                    break
            time.sleep(0.5)
            self.robot.client.stop_recording()
            # try:
            #     step_id = substages.step_id if substages is not None and len(substages) else -1
            # except:
            #     step_id = -1
            step_id = -1
            fail_stage_step= [stage_id, step_id] if success==False else[-1, -1]

            task_info_saved = task_info.copy()
            self.robot.client.SendTaskStatus(success, fail_stage_step)
            if success:
                print(">>>>>>>>>>>>>>>> Success!!!!!!!!!!!!!!!!")
        
            

        return True
