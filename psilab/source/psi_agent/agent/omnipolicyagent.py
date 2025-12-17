# -*- coding: utf-8 -*-
import json
import time
import numpy as np
import collections

from .base import BaseAgent
from robot import Robot
from tqdm import trange
from PIL import Image
import os
import requests
import cv2
import pickle

from scipy.spatial.transform import Rotation
from planner.manip_solver import load_task_solution, generate_action_stages, solve_target_gripper_pose
from base_utils.data_utils import pose_difference

from robot.utils import (
    get_rotation_matrix_from_quaternion,
)

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def quat_wxyz_to_xyzw(quat):
    return np.array(list(quat[-1:]) + list(quat[:-1]))

def resize_img(img, height, width):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img
    
def pose_to_mat(xyz, quat):
    pose = np.eye(4)
    pose[:3, 3] = xyz
    pose[:3, :3] = get_rotation_matrix_from_quaternion(quat)
    return pose

def within(point, position, size):
    return position[0] - size[0] / 2 <= point[0] <= position[0] + size[0] / 2 and position[1] - size[1] / 2 <= point[1] <= position[1] + size[1] / 2 


class PolicyAgent(BaseAgent):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))
        self.attached_obj_ID = None
        self.results = {
            "state": [],
            "action": [],
            "ee_position": [],
            "ee_rotation": [],
            "obj_transform": [],
            "raw_gripper_position": [],
        }

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

    def generate_layout(self, task_file, camera_list):
        self.task_file = task_file
        with open(task_file, "r") as f:
            task_info = json.load(f)
            task_name = task_info["task_name"]

        for object_info in task_info["objects"]:
            object_info['material'] = 'general'
            self.add_object(object_info)
            time.sleep(2)

        self.robot.target_object = task_info["target_object"]
        self.arm = task_info["arm"]

        for obj in task_info["objects"]:
            if obj["object_id"] == task_info["target_object"]:
                self.target_obj_length = np.array(obj["size"])
                self.target_obj_prim = "/World/Objects/" + obj["object_id"]
                self.target_obj_uid = obj['model_path']

            if obj["object_id"] != task_info["target_object"]:
                self.place_obj_length = np.array(obj["size"])
                self.place_obj_prim = "/World/Objects/" + obj["object_id"]
                self.place_obj_id = obj['model_path']
    
        self.data_keys = {
            "camera": {
                "camera_prim_list": camera_list,
                "render_depth": True,
                "render_semantic": True,
            },
            "joint_position": True,
            "gripper": True,
        }
        self.robot.client.get_observation(self.data_keys)
        
        return task_name
    
    def check_task_completion(self, sub_stage, objects):
        active_obj_ID, passive_obj_ID, target_pose_canonical, gripper_action, transform_world, motion_type = sub_stage
        
        current_pose_canonical =  np.linalg.inv(objects[passive_obj_ID].obj_pose) @ objects[active_obj_ID].obj_pose
        if target_pose_canonical is None:
            return False
        
        if len(target_pose_canonical.shape)==3:
            target_pose_canonical = target_pose_canonical[-1]
        pos_diff, angle_diff = pose_difference(current_pose_canonical, target_pose_canonical)

        success = (pos_diff < 0.1) and (angle_diff < 30)
        return success
    
    def update_objects(self, objects):
        # update gripper pose
        objects['gripper'].obj_pose = self.robot.get_ee_pose(ee_type='gripper')

        # update object pose
        for obj_id in objects:
            if obj_id=='gripper':
                continue
            objects[obj_id].obj_pose = self.robot.get_prim_world_pose('/World/Objects/%s'%obj_id)
            if 'simple_place' in objects[obj_id].info and objects[obj_id].info['simple_place']:
                down_direction_world = (np.linalg.inv(objects[obj_id].obj_pose) @ np.array([0,0,-1,1]))[:3]
                down_direction_world = down_direction_world / np.linalg.norm(down_direction_world) * 0.08
                objects[obj_id].elements['active']['place']['direction'] = down_direction_world
   
        return objects

    def get_policy_obs(self, low_dim=False, step_num=None, camera_list=None):
        policy_obs = {}

        # get observation
        observation = self.robot.client.get_observation(self.data_keys)
        def get_rgb_image(observation, camera_prim):
            cam_info = observation["camera"][camera_prim]["camera_info"]
            rgb_image = observation["camera"][camera_prim]["rgb_camera"].reshape(cam_info["height"], cam_info["width"], 4)[:, :, :3]
            return rgb_image
        
        ee_transform = pose_to_mat(observation['gripper']['right']['position'], observation['gripper']['right']['rotation'])
        # ee_transform = self.robot.get_ee_pose()
        print("ee_position", observation['gripper']['right']['position'], "ee_rotation", observation['gripper']['right']['rotation'])
        raw_gripper_position = observation['joint']['right_Left_1_Joint']       # for zhixing gripper
        # raw_gripper_position = observation['joint']['panda_finger_joint1']     # for franka gripper
        print("raw_gripper_position", raw_gripper_position)


        if low_dim:
            obj_transform = np.stack([pose_to_mat(observation['pose']["/World/Objects/" + obj_name]['position'], observation['pose']["/World/Objects/" + obj_name]['rotation']) for obj_name in self.obj_names])
            state_data = np.concatenate([mat_to_pose10d(ee_transform), mat_to_pose10d(obj_transform).flatten(), [raw_gripper_position]])
        else:
            rgb = []
            rgb_images = []
            for camera_name in camera_list:
                rgb_image = get_rgb_image(observation, camera_name)
                rgb_images.append(rgb_image)
                rgb_image = resize_img(rgb_image, 224, 224)
                rgb_image = rgb_image.reshape(3, 224, 224)    # [3, height, width]
                rgb.append(rgb_image)
            rgb = np.concatenate(rgb)
            state_data = np.concatenate([mat_to_pose10d(ee_transform), [raw_gripper_position]])
            policy_obs['rgb'] = rgb

        policy_obs['state'] = state_data
        
        if step_num is not None:
            file_name = f"infer_results/dp_{self.ts_str}"
            os.makedirs(file_name, exist_ok=True)

            # save the image and action to file
            if not low_dim:
                for rgb_image, camera_name in zip(rgb_images, camera_list):
                    camera_name = camera_name.split('/')[-1]
                    img_path = f"infer_results/dp_{self.ts_str}/{step_num}_{camera_name}.png"
                    Image.fromarray(rgb_image).save(img_path)
            
            self.results['state'].append(state_data.tolist())
            self.results['ee_position'].append(observation['gripper']['right']['position'].tolist())
            self.results['ee_rotation'].append(observation['gripper']['right']['rotation'].tolist())
            self.results['raw_gripper_position'].append(raw_gripper_position)

            with open(f"infer_results/dp_{self.ts_str}/results.json", "w") as f:
                json.dump(
                    self.results,
                    f,
                    indent=4
                )

        return policy_obs, ee_transform, raw_gripper_position
    
    def policy_step(self, step_num, low_dim):
        # send request
        start_time = time.time()  # 记录开始时间
        data = {
            "state": np.stack(self.state_deque),
        }
        if not low_dim:
            data.update(
                {
                    "rgb": np.stack(self.rgb_deque),
                }
            )

        response = requests.post("http://0.0.0.0:5000/policy", data=pickle.dumps(data, protocol=4))
        action = pickle.loads(response.content)['action']

        end_time = time.time()  # 记录结束时间
        elapsed_time = (end_time - start_time) * 1000  # 计算运行时间并转换为毫秒
        print(f"Policy executed in {elapsed_time:.3f} ms")

        if step_num is not None:
            self.results['action'].append(action[0].tolist())

            with open(f"infer_results/dp_{self.ts_str}/results.json", "w") as f:
                json.dump(
                    self.results,
                    f,
                    indent=4
                )

        return action[0]

    def run(self, task_folder, camera_list, use_recording, fps=10):
        import glob
        tasks = glob.glob(task_folder + "/*.json")
        
        low_dim = False
        obs_horizon = 2
        img_horizon = 1
        action_steps = 2

        task_len_dict = {
            "fit_lit_on_teapot": 150,
        } 
        for i, task_file in enumerate(tasks):
            print("Start Task:", task_file)

            self.reset()

            task_name = self.generate_layout(task_file, camera_list)

            self.robot.client.DetachObj()

            print("Start Policy Execution")

            for step_num in trange(task_len_dict[task_name]):
                policy_obs, ee_transform, gripper_position = self.get_policy_obs(low_dim, step_num, camera_list)  # get observations every step

                if step_num == 0:
                    self.state_deque = collections.deque(
                        [policy_obs['state']] * obs_horizon, maxlen=obs_horizon)
                    
                    if not low_dim:
                        self.rgb_deque = collections.deque(
                            [policy_obs['rgb']] * img_horizon, maxlen=img_horizon)
                else:
                    self.state_deque.append(policy_obs['state'])
                    if not low_dim:
                        self.rgb_deque.append(policy_obs['rgb'])
                
                if step_num % action_steps == 0:
                    action_list = self.policy_step(step_num, low_dim)

                self.robot.move(
                    {
                        "type": "move_pose_list", 
                        "rot_repr": "euler", 
                        "action_repr": "delta", 
                        "data": action_list[:1],
                        "ee_transform": ee_transform, 
                        "gripper_position": gripper_position
                        }
                    )
                action_list = action_list[1:]

            #     success = self.check_task_completion(task_name)
            #     if success:
            #         print(f"task {i} success, need {step_num} steps")
            #         break

            # if not success:
            #     print(f"task {i} fail")

            self.robot.client.stop_recording()

            return
        