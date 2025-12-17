# -*- coding: utf-8 -*-
import json
import time
import collections
import os
import sys
current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_directory)
import numpy as np
import requests
import cv2
import pickle
from planner import Planner
from .base import BaseAgent
from robot.franka import IsaacSimRpcFranka
from tqdm import trange
from PIL import Image
from scipy.spatial.transform import Rotation

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
        
        
class PolicyAgent(BaseAgent):
    def __init__(self, robot: IsaacSimRpcFranka, planner: Planner):
        super().__init__(robot)
        self.planner = planner
        self.ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))
        self.results = {
            "state": [],
            "action": [],
            "ee_transform": [],
            "obj_transform": [],
        }

    def add_object(self, object_info: dict):
        name = object_info["name"]
        usd_path = object_info["usd"]
        position = np.array(object_info["position"])
        quaternion = np.array(object_info["quaternion"])
        if isinstance(object_info["scale"], list) and len(object_info["scale"]) == 3:
            scale = np.array(object_info["scale"])
        else:
            scale = np.array([object_info["scale"]] * 3)
        color = np.array(object_info["color"])
        material = "Brass" if "material" not in object_info else object_info["material"]
        self.robot.client.add_object(
            usd_path=usd_path,
            prim_path="/World/Objects/%s" % name,
            label_name=name,
            target_position=position,
            target_quaternion=quaternion,
            target_scale=scale,
            material=material,
            color=color,
        )
        
    def generate_layout(self, task_file):
        self.task_file = task_file
        
        with open(task_file, "r") as f:
            task_info = json.load(f)

        for object_info in task_info["objects"]:
            self.add_object(object_info)
        time.sleep(1)

        self.robot.target_object = task_info["target_object"]
        self.arm = task_info["arm"]

        self.obj_names = []
        for obj in task_info["objects"]:
            if obj["name"] == task_info["target_object"]:
                self.target_obj_length = np.array(obj["size"])
                self.target_obj_prim = "/World/Objects/" + obj["name"]
                self.target_obj_uid = obj['uid']

            if obj["name"] != task_info["target_object"]:
                self.place_obj_length = np.array(obj["size"])
                self.place_obj_prim = "/World/Objects/" + obj["name"]
                self.place_obj_id = obj['uid']

            self.obj_names.append(obj['name'])

        # add cameras
        width = 640
        height = 480
        position = [1.2, 0., 0.9]
        rotation = [0.65328, 0.2706, 0.2706, 0.65328]
        self.robot.client.AddCamera(self.head_camera, position, rotation, width, height, 18.14756, 20.955, 15.2908, False)
        position = [0.05, 0., 0.]
        rotation = [0.06163, 0.70442, 0.70442, 0.06163]
        self.robot.client.AddCamera(self.right_camera, position, rotation, width, height, 18.14756, 20.955, 15.2908, True)

        self.data_keys = {
            "camera": {
                "camera_prim_list": [self.head_camera, self.right_camera],
                "render_depth": True,
                "render_semantic": True,
            },
            "pose": ["/World/Objects/" + obj_name for obj_name in self.obj_names],
            "joint_position": True,
            "gripper": True,
        }
        self.get_observation_w_keys(self.data_keys) # avoid the first observation is None

    def reset(self):
        """override the default configurations"""
        self.robot.reset()
        self.head_camera = "/World/Sensors/Head_Camera_1"
        self.right_camera = "/panda/panda_hand/Hand_Camera_1"
        self.robot.base_camera = "/World/Top_Camera"

    def get_observation_w_keys(self, data_keys):
        return self.robot.get_observation(data_keys)

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

        response = requests.post("http://0.0.0.0:4500/policy", data=pickle.dumps(data, protocol=4))
        action = pickle.loads(response.content)['action']

        end_time = time.time()  # 记录结束时间
        elapsed_time = (end_time - start_time) * 1000  # 计算运行时间并转换为毫秒
        print(f"Policy executed in {elapsed_time:.3f} ms")

        return action[0]

    def get_policy_obs(self, low_dim=False, step_num=None):
        policy_obs = {}

        # get observation
        observation = self.get_observation_w_keys(self.data_keys)
        def get_rgb_image(observation, camera_prim):
            cam_info = observation["camera"][camera_prim]["camera_info"]
            rgb_image = observation["camera"][camera_prim]["rgb_camera"].reshape(cam_info["height"], cam_info["width"], 4)[:, :, :3]
            return rgb_image
        
        head_rgb_image = get_rgb_image(observation, self.head_camera)
        right_rgb_image = get_rgb_image(observation, self.right_camera)
        
        ee_transform = pose_to_mat(observation['gripper']['right']['position'], observation['gripper']['right']['rotation'])
        # ee_transform = self.robot.get_ee_pose()
        raw_gripper_position = observation['joint']['panda_finger_joint1']
        # raw_gripper_position = self.robot.client.get_joint_positions().states[-1].position

        obj_transform = np.stack([pose_to_mat(observation['pose']["/World/Objects/" + obj_name]['position'], observation['pose']["/World/Objects/" + obj_name]['rotation']) for obj_name in self.obj_names])
        # obj_transform = np.stack([self.robot.get_prim_world_pose("/World/Objects/" + obj_name) for obj_name in self.obj_names])

        if low_dim:
            state_data = np.concatenate([mat_to_pose10d(ee_transform), mat_to_pose10d(obj_transform).flatten(), [raw_gripper_position]])
        else:
            head_color = resize_img(head_rgb_image, 224, 224)
            right_color = resize_img(right_rgb_image, 224, 224)

            head_color = head_color.reshape(3, 224, 224)    # [3, height, width]
            right_color = right_color.reshape(3, 224, 224)    # [3, height, width]
            rgb = np.concatenate([head_color, right_color])
            state_data = np.concatenate([mat_to_pose10d(ee_transform), [raw_gripper_position]])
            policy_obs['rgb'] = rgb

        policy_obs['state'] = state_data
        
        if step_num is not None:
            file_name = f"infer_results/dp_{self.ts_str}"
            os.makedirs(file_name, exist_ok=True)

            # save the image and action to file
            if not low_dim:
                head_img_path = f"infer_results/dp_{self.ts_str}/{step_num}_head.png"
                right_img_path = f"infer_results/dp_{self.ts_str}/{step_num}_right.png"
                Image.fromarray(head_rgb_image).save(head_img_path)
                Image.fromarray(right_rgb_image).save(right_img_path)
            
            self.results['state'].append(state_data.tolist())
            self.results['ee_transform'].append(ee_transform.tolist())
            self.results['obj_transform'].append(obj_transform.tolist())

            with open(f"infer_results/dp_{self.ts_str}/results.json", "w") as f:
                json.dump(
                    self.results,
                    f,
                    indent=4
                )

        return policy_obs, ee_transform, raw_gripper_position
    
    def check_task_completion(self, task_name):
        if task_name == 'lift':
            obj_pose = self.robot.get_prim_world_pose(self.target_obj_prim)
            success = obj_pose[2, 3] > 0.2
        elif task_name == 'recycle_cube':
            def check_pts_in_bbox(point, bbox):
                px, py, pz = point
                bx, by, bz, H, W, D = bbox
                min_x = bx - H / 2
                max_x = bx + H / 2
                min_y = by - W / 2
                max_y = by + W / 2
                min_z = bz - D / 2
                max_z = bz
                return min_x <= px <= max_x and min_y <= py <= max_y and min_z <= pz <= max_z

            obj_pose = self.robot.get_prim_world_pose(self.target_obj_prim)
            obj_xyz = obj_pose[:3, 3]
            target_xyz = self.robot.get_prim_world_pose(self.place_obj_prim)[:3, 3]
            success = check_pts_in_bbox(obj_xyz, list(target_xyz) + [0.04, 0.04, 0.3])
        elif task_name == 'stack_cube':
            def check_pts_in_bbox(point, bbox):
                px, py, pz = point
                bx, by, bz, H, W, D = bbox
                min_x = bx - H / 2
                max_x = bx + H / 2
                min_y = by - W / 2
                max_y = by + W / 2
                min_z = bz 
                max_z = bz + D / 2
                return min_x <= px <= max_x and min_y <= py <= max_y and min_z <= pz <= max_z

            obj_pose = self.robot.get_prim_world_pose(self.target_obj_prim)
            obj_xyz = obj_pose[:3, 3]
            target_xyz = self.robot.get_prim_world_pose(self.place_obj_prim)[:3, 3]
            success = check_pts_in_bbox(obj_xyz, list(target_xyz) + [0.1, 0.1, 0.1])

        elif task_name == 'insert_pen':
            def check_pts_in_bbox(point, bbox):
                px, py, pz = point
                bx, by, bz, H, W, D = bbox
                min_x = bx - H / 2
                max_x = bx + H / 2
                min_y = by - W / 2
                max_y = by + W / 2
                min_z = bz - D / 2
                max_z = bz + D / 2
                return min_x <= px <= max_x and min_y <= py <= max_y and min_z <= pz <= max_z

            obj_pose = self.robot.get_prim_world_pose(self.target_obj_prim)
            obj_xyz = obj_pose[:3, 3]
            target_xyz = self.robot.get_prim_world_pose(self.place_obj_prim)[:3, 3]
            success = check_pts_in_bbox(obj_xyz, list(target_xyz) + [0.1, 0.1, 0.1])
        return success

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

    def run(self, task_folder, task_name, use_recording=False, fps=30):
        import glob
        tasks = glob.glob(task_folder + "/*.json")

        low_dim = False
        obs_horizon = 2
        img_horizon = 1
        action_steps = 2

        task_len_dict = {
            "stack_cube": 150,
            "recycle_cube": 150,
            "insert_pen": 50,
        } 
        for i, task_file in enumerate(tasks):
            print("Start Task:", task_file)

            self.reset()
            
            if task_name == 'stack_cube' or task_name == 'recycle_cube':
                self.robot.client.set_joint_positions([0, -0.569, 0, -2.809, 0, 3.04, 0.741, 0.04, 0.04], False)
                time.sleep(1)

            self.generate_layout(task_file)

            print("Start Policy Execution")

            if use_recording:
                self.start_recording(task_name="[%s]" % (os.path.basename(task_file).split(".")[0]), 
                                     camera_prim_list=
                                     [
                                        self.head_camera,
                                        self.right_camera,
                                    ],fps=fps)  # TODO 录制判断
                            
            for step_num in trange(task_len_dict[task_name]):
                policy_obs, ee_transform, gripper_position = self.get_policy_obs(low_dim, step_num)  # get observations every step

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

                success = self.check_task_completion(task_name)
                if success:
                    print(f"task {i} success, need {step_num} steps")
                    break

            if not success:
                print(f"task {i} fail")

            self.robot.client.stop_recording()
