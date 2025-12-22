# -*- coding: utf-8 -*-
import json
import time
import numpy as np

from planner import Planner, OmniObject

from .base import BaseAgent
from graspapi import GraspApi
from robot import Robot
from robot.utils import is_local_axis_facing_world_axis, rotate_180_along_axis
import os

from scipy.spatial.transform import Rotation
from robot.utils import get_rotation_matrix_from_quaternion, is_local_axis_facing_world_axis, rotate_180_along_axis, get_rotation_matrix_from_euler, get_quaternion_from_rotation_matrix

class FrankaAgent(BaseAgent):
    def __init__(self, robot: Robot, api: GraspApi, planner: Planner):
        super().__init__(robot)
        self.api = api
        self.planner = planner

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

        for object_info in task_info["objects"]:
            self.add_object(object_info)
        time.sleep(1)

        self.robot.target_object = task_info["target_object"]
        self.arm = task_info["arm"]

        for obj in task_info["objects"]:
            if obj["name"] == task_info["target_object"]:
                self.target_obj_length = np.array(obj["size"])
                self.target_obj_prim = "/World/Objects/" + obj["name"]
                self.target_obj_uid = obj['uid']

            if obj["name"] != task_info["target_object"]:
                self.place_obj_length = np.array(obj["size"])
                self.place_obj_prim = "/World/Objects/" + obj["name"]
                self.place_obj_id = obj['uid']

        width = 640
        height = 480

        position = [1, 0., 0.3]
        rotation = [0.56099, 0.43046, 0.43046, 0.56099]
        self.robot.client.AddCamera("/World/Sensors/Head_Camera", position, rotation, width, height, 18.14756, 20.955, 15.2908, False)
        position = [0.05, 0., 0.]
        rotation = [0.06163, 0.70442, 0.70442, 0.06163]
        self.robot.client.AddCamera("/panda/panda_hand/Hand_Camera_1", position, rotation, width, height, 18.14756, 20.955, 15.2908, True)

    def plan_grasping(self):
        self.get_observation()
        masks, prim_id = (
            self.observation["semantic"]["mask"],
            self.observation['semantic']['prim_id']
        )
        grasp_obj_prim = '/World/Objects/'+self.robot.target_object
        if grasp_obj_prim not in prim_id:
            print('[ERROR] Find object: %s failed!!'%self.robot.target_object)
            return
        target_mask = masks==prim_id[grasp_obj_prim]
        self.observation['target_mask'] = target_mask

        prefer_grasp_direction_wxyz = [0.27, 0.06544, 0.8776, 0.3906]
        prefer_place_direction_wxyz = [0.12383, -0.02889, 0.90963, 0.39548]

        prefer_rot = get_rotation_matrix_from_quaternion(prefer_grasp_direction_wxyz)
        pose2world = np.eye(4)
        pose2world[:3,:3] = prefer_rot
        pose2cam = self.robot.pose_from_world_to_cam(pose2world)
        approaching_vector = np.array([0, 0, 1]) 
        target_vector = np.dot(pose2cam[:3, :3], approaching_vector)
        self.observation['target_vector'] = target_vector

        grasp_result = self.api.run_grasping(
            self.observation,
            grasp_proposal=None,
            grasp_pts_num=60,
            repeat_num=100,
            T0=0.7,
        )
        if grasp_result is None or len(grasp_result) == 0:
            print("No grasp proposal")
            # import ipdb;ipdb.set_trace()
            return None

        grasp_commands_proposal = []
        for i, grasp_gripper_pose_cam in enumerate(grasp_result["pose"]):
            # import ipdb;ipdb.set_trace()
            grasp_pose = self.robot.decode_gripper_pose(grasp_gripper_pose_cam)
            if not is_local_axis_facing_world_axis(grasp_pose, local_axis='y', world_axis='z'):
                grasp_pose = rotate_180_along_axis(grasp_pose, rot_axis='z')

            if self.robot.check_ik(grasp_pose, id=self.arm):
                width = grasp_result["widths"][i] + 0.005  # extent 5mm
                grasp_commands = self.grasp(
                    target_gripper_pose=grasp_pose,
                    gripper_id=self.arm,
                    use_pre_grasp=False,
                    use_pick_up=True,
                    # grasp_width=width,
                    grasp_width=0.1,
                )
                grasp_commands_proposal.append(grasp_commands)
        print(
            "Plan Grasping >>>  %d/%d grasp proposal can pass IK"
            % (len(grasp_commands_proposal), grasp_result["pose"].shape[0]),
            "  obj_prim:" + self.robot.target_object,
        )
        return grasp_commands_proposal
    
    def plan_fixed_grasping(self, add_noise=False):
        # hard-code grasp pose
        obj_pose = self.robot.get_prim_world_pose(self.target_obj_prim)
        gripper_pose = self.robot.get_ee_pose(id=self.arm)
        
        target_grasp_pose = gripper_pose.copy()
        topdown_grasp_rotation_matrix_1 = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [ 0, 0, -1]
        ])
        topdown_grasp_rotation_matrix_2 = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [ 0, 0, -1]
        ])
        obj_euler = Rotation.from_matrix(obj_pose[:3, :3]).as_euler('xyz')
        grasp_euler = np.array([np.pi, 0, obj_euler[2] % np.pi])
        grasp_rotation = Rotation.from_euler('xyz', grasp_euler).as_matrix()
        # grasp_rotation[:2] += np.random.uniform(-np.pi/20, np.pi/20, (2,))
        # grasp_rotation[2] += np.random.uniform(-np.pi/10, np.pi/10)
        target_grasp_pose[:3, :3] = grasp_rotation
        target_grasp_pose[:3, 3] = obj_pose[:3, 3]
        target_grasp_pose[2, 3] += 0.005

        if add_noise:
            target_grasp_pose[:3, 3] += np.random.uniform(-0.008, 0.008, (3,))

        grasp_commands = self.grasp(
            target_gripper_pose=target_grasp_pose,
            gripper_id=self.arm,
            use_pre_grasp=False,
            use_pick_up=True,
            grasp_width=0.1,
        )
        grasp_commands_proposal = [grasp_commands]
        
        return grasp_commands_proposal

    def plan_placing(self):
        obj_pose = self.robot.get_prim_world_pose(self.target_obj_prim)

        active_obj = OmniObject("active_obj", None, type="Active")
        active_obj.set_pose(pose=obj_pose, length=self.target_obj_length)

        target_obj_info = self.search_obj_info(self.target_obj_uid)
        for part in target_obj_info['parts']:
            if part['action'] == 'place' and part['type'] == 'active':
                target_obj_part_info = part['relative_pose']
                target_obj_target_direction = self.compute_part_direction(target_obj_part_info[3:])
                break
        active_obj.set_part(xyz=target_obj_part_info[:3], direction=target_obj_target_direction)

        place_pose = self.robot.get_prim_world_pose(self.place_obj_prim)
        self.place_center = place_pose[:3, 3]
        passive_obj = OmniObject("passive_obj", None, type="Plane")
        passive_obj.set_pose(pose=place_pose, length=self.target_obj_length)

        place_obj_info = self.search_obj_info(self.place_obj_id)
        for part in place_obj_info['parts']:
            if part['action'] == 'place' and part['type'] == 'passive':
                place_obj_part_info = part['relative_pose']
                place_obj_target_direction = self.compute_part_direction(place_obj_part_info[3:])
                break
        passive_obj.set_part(xyz=place_obj_part_info[:3], direction=place_obj_target_direction)

        gripper_pose = self.robot.get_ee_pose(id=self.arm)
        target_obj_pose = self.planner.deduce_target_pose(active_obj, passive_obj, N=18)

        place_commands = self.planner.plan_trajectory(
            active_obj,
            target_obj_pose=target_obj_pose,
            gripper_pose=gripper_pose,
            task="place",
            gripper_id=self.arm,
            ik_checker=self.robot.check_ik,
        )

        print(" %d/%d trajectories can successfully pass IK." % (len(place_commands), 18))
        if len(place_commands) == 0:
            return None
        
        return place_commands[0]

    def plan_fixed_placing(self, add_noise):
        place_pose = self.robot.get_prim_world_pose(self.place_obj_prim)
        xyz = place_pose[:3, 3]
        xyz[2] += np.random.uniform(0.11, 0.13)

        topdown_place_rotation_matrix_2 = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [ 0, 0, -1]
        ])
        target_gripper_pose = np.eye(4)
        target_gripper_pose[:3, 3] = xyz   
        if add_noise:
            target_gripper_pose[:3, 3] += np.random.uniform(-0.005, 0.005, (3,))
        target_gripper_pose[:3, :3] = topdown_place_rotation_matrix_2
        self.place_center = place_pose[:3, 3]

        place_commands = self.planner.place(
                target_gripper_pose=target_gripper_pose,
                gripper_id=self.arm,
                current_gripper_pose=None,
                gripper2part=None

            )
        return place_commands

    def check_task_completion(self):
        obj_pose = self.robot.get_prim_world_pose(self.target_obj_prim)
        success = obj_pose[3, 3] > 0.04
        return success

    def search_obj_info(self, obj_uid):
        import glob
        obj_info_files = glob.glob(self.obj_info_folder + "/*.json")
        for obj_info_file in obj_info_files:
            if obj_uid in obj_info_file:
                with open(obj_info_file, "r") as f:
                    obj_info = json.load(f)
                    return obj_info
    
    def compute_part_direction(self, euler):
        source_matrix = np.eye(4)
        source_matrix[:3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
        target_matrix = np.eye(4)
        target_matrix[2, 3] = 1
        source_matrix = np.dot(source_matrix, target_matrix)
        return source_matrix[:3, 3]    

    def run(self, task_folder, use_recording, fps=10, obj_info_folder=None, fixed_grasp_pose=False, fixed_place_pose=False, add_noise=False):
        self.obj_info_folder = obj_info_folder

        import glob
        tasks = glob.glob(task_folder + "/*.json")
        for task_file in tasks:
            print("Start Task:", task_file)
            self.reset()
            self.robot.client.set_joint_positions([0, -0.569, 0, -2.809, 0, 3.04, 0.741, 0, 0], False)
            time.sleep(1)
            self.generate_layout(task_file)

            grasp_commands_proposal = self.plan_fixed_grasping(add_noise) if fixed_grasp_pose else self.plan_grasping()
            
            if use_recording:
                self.start_recording(task_name="[%s]" % (os.path.basename(task_file).split(".")[0]), 
                                     camera_prim_list=
                                     [
                                        "/World/Sensors/Head_Camera",
                                        "/panda/panda_hand/Hand_Camera_1",
                                    ],fps=fps)  # TODO 录制判断

            if grasp_commands_proposal is None or len(grasp_commands_proposal) == 0:
                print("Failed of generating grasp pose.")
                self.stop_recording(False)
                continue

            grasp_succ = False
            for grasp_commands in grasp_commands_proposal[: min(3, len(grasp_commands_proposal))]:
                grasp_succ = self.execute(grasp_commands)
                if grasp_succ:
                    break

            if not grasp_succ:
                print("Failed of grasping.")
                self.stop_recording(False)
                continue

            self.robot.client.stop_recording()

            success = self.check_task_completion()
            if success:
                print(">>>>>>>>>>>>>>>> Success!!!!!!!!!!!!!!!!")
            
            self.robot.client.SendTaskStatus(success)

        return True
