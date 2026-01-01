# -*- coding: utf-8 -*-
import json
import time
import numpy as np

from planner import Planner, OmniObject

from .base import BaseAgent
from graspapi import GraspApi
from robot import Robot
from robot.utils import get_rotation_matrix_from_quaternion, is_local_axis_facing_world_axis, rotate_180_along_axis
import os


class Agent(BaseAgent):
    def __init__(self, robot: Robot, api: GraspApi, planner: Planner):
        super().__init__(robot)
        self.api = api
        self.planner = planner

    def generate_layout(self, task_file):
        self.task_file = task_file
        task_info = super().generate_layout(task_file)

        N_place = 4
        for obj in task_info["objects"]:
            if "box_place" in obj["name"]:
                pose = self.robot.get_prim_world_pose("/World/Objects/" + obj["name"])
                cx, cy, cz = pose[:3, 3]
                length = obj["size"][0]
                step_length = length / N_place

                place_pose = []
                for i in range(N_place):
                    x = cx - length / 2.0 + (i + 0.5) * step_length
                    y = cy - obj["size"][1] / 4.0
                    z = cz + 0.1
                    prefer_place_direction_wxyz = [0.12383, -0.02889, 0.90963, 0.39548]
                    quat = prefer_place_direction_wxyz
                    place_pose.append([[x, y, z], quat])

            if obj["name"] == task_info["target_object"]:
                self.target_obj_length = np.array(obj["size"])
                self.target_obj_prim = "/World/Objects/" + obj["name"]

        self.place_pose = place_pose[task_info["place_box_id"]]
        self.place_center = place_pose[task_info["place_box_id"]][0]

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

        prefer_grasp_direction_wxyz = [0.27,0.06544,0.8776,0.3906]
        prefer_place_direction_wxyz = [0.12383, -0.02889, 0.90963, 0.39548]

        prefer_rot = get_rotation_matrix_from_quaternion(prefer_grasp_direction_wxyz)
        pose2world = np.eye(4)
        pose2world[:3,:3] = prefer_rot
        pose2cam = self.robot.pose_from_world_to_cam(pose2world)
        approaching_vector = np.array([0, 0, 1]) 
        target_vector = np.dot(pose2cam[:3,:3], approaching_vector)
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
            grasp_pose = self.robot.decode_gripper_pose(grasp_gripper_pose_cam)
            if not is_local_axis_facing_world_axis(grasp_pose, local_axis='y', world_axis='z'):
                grasp_pose = rotate_180_along_axis(grasp_pose, rot_axis='z')

            if self.robot.check_ik(grasp_pose, id=self.arm):
                width = grasp_result["widths"][i] + 0.02  # extent 5mm
                # print("grasp width:", width)
                grasp_commands = self.grasp(
                    target_gripper_pose=grasp_pose,
                    gripper_id=self.arm,
                    use_pre_grasp=True,
                    use_pick_up=True,
                    grasp_width=width,
                )
                grasp_commands_proposal.append(grasp_commands)
        print(
            "Plan Grasping >>>  %d/%d grasp proposal can pass IK"
            % (len(grasp_commands_proposal), grasp_result["pose"].shape[0]),
            "  obj_prim:" + self.robot.target_object,
        )
        return grasp_commands_proposal
    def plan_fixed_placing(self):
            xyz, quat = self.place_pose

            target_gripper_pose = np.eye(4)
            target_gripper_pose[:3,3] = xyz
            target_gripper_pose[:3,:3] = get_rotation_matrix_from_quaternion(quat)

            place_commands = self.planner.place(
                    target_gripper_pose=target_gripper_pose,
                    gripper_id=self.arm,
                    current_gripper_pose=None,
                    gripper2part=None

                )
            return place_commands

    def check_task_completion(self):
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
        target_xyz = self.place_center
        success = check_pts_in_bbox(obj_xyz, target_xyz + [0.13, 0.19, 0.6])
        return success

    def plan_placing(self):
        xyz, quat = self.place_pose

        obj_pose = self.robot.get_prim_world_pose(self.target_obj_prim)

        active_obj = OmniObject("active_obj", None, type="Active")
        active_obj.set_pose(pose=obj_pose, length=self.target_obj_length)
        active_obj.set_part(xyz=[0, 0, 0], direction=[0, 0, 1])

        place_pose = np.eye(4)
        place_pose[:3, 3] = xyz
        passive_obj = OmniObject("passive_obj", None, type="Plane")
        passive_obj.set_pose(pose=place_pose, length=np.array([0.1, 0.1, 0.1]))
        passive_obj.set_part(xyz=[0, 0, 0], direction=[0, 0, 1])

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

    def run(self, task_folder, use_recording):
        import glob
        tasks = glob.glob(task_folder + "/*.json")
        for task_file in tasks:
            print("Start Task:", task_file)
            self.reset()

            self.robot.client.set_joint_positions(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.4, 
                    0,
                    0.1915680953198153,
                    0,
                    -0.6175654402640774,
                    0,0,
                    0,
                     -0.9721613348200921,
                    0,
                    -1.0487244731860523,
                    0,
                    1.5700000000004277,
                    0,
                    0.030076572171027695,
                    0,0,0,0,0,0,0,0,0,0
                ],
                False
            )
            self.generate_layout(task_file)
            grasp_commands_proposal = self.plan_grasping()
            if use_recording:
                self.start_recording(task_name="[%s]" % (os.path.basename(task_file).split(".")[0]), 
                                    camera_prim_list=[
                                    "/World/Raise_A2/base_link/Head_Camera",
                                    "/World/Raise_A2/right_finger_base/Right_Camera",
                                    "/World/Top_Camera_1",
                                    ])  # TODO 录制判断
            if grasp_commands_proposal is None or len(grasp_commands_proposal) == 0:
                print("Failed of grasping.")
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
            # place_commands = self.plan_placing()
            # if place_commands is None or not self.execute(place_commands):
            #     print("Failed of placement.")
            if not self.execute(self.plan_fixed_placing()):
                print('>>>> Failed of fixed placement again.')
                self.stop_recording(False)
                continue
            self.robot.client.stop_recording()
            success = self.check_task_completion()
            if success:
                print(">>>>>>>>>>>>>>>> Success!!!!!!!!!!!!!!!!")
            self.robot.client.SendTaskStatus(success)
        return True
