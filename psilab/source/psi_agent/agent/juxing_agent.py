# -*- coding: utf-8 -*-
import json
import time
import numpy as np

from planner import Planner, OmniObject

from .base import BaseAgent
from graspapi import GraspApi
from robot import Robot
from robot.utils import get_rotation_matrix_from_quaternion, is_local_axis_facing_world_axis, rotate_180_along_axis, get_quaternion_from_euler,get_xyz_euler_from_quaternion
import os


class Agent(BaseAgent):
    def __init__(self, robot: Robot, planner: Planner):
        super().__init__(robot)
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
                    z = cz + 0.2
                    prefer_place_direction_wxyz = [0,0,-1,0]
                    quat = prefer_place_direction_wxyz
                    place_pose.append([[x, y, z], quat])

            if obj["name"] == task_info["target_object"]:
                self.target_obj_length = np.array(obj["size"])
                self.target_obj_prim = "/World/Objects/" + obj["name"]

        self.place_pose = place_pose[task_info["place_box_id"]]
        self.place_center = place_pose[task_info["place_box_id"]][0]

    def plan_grasping(self):
        grasp_obj_prim = '/World/Objects/'+self.robot.target_object
        rotation_x_180 = np.array(
            [
                [1.0, 0.0, 0.0, 0],
                [0.0, -1.0, 0.0, 0],
                [0.0, 0.0, -1.0, 0],
                [0, 0, 0, 1],
            ]
        )
        grasp_commands_proposal = []
        
        grasp_pose = self.robot.get_prim_world_pose(grasp_obj_prim)@ rotation_x_180
        grasp_pose[2, 3] = grasp_pose[2, 3] + 0.125
        quat_wxyz = [0, 0,-1, 0]
        x,y,z = grasp_pose[:3, 3]
        rot_mat = get_rotation_matrix_from_quaternion(quat_wxyz)

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = np.array([x, y, z])
        
        grasp_commands = self.grasp(
            target_gripper_pose=pose,
            gripper_id=self.arm,
            use_pre_grasp=True,
            use_pick_up=False,
            grasp_width=0.1
        )
        grasp_commands_proposal.append(grasp_commands)
        return grasp_commands_proposal
    def plan_fixed_placing(self):
            xyz, quat = self.place_pose
            print(self.place_pose)
            target_gripper_pose = np.eye(4)
            target_gripper_pose[:3,3] = xyz
            target_gripper_pose[:3,:3] = get_rotation_matrix_from_quaternion(quat)

            place_commands = self.planner.place(
                    target_gripper_pose=target_gripper_pose,
                    gripper_id=self.arm,
                    current_gripper_pose=None,
                    gripper2part=None,
                    use_pre_place= False,
                    use_pick_up=False

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
            self.robot.open_gripper()
            self.robot.close_gripper()
            self.generate_layout(task_file)
            grasp_commands_proposal = self.plan_grasping()
            print(grasp_commands_proposal)
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
                    0.4,
                    -1.765740591,
                    1.765740591,
                    -0.6194321746675753,
                    -0.6194321746675753,
                    0,
                    0,
                    -0.9721613348200921,
                    -0.9721613348200921,
                    -1.5700000000004277,
                    -1.5700000000004277,
                    1.5700000000004277,
                    -1.5700000000004277,
                    0.030076572171027695,
                    0.030076572171027695
                ],False
            )
            position, rotation = self.robot.get_ee_pose(return_matrix=False)
            angle = get_xyz_euler_from_quaternion(rotation)
            target_angle = [angle[0],
                            angle[1]+np.random.uniform(-0.74532925,0.74532925),
                            angle[2]+np.random.uniform(-0.74532925,0.74532925)]
            target_rotation = get_quaternion_from_euler(target_angle)
            target_position = [position[0]+np.random.uniform(-0.15,0.15),
                               position[1]+np.random.uniform(-0.15,0.15),
                               position[2]+np.random.uniform(-0.15,0.15)]
            pose = target_position, target_rotation
            self.robot.client.SetTrajectoryList([pose])
            time.sleep(1)
            if use_recording:
                self.start_recording(task_name="[%s]" % (os.path.basename(task_file).split(".")[0]), 
                                    camera_prim_list=[
                                         "/World/Top_Camera_1",
                                         "/Raise_A2_Surface/base/Head_Camera_01",
                                         "/Raise_A2_Surface/right_arm_Link7/Xform/surface_gripper/Right_Camera",
                                         "/Raise_A2_Surface/right_arm_Link7/Xform/surface_gripper/Right_Camera_fisheye",

                                     ]) # TODO 录制判断
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
            # self.robot.client.moveto(init_position, init_rotation, "left", False)
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
                    0.4,
                    -1.765740591,
                    1.765740591,
                    -0.6194321746675753,
                    -0.6194321746675753,
                    0,
                    0,
                    -0.9721613348200921,
                    -0.9721613348200921,
                    -1.5700000000004277,
                    -1.5700000000004277,
                    1.5700000000004277,
                    -1.5700000000004277,
                    0.030076572171027695,
                    0.030076572171027695
                ],True
            )
            time.sleep(1)
            self.robot.client.stop_recording()
            success = self.check_task_completion()
            if success:
                print(">>>>>>>>>>>>>>>> Success!!!!!!!!!!!!!!!!")
            self.robot.client.SendTaskStatus(success)
        return True
