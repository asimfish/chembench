# -*- coding: utf-8 -*-
import numpy as np

from . import Planner

from .utils import (
    OmniObject,
    transform_coordinates_3d,
    calculate_rotation_matrix,
    rotate_around_axis,
)

from scipy.spatial.transform import Rotation as R, Slerp


class ManipulationPlanner(Planner):
    def __init__(self):
        pass

    def set_active_object(self, obj, xyz, direction, relative=True):
        self.active_object = format_object(obj, xyz, direction, type="active", relative=relative)

    def set_passive_object(self, obj, xyz, direction, relative=True):
        self.passive_object = format_object(obj, xyz, direction, type="passive", relative=relative)

    def deduce_target_pose(self, active_obj, passive_obj, N=1):
        self.active_object = active_obj.format_object()
        self.passive_object = passive_obj.format_object()

        R = calculate_rotation_matrix(self.active_object["direction"], self.passive_object["direction"])
        T = self.passive_object["xyz_start_world"] - R @ self.active_object["xyz_start_world"]

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = T

        current_obj_pose = self.active_object["pose"]

        target_obj_pose = transform_matrix @ current_obj_pose

        poses = []
        for angle in [i * 360 / N for i in range(N)]:
            pose_rotated = rotate_around_axis(
                target_obj_pose,
                self.passive_object["xyz_start_world"],
                self.passive_object["direction"],
                angle,
            )
            poses.append(pose_rotated)
        return np.stack(poses)

    def plan_trajectory(
        self,
        active_obj,
        target_obj_pose,
        gripper_pose,
        task,
        gripper_id=None,
        ik_checker=None,
    ):
        """
        gripper_pose: 开始时刻gripper在world-coord下的pose (一般为grasping完成后, 即此时gripper和目标物体呈刚性连接)
        task: execute action
        """
        current_obj_pose = active_obj.obj_pose
        obj2part = active_obj.obj2part

        # 假设gripper与物体呈刚体连接
        gripper2obj = np.linalg.inv(current_obj_pose) @ gripper_pose
        gripper2part = obj2part @ gripper2obj

        if len(target_obj_pose.shape) == 2:
            return_single = True
            target_obj_pose = target_obj_pose[np.newaxis, ...]
        else:
            return_single = False

        target_gripper_pose_proposal = target_obj_pose @ gripper2obj[np.newaxis, ...]

        commands_proposal = []
        for target_gripper_pose in target_gripper_pose_proposal:
            if ik_checker is not None:
                if not ik_checker(target_gripper_pose):
                    continue
            if hasattr(self, task):
                commands = getattr(self, task)(
                    current_gripper_pose=gripper_pose,
                    target_gripper_pose=target_gripper_pose,
                    gripper2part=gripper2part,
                    gripper_id=gripper_id,
                )
                commands_proposal.append(commands)
            else:
                raise ValueError(f"Task '{task}' is not a valid method")
        if return_single:
            commands_proposal = commands_proposal[0]
        return commands_proposal

    def grasp(
        self,
        target_gripper_pose,
        current_gripper_pose=None,
        gripper2part=None,
        gripper_id=None,
        use_pre_grasp=False,
        use_pick_up=False,
    ):
        gripper_id = "left" if gripper_id is None else gripper_id

        pick_up_pose = np.copy(target_gripper_pose)
        pick_up_pose[2, 3] = pick_up_pose[2, 3] + 0.10  # 抓到物体后，先往上方抬10cm

        commands = []

        if use_pre_grasp:
            pre_pose = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, -0.11], [0, 0, 0, 1]])
            pre_grasp_pose = target_gripper_pose @ pre_pose
            commands.append(
                {
                    "action": "move",
                    "content": {
                        "matrix": pre_grasp_pose,
                        "type": "matrix",
                        "comment": "pre_grasp",
                        "trajectory_type": "ObsAvoid",
                    },
                }
            )
        grasp_trajectory_type = "Simple" if use_pre_grasp else "ObsAvoid"
        commands.append(
            {
                "action": "move",
                "content": {
                    "matrix": target_gripper_pose,
                    "type": "matrix",
                    "comment": "grasp",
                    "trajectory_type": grasp_trajectory_type,
                },
            }
        )
        commands.append({"action": "close_gripper", "content": {"gripper": gripper_id, "force": 50}})

        if use_pick_up:
            commands.append(
                {
                    "action": "move",
                    "content": {
                        "matrix": pick_up_pose,
                        "type": "matrix",
                        "comment": "pick_up",
                        "trajectory_type": "Simple",
                    },
                }
            )
        return commands

    def place(self, target_gripper_pose, current_gripper_pose, gripper2part, gripper_id=None, use_pre_place=True, use_pick_up=False):
        gripper_id = "left" if gripper_id is None else gripper_id

        commands = []
        pre_place_pose = np.copy(target_gripper_pose)
        pre_place_pose[2, 3] += 0.15
        move_type = "ObsAvoid"
        if use_pre_place:
            commands.append(
                {
                    "action": "move",
                    "content": {
                        "matrix": pre_place_pose,
                        "type": "matrix",
                        "comment": "pre_place",
                        "trajectory_type": "ObsAvoid",
                    },
                },
            )
            move_type = "Simple"
        commands.append(
            {
                "action": "move",
                "content": {
                    "matrix": target_gripper_pose,
                    "type": "matrix",
                    "comment": "place",
                    "trajectory_type": move_type,
                },
            }

        )
        commands.append(
            {
                "action": "open_gripper",
                "content": {
                    "gripper": gripper_id,
                },
            }

        )
        if use_pick_up:
            commands.append(
                {
                    "action": "move",
                    "content": {
                        "matrix": pre_place_pose,
                        "type": "matrix",
                        "comment": "place",
                        "trajectory_type": "Simple",
                    },
                }

            )
        
        return commands

    def pour(self, target_gripper_pose, current_gripper_pose, gripper2part, gripper_id=None):
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

        target_part_pose = target_gripper_pose @ np.linalg.inv(gripper2part)
        current_part_pose = current_gripper_pose @ np.linalg.inv(gripper2part)

        commands = []
        rotations = interpolate_rotation_matrices(current_part_pose[:3, :3], target_part_pose[:3, :3], 5)
        for i, rotation in enumerate(rotations):
            target_part_pose_step = np.copy(target_part_pose)
            target_part_pose_step[:3, :3] = rotation
            target_gripper_pose_step = target_part_pose_step @ gripper2part

            commands.append(
                {
                    "action": "move",
                    "content": {
                        "matrix": target_gripper_pose_step,
                        "type": "matrix",
                        "comment": "pour_sub_rotate_%d" % i,
                        "trajectory_type": "Simple",
                    },
                }
            )
        return commands


def format_object(obj: OmniObject, xyz, direction, type, relative=True):
    if not isinstance(xyz, np.ndarray):
        xyz = np.array(xyz)

    if obj.type == "plane":
        xyz = np.array([0, 0, 0])
    elif relative:
        xyz = xyz * obj.obj_length / 2.0

    direction = direction / np.linalg.norm(direction) * 0.05

    if type == "active":
        xyz_start = xyz
        xyz_end = xyz_start + direction
    elif type == "passive":
        xyz_end = xyz
        xyz_start = xyz_end - direction

    arrow_in_obj = np.array([xyz_start, xyz_end]).transpose(1, 0)
    arrow_in_world = transform_coordinates_3d(arrow_in_obj, obj.obj_pose).transpose(1, 0)

    xyz_start_world, xyz_end_world = arrow_in_world

    direction_world = xyz_end_world - xyz_start_world
    direction_world = direction_world / np.linalg.norm(direction_world)

    part2obj = np.eye(4)
    part2obj[:3, 3] = xyz_start
    obj2part = np.linalg.inv(part2obj)

    object_world = {
        "pose": obj.obj_pose,
        "length": obj.obj_length,
        "xyz": xyz_start_world,
        "direction": direction_world,
        "obj2part": obj2part,
    }
    return object_world
