# 发送指令
# 001. 拍照
# 002. 左手/右手移动指定位姿
# 003. 全身关节移动到指定角度
# 004. 获取夹爪的位姿
# 005. 获取任意物体的位姿
# 006. 添加物体
import numpy as np
import time

from . import Robot
from .isaac_sim.client import Rpc_Client
from .utils import (
    get_rotation_matrix_from_quaternion,
    matrix_to_euler_angles,
    get_quaternion_from_euler,
    get_quaternion_from_rotation_matrix,
)


class IsaacSimRpcRobot(Robot):
    def __init__(self, robot_cfg, robot_usd, scene_usd,client_host="localhost:50051"):
        self.client = Rpc_Client(client_host)
        self.client.InitRobot(robot_cfg, robot_usd, scene_usd)
        self.cam_info = None
        self.setup()

    def reset(self):
        self.target_object = None
        self.client.reset()
        time.sleep(1)
        self.open_gripper(id='right')
        pass

    def setup(self):
        self.target_object = None

        # set robot init state

    def get_observation(self, data_keys):
        """
        Example
            data_keys = {
                'camera': {
                    'camera_prim_list': [
                        '/World/Raise_A2_W_T1/head_link/D455_Solid/TestCameraDepth'
                    ],
                    'render_depth': True,
                    'render_semantic': True
                },
                'pose': [
                    '/World/Raise_A2_W_T1/head_link/D455_Solid/TestCameraDepth'
                ],
                'joint_position': True,
                'gripper': True
            }
        """

        observation = {}
        observation = self.client.get_observation(data_keys)

        if "camera" in data_keys:
            render_depth = data_keys["camera"]["render_depth"] if "render_depth" in data_keys["camera"] else False
            render_semantic = (
                data_keys["camera"]["render_semantic"] if "render_semantic" in data_keys["camera"] else False
            )

            cam_data = {}
            for cam_prim in data_keys["camera"]["camera_prim_list"]:
                cam_data[cam_prim] = {}
                response = self.client.capture_frame(camera_prim_path=cam_prim)
                # cam_info
                cam_info = {
                    "W": response.color_info.width,
                    "H": response.color_info.height,
                    "K": np.array(
                        [
                            [response.color_info.fx, 0, response.color_info.ppx],
                            [0, response.color_info.fy, response.color_info.ppy],
                            [0, 0, 1],
                        ]
                    ),
                    "scale": 1,
                }
                cam_data[cam_prim]["cam_info"] = cam_info
                # c2w
                c2w = self.get_prim_world_pose(cam_prim, camera=True)
                cam_data[cam_prim]['c2w'] = c2w
                # rgb
                rgb = np.frombuffer(response.color_image.data, dtype=np.uint8).reshape(cam_info["H"], cam_info["W"], 4)[
                    :, :, :3
                ]
                cam_data[cam_prim]["image"] = rgb
                # depth
                if render_depth:
                    depth = np.frombuffer(response.depth_image.data, dtype=np.float32).reshape(
                        cam_info["H"], cam_info["W"]
                    )
                    cam_data[cam_prim]["depth"] = depth

                # semantic
                if render_semantic:
                    response = self.client.capture_semantic_frame(camera_prim_path=cam_prim)
                    prim_id = {}
                    for label in response.label_dict:
                        name, id = label.label_name, label.label_id
                        prim = '/World/Objects/' +name
                        prim_id[prim] = id
                    mask = np.frombuffer(response.semantic_mask.data, dtype=np.int32).reshape(cam_info['H'], cam_info['W'])
                    cam_data[cam_prim]['semantic'] = {
                        'prim_id': prim_id,
                        'mask': mask
                    }

            observation["camera"] = cam_data

        if "pose" in data_keys:
            pose_data = {}
            for obj_prim in data_keys["pose"]:
                pose_data[obj_prim] = self.get_prim_world_pose(prim_path=obj_prim)
            observation["pose"] = pose_data

        if "joint_position" in data_keys:
            # TODO 是否要区分双臂？
            joint_position = self.client.get_joint_positions()
            observation["joint_position"] = joint_position

        if "gripper" in data_keys:
            gripper_state = {}
            gripper_state["left"] = self.client.get_gripper_state(is_right=False)
            gripper_state["right"] = self.client.get_gripper_state(is_right=True)
            observation["gripper"] = gripper_state

        return observation

    def open_gripper(self, id="left", width=0.1):
        is_Right = True if id == "right" else False
        if width is None:
            width = 0.1
        self.client.set_gripper_state(gripper_command="open", is_right=is_Right, opened_width=width)
        
        self.client.DetachObj()

    def close_gripper(self, id="left", force=50):
        is_Right = True if id == "right" else False
        self.client.set_gripper_state(gripper_command="close", is_right=is_Right, opened_width=0.00)
        
        if self.target_object is not None:
            # self.client.DetachObj()
            self.client.AttachObj(prim_paths=['/World/Objects/'+self.target_object])
            

    def move(self, content):
        """
        type: str, 'matrix' or 'joint'
            'pose': np.array, 4x4
            'joint': np.array, 1xN
        """
        type = content["type"]
        if type == "matrix":
            if isinstance(content["matrix"], list):
                content["matrix"] = np.array(content["matrix"])
            R, T = content["matrix"][:3, :3], content["matrix"][:3, 3]
            quat_wxyz = get_quaternion_from_euler(matrix_to_euler_angles(R), order="ZYX")

            if "trajectory_type" in content and content["trajectory_type"] == "Simple":
                is_backend = True
            else:
                is_backend = False
            state = (
                self.client.moveto(
                    target_position=T,
                    target_quaternion=quat_wxyz,
                    arm_name="left",
                    is_backend=is_backend,
                ).errmsg
                == "True"
            )
            # if is_backend == True:
            #     time.sleep(1)
        elif type == "quat":

            if "trajectory_type" in content and content["trajectory_type"] == "Simple":
                is_backend = True
            else:
                is_backend = False
            state = (
                self.client.moveto(
                    target_position=content["xyz"],
                    target_quaternion=content["quat"],
                    arm_name="left",
                    is_backend=is_backend,
                ).errmsg
                == "True"
            )

            # if is_backend == True:
            #     time.sleep(4)
        elif type == "joint":
            state = self.client.set_joint_positions(content["position"])
        else:
            raise NotImplementedError

        return state

    def get_prim_world_pose(self, prim_path, camera=False):
        rotation_x_180 = np.array([[1.0, 0.0, 0.0, 0], [0.0, -1.0, 0.0, 0], [0.0, 0.0, -1.0, 0], [0, 0, 0, 1]])
        response = self.client.get_object_pose(prim_path=prim_path)
        x, y, z = (
            response.object_pose.position.x,
            response.object_pose.position.y,
            response.object_pose.position.z,
        )
        quat_wxyz = np.array(
            [
                response.object_pose.rpy.rw,
                response.object_pose.rpy.rx,
                response.object_pose.rpy.ry,
                response.object_pose.rpy.rz,
            ]
        )
        rot_mat = get_rotation_matrix_from_quaternion(quat_wxyz)

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = np.array([x, y, z])

        if camera:
            pose = pose @ rotation_x_180
        return pose

    def pose_from_cam_to_robot(self, pose2cam):
        """transform pose from cam-coordinate to robot-coordinate"""
        cam2world = self.get_prim_world_pose(prim_path=self.base_camera, camera=True)
        pose2world = cam2world @ pose2cam
        return pose2world

    def pose_from_world_to_cam(self, pose2world):
        """transform pose from world-coordinate to cam-coordinate"""
        cam2world = self.get_prim_world_pose(prim_path=self.base_camera, camera=True)
        pose2cam = np.linalg.inv(cam2world) @ pose2world
        return pose2cam

    def decode_gripper_pose(self, gripper_pose):
        """Decode gripper-pose at cam-coordinate to end-pose at robot-coordinate"""
        gripper_pose = self.pose_from_cam_to_robot(gripper_pose)
        angle = 0  # np.pi / 2
        rot_z = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        flange2gripper = np.eye(4)
        flange2gripper[:3, :3] = rot_z
        flange2gripper[2, 3] = -0.015
        return gripper_pose @ flange2gripper

    def get_ee_pose(self, id="right", return_matrix = True, **kwargs):
        state = self.client.GetEEPose(is_right=id == "right")
        xyz = np.array(
            [
                state.ee_pose.position.x,
                state.ee_pose.position.y,
                state.ee_pose.position.z,
            ]
        )
        quat = np.array(
            [
                state.ee_pose.rpy.rw,
                state.ee_pose.rpy.rx,
                state.ee_pose.rpy.ry,
                state.ee_pose.rpy.rz,
            ]
        )
        pose = np.eye(4)
        pose[:3, 3] = xyz
        pose[:3, :3] = get_rotation_matrix_from_quaternion(quat)
        print(xyz, quat)
        if return_matrix:
            return pose
        else:
            return xyz, quat

    def check_ik(self, pose, id="right"):
        xyz, quat = pose[:3, 3], get_quaternion_from_rotation_matrix(pose[:3, :3])
        state = self.client.GetIKStatus(target_position=xyz, target_rotation=quat, is_right=id == "right")
        return state.isSuccess
