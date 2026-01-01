# 发送指令
# 001. 拍照
# 002. 左手/右手移动指定位姿
# 003. 全身关节移动到指定角度
# 004. 获取夹爪的位姿
# 005. 获取任意物体的位姿
# 006. 添加物体
import numpy as np
import time

from scipy.spatial.transform import Rotation
from . import Robot
from .isaac_sim.client import Rpc_Client
from .utils import (
    get_rotation_matrix_from_quaternion,
    matrix_to_euler_angles,
    get_quaternion_from_euler,
    get_quaternion_from_rotation_matrix,
)

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def pose10d_to_mat(d10):
    pos = d10[..., :3]
    d6 = d10[..., 3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1] + (4, 4), dtype=d10.dtype)
    out[...,:3, :3] = rotmat
    out[...,:3, 3] = pos
    out[...,3, 3] = 1
    return out

class IsaacSimRpcFranka(Robot):
    def __init__(self, scene_usd="Pick_Place_Franka_Yellow_Table.usd", client_host="localhost:50051"):
        self.client = Rpc_Client(client_host)
        self.client.InitRobot(robot_cfg="Franka.json", robot_usd="Franka/franka.usd", scene_usd=scene_usd)
        self.cam_info = None
        self.robot_gripper_2_grasp_gripper = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0]
            ])
        self.setup()

    def reset(self):
        self.target_object = None
        self.client.reset()
        time.sleep(0.5)
        
        # self.client.set_joint_positions([0, -0.569, 0, -2.809, 0, 3.04, 0.741, 0.04, 0.04], False)

        self.open_gripper(id='right')
 
        pass

    def setup(self):
        self.target_object = None

        # set robot init state

    def get_observation(self, data_keys):
        observation = {}
        observation = self.client.get_observation(data_keys)
        return observation

        # joint_datas = observation['joint']
        # print(joint_datas)
        # cam_datas = observation['camera']
        # print(len(cam_datas), type(cam_datas[0]), cam_datas[0].keys())
        # gripper_datas = observation['gripper']
        # ee_pos = gripper_datas['right']['position']
        # ee_rotation = gripper_datas['right']['rotation']
        # print(ee_pos)
        # print(ee_rotation)
    
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
            self.client.DetachObj()
            self.client.AttachObj(prim_paths=['/World/Objects/'+self.target_object])

    def move_pose(self, target_pose, type, **kwargs):
        # import ipdb;ipdb.set_trace()
        if type.lower()=='trajectory':
            content= {
                "type": 'trajectory_4x4_pose',
                "data": target_pose,
            }
        else:
            if type=='AvoidObs':
                type='ObsAvoid'
            elif type=='Normal':
                type='Simple'
      
            content= {
                "type": "matrix",
                "matrix": target_pose,
                'trajectory_type': type
            }
        return self.move(content)
    
    def set_gripper_action(self, action):
        # self.gripper_server.set_gripper_action(action)
        if action is None:
            return
        if action == 'open':
            self.open_gripper(id="right", width=0.1)
        elif action == 'close':
            self.close_gripper(id="right")

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
            print("move  maobo !", T, quat_wxyz, state)

        elif type == "quat":

            if "trajectory_type" in content and content["trajectory_type"] == "Simple":
                is_backend = True
            else:
                is_backend = False
            state = (
                self.client.moveto(
                    target_position=content["xyz"],
                    target_quaternion=content["quat"],
                    arm_name="right",
                    is_backend=is_backend,
                ).errmsg
                == "True"
            )


        elif type == "joint":
            state = self.client.set_joint_positions(content["position"])
        
        elif type == "euler":
            is_backend = True

            T_curr = self.get_ee_pose()
            xyzrpy_input = content["xyzrpy"]
            xyz_curr = T_curr[:3, 3]
            rpy_curr = Rotation.from_matrix(T_curr[:3, :3]).as_euler("xyz")

            incr = content.get("incr", False)
            if incr:
                xyz_tgt = xyz_curr + np.array(xyzrpy_input[:3])
                rpy_tgt = rpy_curr + np.array(xyzrpy_input[3:])
                quat_tgt = get_quaternion_from_rotation_matrix(
                    Rotation.from_euler("xyz", rpy_tgt).as_matrix()
                )
            else:
                raise NotImplementedError

            state = (
                self.client.moveto(
                    target_position=xyz_tgt,
                    target_quaternion=quat_tgt,
                    arm_name="right",
                    is_backend=is_backend,
                ).errmsg
                == "True"
            )
        elif type == 'move_pose_list':
            action_list = content["data"]
            for action in action_list:
                # T_curr = self.get_ee_pose()
                T_curr = content["ee_transform"]
                action_pose10d = action[:-1]
                action_gripper = action[-1]
                xyz_curr = T_curr[:3, 3]
                rotation_curr = T_curr[:3, :3]

                action_repr = content.get("action_repr", "rela")
                if action_repr == 'delta':
                    mat_tgt = pose10d_to_mat(action_pose10d)
                    xyz_delta = mat_tgt[:3, 3]
                    rotation_delta = mat_tgt[:3, :3]
                    xyz_tgt = xyz_curr + np.array(xyz_delta[:3])
                    rotation_tgt = np.dot(rotation_curr, rotation_delta)
                    euler_tgt = Rotation.from_matrix(rotation_tgt).as_euler('xyz')
                    quat_tgt = get_quaternion_from_rotation_matrix(
                        rotation_tgt
                    )
                else:
                    raise RuntimeError()
               
                start_time = time.time()
                state = (
                    self.client.moveto(
                        target_position=xyz_tgt,
                        target_quaternion=quat_tgt,
                        arm_name="right",
                        is_backend=True,
                    ).errmsg
                    == "True"
                )
                xyz_curr = self.get_ee_pose()[:3, 3]
                euler_curr = Rotation.from_matrix(self.get_ee_pose()[:3, :3]).as_euler("xyz")
                euler_tgt = Rotation.from_matrix(rotation_tgt).as_euler("xyz")
                print("xyz dist", np.linalg.norm(xyz_curr - xyz_tgt), "rpy dist", np.linalg.norm(euler_curr - euler_tgt))
                print("move", time.time() - start_time)
                
                # gripper_position = self.client.get_joint_positions().states[-1].position
                gripper_position = content["gripper_position"]
                start_time = time.time()
                if gripper_position > 0.03 and action_gripper < 0.5:
                    self.close_gripper(id="right", force=50)
                    print("Close gripper", time.time() - start_time)
                elif gripper_position < 0.03 and action_gripper > 0.5:
                    self.open_gripper(id="right", width=0.1)
                    # print("Open gripper", time.time() - start_time)
                
        elif type.lower() == 'trajectory':
            action_list = content["data"]

            T_curr = self.get_ee_pose()
            xyz_curr = T_curr[:3, 3]
            rotation_curr = T_curr[:3, :3]

            traj = []
            for action in action_list:
                xyzrpy_input = action[:3]

                action_repr = content.get("action_repr", "rela")
                if action_repr == 'delta':
                    xyz_tgt = xyz_curr + np.array(xyzrpy_input[:3])
                    xyz_curr = xyz_tgt.copy()
                elif action_repr == 'abs':
                    xyz_tgt = np.array(xyzrpy_input[:3])

                action_rotation_matrix = Rotation.from_euler("xyz", np.array([np.pi, 0, np.pi])).as_matrix()
                quat_tgt = Rotation.from_matrix(action_rotation_matrix).as_quat()

                pose = list(xyz_tgt), list(quat_tgt)
                traj.append(pose)

            print("traj", traj)
            start_time = time.time()
            self.client.SetTrajectoryList(traj)
            
            xyz_curr = self.get_ee_pose()[:3, 3]
            print(xyz_curr, xyz_tgt)
            print("xyz dist", np.linalg.norm(xyz_curr - xyz_tgt))
            print("move", time.time() - start_time)

            # time.sleep(100)

        elif type.lower() == 'trajectory_4x4_pose':
            waypoint_list = content["data"]

            traj = []
            waypoint_list = waypoint_list[::len(waypoint_list)//4]
            for pose in waypoint_list:
                xyz = pose[:3,3]
                quat_wxyz = get_quaternion_from_rotation_matrix(pose[:3,:3])
                pose = list(xyz), list(quat_wxyz)
                traj.append(pose)

                state = (
                    self.client.moveto(
                        target_position=xyz,
                        target_quaternion=quat_wxyz,
                        arm_name="right",
                        is_backend=True,
                    ).errmsg
                    == "True"
                )

            # self.client.SetTrajectoryList(traj)           # err: not block
            state = True

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
        flange2gripper[2, 3] = -0.01
        return gripper_pose @ flange2gripper

    def get_ee_pose(self, id="right", **kwargs):
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
        return pose

    def check_ik(self, pose, id="right", **kwargs):
        xyz, quat = pose[:3, 3], get_quaternion_from_rotation_matrix(pose[:3, :3])
        state = self.client.GetIKStatus(target_position=xyz, target_rotation=quat, is_right=id == "right")
        return state.isSuccess

    def solve_ik(self, pose, id="right", **kwargs):
        xyz, quat = pose[:3, 3], get_quaternion_from_rotation_matrix(pose[:3, :3])
        state = self.client.GetIKStatus(target_position=xyz, target_rotation=quat, is_right=id == "right")
        return state.isSuccess, None