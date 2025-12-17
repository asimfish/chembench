
import os
import sys
import json
import time
import numpy as np
current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_directory)



from .utils.object import LayoutObject
from .solver_2d.solver import LayoutSolver2D
from .solver_3d.solver import LayoutSolver3D

from robot.utils import axis_to_quaternion, quaternion_rotate, get_quaternion_wxyz_from_rotation_matrix


def list_to_dict(data:list):
    tmp = {}
    for i in range(len(data)):
        tmp[str(i)] = data[i]
    return tmp

class LayoutGenerator():
    def __init__(self, workspace, obj_infos, objects, key_obj_ids, extra_obj_ids, constraint=None, fix_obj_ids=[]):
        self.workspace = workspace
        self.objects = objects
        self.obj_infos = obj_infos
        
        self.key_obj_ids = key_obj_ids
        self.extra_obj_ids = extra_obj_ids
        self.fix_obj_ids = fix_obj_ids
        self.constraint = constraint

        # Determine which objects are key objects in 2D or 3D based on constraints
        if constraint is None:
            self.key_obj_ids_2d = self.key_obj_ids
            self.key_obj_ids_3d = []
        else:
            self.key_obj_ids_2d = [constraint["passive"]]
            self.key_obj_ids_3d = [constraint["active"]]
        self.constraint = constraint

        # Convert workspace position and size to appropriate units
        workspace_xyz, workspace_size = np.array(workspace['position']), np.array(workspace['size'])
        workspace_size = workspace_size * 1000
        # extra info about workspace

        # Initialize solvers for 2D and 3D layouts
        self.solver_2d = LayoutSolver2D(workspace_xyz, workspace_size, objects, fix_obj_ids=fix_obj_ids, obj_infos=obj_infos)
        self.solver_3d = LayoutSolver3D(workspace_xyz, workspace_size, objects, obj_infos=obj_infos)

        # List to store successfully placed object IDs
        self.succ_obj_ids = []


    def __call__(self):
        ''' Generate Layout '''
        # import pdb;pdb.set_trace()
        if len(self.key_obj_ids_2d) > 0:
            objs_succ = self.solver_2d(self.key_obj_ids_2d, self.succ_obj_ids, object_extent=30, start_with_edge=True, key_obj =True,initial_angle = 0)
            self.update_obj_info(objs_succ)
            print('-- 2d layout done --')

        if len(self.key_obj_ids_3d) > 0:
            objs_succ = self.solver_3d(self.key_obj_ids_3d, self.succ_obj_ids, constraint=self.constraint)
            self.update_obj_info(objs_succ)
            print('-- 3d layout done --')

        if len(self.extra_obj_ids) > 0:
            objs_succ = self.solver_2d(self.extra_obj_ids, self.succ_obj_ids, object_extent=30, start_with_edge=False, key_obj =False)
            self.update_obj_info(objs_succ)
            print('-- extra layout done --')



        ''' Check completion '''
        res_infos = []
        if len(self.key_obj_ids) > 0:
            for obj_id in self.key_obj_ids:
                if obj_id not in self.succ_obj_ids:
                    return None
                res_infos.append(self.obj_infos[obj_id])
            return res_infos
        elif len(self.extra_obj_ids) > 0:
                if len(self.succ_obj_ids) >0:
                    for obj_id in self.succ_obj_ids:
                        res_infos.append(self.obj_infos[obj_id])
                return res_infos
        else:
            return res_infos

    def update_obj_info(self, obj_ids):
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        # Update information for each object ID
        for obj_id in obj_ids:
            pose = self.objects[obj_id].obj_pose
            # Update position and quaternion in obj_infos, converting position to meters
            xyz, quat = pose[:3, 3], get_quaternion_wxyz_from_rotation_matrix(pose[:3, :3])
            self.obj_infos[obj_id]['position'] = (xyz/1000).tolist()
            self.obj_infos[obj_id]['quaternion'] = quat.tolist()
            self.obj_infos[obj_id]["is_key"] = obj_id in self.key_obj_ids
            self.succ_obj_ids.append(obj_id)


class TaskGenerator():
    def __init__(self, task_template):
        # AlbertMao  更改root路径
        # self.data_root = os.path.dirname(os.path.dirname(__file__))+"/Psi"
        self.data_root = task_template["data_root"]
        self.init_info(task_template)


    def _load_json(self, relative_path):
        with open(os.path.join(self.data_root, relative_path), 'r') as file:
            return json.load(file)
        
    def create_robot(self):
        return {
        "prim_path": "/World/Robot",
        "spawn": {
            "usd_path":"/home/zhwang/Albert/psi-isaaclab/Psi/Robots/PsiRobot_DC_01/PsiRobot_DC_01_Tuned.usd"
        },
        "init_state": {
            "pos": [-0.5, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0]
        },
        "device": "cuda:0"
    }

    def create_static_object_cfg(self):
        return {
            "room": {
                "prim_path": "/World/Room",
                "spawn": {
                    "usd_path":"/home/zhwang/Albert/psi-isaaclab/Psi/Envs/psi_garage_2_obj/GarageScene.usd",
                    "scale": [1.0, 1.0, 1.0]
                },
                "init_state": {
                    "pos": [0.0, 0.0, 0.0],
                    "rot": [0.707, 0.707, 0.0, 0.0]
                }
            }
        }

    def create_rigid_objects_cfg(self):
        return {
            "bottle": {
                "prim_path": "/World/Bottle",
                "spawn": {
                    "usd_path": "/home/zhwang/Albert/psi-isaaclab/Psi/Targets/drink-B36-V1/B36.usd",
                    "visual_material": None,
                    "scale": [0.001, 0.001, 0.001]
                },
                "init_state": {
                    "pos": [-0.05, 0.0, 0.9],
                    "rot": [0.707, 0.707, 0.0, 0.0]
                }
            },
            "table": {
                "prim_path": "/World/Table",
                "spawn": {
                    "usd_path": "/home/zhwang/Albert/psi-isaaclab/Psi/Envs/table_1157.usd",
                    "rigid_props": {"kinematic_enabled": True},
                    "visual_material": None
                },
                "init_state": {
                    "pos": [0.15, 0.0, 0.0],
                    "rot": [1.0, 0.0, 0.0, 0.0]
                }
            }
        }

    def create_scene(self):
        return {
            "num_envs": 1,
            "env_spacing": 4.0,
            "replicate_physics": True,
            "light_cfg": {
                "spawn": {
                    "intensity": 3000.0,
                    "color": [0.75, 0.75, 0.75]
                }
            },
            "robot": self.create_robot(),
            "static_object_cfg": self.create_static_object_cfg(),
            "rigid_objects_cfg": self.create_rigid_objects_cfg()
        }

    def create_env_cfg(self):
        return {
            "sim": {
                "dt": 0.008,
                "render_interval": 2,
                "device": "cuda:0",
                "physx": {
                    "solver_type": 1,
                    "max_position_iteration_count": 16,
                    "max_velocity_iteration_count": 0,
                    "bounce_threshold_velocity": 0.002
                }
            },
            "scene": self.create_scene()
        }

    def generate_json(self):
        data = {
            "id": "Task_Grasp_Rigid",
            "robot": "Psi_DC_01",
            "env_cfg": self.create_env_cfg(),
            "teleop": "Vuer",
            "grasp_object_id": "Target",
            "grasp_height": 0.3
        }
        return data
    def merge_taskjson(self,original_json ,merged_json ,i):
        robot_cfg = original_json['env_cfg']['scene']['robot']
        
        robot_cfg['init_state']['pos']=merged_json['robot_init_position']
        robot_cfg['init_state']['rot']=merged_json['robot_init_quaternion']
        static_object_cfg =original_json['env_cfg']['scene']['static_object_cfg']['room']
        static_object_cfg['prim_path']=merged_json['prim_path']
        static_object_cfg['spawn']['usd_path']= merged_json['scene_usd']
        object_infos = merged_json['objects']
        merged_config = {}  # 存储所有对象的配置
        for obj in object_infos:
            formatted_config = self.generate_object_config(obj ,obj["object_id"],obj["prim_path"],obj["obj_path"])
            merged_config.update(formatted_config)  # 合并到总配置   
        print(merged_config)
        original_json['env_cfg']['scene']['rigid_objects_cfg']=merged_config

        return original_json 
    def generate_object_config(self,object_info,object_name ,prim_path , obj_path):
        return {
        object_name: {
            "prim_path": prim_path,
            "spawn": {
                "usd_path": obj_path,
                # "rigid_props": {
                #     "kinematic_enabled": True  # 这里假设 rigid_props 默认存在
                # },
                "visual_material": None , # 这里保持和模板一致
                "scale": object_info["scale"]
            },
            "init_state": {  
                "pos": object_info["position"],
                # "rot": object_info["quaternion"]
                "rot": [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ]
            },
            
        }
    }

# 这里就是json解析   
    def init_info(self, task_template):  
        # Load all objects  & constraints
        self.fix_objs = task_template["objects"].get('fix_objects', [])
        all_objs = task_template["objects"]["task_related_objects"] + task_template["objects"]["extra_objects"] + self.fix_objs
        self.fix_obj_ids = [obj["object_id"] for obj in self.fix_objs]
        

        # Load all object information
        self.key_obj_ids, self.extra_obj_ids = {'0':[]}, {'0':[]}
        for obj in task_template["objects"]["task_related_objects"]:
            ws_id = obj.get("workspace_id", "0")
            if ws_id not in self.key_obj_ids:
                self.key_obj_ids[ws_id] = []
            self.key_obj_ids[ws_id].append(obj["object_id"])
        for obj in task_template["objects"]["extra_objects"]:
            ws_id = obj.get("workspace_id", "0")
            if ws_id not in self.extra_obj_ids:
                self.extra_obj_ids[ws_id] = []
            self.extra_obj_ids[ws_id].append(obj["object_id"])

        # construct LayoutObject 
        obj_infos = {}
        objects = {}
        all_key_objs = [obj_id for ws_id in self.key_obj_ids for obj_id in self.key_obj_ids[ws_id]]
        for obj in all_objs:
            obj_id = obj["object_id"]
            # obj_dir = os.path.join(self.data_root, obj["data_info_dir"])
            # maobo 
            obj_dir = self.data_root + obj["data_info_dir"]
            if "metadata" in obj:
                info=obj["metadata"]["info"]
                info["interaction"]=obj["metadata"]["interaction"]
            else:
                # info = json.load(open(obj_dir + "/object_parameters.json"))
                info = {}
                info["upAxis"]=["y"]
            info["data_info_dir"] = obj_dir
            # info["obj_path"] = obj_dir + "/Aligned.obj"
            # info["obj_path"] = obj_dir + "/Aligned.usd"
            info["obj_path"] = obj_dir  #  在json中添加 
            info['object_id'] = obj_id
            # Albert 
            info['prim_path']=obj['prim_path']
            info['scale']=obj['scale']
            #  Albert  add  for only generate task deatial json  0208 begin 
            # info['upAxis'] =["x","z"]
            #  Albert  add  for only generate task deatial json  0208 begin 
            if 'extent' in obj:
                info['extent'] = obj['extent']
            obj_infos[obj_id] = info
            objects[obj_id] = LayoutObject(info, use_sdf=obj_id in all_key_objs)
            
        self.obj_infos, self.objects = obj_infos, objects

        # fix object info
        self.fix_obj_infos = []
        for fix_obj in self.fix_objs:
            fix_obj['is_key'] = True
            fix_obj.update(obj_infos[fix_obj['object_id']])
            self.fix_obj_infos.append(fix_obj)


        if "robot" not in task_template:
            arm = "right"
            robot_id = "A2D"
        else:
            arm = task_template["robot"]["arm"]
            robot_id = task_template["robot"]["robot_id"]

        scene_info = task_template["scene"]
        self.scene_usd = task_template["scene"]["scene_usd"]     
        self.task_template = {
            "scene_usd": self.scene_usd,
            "arm": arm,
            #Albert psi json   begin 
            "prim_path":task_template["scene"]["prim_path"],
            "grasp_height":task_template["grasp_height"],
            "num_envs":task_template["num_envs"],
            "task_name": task_template["task"],
            "env_spacing":task_template["env_spacing"],
            "robot_id": robot_id,
            # maobo  begin  init robot pose from  taskinfo json 
            "robot_init_position":task_template["robot"]["robot_init_pose"]["table_with_robot"]["position"],
            "robot_init_quaternion":task_template["robot"]["robot_init_pose"]["table_with_robot"]["quaternion"],
            # 定义
            "workspace_position":task_template["objects"]["fix_objects"][0]["position"],
            "workspace_quaternion":task_template["objects"]["fix_objects"][0]["quaternion"],
            
            ##Albert psi json   end 
            # "stages": task_template['stages'],
            "object_with_material": task_template.get('object_with_material', {}),
            "lights": task_template.get('lights', {}),
            "lights": task_template.get('lights', {}),
            "objects": []
        }
        constraint = task_template.get("constraints")
        robot_init_workspace_id = scene_info["scene_id"].split('/')[-1]
        # Retrieve scene information
        self.scene_usd = scene_info["scene_usd"]
        if "function_space_objects" in scene_info:
            workspaces = scene_info["function_space_objects"]
            if robot_init_workspace_id not in task_template["robot"]["robot_init_pose"]:
                 self.robot_init_pose = task_template["robot"]["robot_init_pose"]
            else:
                self.robot_init_pose = task_template["robot"]["robot_init_pose"][robot_init_workspace_id]
        else:
            scene_info = self._load_json(scene_info["scene_info_dir"]+ "/scene_parameters.json")
            workspaces = scene_info["function_space_objects"]
            # Normalize format
            if isinstance(scene_info["robot_init_pose"], list):
                scene_info['robot_init_pose'] = list_to_dict(scene_info["robot_init_pose"])
            self.robot_init_pose = scene_info["robot_init_pose"][robot_init_workspace_id]
        self.robot_init_pose
        if isinstance(workspaces, list):
            workspaces = list_to_dict(workspaces)
            workspaces = {'0': workspaces[robot_init_workspace_id]}
        elif isinstance(workspaces, dict) and 'position' in workspaces:
            workspaces = {'0': workspaces}
        self.layouts = {}
        

        # init workspace and generate object layout info 
        for key in workspaces:
            ws, key_ids, extra_ids = workspaces[key], self.key_obj_ids.get(key, []), self.extra_obj_ids.get(key, [])
            self.layouts[key] = LayoutGenerator(ws, obj_infos, objects, key_ids, extra_ids, constraint=constraint, fix_obj_ids=self.fix_obj_ids)
       

    def generate_tasks(self, save_path, task_num, task_name):
        os.makedirs(save_path, exist_ok=True)
        
        # Generate task json 
        # In order to ensure the effect, for the main operating object, try to ensure that the object can be placed, and ignore the interference if it cannot be placed
        for i in range(task_num):  
            output_file = os.path.join(save_path, f'{task_name}_%d.json'%(i))
            self.task_template['objects'] = [] 
            self.task_template['objects'] += self.fix_obj_infos
            flag_failed = False
            for key in self.layouts:
                obj_infos = self.layouts[key]()
                if not obj_infos:
                    if obj_infos is None:
                        flag_failed = True
                        break
                    continue
                self.task_template['objects'] += obj_infos
            if flag_failed:
                print(f"Failed to place key object, skipping")
                continue
    
            task_info_cfg = self.generate_json()
            final_json = self.merge_taskjson(task_info_cfg ,self.task_template ,i)
            print(final_json)
            print('Saved task json to %s'%output_file)
            with open(output_file, 'w') as f:
                json.dump(final_json, f, indent=4)
