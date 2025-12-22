import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from base_utils.random_utils import RandomUtils

# base_utils.data_utils import pose_difference

ROBOT_OFFSET = 0.65
FIX_OBJECT_HEIGHT = 1.1
FIX_OBJECT_POSITION = [ 0.15,0.0,0.0]
FIX_OBJECT_QUATERNION = [1.0,0.0,0.0,0.0]
DATASET_PATH = "/home/zhwang/Albert/dataset/psi_object_categories/"
# Scene 部分
def generate_scene_config():
    global position, quaternion ,base_path

    base_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
    relative_path = "scene_json/psi_scene_info.json"
    json_path = os.path.join(base_path, relative_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
        scene_info_dict = data['scene'][6]  # AlbertMao  后期改为针对环境进行分组 默认选取第一个
        scene_info_dict['scene_id'] = scene_info_dict['scene_id'] + "/table_with_robot"
        # 提取 workspace_position 和 workspace_quaternion，并修改 z 轴值
        position =scene_info_dict.pop('workspace_position')
        ws_position  = position.copy()
        quaternion =  scene_info_dict.pop('workspace_quaternion')
        ws_position[2] =FIX_OBJECT_HEIGHT # 修改 z 轴值  后期根据桌子高度进行计算
        print(ws_position)
        print(position)
        # 添加 function_space_objects
        scene_info_dict["function_space_objects"] = {
            "table_with_target": {
                "position": ws_position,
                "quaternion": quaternion,
                "size": [0.2, 0.21, 0.3]  # AlbertMao  后期改为随机 默认选取第一个
            },
            "table_with_target_extra": {
                "position": ws_position,
                "quaternion": quaternion,
                "size": [0.4, 0.37, 0.3]  # AlbertMao  后期改为随机 默认选取第一个
            }
        }
        return scene_info_dict     

# Robot 部分
def generate_robot_config():
    robot_position = position.copy() 
    robot_position[0]=robot_position[0]-ROBOT_OFFSET
    robot_quaternion = quaternion 
    return {
        "robot_id": "A2D",
        "robot_cfg": "Franka_120s.json",
        "arm": "left&right",
        "robot_init_pose": {
            "table_with_robot": {
                "position": robot_position,
                "quaternion": robot_quaternion
            }
        }
    }

# Objects 部分
def generate_objects_config():
    object_relative_path = "object_json/psi_object_info.json"
    json_path = os.path.join(base_path, object_relative_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f) 
        # AlbertMao  生成fix_obejct  table 异常scene中 只有一个 目前是 资产json中的 第一个 
        fix_objects_info = data['Object']['fix_object'][0]  
        
        fix_objects_info['position']=FIX_OBJECT_POSITION
        fix_objects_info['quaternion']=FIX_OBJECT_QUATERNION
        #随机产生这个 每个种类 下的 obj 
        target_obj_index = RandomUtils.generate_unique_numbers((0,(len(data['Object']['GS_drink_sub'])-1)) ,1)  # B23V3 需要特殊处理 
        print("========================================================")
        print(target_obj_index[0])
        target_object_info = data['Object']['GS_drink_sub'][target_obj_index[0]]  # AlbertMao  后期改为随机 默认选取第一个
        target_object_info['workspace_id']="table_with_target"
        target_object_info['object_id']='Target'
        
        
        extra_obj_indexs = RandomUtils.generate_unique_numbers((0,(len(data['Object']['GG_ActionFigures'])-1)) ,2)

        # 遍历索引数组，并依次赋值
        extra_objects =[]
        for i,index in enumerate(extra_obj_indexs):
            print("++++++++++++++++++++++++++++++++++++++++++++++")
            print(i)
            print(index)
            extra_obj_indexs[i]=data['Object']['GG_ActionFigures'][index]
            extra_obj_indexs[i]['workspace_id'] = 'table_with_target_extra'
    return {
        "task_related_objects": [target_object_info],
        "constraints": None,
        "fix_objects": [fix_objects_info],
        "extra_objects": extra_obj_indexs
    }

# Task 部分
def generate_task_config():
    return "Grasp"

# 其他配置：如 num_envs, grasp_height 等
def generate_other_config():
    return {
        "num_envs": 1,
        "env_spacing": 4,
        "grasp_height": 0.3
    }

# Recording Setting 配置
def generate_recording_setting():
    return {
        "fps": 30,
        "num_of_episode": 3,
        "camera_list": [
            "/panda/Left_Camera",
            "/panda/Right_Camera",
            "/panda/right_base_link/Hand_Camera"
        ]
    }

# 主函数，整合所有部分
def generate_full_config():
    config = {
        "scene": generate_scene_config(),   # OK
        "robot": generate_robot_config(),
        "data_root": DATASET_PATH,
        "objects": generate_objects_config(),
        "task": generate_task_config(),
        "recording_setting": generate_recording_setting(),
        **generate_other_config()  # 将其他配置部分合并到总配置中
    }
    
    # 返回最终的 JSON 配置
    return config

# 保存 JSON 配置到文件
def save_config_to_json(file_path):
    config = generate_full_config()
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
        print(f"Configuration saved to {file_path}")
             
def main():
    # 使用示例：将生成的配置保存为 `scene_config.json
    current_path = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
    parent_path = os.path.dirname(current_path) 
    target_path = os.path.join(parent_path, "task")
    save_config_to_json(target_path+"/scene_config_psi_test.json")
               
if __name__ == "__main__":
    # args_cli = parse_cli_args()
    main()


