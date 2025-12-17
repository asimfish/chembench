import cv2
import matplotlib.pyplot as plt
import json

# json_path = "/home/zhwang/Albert/psi-isaaclab/source/psi_agent/task/scene_json/scene_base.json"
# with open(json_path, 'r') as f:
#     data = json.load(f)
#     print(data)

import os
import json

# 设置场景数据集的根目录
root_dir = "/home/zhwang/Albert/dataset/scene/"

# 默认字段
default_prim_path = "/World/Room"
default_workspace_position = [0.15, 0.0, 0.0]
default_workspace_quaternion = [1.0, 0.0, 0.0, 0.0]

# 存储最终的 JSON 数据
scene_list = []

# 遍历根目录下的所有子文件夹
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    
    # 确保是一个文件夹
    if os.path.isdir(folder_path):
        # 查找该文件夹下的 .usd 文件
        usd_files = [f for f in os.listdir(folder_path) if f.endswith(".usd")]
        
        # 如果找到 usd 文件，则创建 scene 记录
        for usd_file in usd_files:
            scene_data = {
                "scene_id": folder,  # 以文件夹名作为 scene_id
                "prim_path": default_prim_path,
                "scene_usd": os.path.join(folder_path, usd_file),  # 绝对路径
                "workspace_position": default_workspace_position,
                "workspace_quaternion": default_workspace_quaternion
            }
            scene_list.append(scene_data)

# 生成最终的 JSON 数据
output_data = {"scene": scene_list}

# 保存到文件
output_file = "/home/zhwang/Albert/psi-isaaclab/source/psi_agent/task/scene_json/psi_scene_info.json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"JSON 文件已生成: {output_file}")

