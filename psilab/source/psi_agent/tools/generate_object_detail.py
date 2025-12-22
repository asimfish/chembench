import cv2
import matplotlib.pyplot as plt
import json
import os

# json_path = "/home/zhwang/Albert/psi-isaaclab/source/psi_agent/object_json/object_base.json"
# json_path = "/home/zhwang/Albert/psi-isaaclab0306/source/psi_agent/object_json/psi_object_info.json"
# with open(json_path, 'r') as f:
#     data = json.load(f)
#     print(data)



# 根目录
root_dir = "/home/zhwang/Albert/dataset/psi_object_categories/"

# 用于存储最终的 JSON 结构
json_data = {"Object": {}}

# 遍历二级目录（XXXXXX）
for second_level in os.listdir(root_dir):
    second_level_path = os.path.join(root_dir, second_level)
    print(second_level_path)
    
    if os.path.isdir(second_level_path):  # 确保是文件夹
        json_data["Object"][second_level] = []  # 初始化该类别
        
        # 遍历三级目录（object_id）
        for third_level in os.listdir(second_level_path):
            third_level_path = os.path.join(second_level_path, third_level)
            print(third_level_path)
            
            if os.path.isdir(third_level_path):  # 确保是文件夹
                # 遍历 third_level_path 下的所有文件和文件夹
                for subdir in os.listdir(third_level_path):
                    subdir_path = os.path.join(third_level_path, subdir)

                    # 判断是否是文件夹，并且名称为 "meshes" 或 "usd"
                    if os.path.isdir(subdir_path) and os.path.basename(subdir_path) in ["meshes", "usd"]:
                                 # 查找 .usd 文件
                        usd_files = [f for f in os.listdir(subdir_path) if f.endswith(".usd")]
                        print("second_level")
                        print(second_level)
            
                        
                        obj_scale = [1, 1, 1] 
                        if "fix_" in second_level:
                            obj_scale = [1, 1, 2]
                        if "GS" in second_level :
                            obj_scale = [0.001,0.001,0.001]
                        
                        obj_data = {
                            "object_id": third_level,  # object_id 为三级目录名
                            "prim_path": f"/World/{third_level}",  # prim_path
                            "data_info_dir": os.path.relpath(os.path.join(subdir_path, usd_files[0]), root_dir),  # USD 相对路径
                            "scale": obj_scale  # 默认值
                        }
                        json_data["Object"][second_level].append(obj_data)
                        
# # 输出 JSON 文件
output_file = "/home/zhwang/Albert/psi-isaaclab/source/psi_agent/tools/object_json/psi_object_info.json"
# output_file = "output.json"
with open(output_file, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"JSON 文件已生成: {output_file}")


