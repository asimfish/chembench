import json
import os
from typing import Dict, Any

def convert_to_dict(config: Any) -> dict:
    """
    将配置转换为字典格式
    """
    if isinstance(config, dict):
        return config
    elif isinstance(config, list):
        # 如果是列表，转换为字典，使用对象名称作为键
        result = {}
        for item in config:
            if isinstance(item, dict) and 'name' in item:
                result[item['name']] = item
        return result
    elif isinstance(config, str):
        try:
            parsed = json.loads(config)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

def enrich_object_configs(objects: dict, is_static: bool = True) -> dict:
    """
    为对象添加class_type和spawn.func
    """
    # 确保输入是字典格式
    objects = convert_to_dict(objects)
    
    # 创建结果字典
    result_objects = {}
    
    # 遍历对象
    for obj_name, obj in objects.items():
        if not isinstance(obj, dict):
            continue
            
        # 复制对象配置
        enriched_obj = {}
        
        # 首先添加class_type
        if is_static:
            enriched_obj['class_type'] = None
        else:
            enriched_obj['class_type'] = "isaaclab.assets.rigid_object.rigid_object:RigidObject"
        
        # 复制其他字段
        for key, value in obj.items():
            if key == 'spawn' and isinstance(value, dict):
                # 处理spawn字段
                enriched_obj['spawn'] = value.copy()
                # 确保func被正确设置
                if is_static:
                    enriched_obj['spawn']['func'] = "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd"
                else:
                    enriched_obj['spawn']['func'] = "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd"
            else:
                enriched_obj[key] = value
        
        # 确保spawn字段存在并包含func
        if 'spawn' not in enriched_obj:
            enriched_obj['spawn'] = {}
        if 'func' not in enriched_obj['spawn']:
            if is_static:
                enriched_obj['spawn']['func'] = "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd"
            else:
                enriched_obj['spawn']['func'] = "isaaclab.sim.spawners.from_files"
        
        result_objects[obj_name] = enriched_obj
    
    return result_objects

def generate_grasp_json(grasp_num: int = 1) -> None:
    """
    生成新的Grasp JSON文件
    
    参数:
    grasp_num: 要生成的Grasp文件编号
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current directory:", current_dir)
    
    # 构建文件路径
    template_path = os.path.join(current_dir, 'room_scene_cfg.json')
    grasp_0_path = os.path.join(current_dir, 'Grasp_0.json')
    new_file_path = os.path.join(current_dir, f'Grasp_{grasp_num}.json')
    
    # 检查文件是否存在
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not os.path.exists(grasp_0_path):
        raise FileNotFoundError(f"Grasp_0 file not found: {grasp_0_path}")
    
    # 读取模板文件
    with open(template_path, 'r') as f:
        template_data = json.load(f)
    
    # 读取Grasp_0.json文件
    with open(grasp_0_path, 'r') as f:
        grasp_data = json.load(f)
    
    # 从env_cfg.scene中获取配置
    if 'env_cfg' in grasp_data and 'scene' in grasp_data['env_cfg']:
        scene_data = grasp_data['env_cfg']['scene']
        
        # 更新static_objects_cfg
        if 'static_object_cfg' in scene_data:
            print("\nEnriching static_objects_cfg...")
            template_data['static_objects_cfg'] = enrich_object_configs(
                scene_data['static_object_cfg'],
                is_static=True
            )
        
        # 更新rigid_objects_cfg
        if 'rigid_objects_cfg' in scene_data:
            print("\nEnriching rigid_objects_cfg...")
            template_data['rigid_objects_cfg'] = enrich_object_configs(
                scene_data['rigid_objects_cfg'],
                is_static=False
            )
    else:
        print("Warning: Could not find env_cfg.scene in Grasp_0.json")
    
    # 保存新的JSON文件
    with open(new_file_path, 'w') as f:
        json.dump(template_data, f, indent=4)
    
    print(f"\nSuccessfully generated {new_file_path}")

def main():
    try:
        # 生成Grasp_1.json
        generate_grasp_json(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the required files exist in the correct location.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
