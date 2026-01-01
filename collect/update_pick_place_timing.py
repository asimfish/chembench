#!/usr/bin/env python3
"""
批量更新 object_config.json 中所有物体的 pick_place timing 参数
"""

import json
from pathlib import Path

# 配置文件路径
CONFIG_PATH = Path("/home/psibot/chembench/psilab/source/psilab_tasks/psilab_tasks/imitation_learning/object_config.json")

# 统一的 timing 参数
STANDARD_TIMING = {
    "approach": 0.15,
    "grasp": 0.1,
    "lift": 0.15,
    "transport": 0.15,
    "release": 0.1,
    "retreat": 0.35
}

def update_pick_place_timing():
    """更新所有物体的 pick_place timing 参数"""
    
    # 读取配置文件
    print(f"读取配置文件: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取物体配置（处理不同的结构）
    if "objects" in config:
        objects = config["objects"]
        print(f"配置结构: 包含 'objects' 键")
    else:
        objects = config
        print(f"配置结构: 直接是物体字典")
    
    # 统计信息
    total_objects = len(objects)
    updated_count = 0
    skipped_count = 0
    no_pick_place = 0
    
    print(f"总物体数: {total_objects}\n")
    
    # 遍历所有物体
    for obj_name, obj_config in objects.items():
        # 检查是否有 pick_place 配置
        if "pick_place" in obj_config and obj_config["pick_place"] is not None:
            # 确保 pick_place 是字典类型
            if isinstance(obj_config["pick_place"], dict):
                # 更新或添加 timing 参数
                obj_config["pick_place"]["timing"] = STANDARD_TIMING
                updated_count += 1
                print(f"  ✓ 更新: {obj_name:40s} ({obj_config.get('name_cn', 'N/A')})")
            else:
                skipped_count += 1
                print(f"  ✗ 跳过: {obj_name:40s} (pick_place 不是字典)")
        else:
            # 没有 pick_place 配置，跳过
            no_pick_place += 1
    
    # 创建备份
    backup_path = CONFIG_PATH.with_suffix('.json.backup_timing')
    print(f"\n创建备份: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 保存更新后的配置
    print(f"\n保存更新后的配置...")
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("更新完成！")
    print("=" * 60)
    print(f"总物体数:        {total_objects}")
    print(f"已更新 timing:   {updated_count}")
    print(f"跳过:            {skipped_count}")
    print(f"无 pick_place:   {no_pick_place}")
    print("\n标准 timing 参数:")
    for key, value in STANDARD_TIMING.items():
        print(f"  {key:12s}: {value}")
    print("=" * 60)

if __name__ == "__main__":
    update_pick_place_timing()

