# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-12-13
# Description: 抓取姿态配置文件，存储不同物体的抓取参数

"""
抓取配置参数说明:
- offset: [x, y, z] 相对于物体中心的末端执行器位置偏移 (单位: 米)
    - x: 前后方向 (正值=向前)
    - y: 左右方向 (正值=向右)
    - z: 上下方向 (正值=向上)
    
- euler_deg: [roll, pitch, yaw] 末端执行器的欧拉角 (单位: 度)
    - roll (X轴):  手腕绕前后轴旋转（翻转手掌）
    - pitch (Y轴): 手腕绕左右轴旋转（抬起/放下）
    - yaw (Z轴):   手腕绕上下轴旋转（左右摆动）

常用姿态参考:
- 侧抓: euler_deg ≈ [157, 69, -154]
- 顶抓: euler_deg ≈ [180, 90, 0]
- 正面抓: euler_deg ≈ [90, 0, 0]
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

# 获取 object_config.json 的路径
_CURRENT_DIR = Path(__file__).parent
_OBJECT_CONFIG_PATH = _CURRENT_DIR / "scenes" / "object_config.json"

# 全局缓存，避免重复加载 JSON 文件
_object_config_cache: Optional[Dict[str, Any]] = None


def load_object_config() -> Dict[str, Any]:
    """
    加载 object_config.json 配置文件
    
    Returns:
        完整的配置字典
    """
    global _object_config_cache
    
    if _object_config_cache is not None:
        return _object_config_cache
    
    if not _OBJECT_CONFIG_PATH.exists():
        print(f"[Warning] 配置文件不存在: {_OBJECT_CONFIG_PATH}")
        return {}
    
    with open(_OBJECT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        _object_config_cache = json.load(f)
    
    return _object_config_cache


def get_grasp_config_from_json(object_name: str, operation: str = "grasp") -> Dict[str, Any]:
    """
    从 object_config.json 获取指定物体的操作配置
    
    Args:
        object_name: 物体名称 (如 "glass_beaker_100ml", "mortar")
        operation: 操作类型 (如 "grasp", "handover", "pick_place", "pour" 等)
        
    Returns:
        包含操作配置的字典，包括:
        - offset/grasp_offset: 位置偏移
        - euler_deg/grasp_euler_deg: 欧拉角
        - lift_height: 抬起高度 (仅 grasp 操作)
        - timing: 时序配置
        - 其他操作特定参数
    """
    config = load_object_config()
    
    if not config:
        print(f"[Warning] 无法加载配置文件，使用默认配置")
        return get_default_grasp_config()
    
    objects = config.get("objects", {})
    
    if object_name not in objects:
        print(f"[Warning] 未找到物体 '{object_name}' 的配置，使用默认配置")
        return get_default_grasp_config()
    
    obj_config = objects[object_name]
    operation_config = obj_config.get(operation)
    
    if operation_config is None:
        print(f"[Warning] 物体 '{object_name}' 不支持操作 '{operation}'，使用默认配置")
        return get_default_grasp_config()
    
    # 标准化输出格式
    result = {
        "object_name": object_name,
        "object_name_cn": obj_config.get("name_cn", object_name),
        "description": obj_config.get("description", ""),
        "operation": operation,
    }
    
    # 提取抓取操作特定参数
    if operation == "grasp":
        result["offset"] = operation_config.get("grasp_offset", [-0.02, -0.19, 0.05])
        result["euler_deg"] = operation_config.get("grasp_euler_deg", [157.43, 69.16, -154.0])
        result["lift_height"] = operation_config.get("lift_height", 0.3)
        result["timing"] = operation_config.get("timing", {
            "phases": ["approach", "grasp", "lift"],
            "approach_ratio": 0.4,
            "grasp_ratio": 0.2,
            "lift_ratio": 0.4
        })
    else:
        # 其他操作类型的参数提取
        result["config"] = operation_config
    
    # 添加物理材质和边界框信息
    result["mass"] = obj_config.get("mass", 0.1)
    result["physics_material"] = obj_config.get("physics_material", {})
    result["bbox"] = obj_config.get("bbox", {})
    
    return result


def get_default_grasp_config() -> Dict[str, Any]:
    """获取默认的抓取配置"""
    return {
        "object_name": "default",
        "object_name_cn": "默认物体",
        "description": "默认抓取配置",
        "operation": "grasp",
        "offset": [-0.02, -0.19, 0.05],
        "euler_deg": [157.43, 69.16, -154.0],
        "lift_height": 0.3,
        "timing": {
            "phases": ["approach", "grasp", "lift"],
            "approach_ratio": 0.4,
            "grasp_ratio": 0.2,
            "lift_ratio": 0.4
        },
        "mass": 0.1,
        "physics_material": {},
        "bbox": {}
    }


def list_available_objects() -> list:
    """列出所有可用的物体名称"""
    config = load_object_config()
    if not config:
        return []
    return list(config.get("objects", {}).keys())


def get_object_supported_operations(object_name: str) -> list:
    """获取指定物体支持的操作类型"""
    config = load_object_config()
    if not config:
        return []
    
    objects = config.get("objects", {})
    if object_name not in objects:
        return []
    
    return objects[object_name].get("supported_operations", [])

# ========== 烧杯系列 ==========
BEAKER_CONFIGS = {
    "glass_beaker_50ml": {
        "offset": [-0.14,-0.06,0.001],
        "euler_deg":  [-90.0, 0.0, -90.0],
        "description": "50ml 玻璃烧杯"
    },
    "glass_beaker_100ml": {
        # "offset": [-0.02, -0.19, 0.05],#绝对位置【0.48，-0.2950，0.9105】-0.06,-0.1776909,0.1
        "offset":[-0.14,-0.06,0.001],
        # "euler_deg": [157.43, 80.16, -154.00],
        "euler_deg":  [-90.0, 0.0, -90.0],
        # "euler_deg": [150, 100, -150],
        # "euler_deg": [0.0, 0.0, 0.0],
        "description": "100ml 玻璃烧杯"
    },
    "glass_beaker_250ml": {
        "offset": [-0.14,-0.06,0.001],
        "euler_deg":  [-90.0, 0.0, -90.0],
        "description": "250ml 玻璃烧杯"
    },
    "glass_beaker_500ml": {
        "offset": [-0.14,-0.06,0.001],
        "euler_deg":  [-90.0, 0.0, -90.0],
        "description": "500ml 玻璃烧杯"
    },
    "plastic_beaker_50ml": {
        "offset": [-0.14,-0.06,0.001],
        "euler_deg":  [-90.0, 0.0, -90.0],
        "description": "50ml 塑料烧杯"
    },
}

# ========== 试管系列 ==========
TEST_TUBE_CONFIGS = {
    "glass_test_tube_20ml": {
        "offset": [-0.01, -0.15, 0.03],
        "euler_deg": [180.0, 80.0, -90.0],
        "description": "20ml 玻璃试管"
    },
    "glass_test_tube_50ml": {
        "offset": [-0.01, -0.16, 0.04],
        "euler_deg": [180.0, 80.0, -90.0],
        "description": "50ml 玻璃试管"
    },
}

# ========== 量筒系列 ==========
CYLINDER_CONFIGS = {
    "glass_cylinder_100ml": {
        "offset": [-0.02, -0.20, 0.06],
        "euler_deg": [160.0, 75.0, -150.0],
        "description": "100ml 玻璃量筒"
    },
    "plastic_cylinder_100ml": {
        "offset": [-0.02, -0.20, 0.06],
        "euler_deg": [160.0, 75.0, -150.0],
        "description": "100ml 塑料量筒"
    },
}

# ========== 容量瓶系列 ==========
VOLUMETRIC_FLASK_CONFIGS = {
    "clear_volumetric_flask_250ml": {
        "offset": [-0.02, -0.20, 0.06],
        "euler_deg": [157.43, 69.16, -154.00],
        "description": "250ml 透明容量瓶"
    },
    "clear_volumetric_flask_500ml": {
        "offset": [-0.03, -0.22, 0.07],
        "euler_deg": [157.43, 69.16, -154.00],
        "description": "500ml 透明容量瓶"
    },
    "clear_volumetric_flask_1000ml": {
        "offset": [-0.04, -0.25, 0.08],
        "euler_deg": [157.43, 69.16, -154.00],
        "description": "1000ml 透明容量瓶"
    },
    "brown_volumetric_flask_250ml": {
        "offset": [-0.02, -0.20, 0.06],
        "euler_deg": [157.43, 69.16, -154.00],
        "description": "250ml 棕色容量瓶"
    },
}

# ========== 试剂瓶系列 ==========
REAGENT_BOTTLE_CONFIGS = {
    "clear_reagent_bottle_small": {
        "offset": [-0.02, -0.18, 0.05],
        "euler_deg": [157.43, 69.16, -154.00],
        "description": "透明试剂瓶(小)"
    },
    "clear_reagent_bottle_large": {
        "offset": [-0.03, -0.22, 0.06],
        "euler_deg": [157.43, 69.16, -154.00],
        "description": "透明试剂瓶(大)"
    },
    "brown_reagent_bottle_small": {
        "offset": [-0.02, -0.18, 0.05],
        "euler_deg": [157.43, 69.16, -154.00],
        "description": "棕色试剂瓶(小)"
    },
    "brown_reagent_bottle_large": {
        "offset": [-0.03, -0.22, 0.06],
        "euler_deg": [157.43, 69.16, -154.00],
        "description": "棕色试剂瓶(大)"
    },
}

# ========== 锥形瓶系列 ==========
ERLENMEYER_FLASK_CONFIGS = {
    "erlenmeyer_flask": {
        "offset": [-0.02, -0.20, 0.06],
        "euler_deg": [160.0, 70.0, -150.0],
        "description": "锥形瓶"
    },
}

# ========== 滴定管/滴管系列 ==========
DROPPER_CONFIGS = {
    "dropper": {
        "offset": [-0.01, -0.12, 0.02],
        "euler_deg": [180.0, 85.0, -90.0],
        "description": "胶头滴管"
    },
    "burette": {
        "offset": [-0.01, -0.14, 0.15],
        "euler_deg": [180.0, 88.0, -90.0],
        "description": "滴定管（细长，需要从顶部抓取）"
    },
}

# ========== 冷凝管系列 ==========
CONDENSER_CONFIGS = {
    "spherical_condenser": {
        "offset": [-0.02, -0.18, 0.10],
        "euler_deg": [180.0, 85.0, -90.0],
        "description": "球形冷凝管"
    },
    "straight_condenser": {
        "offset": [-0.02, -0.16, 0.12],
        "euler_deg": [180.0, 85.0, -90.0],
        "description": "直型冷凝管"
    },
    "coiled_condenser": {
        "offset": [-0.02, -0.18, 0.10],
        "euler_deg": [180.0, 85.0, -90.0],
        "description": "蛇型冷凝管"
    },
}

# ========== 支架系列 ==========
STAND_CONFIGS = {
    "iron_stand": {
        "offset": [-0.03, -0.20, 0.15],
        "euler_deg": [160.0, 70.0, -150.0],
        "description": "铁架台"
    },
    "funnel_stand": {
        "offset": [-0.02, -0.18, 0.10],
        "euler_deg": [160.0, 70.0, -150.0],
        "description": "漏斗架"
    },
}

# ========== 实验耗材系列 ==========
CONSUMABLE_CONFIGS = {
    "weighing_paper": {
        "offset": [-0.01, -0.12, 0.01],
        "euler_deg": [180.0, 90.0, 0.0],
        "description": "称量纸（薄片，顶部抓取）"
    },
    "asbestos_gauze": {
        "offset": [-0.02, -0.15, 0.02],
        "euler_deg": [180.0, 90.0, 0.0],
        "description": "石棉网（薄片，顶部抓取）"
    },
}

# ========== 测量仪器系列 ==========
INSTRUMENT_CONFIGS = {
    "hygrothermometer": {
        "offset": [-0.02, -0.16, 0.05],
        "euler_deg": [170.0, 75.0, -140.0],
        "description": "温湿度计"
    },
}

# ========== 研磨器具系列 ==========
MORTAR_CONFIGS = {
    "mortar": {
        # "offset": [-0.03, -0.22, 0.06],
        # "euler_deg": [160.0, 70.0, -150.0],
        "offset": [-0.0574, -0.1388, 0.0266],
        "euler_deg": [-95.43, -43.63, -44.42],
        "description": "坩埚/研钵"
    },
    "pestle": {
        "offset": [-0.01, -0.15, 0.03],
        "euler_deg": [180.0, 80.0, -90.0],
        "description": "坩埚搅拌棒/研杵"
    },
}

# ========== 特殊烧瓶系列 ==========
SPECIAL_FLASK_CONFIGS = {
    "three_neck_flask": {
        "offset": [-0.04, -0.25, 0.08],
        "euler_deg": [160.0, 70.0, -150.0],
        "description": "三口烧瓶"
    },
}

# ========== 汇总所有配置 ==========
GRASP_CONFIGS = {
    **BEAKER_CONFIGS,
    **TEST_TUBE_CONFIGS,
    **CYLINDER_CONFIGS,
    **VOLUMETRIC_FLASK_CONFIGS,
    **REAGENT_BOTTLE_CONFIGS,
    **ERLENMEYER_FLASK_CONFIGS,
    **DROPPER_CONFIGS,
    **CONDENSER_CONFIGS,
    **STAND_CONFIGS,
    **CONSUMABLE_CONFIGS,
    **INSTRUMENT_CONFIGS,
    **MORTAR_CONFIGS,
    **SPECIAL_FLASK_CONFIGS,
}

# ========== 默认配置 ==========
DEFAULT_CONFIG = {
    "offset": [-0.02, -0.19, 0.05],
    "euler_deg": [157.43, 69.16, -154.00],
    "description": "默认抓取配置"
}


def get_grasp_config(object_name: str) -> dict:
    """
    获取指定物体的抓取配置
    
    Args:
        object_name: 物体名称 (如 "glass_beaker_100ml")
        
    Returns:
        包含 offset, euler_deg, description 的配置字典
    """
    if object_name in GRASP_CONFIGS:
        return GRASP_CONFIGS[object_name]
    else:
        print(f"[Warning] 未找到 '{object_name}' 的抓取配置，使用默认配置")
        return DEFAULT_CONFIG


def list_available_configs():
    """列出所有可用的抓取配置"""
    print("=" * 50)
    print("可用的抓取配置:")
    print("=" * 50)
    for name, config in GRASP_CONFIGS.items():
        print(f"  - {name}: {config.get('description', 'N/A')}")
    print("=" * 50)


if __name__ == "__main__":
    # 测试
    list_available_configs()
    
    # 获取配置示例
    config = get_grasp_config("glass_beaker_100ml")
    print(f"\nglass_beaker_100ml 配置:")
    print(f"  offset: {config['offset']}")
    print(f"  euler_deg: {config['euler_deg']}")

