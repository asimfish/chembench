# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-12-18
# Description: 物体配置加载器，从 object_config.json 读取操作参数

"""
配置加载器模块

提供从 object_config.json 读取物体操作参数的功能。
每个任务文件可以直接导入此模块并调用相应函数获取配置。

使用示例:
    from ..config_loader import load_grasp_config, load_handover_config
    
    # 加载抓取配置
    config = load_grasp_config("mortar")
    
    # 访问参数
    offset = config["grasp_offset"]
    euler_deg = config["grasp_euler_deg"]
    lift_height = config["lift_height"]
    timing = config["timing"]
"""

import json
import os
from typing import Any
from functools import lru_cache

# 配置文件路径
_CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "object_config.json")


@lru_cache(maxsize=1)
def _load_config_file() -> dict:
    """
    加载并缓存 JSON 配置文件
    
    Returns:
        完整的配置字典
    """
    with open(_CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_object_names() -> list[str]:
    """
    获取所有可用的物体名称
    
    Returns:
        物体名称列表
    """
    config = _load_config_file()
    return list(config.get("objects", {}).keys())


def get_object_info(object_name: str) -> dict | None:
    """
    获取物体的基本信息
    
    Args:
        object_name: 物体名称
        
    Returns:
        物体信息字典，如果不存在则返回 None
    """
    config = _load_config_file()
    return config.get("objects", {}).get(object_name)


def get_supported_operations(object_name: str) -> list[str]:
    """
    获取物体支持的操作列表
    
    Args:
        object_name: 物体名称
        
    Returns:
        支持的操作列表
    """
    obj_info = get_object_info(object_name)
    if obj_info is None:
        return []
    return obj_info.get("supported_operations", [])


def load_grasp_config(object_name: str) -> dict[str, Any]:
    """
    加载物体的抓取(grasp)配置
    
    Args:
        object_name: 物体名称，如 "mortar", "glass_beaker_100ml" 等
        
    Returns:
        抓取配置字典，包含:
        - grasp_offset: [x, y, z] 抓取位置偏移
        - grasp_euler_deg: [roll, pitch, yaw] 抓取角度
        - lift_height: 抬起高度
        - timing: 时序参数字典
        - name_cn: 物体中文名称（用于输出文件夹命名）
        
    Raises:
        ValueError: 如果物体不存在或不支持抓取操作
    """
    obj_info = get_object_info(object_name)
    
    if obj_info is None:
        raise ValueError(f"物体 '{object_name}' 不存在于配置文件中")
    
    if "grasp" not in obj_info:
        raise ValueError(f"物体 '{object_name}' 不支持抓取(grasp)操作")
    
    grasp = obj_info["grasp"]
    
    return {
        "grasp_offset": grasp.get("grasp_offset", [0, 0, 0]),
        "grasp_euler_deg": grasp.get("grasp_euler_deg", [0, 0, 0]),
        "lift_height": grasp.get("lift_height", 0.3),
        "timing": grasp.get("timing", {
            "approach_ratio": 0.4,
            "grasp_ratio": 0.2,
            "lift_ratio": 0.4
        }),
        "name_cn": obj_info.get("name_cn", object_name),
    }


def get_available_grasp_points(object_name: str) -> list[int]:
    """
    获取物体可用的多抓取点配置数量列表
    
    Args:
        object_name: 物体名称
        
    Returns:
        可用的抓取点数量列表，如 [4, 8, 12]
    """
    obj_info = get_object_info(object_name)
    
    if obj_info is None or "grasp" not in obj_info:
        return []
    
    grasp = obj_info["grasp"]
    available = []
    
    # 查找所有 grasp_points_N 配置
    for key in grasp.keys():
        if key.startswith("grasp_points_"):
            try:
                num_points = int(key.split("_")[-1])
                available.append(num_points)
            except ValueError:
                continue
    
    return sorted(available)


def load_grasp_points(object_name: str, num_points: int) -> dict[str, Any]:
    """
    加载物体的多抓取点配置
    
    Args:
        object_name: 物体名称
        num_points: 抓取点数量（如 4, 6, 8, 12）
        
    Returns:
        多抓取点配置字典，包含:
        - num_points: 抓取点数量
        - rotation_axis: 旋转轴
        - points: 抓取点列表，每个点包含 index, angle_deg, grasp_offset, grasp_euler_deg
        - lift_height: 抬起高度
        - timing: 时序参数
        - name_cn: 物体中文名称
        
    Raises:
        ValueError: 如果物体不存在或没有指定数量的抓取点配置
    """
    obj_info = get_object_info(object_name)
    
    if obj_info is None:
        raise ValueError(f"物体 '{object_name}' 不存在于配置文件中")
    
    if "grasp" not in obj_info:
        raise ValueError(f"物体 '{object_name}' 不支持抓取(grasp)操作")
    
    grasp = obj_info["grasp"]
    key = f"grasp_points_{num_points}"
    
    if key not in grasp:
        available = get_available_grasp_points(object_name)
        raise ValueError(
            f"物体 '{object_name}' 没有 {num_points} 点抓取配置。"
            f"可用配置: {available if available else '无'}"
        )
    
    grasp_points_config = grasp[key]
    
    return {
        "num_points": grasp_points_config.get("num_points", num_points),
        "rotation_axis": grasp_points_config.get("rotation_axis", "z"),
        "points": grasp_points_config.get("points", []),
        "lift_height": grasp.get("lift_height", 0.3),
        "timing": grasp.get("timing", {
            "approach_ratio": 0.4,
            "grasp_ratio": 0.2,
            "lift_ratio": 0.4
        }),
        "name_cn": obj_info.get("name_cn", object_name),
    }


def get_grasp_point_by_index(object_name: str, num_points: int, point_index: int) -> dict[str, Any]:
    """
    获取指定索引的单个抓取点配置
    
    Args:
        object_name: 物体名称
        num_points: 抓取点总数（配置版本）
        point_index: 抓取点索引（0 到 num_points-1）
        
    Returns:
        单个抓取点配置:
        - grasp_offset: [x, y, z]
        - grasp_euler_deg: [roll, pitch, yaw]
        - angle_deg: 旋转角度
        - index: 点索引
        - lift_height: 抬起高度
        - timing: 时序参数
        - name_cn: 物体中文名称
    """
    config = load_grasp_points(object_name, num_points)
    points = config["points"]
    
    if not points:
        raise ValueError(f"物体 '{object_name}' 的 {num_points} 点配置为空")
    
    # 使用取模实现周期性索引
    actual_index = point_index % len(points)
    point = points[actual_index]
    
    return {
        "grasp_offset": point["grasp_offset"],
        "grasp_euler_deg": point["grasp_euler_deg"],
        "angle_deg": point["angle_deg"],
        "index": point["index"],
        "lift_height": config["lift_height"],
        "timing": config["timing"],
        "name_cn": config["name_cn"],
    }


def load_handover_config(object_name: str) -> dict[str, Any]:
    """
    加载物体的双手交接(handover)配置
    
    Args:
        object_name: 物体名称
        
    Returns:
        交接配置字典
        
    Raises:
        ValueError: 如果物体不存在或不支持交接操作
    """
    obj_info = get_object_info(object_name)
    
    if obj_info is None:
        raise ValueError(f"物体 '{object_name}' 不存在于配置文件中")
    
    if "handover" not in obj_info:
        raise ValueError(f"物体 '{object_name}' 不支持交接(handover)操作")
    
    handover = obj_info["handover"]
    
    return {
        "right_grasp_offset": handover.get("right_grasp_offset", [0, 0, 0]),
        "right_grasp_euler_deg": handover.get("right_grasp_euler_deg", [0, 0, 0]),
        "handover_position": handover.get("handover_position", [0.4, 0.0, 1.0]),
        "right_handover_euler_deg": handover.get("right_handover_euler_deg", [90, 45, 0]),
        "left_grasp_offset": handover.get("left_grasp_offset", [0, 0, 0]),
        "left_grasp_euler_deg": handover.get("left_grasp_euler_deg", [-90, 45, 0]),
        "left_target_position": handover.get("left_target_position", [0.4, 0.25, 0.95]),
        "left_target_euler_deg": handover.get("left_target_euler_deg", [-90, 45, 0]),
        "right_reset_position": handover.get("right_reset_position", [0.4, 0.25, 0.95]),
        "right_reset_euler_deg": handover.get("right_reset_euler_deg", [-90, 45, 0]),
        "timing": handover.get("timing", {}),
        "name_cn": obj_info.get("name_cn", object_name),
    }


def load_pick_place_config(object_name: str) -> dict[str, Any]:
    """
    加载物体的抓放(pick_place)配置
    
    Args:
        object_name: 物体名称
        
    Returns:
        抓放配置字典
        
    Raises:
        ValueError: 如果物体不存在或不支持抓放操作
    """
    obj_info = get_object_info(object_name)
    
    if obj_info is None:
        raise ValueError(f"物体 '{object_name}' 不存在于配置文件中")
    
    if "pick_place" not in obj_info:
        raise ValueError(f"物体 '{object_name}' 不支持抓放(pick_place)操作")
    
    pick_place = obj_info["pick_place"]
    
    return {
        "right_grasp_offset": pick_place.get("right_grasp_offset", [0, 0, 0]),
        "right_grasp_euler_deg": pick_place.get("right_grasp_euler_deg", [0, 0, 0]),
        "place_offset": pick_place.get("place_offset", [0, 0, 0]),
        "place_euler_deg": pick_place.get("place_euler_deg", [0, 0, 0]),
        "right_reset_position": pick_place.get("right_reset_position", [0.4, 0.25, 0.95]),
        "right_reset_euler_deg": pick_place.get("right_reset_euler_deg", [-90, 45, 0]),
        "timing": pick_place.get("timing", {}),
        "name_cn": obj_info.get("name_cn", object_name),
    }


def load_pour_config(object_name: str) -> dict[str, Any]:
    """
    加载物体的倾倒(pour)配置
    
    Args:
        object_name: 物体名称
        
    Returns:
        倾倒配置字典
        
    Raises:
        ValueError: 如果物体不存在或不支持倾倒操作
    """
    obj_info = get_object_info(object_name)
    
    if obj_info is None:
        raise ValueError(f"物体 '{object_name}' 不存在于配置文件中")
    
    if "pour" not in obj_info:
        raise ValueError(f"物体 '{object_name}' 不支持倾倒(pour)操作")
    
    pour = obj_info["pour"]
    
    return {
        "right_grasp_offset": pour.get("right_grasp_offset", [0, 0, 0]),
        "right_grasp_euler_deg": pour.get("right_grasp_euler_deg", [0, 0, 0]),
        "pour_position_offset": pour.get("pour_position_offset", [0, 0, 0]),
        "pour_euler_deg": pour.get("pour_euler_deg", [0, 0, 0]),
        "right_reset_position": pour.get("right_reset_position", [0.4, 0.25, 0.95]),
        "right_reset_euler_deg": pour.get("right_reset_euler_deg", [-90, 45, 0]),
        "timing": pour.get("timing", {}),
        "name_cn": obj_info.get("name_cn", object_name),
    }


def load_operation_config(object_name: str, operation: str) -> dict[str, Any]:
    """
    通用操作配置加载函数
    
    Args:
        object_name: 物体名称
        operation: 操作类型 ("grasp", "handover", "pick_place", "pour" 等)
        
    Returns:
        操作配置字典
        
    Raises:
        ValueError: 如果物体或操作不存在
    """
    loaders = {
        "grasp": load_grasp_config,
        "handover": load_handover_config,
        "pick_place": load_pick_place_config,
        "pour": load_pour_config,
    }
    
    loader = loaders.get(operation)
    if loader is None:
        raise ValueError(f"不支持的操作类型: {operation}")
    
    return loader(object_name)


def get_metadata() -> dict:
    """
    获取配置文件的元数据
    
    Returns:
        元数据字典
    """
    config = _load_config_file()
    return config.get("metadata", {})


def get_default_positions() -> dict:
    """
    获取默认位置配置
    
    Returns:
        默认位置字典
    """
    metadata = get_metadata()
    return metadata.get("default_positions", {})

