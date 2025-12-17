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
        "offset": [-0.03, -0.22, 0.06],
        "euler_deg": [160.0, 70.0, -150.0],
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

