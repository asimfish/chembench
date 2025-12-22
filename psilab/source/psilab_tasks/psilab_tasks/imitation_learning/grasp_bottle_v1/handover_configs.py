# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-12-13
# Description: 双手交接配置文件，存储不同物体的双手交接参数

"""
双手交接配置参数说明:

交接过程分为几个阶段:
1. 抓取阶段 (grasp): 主手抓取物体
2. 移动阶段 (transfer): 主手移动到交接位置
3. 接收阶段 (receive): 副手准备接收
4. 释放阶段 (release): 主手释放，副手抓取

配置参数:
- grasp_hand: 首先抓取物体的手 ("left" 或 "right")
- receive_hand: 接收物体的手 ("left" 或 "right")

- grasp_offset: [x, y, z] 抓取时相对于物体中心的偏移 (单位: 米)
- grasp_euler_deg: [roll, pitch, yaw] 抓取时末端执行器的欧拉角 (单位: 度)

- handover_pos: [x, y, z] 交接位置（相对于机器人基座）(单位: 米)
- handover_euler_deg_giver: [roll, pitch, yaw] 交出方的末端执行器欧拉角
- handover_euler_deg_receiver: [roll, pitch, yaw] 接收方的末端执行器欧拉角

- receive_offset: [x, y, z] 接收时相对于物体中心的偏移 (单位: 米)

- timing: 时间配置 (占总时间的比例)
    - grasp_ratio: 抓取阶段占比
    - transfer_ratio: 移动阶段占比
    - handover_ratio: 交接阶段占比
    - retreat_ratio: 撤退阶段占比
"""

# ========== 烧杯系列 ==========
BEAKER_HANDOVER_CONFIGS = {
    "glass_beaker_50ml": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.02, -0.18, 0.04],
        "grasp_euler_deg": [157.43, 69.16, -154.00],
        "handover_pos": [0.0, 0.0, 1.0],
        "handover_euler_deg_giver": [90.0, 45.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 45.0, 0.0],
        "receive_offset": [0.02, 0.18, 0.04],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "50ml 玻璃烧杯双手交接"
    },
    "glass_beaker_100ml": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.02, -0.19, 0.05],
        "grasp_euler_deg": [157.43, 80.16, -154.00],
        "handover_pos": [0.0, 0.0, 1.0],
        "handover_euler_deg_giver": [90.0, 45.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 45.0, 0.0],
        "receive_offset": [0.02, 0.19, 0.05],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "100ml 玻璃烧杯双手交接"
    },
    "glass_beaker_250ml": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.03, -0.22, 0.06],
        "grasp_euler_deg": [160.0, 70.0, -150.0],
        "handover_pos": [0.0, 0.0, 1.05],
        "handover_euler_deg_giver": [90.0, 45.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 45.0, 0.0],
        "receive_offset": [0.03, 0.22, 0.06],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "250ml 玻璃烧杯双手交接"
    },
    "glass_beaker_500ml": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.04, -0.25, 0.08],
        "grasp_euler_deg": [160.0, 70.0, -150.0],
        "handover_pos": [0.0, 0.0, 1.1],
        "handover_euler_deg_giver": [90.0, 45.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 45.0, 0.0],
        "receive_offset": [0.04, 0.25, 0.08],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "500ml 玻璃烧杯双手交接"
    },
}

# ========== 试管系列 ==========
TEST_TUBE_HANDOVER_CONFIGS = {
    "glass_test_tube_20ml": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.01, -0.15, 0.03],
        "grasp_euler_deg": [180.0, 80.0, -90.0],
        "handover_pos": [0.0, 0.0, 0.95],
        "handover_euler_deg_giver": [90.0, 60.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 60.0, 0.0],
        "receive_offset": [0.01, 0.15, 0.03],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "20ml 玻璃试管双手交接"
    },
    "glass_test_tube_50ml": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.01, -0.16, 0.04],
        "grasp_euler_deg": [180.0, 80.0, -90.0],
        "handover_pos": [0.0, 0.0, 0.95],
        "handover_euler_deg_giver": [90.0, 60.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 60.0, 0.0],
        "receive_offset": [0.01, 0.16, 0.04],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "50ml 玻璃试管双手交接"
    },
}

# ========== 容量瓶系列 ==========
VOLUMETRIC_FLASK_HANDOVER_CONFIGS = {
    "clear_volumetric_flask_250ml": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.02, -0.20, 0.06],
        "grasp_euler_deg": [157.43, 69.16, -154.00],
        "handover_pos": [0.0, 0.0, 1.0],
        "handover_euler_deg_giver": [90.0, 45.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 45.0, 0.0],
        "receive_offset": [0.02, 0.20, 0.06],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "250ml 透明容量瓶双手交接"
    },
    "clear_volumetric_flask_500ml": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.03, -0.22, 0.07],
        "grasp_euler_deg": [157.43, 69.16, -154.00],
        "handover_pos": [0.0, 0.0, 1.05],
        "handover_euler_deg_giver": [90.0, 45.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 45.0, 0.0],
        "receive_offset": [0.03, 0.22, 0.07],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "500ml 透明容量瓶双手交接"
    },
}

# ========== 锥形瓶系列 ==========
ERLENMEYER_FLASK_HANDOVER_CONFIGS = {
    "erlenmeyer_flask": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.02, -0.20, 0.06],
        "grasp_euler_deg": [160.0, 70.0, -150.0],
        "handover_pos": [0.0, 0.0, 1.0],
        "handover_euler_deg_giver": [90.0, 45.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 45.0, 0.0],
        "receive_offset": [0.02, 0.20, 0.06],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "锥形瓶双手交接"
    },
}

# ========== 其他器材 ==========
OTHER_HANDOVER_CONFIGS = {
    "dropper": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.01, -0.12, 0.02],
        "grasp_euler_deg": [180.0, 85.0, -90.0],
        "handover_pos": [0.0, 0.0, 0.9],
        "handover_euler_deg_giver": [90.0, 70.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 70.0, 0.0],
        "receive_offset": [0.01, 0.12, 0.02],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "胶头滴管双手交接"
    },
    "pestle": {
        "grasp_hand": "right",
        "receive_hand": "left",
        "grasp_offset": [-0.01, -0.15, 0.03],
        "grasp_euler_deg": [180.0, 80.0, -90.0],
        "handover_pos": [0.0, 0.0, 0.95],
        "handover_euler_deg_giver": [90.0, 60.0, 0.0],
        "handover_euler_deg_receiver": [-90.0, 60.0, 0.0],
        "receive_offset": [0.01, 0.15, 0.03],
        "timing": {
            "grasp_ratio": 0.25,
            "transfer_ratio": 0.25,
            "handover_ratio": 0.25,
            "retreat_ratio": 0.25
        },
        "description": "研杵双手交接"
    },
}

# ========== 汇总所有配置 ==========
HANDOVER_CONFIGS = {
    **BEAKER_HANDOVER_CONFIGS,
    **TEST_TUBE_HANDOVER_CONFIGS,
    **VOLUMETRIC_FLASK_HANDOVER_CONFIGS,
    **ERLENMEYER_FLASK_HANDOVER_CONFIGS,
    **OTHER_HANDOVER_CONFIGS,
}

# ========== 默认配置 ==========
DEFAULT_HANDOVER_CONFIG = {
    "grasp_hand": "right",
    "receive_hand": "left",
    "grasp_offset": [-0.02, -0.19, 0.05],
    "grasp_euler_deg": [157.43, 69.16, -154.00],
    "handover_pos": [0.0, 0.0, 1.0],
    "handover_euler_deg_giver": [90.0, 45.0, 0.0],
    "handover_euler_deg_receiver": [-90.0, 45.0, 0.0],
    "receive_offset": [0.02, 0.19, 0.05],
    "timing": {
        "grasp_ratio": 0.25,
        "transfer_ratio": 0.25,
        "handover_ratio": 0.25,
        "retreat_ratio": 0.25
    },
    "description": "默认双手交接配置"
}


def get_handover_config(object_name: str) -> dict:
    """
    获取指定物体的双手交接配置
    
    Args:
        object_name: 物体名称 (如 "glass_beaker_100ml")
        
    Returns:
        包含交接配置的字典
    """
    if object_name in HANDOVER_CONFIGS:
        return HANDOVER_CONFIGS[object_name]
    else:
        print(f"[Warning] 未找到 '{object_name}' 的双手交接配置，使用默认配置")
        return DEFAULT_HANDOVER_CONFIG


def list_available_handover_configs():
    """列出所有可用的双手交接配置"""
    print("=" * 60)
    print("可用的双手交接配置:")
    print("=" * 60)
    for name, config in HANDOVER_CONFIGS.items():
        print(f"  - {name}: {config.get('description', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    # 测试
    list_available_handover_configs()
    
    # 获取配置示例
    config = get_handover_config("glass_beaker_100ml")
    print(f"\nglass_beaker_100ml 双手交接配置:")
    print(f"  grasp_hand: {config['grasp_hand']}")
    print(f"  receive_hand: {config['receive_hand']}")
    print(f"  handover_pos: {config['handover_pos']}")
    print(f"  timing: {config['timing']}")

