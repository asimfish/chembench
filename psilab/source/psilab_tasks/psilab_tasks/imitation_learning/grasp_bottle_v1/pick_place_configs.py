# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-12-13
# Description: Pick and Place 配置文件，存储不同物体的抓取放置参数

"""
Pick and Place 配置参数说明:

任务过程分为几个阶段:
1. 接近阶段 (approach): 移动到物体上方
2. 抓取阶段 (grasp): 下降并抓取物体
3. 抬起阶段 (lift): 抬起物体
4. 移动阶段 (transfer): 移动到目标位置上方
5. 放置阶段 (place): 下降并放置物体
6. 撤退阶段 (retreat): 抬起并撤退

配置参数:
- pick_offset: [x, y, z] 抓取时相对于物体中心的偏移 (单位: 米)
- pick_euler_deg: [roll, pitch, yaw] 抓取时末端执行器的欧拉角 (单位: 度)
- pick_approach_height: 接近物体时的高度 (单位: 米)

- place_offset: [x, y, z] 放置时相对于目标位置的偏移 (单位: 米)
- place_euler_deg: [roll, pitch, yaw] 放置时末端执行器的欧拉角 (单位: 度)
- place_approach_height: 接近放置点时的高度 (单位: 米)

- lift_height: 抬起物体的高度 (单位: 米)
- transfer_height: 移动过程中的高度 (单位: 米)

- gripper_open_width: 手指张开宽度比例 (0-1)
- gripper_close_width: 手指闭合宽度比例 (0-1)

- timing: 时间配置 (占总时间的比例)
    - approach_ratio: 接近阶段占比
    - grasp_ratio: 抓取阶段占比
    - lift_ratio: 抬起阶段占比
    - transfer_ratio: 移动阶段占比
    - place_ratio: 放置阶段占比
    - retreat_ratio: 撤退阶段占比

- speed: 速度配置
    - approach_speed: 接近速度 (单位: m/s)
    - transfer_speed: 移动速度 (单位: m/s)
    - place_speed: 放置速度 (单位: m/s)
"""

# ========== 烧杯系列 ==========
BEAKER_PICK_PLACE_CONFIGS = {
    "glass_beaker_50ml": {
        "pick_offset": [-0.02, -0.18, 0.04],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.15,
        "place_offset": [-0.02, -0.18, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.15,
        "lift_height": 0.10,
        "transfer_height": 0.20,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.3,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.2,
            "transfer_speed": 0.3,
            "place_speed": 0.15
        },
        "description": "50ml 玻璃烧杯 Pick and Place"
    },
    "glass_beaker_100ml": {
        "pick_offset": [-0.02, -0.19, 0.05],
        "pick_euler_deg": [157.43, 80.16, -154.00],
        "pick_approach_height": 0.15,
        "place_offset": [-0.02, -0.19, 0.02],
        "place_euler_deg": [157.43, 80.16, -154.00],
        "place_approach_height": 0.15,
        "lift_height": 0.12,
        "transfer_height": 0.22,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.35,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.2,
            "transfer_speed": 0.3,
            "place_speed": 0.15
        },
        "description": "100ml 玻璃烧杯 Pick and Place"
    },
    "glass_beaker_250ml": {
        "pick_offset": [-0.03, -0.22, 0.06],
        "pick_euler_deg": [160.0, 70.0, -150.0],
        "pick_approach_height": 0.18,
        "place_offset": [-0.03, -0.22, 0.02],
        "place_euler_deg": [160.0, 70.0, -150.0],
        "place_approach_height": 0.18,
        "lift_height": 0.15,
        "transfer_height": 0.25,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.4,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.18,
            "transfer_speed": 0.25,
            "place_speed": 0.12
        },
        "description": "250ml 玻璃烧杯 Pick and Place"
    },
    "glass_beaker_500ml": {
        "pick_offset": [-0.04, -0.25, 0.08],
        "pick_euler_deg": [160.0, 70.0, -150.0],
        "pick_approach_height": 0.20,
        "place_offset": [-0.04, -0.25, 0.02],
        "place_euler_deg": [160.0, 70.0, -150.0],
        "place_approach_height": 0.20,
        "lift_height": 0.18,
        "transfer_height": 0.28,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.5,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.15,
            "transfer_speed": 0.2,
            "place_speed": 0.1
        },
        "description": "500ml 玻璃烧杯 Pick and Place"
    },
    "plastic_beaker_50ml": {
        "pick_offset": [-0.02, -0.18, 0.04],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.15,
        "place_offset": [-0.02, -0.18, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.15,
        "lift_height": 0.10,
        "transfer_height": 0.20,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.3,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.2,
            "transfer_speed": 0.3,
            "place_speed": 0.15
        },
        "description": "50ml 塑料烧杯 Pick and Place"
    },
}

# ========== 试管系列 ==========
TEST_TUBE_PICK_PLACE_CONFIGS = {
    "glass_test_tube_20ml": {
        "pick_offset": [-0.01, -0.15, 0.03],
        "pick_euler_deg": [180.0, 80.0, -90.0],
        "pick_approach_height": 0.12,
        "place_offset": [-0.01, -0.15, 0.01],
        "place_euler_deg": [180.0, 80.0, -90.0],
        "place_approach_height": 0.12,
        "lift_height": 0.10,
        "transfer_height": 0.18,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.2,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.12,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.28,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.15,
            "transfer_speed": 0.2,
            "place_speed": 0.1
        },
        "description": "20ml 玻璃试管 Pick and Place (需小心处理)"
    },
    "glass_test_tube_50ml": {
        "pick_offset": [-0.01, -0.16, 0.04],
        "pick_euler_deg": [180.0, 80.0, -90.0],
        "pick_approach_height": 0.14,
        "place_offset": [-0.01, -0.16, 0.01],
        "place_euler_deg": [180.0, 80.0, -90.0],
        "place_approach_height": 0.14,
        "lift_height": 0.12,
        "transfer_height": 0.20,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.25,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.12,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.28,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.15,
            "transfer_speed": 0.2,
            "place_speed": 0.1
        },
        "description": "50ml 玻璃试管 Pick and Place (需小心处理)"
    },
}

# ========== 量筒系列 ==========
CYLINDER_PICK_PLACE_CONFIGS = {
    "glass_cylinder_100ml": {
        "pick_offset": [-0.02, -0.20, 0.06],
        "pick_euler_deg": [160.0, 75.0, -150.0],
        "pick_approach_height": 0.18,
        "place_offset": [-0.02, -0.20, 0.02],
        "place_euler_deg": [160.0, 75.0, -150.0],
        "place_approach_height": 0.18,
        "lift_height": 0.15,
        "transfer_height": 0.25,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.35,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.15,
            "transfer_speed": 0.2,
            "place_speed": 0.1
        },
        "description": "100ml 玻璃量筒 Pick and Place"
    },
    "plastic_cylinder_100ml": {
        "pick_offset": [-0.02, -0.20, 0.06],
        "pick_euler_deg": [160.0, 75.0, -150.0],
        "pick_approach_height": 0.18,
        "place_offset": [-0.02, -0.20, 0.02],
        "place_euler_deg": [160.0, 75.0, -150.0],
        "place_approach_height": 0.18,
        "lift_height": 0.15,
        "transfer_height": 0.25,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.35,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.18,
            "transfer_speed": 0.25,
            "place_speed": 0.12
        },
        "description": "100ml 塑料量筒 Pick and Place"
    },
}

# ========== 容量瓶系列 ==========
VOLUMETRIC_FLASK_PICK_PLACE_CONFIGS = {
    "clear_volumetric_flask_250ml": {
        "pick_offset": [-0.02, -0.20, 0.06],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.18,
        "place_offset": [-0.02, -0.20, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.18,
        "lift_height": 0.15,
        "transfer_height": 0.25,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.35,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.15,
            "transfer_speed": 0.2,
            "place_speed": 0.1
        },
        "description": "250ml 透明容量瓶 Pick and Place"
    },
    "clear_volumetric_flask_500ml": {
        "pick_offset": [-0.03, -0.22, 0.07],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.20,
        "place_offset": [-0.03, -0.22, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.20,
        "lift_height": 0.18,
        "transfer_height": 0.28,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.4,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.12,
            "transfer_speed": 0.18,
            "place_speed": 0.08
        },
        "description": "500ml 透明容量瓶 Pick and Place"
    },
    "clear_volumetric_flask_1000ml": {
        "pick_offset": [-0.04, -0.25, 0.08],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.22,
        "place_offset": [-0.04, -0.25, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.22,
        "lift_height": 0.20,
        "transfer_height": 0.30,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.5,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.1,
            "transfer_speed": 0.15,
            "place_speed": 0.06
        },
        "description": "1000ml 透明容量瓶 Pick and Place (大物体，需缓慢操作)"
    },
}

# ========== 试剂瓶系列 ==========
REAGENT_BOTTLE_PICK_PLACE_CONFIGS = {
    "clear_reagent_bottle_small": {
        "pick_offset": [-0.02, -0.18, 0.05],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.15,
        "place_offset": [-0.02, -0.18, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.15,
        "lift_height": 0.12,
        "transfer_height": 0.22,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.3,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.2,
            "transfer_speed": 0.25,
            "place_speed": 0.12
        },
        "description": "透明试剂瓶(小) Pick and Place"
    },
    "clear_reagent_bottle_large": {
        "pick_offset": [-0.03, -0.22, 0.06],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.18,
        "place_offset": [-0.03, -0.22, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.18,
        "lift_height": 0.15,
        "transfer_height": 0.25,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.4,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.15,
            "transfer_speed": 0.2,
            "place_speed": 0.1
        },
        "description": "透明试剂瓶(大) Pick and Place"
    },
    "brown_reagent_bottle_small": {
        "pick_offset": [-0.02, -0.18, 0.05],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.15,
        "place_offset": [-0.02, -0.18, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.15,
        "lift_height": 0.12,
        "transfer_height": 0.22,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.3,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.2,
            "transfer_speed": 0.25,
            "place_speed": 0.12
        },
        "description": "棕色试剂瓶(小) Pick and Place"
    },
    "brown_reagent_bottle_large": {
        "pick_offset": [-0.03, -0.22, 0.06],
        "pick_euler_deg": [157.43, 69.16, -154.00],
        "pick_approach_height": 0.18,
        "place_offset": [-0.03, -0.22, 0.02],
        "place_euler_deg": [157.43, 69.16, -154.00],
        "place_approach_height": 0.18,
        "lift_height": 0.15,
        "transfer_height": 0.25,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.4,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.15,
            "transfer_speed": 0.2,
            "place_speed": 0.1
        },
        "description": "棕色试剂瓶(大) Pick and Place"
    },
}

# ========== 锥形瓶系列 ==========
ERLENMEYER_FLASK_PICK_PLACE_CONFIGS = {
    "erlenmeyer_flask": {
        "pick_offset": [-0.02, -0.20, 0.06],
        "pick_euler_deg": [160.0, 70.0, -150.0],
        "pick_approach_height": 0.18,
        "place_offset": [-0.02, -0.20, 0.02],
        "place_euler_deg": [160.0, 70.0, -150.0],
        "place_approach_height": 0.18,
        "lift_height": 0.15,
        "transfer_height": 0.25,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.35,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.15,
            "transfer_speed": 0.2,
            "place_speed": 0.1
        },
        "description": "锥形瓶 Pick and Place"
    },
}

# ========== 其他器材 ==========
OTHER_PICK_PLACE_CONFIGS = {
    "dropper": {
        "pick_offset": [-0.01, -0.12, 0.02],
        "pick_euler_deg": [180.0, 85.0, -90.0],
        "pick_approach_height": 0.10,
        "place_offset": [-0.01, -0.12, 0.01],
        "place_euler_deg": [180.0, 85.0, -90.0],
        "place_approach_height": 0.10,
        "lift_height": 0.08,
        "transfer_height": 0.15,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.15,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.12,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.28,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.12,
            "transfer_speed": 0.15,
            "place_speed": 0.08
        },
        "description": "胶头滴管 Pick and Place (小物体)"
    },
    "mortar": {
        "pick_offset": [-0.03, -0.22, 0.06],
        "pick_euler_deg": [160.0, 70.0, -150.0],
        "pick_approach_height": 0.18,
        "place_offset": [-0.03, -0.22, 0.02],
        "place_euler_deg": [160.0, 70.0, -150.0],
        "place_approach_height": 0.18,
        "lift_height": 0.15,
        "transfer_height": 0.25,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.45,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.12,
            "transfer_speed": 0.15,
            "place_speed": 0.08
        },
        "description": "坩埚/研钵 Pick and Place (较重)"
    },
    "pestle": {
        "pick_offset": [-0.01, -0.15, 0.03],
        "pick_euler_deg": [180.0, 80.0, -90.0],
        "pick_approach_height": 0.12,
        "place_offset": [-0.01, -0.15, 0.01],
        "place_euler_deg": [180.0, 80.0, -90.0],
        "place_approach_height": 0.12,
        "lift_height": 0.10,
        "transfer_height": 0.18,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.25,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.18,
            "transfer_speed": 0.22,
            "place_speed": 0.1
        },
        "description": "研杵 Pick and Place"
    },
    "three_neck_flask": {
        "pick_offset": [-0.04, -0.25, 0.08],
        "pick_euler_deg": [160.0, 70.0, -150.0],
        "pick_approach_height": 0.22,
        "place_offset": [-0.04, -0.25, 0.02],
        "place_euler_deg": [160.0, 70.0, -150.0],
        "place_approach_height": 0.22,
        "lift_height": 0.20,
        "transfer_height": 0.30,
        "gripper_open_width": 1.0,
        "gripper_close_width": 0.55,
        "timing": {
            "approach_ratio": 0.15,
            "grasp_ratio": 0.10,
            "lift_ratio": 0.15,
            "transfer_ratio": 0.30,
            "place_ratio": 0.15,
            "retreat_ratio": 0.15
        },
        "speed": {
            "approach_speed": 0.1,
            "transfer_speed": 0.12,
            "place_speed": 0.06
        },
        "description": "三口烧瓶 Pick and Place (大物体，需缓慢操作)"
    },
}

# ========== 汇总所有配置 ==========
PICK_PLACE_CONFIGS = {
    **BEAKER_PICK_PLACE_CONFIGS,
    **TEST_TUBE_PICK_PLACE_CONFIGS,
    **CYLINDER_PICK_PLACE_CONFIGS,
    **VOLUMETRIC_FLASK_PICK_PLACE_CONFIGS,
    **REAGENT_BOTTLE_PICK_PLACE_CONFIGS,
    **ERLENMEYER_FLASK_PICK_PLACE_CONFIGS,
    **OTHER_PICK_PLACE_CONFIGS,
}

# ========== 默认配置 ==========
DEFAULT_PICK_PLACE_CONFIG = {
    "pick_offset": [-0.02, -0.19, 0.05],
    "pick_euler_deg": [157.43, 69.16, -154.00],
    "pick_approach_height": 0.15,
    "place_offset": [-0.02, -0.19, 0.02],
    "place_euler_deg": [157.43, 69.16, -154.00],
    "place_approach_height": 0.15,
    "lift_height": 0.12,
    "transfer_height": 0.22,
    "gripper_open_width": 1.0,
    "gripper_close_width": 0.35,
    "timing": {
        "approach_ratio": 0.15,
        "grasp_ratio": 0.10,
        "lift_ratio": 0.15,
        "transfer_ratio": 0.30,
        "place_ratio": 0.15,
        "retreat_ratio": 0.15
    },
    "speed": {
        "approach_speed": 0.2,
        "transfer_speed": 0.25,
        "place_speed": 0.12
    },
    "description": "默认 Pick and Place 配置"
}


def get_pick_place_config(object_name: str) -> dict:
    """
    获取指定物体的 Pick and Place 配置
    
    Args:
        object_name: 物体名称 (如 "glass_beaker_100ml")
        
    Returns:
        包含 Pick and Place 配置的字典
    """
    if object_name in PICK_PLACE_CONFIGS:
        return PICK_PLACE_CONFIGS[object_name]
    else:
        print(f"[Warning] 未找到 '{object_name}' 的 Pick and Place 配置，使用默认配置")
        return DEFAULT_PICK_PLACE_CONFIG


def list_available_pick_place_configs():
    """列出所有可用的 Pick and Place 配置"""
    print("=" * 60)
    print("可用的 Pick and Place 配置:")
    print("=" * 60)
    for name, config in PICK_PLACE_CONFIGS.items():
        print(f"  - {name}: {config.get('description', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    # 测试
    list_available_pick_place_configs()
    
    # 获取配置示例
    config = get_pick_place_config("glass_beaker_100ml")
    print(f"\nglass_beaker_100ml Pick and Place 配置:")
    print(f"  pick_offset: {config['pick_offset']}")
    print(f"  pick_euler_deg: {config['pick_euler_deg']}")
    print(f"  lift_height: {config['lift_height']}")
    print(f"  timing: {config['timing']}")
    print(f"  speed: {config['speed']}")

