"""
对称物体多抓取点生成器

基于一个已标注的抓取点，围绕物体中心轴（Z轴）旋转生成多个均匀分布的抓取点。
适用于圆柱形对称物体（如烧杯、试管等）。

使用方法:
    # 从 object_config.json 读取并写回
    python grasp_point_generator.py --object glass_beaker_100ml --task grasp --num_points 8
    
    # 生成所有变体 (4/6/8/12点)
    python grasp_point_generator.py --object glass_beaker_100ml --task grasp --all
    
    # 调试模式（可视化验证，不写入文件）
    python grasp_point_generator.py --object glass_beaker_100ml --task grasp --debug --dry_run
    
    # 列出 object_config.json 中所有物体
    python grasp_point_generator.py --list_objects
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


# ==================== 配置 ====================

# object_config.json 文件路径
SCRIPT_DIR = Path(__file__).parent
OBJECT_CONFIG_PATH = SCRIPT_DIR / "object_config.json"

# 数据输出根目录（备用）
DATA_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent / "data"

# 支持的任务类型
SUPPORTED_TASKS = ["grasp", "handover", "pour", "place", "pick_place"]


# ==================== object_config.json 操作 ====================

def load_object_config() -> Dict[str, Any]:
    """加载 object_config.json 的完整内容"""
    if not OBJECT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"找不到配置文件: {OBJECT_CONFIG_PATH}")
    
    with open(OBJECT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_objects_dict() -> Dict[str, Any]:
    """加载 object_config.json 中的 objects 字典"""
    config = load_object_config()
    return config.get('objects', config)  # 兼容两种格式


def save_object_config(config: Dict[str, Any]):
    """保存 object_config.json"""
    with open(OBJECT_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"已保存到: {OBJECT_CONFIG_PATH}")


def get_object_config(object_id: str) -> Dict[str, Any]:
    """获取指定物体的配置"""
    objects = load_objects_dict()
    
    if object_id not in objects:
        raise ValueError(f"物体 '{object_id}' 不在 object_config.json 中")
    
    return objects[object_id]


def get_task_config(object_id: str, task: str) -> Dict[str, Any]:
    """获取指定物体的任务配置"""
    obj_config = get_object_config(object_id)
    
    if task not in obj_config:
        raise ValueError(f"物体 '{object_id}' 不支持任务 '{task}'")
    
    task_config = obj_config[task]
    if task_config is None:
        raise ValueError(f"物体 '{object_id}' 的任务 '{task}' 配置为空")
    
    return task_config


def list_all_objects() -> List[Dict[str, Any]]:
    """列出所有物体及其支持的任务"""
    objects = load_objects_dict()
    result = []
    
    for obj_id, obj_config in objects.items():
        if isinstance(obj_config, dict) and 'name' in obj_config:
            result.append({
                'id': obj_id,
                'name': obj_config.get('name', obj_id),
                'name_cn': obj_config.get('name_cn', ''),
                'supported_operations': obj_config.get('supported_operations', [])
            })
    
    return result


# ==================== 核心函数 ====================

def generate_symmetric_grasp_points(
    base_grasp_offset: List[float],
    base_grasp_euler_deg: List[float],
    num_points: int = 12,
    rotation_axis: str = 'z'
) -> List[Dict[str, Any]]:
    """
    基于一个基础抓取点，生成围绕物体中心轴旋转的多个抓取点。
    
    Args:
        base_grasp_offset: 基础抓取偏移量 [x, y, z]，相对于物体中心
        base_grasp_euler_deg: 基础抓取欧拉角 [rx, ry, rz]，单位：度
        num_points: 生成的抓取点数量（4, 6, 8, 12 等）
        rotation_axis: 旋转轴，默认 'z'（对于竖直放置的圆柱形物体）
    
    Returns:
        包含所有抓取点配置的列表
    """
    grasp_points = []
    base_offset = np.array(base_grasp_offset)
    base_euler = np.array(base_grasp_euler_deg)
    
    angle_step = 360.0 / num_points
    
    for i in range(num_points):
        angle_deg = i * angle_step
        angle_rad = np.radians(angle_deg)
        
        # 1. 旋转 grasp_offset
        if rotation_axis == 'z':
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])
        elif rotation_axis == 'x':
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [1, 0,      0     ],
                [0, cos_a, -sin_a],
                [0, sin_a,  cos_a]
            ])
        elif rotation_axis == 'y':
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a,  0, sin_a],
                [0,      1, 0    ],
                [-sin_a, 0, cos_a]
            ])
        else:
            raise ValueError(f"不支持的旋转轴: {rotation_axis}")
        
        new_offset = rotation_matrix @ base_offset
        
        # 2. 旋转 grasp_euler
        base_rot = R.from_euler('xyz', base_euler, degrees=True)
        delta_rot = R.from_euler(rotation_axis, angle_deg, degrees=True)
        new_rot = delta_rot * base_rot
        new_euler = new_rot.as_euler('xyz', degrees=True)
        
        grasp_points.append({
            'index': i,
            'angle_deg': round(angle_deg, 2),
            'grasp_offset': [round(v, 6) for v in new_offset.tolist()],
            'grasp_euler_deg': [round(v, 4) for v in new_euler.tolist()]
        })
    
    return grasp_points


# ==================== 写回 object_config.json ====================

def update_object_config_with_grasp_points(
    object_id: str,
    task: str,
    num_points: int,
    grasp_points: List[Dict[str, Any]],
    dry_run: bool = False
) -> bool:
    """
    将计算的抓取点写回 object_config.json
    
    Args:
        object_id: 物体ID (如 glass_beaker_100ml)
        task: 任务类型 (如 grasp)
        num_points: 点数量
        grasp_points: 计算出的抓取点列表
        dry_run: 如果为 True，只打印不写入
    
    Returns:
        是否成功
    """
    config = load_object_config()
    
    # 处理带 objects 键的结构
    if 'objects' in config:
        objects = config['objects']
    else:
        objects = config
    
    if object_id not in objects:
        print(f"错误: 物体 '{object_id}' 不在配置中")
        return False
    
    if task not in objects[object_id] or objects[object_id][task] is None:
        print(f"错误: 物体 '{object_id}' 不支持任务 '{task}'")
        return False
    
    # 字段名格式: grasp_points_8
    field_name = f"grasp_points_{num_points}"
    
    # 添加元数据
    grasp_points_with_meta = {
        "num_points": num_points,
        "rotation_axis": "z",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "points": grasp_points
    }
    
    if dry_run:
        print(f"\n[Dry Run] 将写入 {object_id}.{task}.{field_name}:")
        print(json.dumps(grasp_points_with_meta, indent=2, ensure_ascii=False))
        return True
    
    # 写入配置
    objects[object_id][task][field_name] = grasp_points_with_meta
    save_object_config(config)
    
    print(f"已写入: {object_id}.{task}.{field_name} ({num_points} 个点)")
    return True


def process_object(
    object_id: str,
    task: str,
    num_points_list: List[int],
    debug: bool = False,
    dry_run: bool = False
):
    """
    处理单个物体：读取配置、计算抓取点、写回配置
    
    Args:
        object_id: 物体ID
        task: 任务类型
        num_points_list: 要生成的点数列表
        debug: 是否显示调试可视化
        dry_run: 是否只预览不写入
    """
    print("=" * 60)
    print(f"对称物体多抓取点生成器")
    print("=" * 60)
    
    # 获取物体配置
    try:
        obj_config = get_object_config(object_id)
        task_config = get_task_config(object_id, task)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 获取基础抓取配置
    base_grasp_offset = task_config.get('grasp_offset')
    base_grasp_euler_deg = task_config.get('grasp_euler_deg')
    
    if base_grasp_offset is None or base_grasp_euler_deg is None:
        print(f"错误: 物体 '{object_id}' 的 '{task}' 任务缺少 grasp_offset 或 grasp_euler_deg")
        sys.exit(1)
    
    print(f"\n物体ID: {object_id}")
    print(f"物体名称: {obj_config.get('name_cn', obj_config.get('name', object_id))}")
    print(f"任务: {task}")
    print(f"\n基础配置 (从 object_config.json 读取):")
    print(f"  grasp_offset:    {base_grasp_offset}")
    print(f"  grasp_euler_deg: {base_grasp_euler_deg}")
    print(f"\n生成点数: {num_points_list}")
    
    if dry_run:
        print("\n[Dry Run 模式] 不会实际写入文件")
    
    # 对每个点数生成并写回
    for num_points in num_points_list:
        print(f"\n{'='*40}")
        print(f"生成 {num_points} 个抓取点 (每 {360//num_points}°)")
        print(f"{'='*40}")
        
        # 生成抓取点
        grasp_points = generate_symmetric_grasp_points(
            base_grasp_offset, base_grasp_euler_deg, num_points
        )
        
        # 调试可视化
        if debug:
            debug_visualize(grasp_points, f"{object_id} ({num_points}点)")
        
        # 写回配置
        update_object_config_with_grasp_points(
            object_id, task, num_points, grasp_points, dry_run
        )
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


# ==================== 调试/可视化 ====================

def debug_visualize(
    grasp_points: List[Dict[str, Any]],
    object_name: str
):
    """调试可视化 - 在终端中用 ASCII 艺术显示抓取点分布"""
    print("\n" + "-" * 60)
    print(f"调试可视化: {object_name}")
    print("-" * 60)
    
    # 创建一个简单的 ASCII 图
    size = 21
    center = size // 2
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    
    # 画坐标轴
    for i in range(size):
        grid[center][i] = '-'
        grid[i][center] = '|'
    grid[center][center] = '+'
    
    grid[0][center] = 'Y'
    grid[center][size-1] = 'X'
    
    # 计算缩放因子
    max_offset = max(
        max(abs(p['grasp_offset'][0]), abs(p['grasp_offset'][1]))
        for p in grasp_points
    )
    scale = (center - 2) / max_offset if max_offset > 0 else 1
    
    # 画抓取点
    for point in grasp_points:
        x = int(point['grasp_offset'][0] * scale) + center
        y = int(point['grasp_offset'][1] * scale) + center
        if 0 <= x < size and 0 <= y < size:
            idx = point['index']
            marker = str(idx) if idx < 10 else chr(ord('A') + idx - 10)
            grid[size - 1 - y][x] = marker
    
    print("\nXY 平面俯视图 (物体中心在 '+'):")
    for row in grid:
        print('  ' + ''.join(row))
    
    # 打印详细数据表
    print("\n详细数据:")
    print("-" * 70)
    print(f"{'点':>3} | {'角度':>7} | {'offset_x':>10} | {'offset_y':>10} | {'offset_z':>10}")
    print("-" * 70)
    for point in grasp_points:
        print(f"{point['index']:>3} | {point['angle_deg']:>6.1f}° | "
              f"{point['grasp_offset'][0]:>10.4f} | "
              f"{point['grasp_offset'][1]:>10.4f} | "
              f"{point['grasp_offset'][2]:>10.4f}")
    print("-" * 70)
    
    # 验证距离一致性
    distances = [
        np.sqrt(p['grasp_offset'][0]**2 + p['grasp_offset'][1]**2)
        for p in grasp_points
    ]
    print(f"\n验证: XY 距离 = {distances[0]:.4f} ", end="")
    if max(distances) - min(distances) < 1e-6:
        print("✓")
    else:
        print("✗ 不一致!")


# ==================== 命令行接口 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="对称物体多抓取点生成器 - 从 object_config.json 读取并写回",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 8 个抓取点
  python grasp_point_generator.py --object glass_beaker_100ml --task grasp --num_points 8
  
  # 生成所有变体 (4/6/8/12点)
  python grasp_point_generator.py --object glass_beaker_100ml --task grasp --all
  
  # 调试模式 + 预览（不写入）
  python grasp_point_generator.py --object glass_beaker_100ml --task grasp --debug --dry_run
  
  # 列出所有物体
  python grasp_point_generator.py --list_objects
"""
    )
    
    parser.add_argument("--object", "-o", type=str, default=None,
                        help="物体ID (如 glass_beaker_100ml)")
    parser.add_argument("--task", "-t", type=str, default="grasp",
                        help=f"任务类型 (默认: grasp)")
    parser.add_argument("--num_points", "-n", type=int, default=8,
                        help="抓取点数量 (默认: 8)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="生成所有变体 (4/6/8/12点)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="启用调试可视化")
    parser.add_argument("--dry_run", action="store_true",
                        help="预览模式，不实际写入文件")
    parser.add_argument("--list_objects", "-l", action="store_true",
                        help="列出 object_config.json 中所有物体")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 列出所有物体
    if args.list_objects:
        print("\nobject_config.json 中的物体:")
        print("-" * 80)
        print(f"{'ID':<30} | {'中文名':<20} | {'支持的操作'}")
        print("-" * 80)
        
        for obj in list_all_objects():
            ops = ', '.join(obj['supported_operations'][:4])
            if len(obj['supported_operations']) > 4:
                ops += '...'
            print(f"{obj['id']:<30} | {obj['name_cn']:<20} | {ops}")
        
        print("-" * 80)
        print(f"\n配置文件: {OBJECT_CONFIG_PATH}")
        return
    
    # 检查必需参数
    if args.object is None:
        print("错误: 请使用 --object 指定物体ID")
        print("使用 --list_objects 查看所有可用物体")
        sys.exit(1)
    
    # 确定要生成的点数列表
    if args.all:
        num_points_list = [4, 6, 8, 12]
    else:
        num_points_list = [args.num_points]
    
    # 处理物体
    process_object(
        object_id=args.object,
        task=args.task,
        num_points_list=num_points_list,
        debug=args.debug,
        dry_run=args.dry_run
    )


# ==================== API 接口（供其他模块调用）====================

def generate_for_object(
    object_id: str,
    task: str = "grasp",
    num_points: int = 8,
    write_back: bool = True
) -> List[Dict[str, Any]]:
    """
    API 接口：为指定物体生成抓取点
    
    Args:
        object_id: 物体ID (如 glass_beaker_100ml)
        task: 任务类型
        num_points: 点数量
        write_back: 是否写回 object_config.json
    
    Returns:
        生成的抓取点列表
    """
    task_config = get_task_config(object_id, task)
    
    base_grasp_offset = task_config['grasp_offset']
    base_grasp_euler_deg = task_config['grasp_euler_deg']
    
    grasp_points = generate_symmetric_grasp_points(
        base_grasp_offset, base_grasp_euler_deg, num_points
    )
    
    if write_back:
        update_object_config_with_grasp_points(
            object_id, task, num_points, grasp_points
        )
    
    return grasp_points


if __name__ == "__main__":
    main()
