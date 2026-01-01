#!/usr/bin/env python3
"""
验证配置文件是否正确修改
"""

import sys
from pathlib import Path

GRASP_MP_PATH = Path("/home/psibot/chembench/psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp/grasp_mp.py")
ROOM_CFG_PATH = Path("/home/psibot/chembench/psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp/scenes/room_cfg.py")

print("=" * 80)
print("验证配置文件")
print("=" * 80)

# 检查 grasp_mp.py
print("\n1. 检查 grasp_mp.py 中的 TARGET_OBJECT_NAME:")
print("-" * 80)
with open(GRASP_MP_PATH, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if 'TARGET_OBJECT_NAME' in line and '=' in line and not line.strip().startswith('#'):
            if 'TASK_TYPE' not in line and 'target_object_name:' not in line:
                # 全局变量
                print(f"  行 {i}: {line.rstrip()}")
        elif 'target_object_name:' in line and 'str' in line and not line.strip().startswith('#'):
            # 类属性
            print(f"  行 {i}: {line.rstrip()}")

# 检查 room_cfg.py
print("\n2. 检查 room_cfg.py 中的 bottle usd_path:")
print("-" * 80)
with open(ROOM_CFG_PATH, 'r') as f:
    content = f.read()
    lines = content.split('\n')
    
    in_psi_dc_grasp = False
    in_bottle = False
    usd_path_count = 0
    
    for i, line in enumerate(lines, 1):
        if 'PSI_DC_Grasp_CFG' in line and 'replace' in line:
            in_psi_dc_grasp = True
        
        if in_psi_dc_grasp and '"bottle"' in line and 'RigidObjectCfg' in line:
            in_bottle = True
            print(f"  行 {i}: 找到 bottle 定义")
        
        if in_bottle and 'usd_path' in line and not line.strip().startswith('#'):
            print(f"  行 {i}: {line.rstrip()}")
            usd_path_count += 1
            if usd_path_count >= 1:  # 只显示第一个 usd_path
                in_bottle = False

# 检查 Python 缓存
print("\n3. 检查 Python 缓存文件:")
print("-" * 80)

pycache_grasp = GRASP_MP_PATH.parent / "__pycache__"
if pycache_grasp.exists():
    pyc_files = list(pycache_grasp.glob("grasp_mp.*.pyc"))
    if pyc_files:
        print(f"  ❌ 发现 grasp_mp 缓存文件: {len(pyc_files)} 个")
        for pyc in pyc_files:
            print(f"     - {pyc.name}")
    else:
        print(f"  ✅ grasp_mp 无缓存文件")
else:
    print(f"  ✅ grasp_mp __pycache__ 目录不存在")

pycache_room = ROOM_CFG_PATH.parent / "__pycache__"
if pycache_room.exists():
    pyc_files = list(pycache_room.glob("room_cfg.*.pyc"))
    if pyc_files:
        print(f"  ❌ 发现 room_cfg 缓存文件: {len(pyc_files)} 个")
        for pyc in pyc_files:
            print(f"     - {pyc.name}")
    else:
        print(f"  ✅ room_cfg 无缓存文件")
else:
    print(f"  ✅ room_cfg __pycache__ 目录不存在")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)

