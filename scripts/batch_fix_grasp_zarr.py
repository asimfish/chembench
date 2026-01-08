#!/usr/bin/env python3
"""
批量修复 Grasp 数据转换问题
将所有 Grasp 数据从双手任务转换为单手任务（移除左手数据）
"""

import zarr
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_bimanual_to_single_hand(zarr_path: str, backup: bool = True, dry_run: bool = False):
    """
    将包含双手数据的 Zarr 文件就地转换为只包含右手数据的版本
    
    Args:
        zarr_path: Zarr 文件路径
        backup: 是否备份原文件
        dry_run: 仅检查不修改
    """
    print(f"\n{'='*60}")
    print(f"处理: {Path(zarr_path).name}")
    print(f"{'='*60}")
    
    # 打开 Zarr
    root = zarr.open(zarr_path, mode='r')
    data_group = root['data']
    
    # 检查 action 维度
    action_data = data_group['action'][:]
    print(f"  Action 形状: {action_data.shape}")
    
    if action_data.shape[1] != 26:
        print(f"  ✓ 已是单手数据 ({action_data.shape[1]}维)，跳过")
        return False
    
    print(f"  ⚠️  检测到双手数据 (26维)")
    
    # 检查是否有左手数据
    left_hand_keys = ['arm1_pos', 'arm1_vel', 'hand1_pos', 'hand1_vel', 
                      'arm1_eef_pos', 'arm1_eef_quat']
    has_left_hand = any(key in data_group for key in left_hand_keys)
    
    if has_left_hand:
        found_keys = [key for key in left_hand_keys if key in data_group]
        print(f"  ⚠️  发现左手数据: {found_keys}")
    
    if dry_run:
        print(f"  [DRY RUN] 将会转换为单手数据")
        return True
    
    # 创建备份
    if backup:
        backup_path = str(Path(zarr_path).parent / (Path(zarr_path).stem + "_backup.zarr"))
        if os.path.exists(backup_path):
            print(f"  跳过备份 (已存在): {Path(backup_path).name}")
        else:
            print(f"  创建备份: {Path(backup_path).name}")
            shutil.copytree(zarr_path, backup_path)
    
    # 重新打开为写模式
    root = zarr.open(zarr_path, mode='r+')
    data_group = root['data']
    
    # 转换 action: 只保留右手 [0:13]
    print(f"  转换 action: [26] -> [13]")
    action_single = action_data[:, :13]
    del data_group['action']
    data_group.create_dataset('action', data=action_single)
    
    # 删除左手数据
    for key in left_hand_keys:
        if key in data_group:
            print(f"  删除左手数据: {key}")
            del data_group[key]
    
    print(f"  ✅ 转换完成!")
    return True

def batch_convert_grasp_data(grasp_dir: str, backup: bool = True, dry_run: bool = False, auto_yes: bool = False):
    """
    批量转换 Grasp 目录下的所有 Zarr 文件
    
    Args:
        grasp_dir: Grasp 数据目录
        backup: 是否备份原文件
        dry_run: 仅检查不修改
    """
    grasp_path = Path(grasp_dir)
    
    if not grasp_path.exists():
        print(f"❌ 目录不存在: {grasp_dir}")
        return
    
    # 查找所有 .zarr 目录
    zarr_dirs = sorted([d for d in grasp_path.iterdir() 
                       if d.is_dir() and d.suffix == '.zarr'])
    
    if not zarr_dirs:
        print(f"❌ 未找到 .zarr 文件")
        return
    
    print(f"\n{'='*60}")
    print(f"批量转换 Grasp 数据")
    print(f"{'='*60}")
    print(f"目录: {grasp_dir}")
    print(f"找到 {len(zarr_dirs)} 个 Zarr 文件")
    print(f"备份: {'是' if backup else '否'}")
    print(f"模式: {'检查模式 (不修改)' if dry_run else '转换模式'}")
    print(f"{'='*60}\n")
    
    if not dry_run and not auto_yes:
        response = input("确认开始转换? (y/n): ")
        if response.lower() != 'y':
            print("取消操作")
            return
    
    # 统计
    converted_count = 0
    skipped_count = 0
    
    # 批量转换
    for zarr_dir in tqdm(zarr_dirs, desc="转换进度"):
        try:
            was_converted = convert_bimanual_to_single_hand(
                str(zarr_dir), 
                backup=backup, 
                dry_run=dry_run
            )
            if was_converted:
                converted_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ❌ 转换失败: {e}")
    
    # 总结
    print(f"\n{'='*60}")
    print(f"批量转换完成!")
    print(f"{'='*60}")
    print(f"  转换: {converted_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件")
    print(f"  总计: {len(zarr_dirs)} 个文件")
    
    if dry_run:
        print(f"\n💡 这是检查模式，未实际修改文件")
        print(f"   如需转换，请移除 --dry-run 参数")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量转换 Grasp 数据为单手版本')
    parser.add_argument('grasp_dir', nargs='?', 
                       default='/home/psibot/chembench/data/zarr_point_cloud/motion_plan/grasp',
                       help='Grasp 数据目录 (默认: %(default)s)')
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份文件')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅检查不修改文件')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='自动确认，不询问')
    
    args = parser.parse_args()
    
    batch_convert_grasp_data(
        args.grasp_dir,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        auto_yes=args.yes
    )

