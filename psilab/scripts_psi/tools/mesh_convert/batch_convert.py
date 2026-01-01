# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Qiao Kai-Sa
# Date: 2025-07-29
# Vesion: 1.2

#!/usr/bin/env python3
"""
批量网格转换脚本

功能：
1. 自动整理材质文件到 textures 文件夹
2. 批量转换 .obj 文件为 .usd 文件
3. 支持递归搜索文件夹
4. 将转换结果按前缀分组存储到新的目录结构中

使用方法：
python batch_convert.py /path/to/parent/folder

输出结构：
/path/to/parent/folder_usd/
├── condiment/
│   ├── C13/
│   │   ├── model.usd
│   │   └── textures/
│   └── C14/
│       ├── model.usd
│       └── textures/
└── other_prefix/
    └── ...
"""

import os
import shutil
import subprocess
import argparse
import time
from pathlib import Path
import re

def parse_folder_name(folder_name):
    """
    解析文件夹名称，提取前缀和中间名称
    
    Args:
        folder_name: 文件夹名称，如 "condiment-C13-V1"
        
    Returns:
        tuple: (前缀, 中间名称) 如 ("condiment", "C13")
    """
    # 使用正则表达式分割文件夹名称
    parts = folder_name.split('-')
    
    if len(parts) >= 2:
        prefix = parts[0]  # 第一部分作为前缀
        middle = parts[1]  # 第二部分作为中间名称
        return prefix, middle
    else:
        # 如果格式不符合预期，使用整个名称作为前缀，"default"作为中间名称
        return folder_name, "default"


def organize_material_files(source_folder, target_folder):
    """
    将材质文件从源文件夹整理到目标文件夹的 textures 子文件夹中
    
    Args:
        source_folder: 源文件夹路径（包含 .obj 文件的原始文件夹）
        target_folder: 目标文件夹路径（将要存放 .usd 文件的文件夹）
        
    Returns:
        tuple: (是否成功, 移动的文件数量)
    """
    print(f"整理材质文件: {source_folder} -> {target_folder}/textures")
    
    # 定义需要移动的材质文件关键词（不区分大小写）
    material_keywords = {
        'normal',
        'albedo',
        'ambient_occlusion_map',
        'ambientocclusionmap',
        'ambient occlusion map',
        'ao',
        'metalness',
        'metallic',
        'roughness'
    }
    
    # 创建目标 textures 文件夹
    materials_folder_name = "textures"
    target_textures_dir = os.path.join(target_folder, materials_folder_name)
    if not os.path.exists(target_textures_dir):
        os.makedirs(target_textures_dir)
        print(f"  创建 {materials_folder_name} 文件夹: {target_textures_dir}")
    
    copied_files = 0
    skipped_files = 0
    
    try:
        # 使用 Path 对象，更安全的文件操作
        source_folder_obj = Path(source_folder)
        
        for file_path in source_folder_obj.iterdir():
            # 跳过文件夹和隐藏文件
            if file_path.is_dir() or file_path.name.startswith('.'):
                continue
                
            # 获取文件名（不含扩展名）并转换为小写
            file_stem = file_path.stem.lower()
            
            # 检查文件名是否匹配材质关键词
            is_material_file = False
            for keyword in material_keywords:
                # 检查完全匹配或包含关键词
                if (file_stem == keyword or 
                    f'_{keyword}' in file_stem or 
                    f'{keyword}_' in file_stem or
                    file_stem.endswith(f'_{keyword}') or
                    file_stem.startswith(f'{keyword}_')):
                    is_material_file = True
                    break
            
            # 特殊处理 AO (Ambient Occlusion) 的各种变体
            if not is_material_file:
                if any(ao_variant in file_stem for ao_variant in ['_ao', 'ao_', '_ambientocclusion', 'ambientocclusion_']):
                    is_material_file = True
            
            if is_material_file:
                target_path = Path(target_textures_dir) / file_path.name
                
                # 如果目标文件不存在，则复制
                if not target_path.exists():
                    try:
                        shutil.copy2(file_path, target_path)
                        # print(f"  复制: {file_path.name} -> {materials_folder_name}/")
                        copied_files += 1
                    except OSError as e:
                        print(f"  复制失败 {file_path.name}: {e}")
                else:
                    print(f"  跳过已存在: {materials_folder_name}/{file_path.name}")
                    skipped_files += 1
    
    except OSError as e:
        print(f"  访问文件夹失败: {e}")
        return False, 0
    
    print(f"  完成材质整理，复制了 {copied_files} 个文件，跳过 {skipped_files} 个")
    return True, copied_files

def find_obj_folders(parent_folder):
    """
    递归查找包含 .obj 文件的文件夹，并解析文件夹名称
    
    Args:
        parent_folder: 父文件夹路径
        
    Returns:
        list: 包含 .obj 文件的文件夹信息列表
    """
    obj_folders = []
    
    print(f"搜索包含 .obj 文件的文件夹: {parent_folder}")
    
    # 递归搜索 .obj 文件
    for root, dirs, files in os.walk(parent_folder):
        # 查找 .obj 文件
        obj_files = [f for f in files if f.lower().endswith('.obj')]
        
        if obj_files:
            # 获取文件夹名称
            folder_name = os.path.basename(root)
            prefix, middle_name = parse_folder_name(folder_name)
            
            obj_folders.append({
                'folder': root,
                'folder_name': folder_name,
                'prefix': prefix,
                'middle_name': middle_name,
                'obj_files': obj_files
            })
            print(f"找到: {root} (包含 {len(obj_files)} 个 .obj 文件)")
            print(f"  解析: 前缀={prefix}, 中间名={middle_name}")
    
    print(f"总共找到 {len(obj_folders)} 个包含 .obj 文件的文件夹")
    return obj_folders

def convert_obj_to_usd(obj_path, usd_path, convert_script_path, timeout=300):
    """
    调用 convert_mesh.py 转换 .obj 为 .usd
    
    Args:
        obj_path: .obj 文件路径
        usd_path: 输出 .usd 文件路径
        convert_script_path: convert_mesh.py 脚本路径
        timeout: 转换超时时间（秒）
        
    Returns:
        bool: 转换是否成功
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(obj_path):
            print(f"输入文件不存在: {obj_path}")
            return False
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(usd_path), exist_ok=True)
        
        # 获取绝对路径
        abs_obj_path = os.path.abspath(obj_path)
        # print(f"DEBUG abs_obj_path: {abs_obj_path}")
        abs_usd_path = os.path.abspath(usd_path)
        abs_script_path = os.path.abspath(convert_script_path)
        
        # 构建转换命令
        cmd = [
            'python', abs_script_path,
            abs_obj_path,  # 使用绝对路径
            abs_usd_path,  # 使用绝对路径
            '--headless'  # 添加无头模式参数，避免 GUI 阻塞
        ]
        
        print(f"转换: {os.path.basename(obj_path)} -> {os.path.basename(usd_path)}")
        print(f"  源文件: {abs_obj_path}")
        print(f"  目标文件: {abs_usd_path}")
        
        # 执行转换，在 .obj 文件所在目录执行
        working_dir = os.path.dirname(abs_obj_path)
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=working_dir  # 在 .obj 文件所在目录执行
        )
        
        if result.returncode == 0:
            # 验证输出文件是否真的创建了
            if os.path.exists(usd_path):
                print(f"     转换成功")
                return True
            else:
                print(f"    转换脚本成功但未生成文件")
                print(f"  检查路径: {abs_usd_path}")
                # 打印更多调试信息
                if result.stdout:
                    print(f"  标准输出: {result.stdout[:200]}...")
                return False
        else:
            print(f"转换失败 (返回码: {result.returncode})")
            if result.stderr:
                # 只显示错误的前几行，避免日志过长
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[:3]:  # 只显示前3行错误
                    print(f"     {line}")
                if len(error_lines) > 3:
                    print(f"     ... (还有 {len(error_lines) - 3} 行错误)")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"转换超时 (超过 {timeout} 秒)")
        return False
    except FileNotFoundError:
        print(f"找不到 Python 或转换脚本")
        return False
    except Exception as e:
        print(f"转换出错: {e}")
        return False

def process_single_folder(folder_info, output_base_dir, convert_script_path, force_reconvert=False):
    """
    处理单个包含 .obj 文件的文件夹
    
    Args:
        folder_info: 文件夹信息字典
        output_base_dir: 输出基础目录 (如 /path/to/parent/folder_usd)
        convert_script_path: convert_mesh.py 脚本路径
        force_reconvert: 是否强制重新转换已存在的 .usd 文件
        
    Returns:
        dict: 处理结果统计
    """
    source_folder = folder_info['folder']
    prefix = folder_info['prefix']
    middle_name = folder_info['middle_name']
    obj_files = folder_info['obj_files']
    
    # 构建目标文件夹路径：output_base_dir/prefix/middle_name/
    target_folder = os.path.join(output_base_dir, prefix, middle_name)
    
    print(f"\n{'='*60}")
    print(f"处理文件夹: {source_folder}")
    print(f"输出到: {target_folder}")
    print(f"{'='*60}")
    
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    
    # 1. 先复制材质文件到目标文件夹的 textures 子文件夹
    # 这样 convert_mesh 在转换时可以使用相对路径找到材质
    success, copied_count = organize_material_files(source_folder, target_folder)
    if not success:
        print("材质文件整理失败，但继续转换过程")

    # 验证 textures 文件夹内容
    textures_dir = os.path.join(target_folder, "textures")
    if os.path.exists(textures_dir):
        texture_files = os.listdir(textures_dir)
        print(f"  textures 文件夹包含 {len(texture_files)} 个文件:")
        for tf in texture_files[:6]:  # 只显示前5个
            print(f"    - {tf}")
        if len(texture_files) > 6:
            print(f"    ... 还有 {len(texture_files) - 6} 个文件")
    
    # 3. 转换 .obj 文件
    results = {
        'total': len(obj_files),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'copied_materials': copied_count
    }
    
    for obj_file in obj_files:
        source_obj_path = os.path.join(source_folder, obj_file)
        
        # 确定输出文件名 - 保持与 .obj 文件相同的名称
        usd_file = os.path.splitext(obj_file)[0] + '.usd'
        usd_path = os.path.join(target_folder, usd_file)
        
        # 检查是否已存在 .usd 文件
        if os.path.exists(usd_path) and not force_reconvert:
            print(f"跳过已存在: {usd_file}")
            results['skipped'] += 1
            continue
        
        # 执行转换（使用原文件夹中的 .obj 文件）
        if convert_obj_to_usd(source_obj_path, usd_path, convert_script_path):
            results['success'] += 1
        else:
            results['failed'] += 1

    # 移除 config.yaml 文件
    config_path = os.path.join(target_folder, "config.yaml")
    if os.path.exists(config_path):
            os.remove(config_path)
    
    print(f"\n文件夹处理完成:")
    print(f"  材质文件复制: {results['copied_materials']}")
    print(f"  转换总计: {results['total']}")
    print(f"  转换成功: {results['success']}")
    print(f"  跳过文件: {results['skipped']}")
    print(f"  转换失败: {results['failed']}")
    
    return results

def batch_convert_meshes(parent_folder, convert_script_path="scripts_psi/tools/convert_mesh.py", force_reconvert=False):
    """
    批量转换网格文件
    
    Args:
        parent_folder: 包含多个模型文件夹的父文件夹
        convert_script_path: convert_mesh.py 脚本路径
        force_reconvert: 是否强制重新转换
    """
    print(f"开始批量转换: {parent_folder}")
    print(f"转换脚本: {convert_script_path}")
    
    # 检查转换脚本是否存在
    if not os.path.exists(convert_script_path):
        print(f"  转换脚本不存在: {convert_script_path}")
        print(f"  请检查路径，或使用 --convert-script 参数指定正确路径")
        return
    
    # 创建输出目录
    output_dir = parent_folder.rstrip('/') + '_usd'
    print(f"输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找包含 .obj 文件的文件夹
    obj_folders = find_obj_folders(parent_folder)
    
    if not obj_folders:
        print("  未找到包含 .obj 文件的文件夹")
        return
    
    # 按前缀分组显示
    prefix_groups = {}
    for folder_info in obj_folders:
        prefix = folder_info['prefix']
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(folder_info)
    
    print(f"\n按前缀分组:")
    for prefix, folders in prefix_groups.items():
        print(f"  {prefix}: {len(folders)} 个文件夹")
    
    # 处理每个文件夹
    total_results = {
        'folders': 0,
        'total_files': 0,
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'copied_materials': 0
    }
    
    start_time = time.time()
    
    for i, folder_info in enumerate(obj_folders, 1):
        print(f"\n进度: [{i}/{len(obj_folders)}]")
        
        try:
            results = process_single_folder(folder_info, output_dir, convert_script_path, force_reconvert)
            
            total_results['folders'] += 1
            total_results['total_files'] += results['total']
            total_results['success'] += results['success']
            total_results['skipped'] += results['skipped']
            total_results['failed'] += results['failed']
            total_results['copied_materials'] += results.get('copied_materials', 0)
            
        except KeyboardInterrupt:
            print("\n  用户中断，正在停止...")
            break
        except Exception as e:
            print(f"  处理文件夹时出错: {e}")
            continue
    
    # 计算总用时
    elapsed_time = time.time() - start_time
    
    # 打印总结
    print(f"\n{'='*60}")
    print("批量转换完成!")
    print(f"{'='*60}")
    print(f"处理时间: {elapsed_time:.1f} 秒")
    print(f"输出目录: {output_dir}")
    print(f"处理文件夹: {total_results['folders']}")
    print(f"复制材质文件: {total_results['copied_materials']}")
    print(f"总文件数: {total_results['total_files']}")
    print(f"成功转换: {total_results['success']}")
    print(f"跳过文件: {total_results['skipped']}")
    print(f"转换失败: {total_results['failed']}")
    
    if total_results['failed'] > 0:
        print(f"\n  有 {total_results['failed']} 个文件转换失败，请检查日志")
    elif total_results['success'] > 0:
        print(f"\n  所有文件转换成功！")

def main():
    parser = argparse.ArgumentParser(description="批量转换 .obj 文件为 .usd 文件")
    parser.add_argument("folder", help="包含模型文件夹的父目录路径")
    parser.add_argument(
        "--convert-script", 
        default="scripts_psi/tools/convert_mesh.py",
        help="convert_mesh.py 脚本路径 (默认: scripts_psi/tools/convert_mesh.py)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="强制重新转换已存在的 .usd 文件"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="仅显示将要处理的文件，不执行实际转换"
    )
    parser.add_argument(
        "--debug",
        action="store_true", 
        help="显示详细的调试信息"
    )
    
    args = parser.parse_args()
    
    # 设置全局调试标志
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.folder):
        print(f"  文件夹不存在: {args.folder}")
        return
    
    # Dry run 模式
    if args.dry_run:
        print("  Dry Run 模式 - 仅显示将要处理的文件")
        output_dir = args.folder.rstrip('/') + '_usd'
        print(f"输出将保存到: {output_dir}")
        
        obj_folders = find_obj_folders(args.folder)
        for folder_info in obj_folders:
            source_folder = folder_info['folder']
            prefix = folder_info['prefix']
            middle_name = folder_info['middle_name']
            target_folder = os.path.join(output_dir, prefix, middle_name)
            
            print(f"\n源文件夹: {source_folder}")
            print(f"目标文件夹: {target_folder}")
            for obj_file in folder_info['obj_files']:
                usd_file = os.path.splitext(obj_file)[0] + '.usd'
                print(f"  {obj_file} -> {usd_file}")
        return
    
    # 执行批量转换
    batch_convert_meshes(args.folder, args.convert_script, args.force)

if __name__ == "__main__":
    main()