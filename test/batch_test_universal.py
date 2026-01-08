#!/usr/bin/env python3
"""
通用批量测试脚本 - 支持 grasp, pick_place, handover 任务
自动化修改配置文件并串行执行模型测试流程
"""

import os
import sys
import subprocess
import time
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ========== 配置区域 ==========
# 工作空间根目录
WORKSPACE_ROOT = Path("/home/psibot/chembench")

# 任务类型配置
TASK_CONFIGS = {
    "grasp": {
        "room_cfg_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp/scenes/room_cfg.py",
        "task_il_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp/grasp_il.py",
        "task_name": "Psi-IL-Grasp-v1",
        "scene_name": "room_cfg:PSI_DC_Grasp_CFG",
        "launch_json_name": "IL-Grasp-v1:Play",
    },
    "pick_place": {
        "room_cfg_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/pick_place/scenes/room_cfg.py",
        "task_il_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/pick_place/pick_place_il.py",
        "task_name": "Psi-IL-PickPlace-v1",
        "scene_name": "room_cfg:PSI_DC_PickPlace_CFG",
        "launch_json_name": "IL-PickPlace-v1:Play",
    },
    "handover": {
        "room_cfg_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/handover/scenes/room_cfg.py",
        "task_il_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/handover/handover_il.py",
        "task_name": "Psi-IL-Handover-v1",
        "scene_name": "room_cfg:PSI_DC_Handover_CFG",
        "launch_json_name": "IL-Handover-v1:Play",
    }
}

# 配置文件路径
LAUNCH_JSON_PATH = WORKSPACE_ROOT / ".vscode/launch.json"
PLAY_SCRIPT_PATH = WORKSPACE_ROOT / "psilab/scripts_psi/workflows/imitation_learning/play.py"

# 日志输出目录
LOG_OUTPUT_DIR = WORKSPACE_ROOT / "test/test_logs"

# ========== 工具函数 ==========
def log(message, level="INFO"):
    """打印日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def clear_python_cache(py_file_path):
    """清理 Python 缓存文件 (.pyc 和 __pycache__)"""
    py_file = Path(py_file_path)
    
    # 清理同目录下的 __pycache__
    pycache_dir = py_file.parent / "__pycache__"
    if pycache_dir.exists():
        module_name = py_file.stem
        cache_files = list(pycache_dir.glob(f"{module_name}.*.pyc"))
        for pyc_file in cache_files:
            try:
                pyc_file.unlink()
                log(f"  ✓ 已删除缓存: {pyc_file.name}")
            except Exception as e:
                log(f"  ✗ 删除缓存失败 {pyc_file.name}: {e}", level="WARNING")
        
        if cache_files:
            log(f"清理了 {len(cache_files)} 个缓存文件")


def save_results_to_log(results, test_settings, log_file_path, generalization_tests=None, incremental=False):
    """
    将测试结果保存到日志文件
    
    Args:
        results: 测试结果列表
        test_settings: 测试配置
        log_file_path: 日志文件路径
        generalization_tests: 泛化测试配置（可选）
        incremental: 是否为增量保存模式（仅追加最新结果）
    """
    try:
        # 增量模式：追加模式打开，只写入最新的结果摘要
        if incremental and len(results) > 0:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                latest_result = results[-1]
                f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
                f.write(f"完成: {latest_result['task']} ")
                
                if latest_result['success']:
                    stats = latest_result.get('stats', {})
                    if stats.get('success_rate') is not None:
                        f.write(f"✅ 成功率: {stats['success_rate']*100:.2f}%")
                    else:
                        f.write("✅ 完成（无统计数据）")
                elif latest_result.get('error') == '文件缺失':
                    f.write(f"⚠️  跳过（文件缺失）")
                else:
                    f.write(f"❌ 失败")
                
                f.write(f" ({len(results)} 个任务已完成)\n")
            
            log(f"  ✅ 增量保存: 任务 {len(results)} 已记录")
            return True
        
        # 完整模式：覆盖写入完整报告
        with open(log_file_path, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write("=" * 80 + "\n")
            f.write("批量测试结果报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试配置: num_envs={test_settings.get('num_envs', 1)}, ")
            f.write(f"max_episode={test_settings.get('max_episode', 100)}, ")
            f.write(f"max_step={test_settings.get('max_step', 500)}\n")
            f.write("=" * 80 + "\n\n")
            
            # 如果有泛化测试，显示泛化测试矩阵
            if generalization_tests:
                f.write("=" * 80 + "\n")
                f.write("【泛化测试矩阵】\n")
                f.write("=" * 80 + "\n\n")
                
                # 按任务和泛化级别组织结果
                task_results = {}
                for r in results:
                    task_name = r.get('base_task_name', r['task'])
                    gen_name = r.get('generalization_name', 'default')
                    
                    if task_name not in task_results:
                        task_results[task_name] = {
                            'chinese_name': r.get('chinese_name', ''),
                            'results': {}
                        }
                    
                    task_results[task_name]['results'][gen_name] = r.get('stats', {}).get('success_rate')
                
                # 生成表格
                gen_names = [gt['name'] for gt in generalization_tests]
                gen_chinese_names = [gt.get('chinese_name', gt['name']) for gt in generalization_tests]
                
                # 表头
                header = f"{'任务名称':<30}"
                for gcn in gen_chinese_names:
                    header += f" | {gcn:^12}"
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                
                # 表格内容
                for task_name, task_data in sorted(task_results.items()):
                    chinese_name = task_data['chinese_name']
                    display_name = f"{chinese_name}" if chinese_name else task_name
                    row = f"{display_name:<30}"
                    
                    for gen_name in gen_names:
                        rate = task_data['results'].get(gen_name)
                        if rate is not None:
                            row += f" | {rate*100:^10.1f}% "
                        else:
                            row += f" | {'N/A':^12}"
                    
                    f.write(row + "\n")
                
                f.write("\n")
                
                # 计算每个泛化级别的平均成功率
                f.write("各泛化级别平均成功率:\n")
                for i, gen_name in enumerate(gen_names):
                    rates = [task_data['results'].get(gen_name) 
                             for task_data in task_results.values() 
                             if task_data['results'].get(gen_name) is not None]
                    if rates:
                        avg_rate = sum(rates) / len(rates)
                        f.write(f"  {gen_chinese_names[i]}: {avg_rate*100:.2f}% (测试{len(rates)}个任务)\n")
                
                f.write("\n")
            
            # 统计概览
            success_count = sum(1 for r in results if r["success"])
            file_missing_count = sum(1 for r in results if not r["success"] and r.get("error") == "文件缺失")
            test_failed_count = len(results) - success_count - file_missing_count
            
            f.write("=" * 80 + "\n")
            f.write("【统计概览】\n")
            f.write("=" * 80 + "\n")
            f.write(f"总计测试次数: {len(results)}\n")
            f.write(f"成功测试次数: {success_count}\n")
            f.write(f"文件缺失(跳过): {file_missing_count}\n")
            f.write(f"测试失败次数: {test_failed_count}\n")
            f.write(f"实际执行率: {(len(results) - file_missing_count)/len(results)*100:.2f}%\n")
            if len(results) - file_missing_count > 0:
                f.write(f"测试成功率: {success_count/(len(results) - file_missing_count)*100:.2f}%\n")
            f.write("\n")
            
            # 详细测试结果
            f.write("=" * 80 + "\n")
            f.write("【详细测试结果】\n")
            f.write("=" * 80 + "\n\n")
            
            for i, r in enumerate(results, 1):
                status = "✅ 成功" if r["success"] else "❌ 失败"
                chinese_name = r.get('chinese_name', '')
                
                f.write(f"{i}. 任务: {r['task']}\n")
                if chinese_name:
                    f.write(f"   中文名称: {chinese_name}\n")
                f.write(f"   状态: {status}\n")
                
                # 显示统计信息
                stats = r.get("stats", {})
                if stats and stats.get("success_rate") is not None:
                    f.write(f"   📊 成功率: {stats['success_rate']*100:.2f}%\n")
                    if stats.get("success_episodes") is not None and stats.get("total_episodes") is not None:
                        f.write(f"   成功集数: {stats['success_episodes']}/{stats['total_episodes']}\n")
                    if stats.get("avg_steps") is not None:
                        f.write(f"   平均步数: {stats['avg_steps']:.2f}\n")
                else:
                    f.write("   (未能提取统计信息)\n")
                
                if r.get("error"):
                    f.write(f"   错误信息: {r['error']}\n")
                
                f.write("\n")
            
            # 成功率汇总
            f.write("=" * 80 + "\n")
            f.write("【成功率汇总】\n")
            f.write("=" * 80 + "\n\n")
            
            valid_rates = [(r['task'], r.get('chinese_name', ''), r['stats'].get('success_rate')) 
                           for r in results if r.get('stats', {}).get('success_rate') is not None]
            
            if valid_rates:
                # 按成功率降序排列
                valid_rates.sort(key=lambda x: x[2], reverse=True)
                
                f.write("任务成功率排名:\n")
                for rank, (task_name, chinese_name, rate) in enumerate(valid_rates, 1):
                    chinese_suffix = f" ({chinese_name})" if chinese_name else ""
                    f.write(f"  {rank}. {task_name}{chinese_suffix}: {rate*100:.2f}%\n")
                
                # 计算平均成功率
                avg_rate = sum(rate for _, _, rate in valid_rates) / len(valid_rates)
                f.write(f"\n平均成功率: {avg_rate*100:.2f}%\n")
                
                # 成功率分布
                high_success = [t for t, c, r in valid_rates if r >= 0.8]
                medium_success = [t for t, c, r in valid_rates if 0.5 <= r < 0.8]
                low_success = [t for t, c, r in valid_rates if r < 0.5]
                
                f.write(f"\n成功率分布:\n")
                f.write(f"  高成功率 (≥80%): {len(high_success)} 个任务\n")
                f.write(f"  中等成功率 (50%-80%): {len(medium_success)} 个任务\n")
                f.write(f"  低成功率 (<50%): {len(low_success)} 个任务\n")
            else:
                f.write("未能提取到任何成功率数据\n")
            
            # 失败任务列表
            if success_count < len(results):
                f.write("\n" + "=" * 80 + "\n")
                f.write("【失败的任务】\n")
                f.write("=" * 80 + "\n\n")
                
                failed_tasks = [r for r in results if not r["success"]]
                
                # 分类失败任务
                file_missing_tasks = [r for r in failed_tasks if r.get("error") == "文件缺失"]
                other_failed_tasks = [r for r in failed_tasks if r.get("error") != "文件缺失"]
                
                # 显示文件缺失的任务
                if file_missing_tasks:
                    f.write("【文件缺失 (跳过测试)】\n")
                    for i, r in enumerate(file_missing_tasks, 1):
                        chinese_name = f" ({r['chinese_name']})" if r.get('chinese_name') else ""
                        f.write(f"  {i}. {r['task']}{chinese_name}\n")
                        
                        # 显示缺失的文件详情
                        missing_files = r.get("missing_files", [])
                        for file_type, file_path in missing_files:
                            f.write(f"     - {file_type}: {file_path}\n")
                    f.write("\n")
                
                # 显示其他失败的任务
                if other_failed_tasks:
                    f.write("【测试失败】\n")
                    for i, r in enumerate(other_failed_tasks, 1):
                        chinese_name = f" ({r['chinese_name']})" if r.get('chinese_name') else ""
                        f.write(f"  {i}. {r['task']}{chinese_name}\n")
                        if r.get("error") and r.get("error") != "文件缺失":
                            f.write(f"     错误: {r['error']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告结束\n")
            f.write("=" * 80 + "\n")
        
        log(f"✅ 测试结果已保存到: {log_file_path}")
        return True
        
    except Exception as e:
        log(f"❌ 保存日志文件失败: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return False


def load_config_from_yaml(config_file):
    """从 YAML 文件加载配置"""
    if not YAML_AVAILABLE:
        log("PyYAML 未安装，使用默认配置。安装命令: pip install pyyaml", level="WARNING")
        return None
    
    if not os.path.exists(config_file):
        log(f"配置文件不存在: {config_file}", level="ERROR")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log(f"成功加载配置文件: {config_file}")
        return config
    except Exception as e:
        log(f"加载配置文件失败: {e}", level="ERROR")
        return None


def backup_file(file_path):
    """备份文件"""
    if not os.path.exists(file_path):
        return None
    
    backup_path = f"{file_path}.backup_{int(time.time())}"
    shutil.copy2(file_path, backup_path)
    log(f"已备份文件: {file_path.name if hasattr(file_path, 'name') else file_path}")
    return backup_path


def restore_file(file_path, backup_path):
    """恢复文件"""
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        os.remove(backup_path)
        log(f"已恢复文件: {file_path.name if hasattr(file_path, 'name') else file_path}")


def modify_launch_json(checkpoint_path, launch_json_name):
    """修改 launch.json 中的 checkpoint 路径"""
    log(f"修改 launch.json 中 {launch_json_name} 的 checkpoint 路径")
    log(f"目标路径: {checkpoint_path}")
    
    with open(LAUNCH_JSON_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        # 解析 JSON（带注释的 JSON）
        # 由于 launch.json 可能包含注释，我们使用文本替换方式
        lines = content.split('\n')
        new_lines = []
        
        in_target_config = False
        checkpoint_found = False
        
        for i, line in enumerate(lines):
            # 找到目标配置块
            if f'"name": "{launch_json_name}"' in line:
                in_target_config = True
                new_lines.append(line)
                continue
            
            # 在目标配置块中处理
            if in_target_config:
                # 找到 checkpoint 参数
                if '"--checkpoint"' in line:
                    new_lines.append(line)
                    checkpoint_found = True
                    continue
                
                # 修改 checkpoint 路径（下一行）
                if checkpoint_found and not line.strip().startswith('//'):
                    # 提取缩进
                    indent = len(line) - len(line.lstrip())
                    # 添加新的 checkpoint 路径
                    new_lines.append(' ' * indent + f'"{checkpoint_path}",')
                    checkpoint_found = False
                    continue
                
                # 结束配置块
                if line.strip().startswith('}') and in_target_config:
                    # 检查是否是该配置块的结束
                    in_target_config = False
                
                new_lines.append(line)
            else:
                new_lines.append(line)
        
        modified_content = '\n'.join(new_lines)
        
        with open(LAUNCH_JSON_PATH, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        log("launch.json 修改完成")
        
    except Exception as e:
        log(f"修改 launch.json 失败: {e}", level="ERROR")
        raise


def modify_room_cfg(room_cfg_path, usd_path, task_type, offset_range=None):
    """修改 room_cfg.py 中的 bottle USD 路径和随机化范围"""
    log(f"修改 {task_type} room_cfg.py 中的 bottle usd_path")
    log(f"目标路径: {usd_path}")
    if offset_range:
        log(f"设置 offset_range: {offset_range}")
    
    with open(room_cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    
    # 根据任务类型查找正确的配置块
    if task_type == "grasp":
        target_cfg = "PSI_DC_Grasp_CFG"
    elif task_type == "pick_place":
        target_cfg = "PSI_DC_PickPlace_CFG"
    elif task_type == "handover":
        target_cfg = "PSI_DC_Handover_CFG"
    else:
        raise ValueError(f"未知任务类型: {task_type}")
    
    in_target_cfg = False
    in_rigid_objects = False
    in_bottle_spawn = False
    in_position_random = False
    usd_path_replaced = False
    offset_range_replaced = False
    
    for i, line in enumerate(lines):
        # 检测目标配置块
        if target_cfg in line and '=' in line and 'replace' in line:
            in_target_cfg = True
            new_lines.append(line)
            continue
        
        # 只在目标配置块中处理
        if in_target_cfg:
            # 查找 rigid_objects_cfg
            if 'rigid_objects_cfg' in line and '=' in line and '{' in line:
                in_rigid_objects = True
                new_lines.append(line)
                continue
            
            # 在 rigid_objects_cfg 中找到 bottle
            if in_rigid_objects and '"bottle"' in line and 'RigidObjectCfg' in line:
                in_bottle_spawn = True
                new_lines.append(line)
                continue
            
            # 在 bottle 的配置中处理 usd_path 和 offset_range
            if in_bottle_spawn:
                if 'usd_path' in line and '=' in line and not line.strip().startswith('#'):
                    # 找到第一个未注释的 usd_path，进行替换
                    indent = len(line) - len(line.lstrip())
                    if not usd_path_replaced:
                        new_lines.append(' ' * indent + f'usd_path = "{usd_path}",')
                        usd_path_replaced = True
                    continue
                
                # 检测 PositionRandomCfg 块
                if 'PositionRandomCfg' in line:
                    in_position_random = True
                    new_lines.append(line)
                    continue
                
                # 在 PositionRandomCfg 中处理 offset_range
                if in_position_random and offset_range is not None:
                    if 'offset_range' in line and '=' in line:
                        indent = len(line) - len(line.lstrip())
                        if not offset_range_replaced and not line.strip().startswith('#'):
                            new_lines.append(' ' * indent + f'offset_range={offset_range},')
                            offset_range_replaced = True
                        continue
                    
                    # 检测 PositionRandomCfg 结束
                    if ')' in line and line.strip().startswith(')'):
                        in_position_random = False
                        new_lines.append(line)
                        continue
                
                new_lines.append(line)
                    
                # 检查是否结束 bottle 的配置
                if 'scale=' in line or 'rigid_props=' in line:
                    in_bottle_spawn = False
            else:
                new_lines.append(line)
                
                # 检查是否结束 rigid_objects_cfg 块
                if in_rigid_objects and line.strip() == '},':
                    in_rigid_objects = False
                
                # 检查是否结束目标配置块
                if in_target_cfg and line.strip() == ')' and not in_rigid_objects:
                    in_target_cfg = False
        else:
            new_lines.append(line)
    
    modified_content = '\n'.join(new_lines)
    
    with open(room_cfg_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    log(f"{task_type} room_cfg.py 修改完成")
    
    # 清理 Python 缓存
    log(f"清理 {task_type} room_cfg.py 的 Python 缓存...")
    clear_python_cache(room_cfg_path)


def modify_task_il(task_il_path, obs_mode, task_type, ground_truth_usd_path=None):
    """修改任务 IL 文件中的 obs_mode 和 ground_truth_usd_path"""
    log(f"修改 {task_type}_il.py 中的参数")
    log(f"  obs_mode: {obs_mode}")
    if ground_truth_usd_path:
        log(f"  ground_truth_usd_path: {ground_truth_usd_path}")
    
    with open(task_il_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    obs_mode_replaced = False
    gt_usd_path_replaced = False
    
    for i, line in enumerate(lines):
        # 查找 obs_mode 定义行
        if 'obs_mode:' in line and 'Literal' in line and '=' in line and not line.strip().startswith('#'):
            # 提取缩进
            indent = len(line) - len(line.lstrip())
            
            if not obs_mode_replaced:
                # 构造新的 obs_mode 行
                new_lines.append(' ' * indent + f'obs_mode: Literal["rgb", "rgbm", "nd", "rgbnd", "state", "rgb_masked", "rgb_masked_rgb", "point_cloud"] = "{obs_mode}"')
                obs_mode_replaced = True
            continue
        
        # 查找 ground_truth_usd_path 定义行（如果提供了该参数）
        elif ground_truth_usd_path and 'ground_truth_usd_path:' in line and 'str' in line and '=' in line and not line.strip().startswith('#'):
            # 提取缩进
            indent = len(line) - len(line.lstrip())
            
            if not gt_usd_path_replaced:
                # 构造新的 ground_truth_usd_path 行
                new_lines.append(' ' * indent + f'ground_truth_usd_path: str = "{ground_truth_usd_path}"  # USD 文件路径（例如："/path/to/object.usd"）')
                gt_usd_path_replaced = True
            continue
        else:
            new_lines.append(line)
    
    modified_content = '\n'.join(new_lines)
    
    with open(task_il_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    log(f"{task_type}_il.py 修改完成")
    
    # 清理 Python 缓存
    log(f"清理 {task_type}_il.py 的 Python 缓存...")
    clear_python_cache(task_il_path)


def cleanup_isaac_sim():
    """清理 IsaacSim 相关进程"""
    log("清理 IsaacSim 进程...")
    
    processes_to_kill = [
        "isaac-sim",
        "omniverse",
        "kit",
        "vulkan"
    ]
    
    for proc_name in processes_to_kill:
        try:
            subprocess.run(["pkill", "-f", proc_name], capture_output=True, timeout=5)
            subprocess.run(["pkill", "-9", "-f", proc_name], capture_output=True, timeout=5)
        except Exception:
            pass
    
    # 清理 GPU 进程
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    try:
                        subprocess.run(["kill", "-9", pid.strip()], timeout=2)
                    except Exception:
                        pass
    except Exception:
        pass
    
    log("IsaacSim 进程清理完成")


def wait_for_process_exit(timeout=30):
    """等待 Python 子进程完全退出"""
    log(f"等待子进程退出（超时: {timeout}秒）...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 检查是否还有 play.py 进程在运行
        try:
            result = subprocess.run(
                ["pgrep", "-f", "play.py"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:  # 没有找到进程
                log("子进程已退出")
                return True
        except Exception:
            pass
        
        time.sleep(2)
    
    log("等待超时，强制清理进程", level="WARNING")
    return False


def run_test(args):
    """运行测试，返回 (success, stats) 元组"""
    log("开始测试...")
    
    cmd = [sys.executable, str(PLAY_SCRIPT_PATH)] + args
    log(f"执行命令: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        success_marker = False
        cleanup_started = False
        stats = {
            "success_rate": None,
            "total_episodes": None,
            "success_episodes": None,
            "avg_steps": None
        }
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')
                
                # 检测成功标记（更宽松的匹配）
                if ('测试完成' in line or '平均成功率' in line or '总成功率' in line or 
                    '成功率:' in line or 'Statistics' in line or 
                    'Simulation is stopped' in line):
                    success_marker = True
                    log("✅ 检测到测试完成标记", level="INFO")
                
                # 检测清理开始标记（与数据采集脚本类似）
                if ('Replicator:Annotators' in line or 'Replicator:Core' in line or 
                    'closing' in line.lower() or 
                    'app will keep running' in line.lower() or
                    'Press Ctrl+C or close the window' in line):
                    cleanup_started = True
                
                # ⭐ 关键修复：检测到测试完成后，主动终止进程（特别是有头模式）
                if success_marker and cleanup_started:
                    log("检测到测试完成且进程正在清理，等待 5 秒后强制终止...", level="INFO")
                    time.sleep(5)
                    
                    if process.poll() is None:
                        log("进程仍在运行，强制终止（避免有头模式下进程挂起）", level="WARNING")
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            log("温和终止超时，使用 SIGKILL 强制终止", level="WARNING")
                            process.kill()
                            process.wait()
                    break
                
                # 解析成功率信息
                if '平均成功率' in line or '总成功率' in line or 'Success Rate' in line or '成功率' in line:
                    # 尝试提取成功率
                    import re
                    
                    # 格式1: "成功率: 1/1 次" 或 "成功率：10/20 次" 或 "成功率: 10/20"
                    match_fraction = re.search(r'成功率[:：]\s*(\d+)/(\d+)\s*(?:次)?', line)
                    if match_fraction:
                        success = int(match_fraction.group(1))
                        total = int(match_fraction.group(2))
                        if total > 0:
                            rate_value = success / total
                            stats["success_rate"] = rate_value
                            stats["success_episodes"] = success
                            stats["total_episodes"] = total
                            log(f"📊 提取到成功率: {success}/{total} = {rate_value*100:.2f}%")
                            continue
                    
                    # 格式2: "平均成功率: 0.85" 或 "Success Rate: 85%" 或 "成功率：85.5%"
                    match_decimal = re.search(r'[:：]\s*(\d+\.?\d*)\s*%', line)
                    if match_decimal:
                        rate_value = float(match_decimal.group(1))
                        # 如果是百分比形式（大于1），转换为小数
                        if rate_value > 1:
                            rate_value = rate_value / 100.0
                        stats["success_rate"] = rate_value
                        log(f"📊 提取到成功率: {rate_value*100:.2f}%")
                        continue
                    
                    # 格式3: "成功率: 0.85" （小数形式，无百分号）
                    match_plain = re.search(r'[:：]\s*(\d+\.?\d*)\s*$', line)
                    if match_plain:
                        rate_value = float(match_plain.group(1))
                        if rate_value <= 1:  # 只有小于等于1的才是小数形式
                            stats["success_rate"] = rate_value
                            log(f"📊 提取到成功率: {rate_value*100:.2f}%")
                
                # 解析总集数（优先从 "运行次数" 提取）
                if '运行次数' in line or '总集数' in line or 'Total Episodes' in line or 'max_episode' in line:
                    import re
                    # 匹配 "运行次数: 1 次数" 或 "运行次数: 10 次" 或 "总集数: 10"
                    match = re.search(r'[:：]\s*(\d+)\s*(?:次数?)?', line)
                    if match:
                        stats["total_episodes"] = int(match.group(1))
                        log(f"📊 提取到总集数: {match.group(1)}")
                
                # 解析成功集数（如果单独提供）
                if '成功集数' in line or 'Success Episodes' in line:
                    import re
                    match = re.search(r'[:：]\s*(\d+)', line)
                    if match and stats.get("success_episodes") is None:  # 避免覆盖从成功率中提取的值
                        stats["success_episodes"] = int(match.group(1))
                
                # 解析平均步数
                if '平均步数' in line or 'Average Steps' in line:
                    import re
                    match = re.search(r'[:：]\s*(\d+\.?\d*)', line)
                    if match:
                        stats["avg_steps"] = float(match.group(1))
        
        # ⭐ 添加超时机制：如果进程还在运行，等待最多10秒
        if process.poll() is None:
            try:
                returncode = process.wait(timeout=10)
                log(f"测试进程退出，返回码: {returncode}")
            except subprocess.TimeoutExpired:
                log("测试进程超时未退出，强制终止", level="WARNING")
                process.kill()
                process.wait()
                returncode = process.returncode
        else:
            returncode = process.returncode
            log(f"测试进程已退出，返回码: {returncode}")
        
        if success_marker:
            log("✅ 测试完成（基于成功标记）")
            return True, stats
        elif returncode == 0:
            log("✅ 测试完成（基于返回码）")
            return True, stats
        else:
            log(f"❌ 测试失败，返回码: {returncode}，未检测到成功标记", level="ERROR")
            return False, stats
            
    except Exception as e:
        log(f"❌ 测试异常: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return False, {"success_rate": None, "total_episodes": None, "success_episodes": None, "avg_steps": None}


def process_test_task(config, index, total, task_config, test_settings, offset_range=None, generalization_name="default", auto_cleanup=True):
    """处理单个测试任务"""
    task_name = config["name"]
    checkpoint = config["checkpoint"]
    usd_path = config["usd_path"]
    obs_mode = config["obs_mode"]
    task_type = config["task_type"]
    chinese_name = config.get("chinese_name", task_name)
    # 获取 ground_truth_usd_path，如果没有则使用 usd_path
    ground_truth_usd_path = config.get("ground_truth_usd_path", usd_path)
    
    log(f"{'='*60}")
    log(f"[{task_type}] 测试任务 [{index+1}/{total}]: {task_name}")
    if generalization_name != "default":
        log(f"泛化测试: {generalization_name}, offset_range: {offset_range}")
    log(f"中文名称: {chinese_name}")
    log(f"模型路径: {checkpoint}")
    log(f"USD 路径: {usd_path}")
    log(f"GT点云 USD 路径: {ground_truth_usd_path}")
    log(f"观测模式: {obs_mode}")
    log(f"{'='*60}")
    
    # ⭐ 文件存在性检查
    missing_files = []
    
    # 检查模型文件
    if not os.path.exists(checkpoint):
        missing_files.append(("模型文件", checkpoint))
        log(f"❌ 模型文件不存在: {checkpoint}", level="ERROR")
    
    # 检查 USD 文件
    if not os.path.exists(usd_path):
        missing_files.append(("USD文件", usd_path))
        log(f"❌ USD文件不存在: {usd_path}", level="ERROR")
    
    # 检查 Ground Truth USD 文件（如果与 usd_path 不同）
    if ground_truth_usd_path != usd_path and not os.path.exists(ground_truth_usd_path):
        missing_files.append(("GT点云USD文件", ground_truth_usd_path))
        log(f"❌ GT点云USD文件不存在: {ground_truth_usd_path}", level="ERROR")
    
    # 如果有文件缺失，跳过该任务
    if missing_files:
        error_msg = "; ".join([f"{file_type}缺失: {path}" for file_type, path in missing_files])
        log(f"⚠️  跳过任务 {task_name}: 文件缺失", level="WARNING")
        return False, {
            "success_rate": None,
            "total_episodes": None,
            "success_episodes": None,
            "avg_steps": None,
            "error_type": "file_missing",
            "missing_files": missing_files
        }
    
    # 开始前清理残留进程
    if auto_cleanup and index > 0:
        log("🧹 开始前清理所有残留进程...")
        cleanup_isaac_sim()
        time.sleep(3)
    
    # 备份文件
    launch_json_backup = backup_file(LAUNCH_JSON_PATH)
    room_cfg_backup = backup_file(task_config["room_cfg_path"])
    task_il_backup = backup_file(task_config["task_il_path"])
    
    try:
        # 1. 修改配置文件
        modify_launch_json(checkpoint, task_config["launch_json_name"])
        modify_room_cfg(task_config["room_cfg_path"], usd_path, task_type, offset_range)
        modify_task_il(task_config["task_il_path"], obs_mode, task_type, ground_truth_usd_path)
        
        # 等待配置文件同步
        log("等待配置文件同步...")
        time.sleep(2)
        
        # 2. 构造测试参数
        test_args = [
            "--task", task_config["task_name"],
            "--num_envs", str(test_settings.get("num_envs", 1)),
            "--seed", str(test_settings.get("seed", 17)),
            "--scene", task_config["scene_name"],
            "--enable_cameras",
            "--async_reset",
            "--enable_eval",
            "--enable_random",
            "--checkpoint", checkpoint,
            "--max_step", str(test_settings.get("max_step", 500)),
            "--max_episode", str(test_settings.get("max_episode", 100)),
        ]
        
        if test_settings.get("headless", False):
            test_args.append("--headless")
        
        # 3. 运行测试
        success, stats = run_test(test_args)
        if not success:
            log(f"任务 {task_name} 测试失败", level="ERROR")
            return False, stats
        
        # ⭐ 添加进程退出等待（参考数据采集脚本）
        if auto_cleanup:
            wait_for_process_exit(timeout=30)
            
            # 清理进程
            time.sleep(3)
            cleanup_isaac_sim()
            log("等待资源完全释放...")
            time.sleep(5)
        
        log(f"任务 {task_name} 测试完成！")
        if stats["success_rate"] is not None:
            log(f"📊 成功率: {stats['success_rate']*100:.2f}%")
        return True, stats
        
    except Exception as e:
        log(f"处理任务 {task_name} 时出错: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return False, {"success_rate": None, "total_episodes": None, "success_episodes": None, "avg_steps": None}
    
    finally:
        # 恢复备份文件
        restore_file(LAUNCH_JSON_PATH, launch_json_backup)
        restore_file(task_config["room_cfg_path"], room_cfg_backup)
        restore_file(task_config["task_il_path"], task_il_backup)
        
        # 恢复后清理缓存
        log("清理配置文件的 Python 缓存（恢复后）...")
        clear_python_cache(task_config["room_cfg_path"])
        clear_python_cache(task_config["task_il_path"])
        
        time.sleep(1)
        
        if auto_cleanup:
            cleanup_isaac_sim()
            log("最终清理完成，等待环境稳定...")
            time.sleep(3)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="通用批量测试脚本")
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径 (YAML 格式)")
    parser.add_argument("--tasks", type=str, nargs="+",
                        help="指定要测试的任务名称（不指定则测试全部）")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="不自动清理 IsaacSim 进程（调试用）")
    parser.add_argument("--wait-time", type=int, default=10,
                        help="任务之间的等待时间（秒），默认10秒")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将要测试的任务，不实际执行")
    parser.add_argument("--log-file", type=str, default=None,
                        help="日志文件路径（默认自动生成）")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config_from_yaml(args.config)
    if not config:
        log("无法加载配置文件，退出", level="ERROR")
        sys.exit(1)
    
    test_tasks = config.get("test_tasks", [])
    test_settings = config.get("test_settings", {})
    generalization_tests = config.get("generalization_tests", [])
    
    # 过滤指定的任务
    if args.tasks:
        test_tasks = [task for task in test_tasks if task["name"] in args.tasks]
    
    if not test_tasks:
        log("没有要测试的任务", level="WARNING")
        sys.exit(0)
    
    log(f"将测试 {len(test_tasks)} 个任务:")
    for task in test_tasks:
        log(f"  - {task['name']} ({task.get('chinese_name', '')})")
    
    # 如果有泛化测试配置，显示泛化测试信息
    if generalization_tests:
        log(f"\n将进行 {len(generalization_tests)} 个泛化级别的测试:")
        for gt in generalization_tests:
            log(f"  - {gt.get('chinese_name', gt['name'])}: offset_range={gt['offset_range']}")
        log(f"\n总计将执行 {len(test_tasks) * len(generalization_tests)} 次测试")
    
    if args.dry_run:
        log("Dry run 模式，不执行实际操作")
        sys.exit(0)
    
    # 创建日志目录
    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件路径
    if args.log_file:
        log_file_path = Path(args.log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = LOG_OUTPUT_DIR / f"test_results_{timestamp}.log"
    
    log(f"测试结果将保存到: {log_file_path}")
    
    # ⭐ 初始化日志文件（创建文件头）
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("批量测试结果报告 (实时更新)\n")
            f.write("=" * 80 + "\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试配置: num_envs={test_settings.get('num_envs', 1)}, ")
            f.write(f"max_episode={test_settings.get('max_episode', 100)}, ")
            f.write(f"max_step={test_settings.get('max_step', 500)}\n")
            
            if generalization_tests and len(generalization_tests) > 1:
                f.write(f"\n泛化测试级别: {len(generalization_tests)} 个\n")
                for gt in generalization_tests:
                    f.write(f"  - {gt.get('chinese_name', gt['name'])}: offset_range={gt['offset_range']}\n")
            
            f.write(f"\n总计任务数: {len(test_tasks)}\n")
            if generalization_tests:
                f.write(f"总计测试数: {len(test_tasks) * len(generalization_tests)}\n")
            
            f.write("=" * 80 + "\n")
            f.write("\n【实时测试进度】\n")
            f.write("-" * 80 + "\n")
        
        log(f"✅ 日志文件已初始化")
    except Exception as e:
        log(f"⚠️  初始化日志文件失败: {e}", level="WARNING")
    
    # 批量测试
    auto_cleanup = not args.no_cleanup
    results = []
    test_counter = 0
    
    # 如果没有泛化测试配置，使用默认配置
    if not generalization_tests:
        generalization_tests = [{"name": "default", "offset_range": None, "chinese_name": "默认"}]
    
    # 嵌套循环：外层是泛化级别，内层是任务
    for gen_idx, gen_test in enumerate(generalization_tests):
        gen_name = gen_test["name"]
        gen_chinese = gen_test.get("chinese_name", gen_name)
        offset_range = gen_test.get("offset_range")
        
        log(f"\n{'='*80}")
        log(f"开始泛化测试: {gen_chinese} ({gen_name})")
        log(f"offset_range: {offset_range}")
        log(f"{'='*80}\n")
        
        for task_idx, test_task in enumerate(test_tasks):
            test_counter += 1
            task_type = test_task["task_type"]
            
            if task_type not in TASK_CONFIGS:
                log(f"未知任务类型: {task_type}，跳过", level="ERROR")
                results.append({
                    "task": f"{test_task['name']}_{gen_name}",
                    "base_task_name": test_task["name"],
                    "generalization_name": gen_name,
                    "success": False,
                    "error": "未知任务类型"
                })
                continue
            
            task_config = TASK_CONFIGS[task_type]
            
            success, stats = process_test_task(
                test_task, test_counter - 1, len(test_tasks) * len(generalization_tests),
                task_config, test_settings, offset_range, gen_name, auto_cleanup
            )
            
            # 构建结果记录
            result = {
                "task": f"{test_task['name']}_{gen_name}",
                "base_task_name": test_task["name"],
                "generalization_name": gen_name,
                "chinese_name": test_task.get("chinese_name", ""),
                "success": success,
                "stats": stats
            }
            
            # 如果有文件缺失信息，添加到结果中
            if stats.get("error_type") == "file_missing":
                result["error"] = "文件缺失"
                result["missing_files"] = stats.get("missing_files", [])
            
            results.append(result)
            
            # ⭐ 增量保存：每完成一个任务就保存一次
            try:
                save_results_to_log(results, test_settings, log_file_path, 
                                   generalization_tests if len(generalization_tests) > 1 else None, 
                                   incremental=True)
            except Exception as e:
                log(f"⚠️  增量保存失败: {e}", level="WARNING")
            
            completed = len(results)
            total_tests = len(test_tasks) * len(generalization_tests)
            log(f"总进度: {completed}/{total_tests} 完成")
            
            if completed < total_tests:
                log(f"等待 {args.wait_time} 秒后继续下一个测试...")
                time.sleep(args.wait_time)
                
                if auto_cleanup:
                    log("⚠️  开始处理下一个测试前，再次彻底清理进程...")
                    cleanup_isaac_sim()
                    time.sleep(5)
                    
                    # 验证清理结果
                    result = subprocess.run(
                        ["pgrep", "-f", "play.py"],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        log("⚠️  发现残留进程，强制清理", level="WARNING")
                        subprocess.run(["pkill", "-9", "-f", "play.py"], timeout=5)
                        time.sleep(2)
                    else:
                        log("✅ 进程清理验证通过")
    
    # 显示总结
    log("=" * 60)
    log("批量测试完成！")
    log("=" * 60)
    log(f"总计: {len(results)} 个测试")
    success_count = sum(1 for r in results if r["success"])
    file_missing_count = sum(1 for r in results if not r["success"] and r.get("error") == "文件缺失")
    test_failed_count = len(results) - success_count - file_missing_count
    
    log(f"成功: {success_count}")
    log(f"文件缺失(跳过): {file_missing_count}")
    log(f"测试失败: {test_failed_count}")
    
    log("\n" + "=" * 60)
    log("详细测试结果:")
    log("=" * 60)
    for r in results:
        status = "✅ 成功" if r["success"] else ("⚠️  跳过" if r.get("error") == "文件缺失" else "❌ 失败")
        chinese_name = f" ({r['chinese_name']})" if r.get('chinese_name') else ""
        log(f"\n任务: {r['task']}{chinese_name}")
        log(f"状态: {status}")
        
        # 如果是文件缺失，显示缺失文件
        if r.get("error") == "文件缺失":
            log("   原因: 文件缺失")
            missing_files = r.get("missing_files", [])
            for file_type, file_path in missing_files:
                log(f"   - {file_type}: {file_path}")
            continue
        
        # 显示统计信息
        stats = r.get("stats", {})
        if stats and stats.get("success_rate") is not None:
            log(f"📊 成功率: {stats['success_rate']*100:.2f}%")
            if stats.get("success_episodes") is not None and stats.get("total_episodes") is not None:
                log(f"   成功集数: {stats['success_episodes']}/{stats['total_episodes']}")
            if stats.get("avg_steps") is not None:
                log(f"   平均步数: {stats['avg_steps']:.2f}")
        else:
            log("   (未能提取统计信息)")
    
    # 汇总成功率
    log("\n" + "=" * 60)
    log("成功率汇总:")
    log("=" * 60)
    valid_rates = [(r['task'], r.get('chinese_name', ''), r['stats'].get('success_rate')) 
                   for r in results if r['stats'].get('success_rate') is not None]
    
    if valid_rates:
        for task_name, chinese_name, rate in valid_rates:
            chinese_suffix = f" ({chinese_name})" if chinese_name else ""
            log(f"  {task_name}{chinese_suffix}: {rate*100:.2f}%")
        
        # 计算平均成功率
        avg_rate = sum(rate for _, _, rate in valid_rates) / len(valid_rates)
        log(f"\n  平均成功率: {avg_rate*100:.2f}%")
    else:
        log("  未能提取到任何成功率数据")
    
    if success_count < len(results):
        # 文件缺失的任务
        file_missing_tasks = [r for r in results if not r["success"] and r.get("error") == "文件缺失"]
        if file_missing_tasks:
            log("\n" + "=" * 60)
            log("文件缺失 (跳过的任务):")
            log("=" * 60)
            for r in file_missing_tasks:
                chinese_name = f" ({r['chinese_name']})" if r.get('chinese_name') else ""
                log(f"  - {r['task']}{chinese_name}")
                missing_files = r.get("missing_files", [])
                for file_type, file_path in missing_files:
                    log(f"    • {file_type}: {file_path}")
        
        # 测试失败的任务
        test_failed_tasks = [r for r in results if not r["success"] and r.get("error") != "文件缺失"]
        if test_failed_tasks:
            log("\n" + "=" * 60)
            log("测试失败的任务:")
            log("=" * 60)
            for r in test_failed_tasks:
                chinese_name = f" ({r['chinese_name']})" if r.get('chinese_name') else ""
                log(f"  - {r['task']}{chinese_name}")
    
    # 保存结果到日志文件
    log("\n" + "=" * 60)
    log("正在生成完整测试报告...")
    
    # ⭐ 追加完成标记到日志
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"所有测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        log("✅ 测试完成标记已记录")
    except Exception as e:
        log(f"⚠️  记录完成标记失败: {e}", level="WARNING")
    
    # 如果有泛化测试，传递泛化测试配置
    gen_tests_for_log = generalization_tests if len(generalization_tests) > 1 or generalization_tests[0]["name"] != "default" else None
    save_results_to_log(results, test_settings, log_file_path, gen_tests_for_log, incremental=False)


if __name__ == "__main__":
    main()

