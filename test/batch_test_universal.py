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


def save_results_to_log(results, test_settings, log_file_path):
    """将测试结果保存到日志文件"""
    try:
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
            
            # 统计概览
            success_count = sum(1 for r in results if r["success"])
            f.write("【统计概览】\n")
            f.write(f"总计任务数: {len(results)}\n")
            f.write(f"成功任务数: {success_count}\n")
            f.write(f"失败任务数: {len(results) - success_count}\n")
            f.write(f"任务完成率: {success_count/len(results)*100:.2f}%\n\n")
            
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
                for i, r in enumerate(failed_tasks, 1):
                    chinese_name = f" ({r['chinese_name']})" if r.get('chinese_name') else ""
                    f.write(f"  {i}. {r['task']}{chinese_name}\n")
                    if r.get("error"):
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


def modify_room_cfg(room_cfg_path, usd_path, task_type):
    """修改 room_cfg.py 中的 bottle USD 路径"""
    log(f"修改 {task_type} room_cfg.py 中的 bottle usd_path")
    log(f"目标路径: {usd_path}")
    
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
    usd_path_replaced = False
    
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
            
            # 在 bottle 的配置中处理 usd_path
            if in_bottle_spawn:
                if 'usd_path' in line and '=' in line and not line.strip().startswith('#'):
                    # 找到第一个未注释的 usd_path，进行替换
                    indent = len(line) - len(line.lstrip())
                    if not usd_path_replaced:
                        new_lines.append(' ' * indent + f'usd_path = "{usd_path}",')
                        usd_path_replaced = True
                    continue
                else:
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


def modify_task_il(task_il_path, obs_mode, task_type):
    """修改任务 IL 文件中的 obs_mode"""
    log(f"修改 {task_type}_il.py 中的 obs_mode: {obs_mode}")
    
    with open(task_il_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    obs_mode_replaced = False
    
    for i, line in enumerate(lines):
        # 查找 obs_mode 定义行
        if 'obs_mode:' in line and 'Literal' in line and '=' in line and not line.strip().startswith('#'):
            # 提取缩进
            indent = len(line) - len(line.lstrip())
            
            if not obs_mode_replaced:
                # 构造新的 obs_mode 行
                new_lines.append(' ' * indent + f'obs_mode: Literal["rgb", "rgbm", "nd", "rgbnd", "state", "rgb_masked", "rgb_masked_rgb"] = "{obs_mode}"')
                obs_mode_replaced = True
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
        stats = {
            "success_rate": None,
            "total_episodes": None,
            "success_episodes": None,
            "avg_steps": None
        }
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')
                
                # 检测成功标记
                if '测试完成' in line or '平均成功率' in line or '总成功率' in line:
                    success_marker = True
                    log("✅ 检测到测试完成标记", level="INFO")
                
                # 解析成功率信息
                if '平均成功率' in line or '总成功率' in line or 'Success Rate' in line:
                    # 尝试提取百分比或小数形式的成功率
                    import re
                    # 匹配格式如: "平均成功率: 0.85" 或 "Success Rate: 85%" 或 "成功率：0.85"
                    match = re.search(r'[:：]\s*(\d+\.?\d*)\s*%?', line)
                    if match:
                        rate_value = float(match.group(1))
                        # 如果是百分比形式（大于1），转换为小数
                        if rate_value > 1:
                            rate_value = rate_value / 100.0
                        stats["success_rate"] = rate_value
                        log(f"📊 提取到成功率: {rate_value*100:.2f}%")
                
                # 解析总集数
                if '总集数' in line or 'Total Episodes' in line or 'max_episode' in line:
                    import re
                    match = re.search(r'[:：]\s*(\d+)', line)
                    if match:
                        stats["total_episodes"] = int(match.group(1))
                
                # 解析成功集数
                if '成功集数' in line or 'Success Episodes' in line:
                    import re
                    match = re.search(r'[:：]\s*(\d+)', line)
                    if match:
                        stats["success_episodes"] = int(match.group(1))
                
                # 解析平均步数
                if '平均步数' in line or 'Average Steps' in line:
                    import re
                    match = re.search(r'[:：]\s*(\d+\.?\d*)', line)
                    if match:
                        stats["avg_steps"] = float(match.group(1))
        
        returncode = process.wait()
        log(f"测试进程退出，返回码: {returncode}")
        
        if success_marker or returncode == 0:
            log("✅ 测试完成")
            return True, stats
        else:
            log(f"❌ 测试失败，返回码: {returncode}", level="ERROR")
            return False, stats
            
    except Exception as e:
        log(f"❌ 测试异常: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return False, {"success_rate": None, "total_episodes": None, "success_episodes": None, "avg_steps": None}


def process_test_task(config, index, total, task_config, test_settings, auto_cleanup=True):
    """处理单个测试任务"""
    task_name = config["name"]
    checkpoint = config["checkpoint"]
    usd_path = config["usd_path"]
    obs_mode = config["obs_mode"]
    task_type = config["task_type"]
    chinese_name = config.get("chinese_name", task_name)
    
    log(f"{'='*60}")
    log(f"[{task_type}] 测试任务 [{index+1}/{total}]: {task_name}")
    log(f"中文名称: {chinese_name}")
    log(f"模型路径: {checkpoint}")
    log(f"USD 路径: {usd_path}")
    log(f"观测模式: {obs_mode}")
    log(f"{'='*60}")
    
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
        modify_room_cfg(task_config["room_cfg_path"], usd_path, task_type)
        modify_task_il(task_config["task_il_path"], obs_mode, task_type)
        
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
        
        # 清理进程
        if auto_cleanup:
            time.sleep(5)
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
    
    # 过滤指定的任务
    if args.tasks:
        test_tasks = [task for task in test_tasks if task["name"] in args.tasks]
    
    if not test_tasks:
        log("没有要测试的任务", level="WARNING")
        sys.exit(0)
    
    log(f"将测试 {len(test_tasks)} 个任务:")
    for task in test_tasks:
        log(f"  - {task['name']} ({task.get('chinese_name', '')})")
    
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
    
    # 批量测试
    auto_cleanup = not args.no_cleanup
    results = []
    
    for i, test_task in enumerate(test_tasks):
        task_type = test_task["task_type"]
        
        if task_type not in TASK_CONFIGS:
            log(f"未知任务类型: {task_type}，跳过", level="ERROR")
            results.append({
                "task": test_task["name"],
                "success": False,
                "error": "未知任务类型"
            })
            continue
        
        task_config = TASK_CONFIGS[task_type]
        
        success, stats = process_test_task(
            test_task, i, len(test_tasks),
            task_config, test_settings, auto_cleanup
        )
        
        results.append({
            "task": test_task["name"],
            "chinese_name": test_task.get("chinese_name", ""),
            "success": success,
            "stats": stats
        })
        
        completed = len(results)
        log(f"进度: {completed}/{len(test_tasks)} 完成")
        
        if completed < len(test_tasks):
            log(f"等待 {args.wait_time} 秒后继续下一个任务...")
            time.sleep(args.wait_time)
            
            if auto_cleanup:
                log("⚠️  开始处理下一个任务前，再次彻底清理进程...")
                cleanup_isaac_sim()
                time.sleep(5)
    
    # 显示总结
    log("=" * 60)
    log("批量测试完成！")
    log("=" * 60)
    log(f"总计: {len(results)} 个任务")
    success_count = sum(1 for r in results if r["success"])
    log(f"成功: {success_count}")
    log(f"失败: {len(results) - success_count}")
    
    log("\n" + "=" * 60)
    log("详细测试结果:")
    log("=" * 60)
    for r in results:
        status = "✅ 成功" if r["success"] else "❌ 失败"
        chinese_name = f" ({r['chinese_name']})" if r.get('chinese_name') else ""
        log(f"\n任务: {r['task']}{chinese_name}")
        log(f"状态: {status}")
        
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
        log("\n" + "=" * 60)
        log("失败的任务:")
        log("=" * 60)
        for r in results:
            if not r["success"]:
                chinese_name = f" ({r['chinese_name']})" if r.get('chinese_name') else ""
                log(f"  - {r['task']}{chinese_name}")
    
    # 保存结果到日志文件
    log("\n" + "=" * 60)
    log("正在保存测试结果到日志文件...")
    save_results_to_log(results, test_settings, log_file_path)


if __name__ == "__main__":
    main()

