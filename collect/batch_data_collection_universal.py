#!/usr/bin/env python3
"""
通用批量数据采集脚本 - 支持 grasp 和 pick_place 任务
自动化修改配置文件并串行执行数据采集和转换流程
"""

import os
import sys
import subprocess
import time
import re
import argparse
from pathlib import Path
from datetime import datetime
import glob
import shutil

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
        "task_mp_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp/grasp_mp.py",
        "task_name": "Psi-MP-Grasp-v1",
        "scene_name": "room_cfg:PSI_DC_Grasp_CFG",
        "data_subdir": "grasp",
        "target_success_var": "target_success_count",
    },
    "pick_place": {
        "room_cfg_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/pick_place/scenes/room_cfg.py",
        "task_mp_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/pick_place/pick_place_mp.py",
        "task_name": "Psi-MP-PickPlace-v1",
        "scene_name": "room_cfg:PSI_DC_PickPlace_CFG",
        "data_subdir": "pick_place",
        "target_success_var": "target_success_count",
    },
    "handover": {
        "room_cfg_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/handover/scenes/room_cfg.py",
        "task_mp_path": WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/handover/handover_mp.py",
        "task_name": "Psi-MP-Handover-v1",
        "scene_name": "room_cfg:PSI_DC_Handover_CFG",
        "data_subdir": "handover",
        "target_success_var": "target_success_count",
    }
}

# 脚本路径
PLAY_SCRIPT_PATH = WORKSPACE_ROOT / "psilab/scripts_psi/workflows/motion_planning/play.py"
ZARR_UTILS_PATH = WORKSPACE_ROOT / "psilab/source/psilab/psilab/utils/zarr_utils.py"

# 默认配置文件路径
DEFAULT_CONFIG_FILE = WORKSPACE_ROOT / "objects_config.yaml"

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


def load_config_from_yaml(config_file):
    """从 YAML 文件加载配置"""
    if not YAML_AVAILABLE:
        log("PyYAML 未安装，使用默认配置。安装命令: pip install pyyaml", level="WARNING")
        return None
    
    if not os.path.exists(config_file):
        log(f"配置文件不存在: {config_file}，使用默认配置", level="WARNING")
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
    log(f"已备份文件: {file_path} -> {backup_path}")
    return backup_path


def restore_file(file_path, backup_path):
    """恢复文件"""
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        os.remove(backup_path)
        log(f"已恢复文件: {file_path}")


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
    usd_path_added = False
    
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
                if 'usd_path' in line and '=' in line:
                    stripped = line.strip()
                    indent = len(line) - len(line.lstrip())
                    
                    if not usd_path_added:
                        new_lines.append(' ' * indent + f'usd_path="{usd_path}",')
                        usd_path_added = True
                    else:
                        if not stripped.startswith('#'):
                            new_lines.append(' ' * indent + '# ' + stripped)
                        else:
                            new_lines.append(line)
                else:
                    new_lines.append(line)
                    
                    # 检查是否结束 bottle 的配置
                    if 'scale=' in line or 'rigid_props=' in line:
                        in_bottle_spawn = False
                        usd_path_added = False
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


def modify_task_mp(task_mp_path, target_object_name, task_type, finger_grasp_mode=None, pre_grasp_offset=None, left_pre_grasp_offset=None, right_post_release_offset=None):
    """修改任务 MP 文件中的 TARGET_OBJECT_NAME, finger_grasp_mode, pre_grasp_offset, left_pre_grasp_offset 和 right_post_release_offset"""
    log(f"修改 {task_type}_mp.py 中的 TARGET_OBJECT_NAME: {target_object_name}")
    if finger_grasp_mode:
        log(f"  设置 finger_grasp_mode: {finger_grasp_mode}")
    if pre_grasp_offset:
        log(f"  设置 pre_grasp_offset (右手): {pre_grasp_offset}")
    if left_pre_grasp_offset:
        log(f"  设置 left_pre_grasp_offset (左手): {left_pre_grasp_offset}")
    if right_post_release_offset:
        log(f"  设置 right_post_release_offset (右手释放后): {right_post_release_offset}")
    
    with open(task_mp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    replaced_global = False
    replaced_class_attr = False
    replaced_finger_mode = False
    replaced_pre_grasp_x = False
    replaced_pre_grasp_y = False
    replaced_pre_grasp_height = False
    replaced_left_pre_grasp_x = False
    replaced_left_pre_grasp_y = False
    replaced_left_pre_grasp_height = False
    replaced_right_post_release_x = False
    replaced_right_post_release_y = False
    replaced_right_post_release_height = False
    
    for i, line in enumerate(lines):
        # 处理全局变量 TARGET_OBJECT_NAME
        if 'TARGET_OBJECT_NAME' in line and '=' in line and 'TASK_TYPE' not in line and 'target_object_name:' not in line:
            stripped = line.strip()
            indent_count = len(line) - len(line.lstrip())
            if indent_count == 0:  # 全局变量
                if not replaced_global and not stripped.startswith('#'):
                    new_lines.append(f'TARGET_OBJECT_NAME = "{target_object_name}"  # 目标物体名称')
                    replaced_global = True
                else:
                    if not stripped.startswith('#'):
                        new_lines.append('# ' + line)
                    else:
                        new_lines.append(line)
            else:
                new_lines.append(line)
        # 处理类属性 target_object_name
        elif 'target_object_name:' in line and 'str' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not replaced_class_attr:
                if original_indent == 0:
                    original_indent = 4
                new_lines.append(' ' * original_indent + 'target_object_name: str = TARGET_OBJECT_NAME')
                replaced_class_attr = True
            else:
                if not stripped.startswith('#'):
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    new_lines.append(line)
        # 处理 finger_grasp_mode (仅对 grasp 任务)
        elif finger_grasp_mode and 'finger_grasp_mode:' in line and 'str' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not replaced_finger_mode and not stripped.startswith('#'):
                new_lines.append(' ' * original_indent + f'finger_grasp_mode: str = "{finger_grasp_mode}"')
                replaced_finger_mode = True
            else:
                if not stripped.startswith('#'):
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    new_lines.append(line)
        # ⭐ 重要：先处理 left_pre_grasp_offset（handover 任务）
        # 因为 'left_pre_grasp_x_offset' 也包含 'pre_grasp_x_offset'，必须先匹配更具体的模式
        elif left_pre_grasp_offset and task_type == "handover" and 'left_pre_grasp_x_offset:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            # 检查是否是未注释的行
            if not stripped.startswith('#'):
                if not replaced_left_pre_grasp_x:
                    new_lines.append(' ' * original_indent + f'left_pre_grasp_x_offset: float = {left_pre_grasp_offset[0]}')
                    replaced_left_pre_grasp_x = True
                else:
                    # 已经替换过了，注释掉这行
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                # 保持注释行不变
                new_lines.append(line)
        elif left_pre_grasp_offset and task_type == "handover" and 'left_pre_grasp_y_offset:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not stripped.startswith('#'):
                if not replaced_left_pre_grasp_y:
                    new_lines.append(' ' * original_indent + f'left_pre_grasp_y_offset: float = {left_pre_grasp_offset[1]}')
                    replaced_left_pre_grasp_y = True
                else:
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                new_lines.append(line)
        elif left_pre_grasp_offset and task_type == "handover" and 'left_pre_grasp_height:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not stripped.startswith('#'):
                if not replaced_left_pre_grasp_height:
                    new_lines.append(' ' * original_indent + f'left_pre_grasp_height: float = {left_pre_grasp_offset[2]}')
                    replaced_left_pre_grasp_height = True
                else:
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                new_lines.append(line)
        # 处理 pre_grasp_offset
        # 对于 handover 任务，只修改 right_pre_grasp_* 参数
        # 对于其他任务，修改 pre_grasp_* 参数
        elif pre_grasp_offset and 'pre_grasp_x_offset:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            # handover 任务需要修改 right_pre_grasp_x_offset
            is_right_hand = 'right_pre_grasp_x_offset' in line
            # 注意：left_pre_grasp_x_offset 已经在前面的 elif 中处理了，这里不会匹配到
            is_target_line = (task_type == "handover" and is_right_hand) or (task_type != "handover" and 'left_pre_grasp_x_offset' not in line and 'right_pre_grasp_x_offset' not in line)
            
            # 先判断是否是注释行
            if not stripped.startswith('#'):
                # 未注释行：检查是否是目标行且未被替换
                if is_target_line and not replaced_pre_grasp_x:
                    param_name = 'right_pre_grasp_x_offset' if task_type == "handover" else 'pre_grasp_x_offset'
                    new_lines.append(' ' * original_indent + f'{param_name}: float = {pre_grasp_offset[0]}')
                    replaced_pre_grasp_x = True
                elif is_target_line:
                    # 目标行但已经替换过了，注释掉
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    # 其他情况：注释掉
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                # 注释行保持不变
                new_lines.append(line)
        elif pre_grasp_offset and 'pre_grasp_y_offset:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            is_right_hand = 'right_pre_grasp_y_offset' in line
            is_target_line = (task_type == "handover" and is_right_hand) or (task_type != "handover" and 'left_pre_grasp_y_offset' not in line and 'right_pre_grasp_y_offset' not in line)
            
            if not stripped.startswith('#'):
                if is_target_line and not replaced_pre_grasp_y:
                    param_name = 'right_pre_grasp_y_offset' if task_type == "handover" else 'pre_grasp_y_offset'
                    new_lines.append(' ' * original_indent + f'{param_name}: float = {pre_grasp_offset[1]}')
                    replaced_pre_grasp_y = True
                elif is_target_line:
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                new_lines.append(line)
        elif pre_grasp_offset and 'pre_grasp_height:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            is_right_hand = 'right_pre_grasp_height' in line
            is_target_line = (task_type == "handover" and is_right_hand) or (task_type != "handover" and 'left_pre_grasp_height' not in line and 'right_pre_grasp_height' not in line)
            
            if not stripped.startswith('#'):
                if is_target_line and not replaced_pre_grasp_height:
                    param_name = 'right_pre_grasp_height' if task_type == "handover" else 'pre_grasp_height'
                    new_lines.append(' ' * original_indent + f'{param_name}: float = {pre_grasp_offset[2]}')
                    replaced_pre_grasp_height = True
                elif is_target_line:
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                new_lines.append(line)
        # 处理 right_post_release_offset (仅对 handover 任务)
        elif right_post_release_offset and task_type == "handover" and 'right_post_release_x_offset:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not stripped.startswith('#'):
                if not replaced_right_post_release_x:
                    new_lines.append(' ' * original_indent + f'right_post_release_x_offset: float = {right_post_release_offset[0]}')
                    replaced_right_post_release_x = True
                else:
                    # 已经替换过了，注释掉这行
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                # 保持注释行不变
                new_lines.append(line)
        elif right_post_release_offset and task_type == "handover" and 'right_post_release_y_offset:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not stripped.startswith('#'):
                if not replaced_right_post_release_y:
                    new_lines.append(' ' * original_indent + f'right_post_release_y_offset: float = {right_post_release_offset[1]}')
                    replaced_right_post_release_y = True
                else:
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                new_lines.append(line)
        elif right_post_release_offset and task_type == "handover" and 'right_post_release_height:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not stripped.startswith('#'):
                if not replaced_right_post_release_height:
                    new_lines.append(' ' * original_indent + f'right_post_release_height: float = {right_post_release_offset[2]}')
                    replaced_right_post_release_height = True
                else:
                    new_lines.append(' ' * original_indent + '# ' + stripped)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    modified_content = '\n'.join(new_lines)
    
    with open(task_mp_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    log(f"{task_type}_mp.py 修改完成")
    
    # 清理 Python 缓存
    log(f"清理 {task_type}_mp.py 的 Python 缓存...")
    clear_python_cache(task_mp_path)


def cleanup_isaac_sim():
    """清理 IsaacSim 相关进程"""
    log("清理 IsaacSim 进程...")
    
    try:
        # 查找并终止 Isaac Sim 相关进程
        processes_to_kill = [
            "isaac-sim",
            "omniverse",
            "kit",
            "vulkan"
        ]
        
        for proc_name in processes_to_kill:
            try:
                # 使用 pkill 终止进程 (温和方式)
                result = subprocess.run(
                    ["pkill", "-f", proc_name],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    log(f"终止进程: {proc_name}")
            except Exception as e:
                log(f"清理进程 {proc_name} 时出错: {e}", level="WARNING")
        
        # 等待进程完全退出
        log("等待进程完全退出...")
        time.sleep(5)
        
        # 如果还有顽固进程，使用 SIGKILL 强制终止
        for proc_name in processes_to_kill:
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", proc_name],
                    capture_output=True,
                    timeout=5
                )
            except Exception:
                pass
        
        # 额外清理：查找占用 GPU 的 Python 进程
        try:
            # 使用 nvidia-smi 查找使用 GPU 的进程
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
                            log(f"终止 GPU 进程: {pid}")
                        except Exception:
                            pass
        except Exception as e:
            log(f"清理 GPU 进程时出错: {e}", level="WARNING")
        
        log("IsaacSim 进程清理完成")
        
    except Exception as e:
        log(f"清理 IsaacSim 进程时出错: {e}", level="WARNING")


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


def run_data_collection(args):
    """运行数据采集"""
    log("开始数据采集...")
    
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
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')
                
                if '已达到目标成功次数' in line or '🎉' in line or '最终成功率' in line:
                    success_marker = True
                    log("✅ 检测到成功标记，数据采集完成", level="INFO")
                
                if 'Replicator:Annotators' in line or 'Replicator:Core' in line:
                    cleanup_started = True
                
                if success_marker and cleanup_started:
                    log("检测到进程正在清理，等待 5 秒后强制终止...", level="INFO")
                    time.sleep(5)
                    
                    if process.poll() is None:
                        log("进程仍在运行，强制终止", level="WARNING")
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    break
        
        if process.poll() is None:
            try:
                returncode = process.wait(timeout=10)
                log(f"数据采集进程退出，返回码: {returncode}")
            except subprocess.TimeoutExpired:
                log("数据采集进程超时，强制终止", level="WARNING")
                process.kill()
                process.wait()
        else:
            returncode = process.returncode
            log(f"数据采集进程已退出，返回码: {returncode}")
        
        if success_marker:
            log("✅ 数据采集完成（基于成功标记）")
            return True
        elif returncode == 0:
            log("✅ 数据采集完成（基于返回码）")
            return True
        else:
            log(f"❌ 数据采集失败，返回码: {returncode}，未检测到成功标记", level="ERROR")
            return False
            
    except Exception as e:
        log(f"❌ 数据采集异常: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return False


def find_latest_data_folder(target_object_name, object_name_map, task_type):
    """查找最新生成的数据文件夹"""
    log(f"查找最新的数据文件夹: {target_object_name}")
    
    chinese_name = object_name_map.get(target_object_name, target_object_name)
    
    # 查找数据文件夹
    data_base_path = WORKSPACE_ROOT / f"data/motion_plan/{task_type}"
    
    # 先尝试中文名称
    search_pattern = str(data_base_path / chinese_name / "*")
    folders = glob.glob(search_pattern)
    
    # 如果没找到，尝试英文名称
    if not folders:
        search_pattern = str(data_base_path / target_object_name / "*")
        folders = glob.glob(search_pattern)
    
    if folders:
        # 找到最新的文件夹（按时间戳）
        latest_folder = max(folders, key=os.path.getmtime)
        log(f"找到最新数据文件夹: {latest_folder}")
        return latest_folder
    else:
        log(f"未找到数据文件夹: {chinese_name} 或 {target_object_name}", level="ERROR")
        return None


def run_zarr_conversion(h5_dir, zarr_dir, args):
    """运行 Zarr 转换"""
    log(f"开始 Zarr 转换: {h5_dir}")
    
    cmd = [
        sys.executable, str(ZARR_UTILS_PATH),
        "--h5_dir", str(h5_dir),
        "--zarr_dir", str(zarr_dir),
    ] + args
    
    log(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT))
        log("Zarr 转换完成")
        return True
    except subprocess.CalledProcessError as e:
        log(f"Zarr 转换失败: {e}", level="ERROR")
        return False


def process_object(config, index, total, task_config, data_collect_args, zarr_dir, zarr_convert_args, object_name_map, skip_conversion=False, auto_cleanup=True):
    """处理单个物体的数据采集流程"""
    target_object_name = config["target_object_name"]
    usd_path = config["usd_path"]
    task_type = task_config["task_type"]
    finger_grasp_mode = config.get("finger_grasp_mode")  # 可选
    pre_grasp_offset = config.get("pre_grasp_offset")    # 可选（右手）
    left_pre_grasp_offset = config.get("left_pre_grasp_offset")  # 可选（左手，仅 handover）
    right_post_release_offset = config.get("right_post_release_offset")  # 可选（右手释放后，仅 handover）
    
    log(f"{'='*60}")
    log(f"[{task_type}] 处理物体 [{index+1}/{total}]: {target_object_name}")
    log(f"USD 路径: {usd_path}")
    if finger_grasp_mode:
        log(f"手指抓取模式: {finger_grasp_mode}")
    if pre_grasp_offset:
        log(f"预抓取偏移 (右手): {pre_grasp_offset}")
    if left_pre_grasp_offset:
        log(f"预抓取偏移 (左手): {left_pre_grasp_offset}")
    if right_post_release_offset:
        log(f"释放后偏移 (右手): {right_post_release_offset}")
    log(f"{'='*60}")
    
    # ⭐ 重要：在开始前先清理所有残留进程
    if auto_cleanup and index > 0:
        log("🧹 开始前清理所有残留进程...")
        cleanup_isaac_sim()
        time.sleep(3)
    
    # 备份文件
    room_cfg_backup = backup_file(task_config["room_cfg_path"])
    task_mp_backup = backup_file(task_config["task_mp_path"])
    
    try:
        # 1. 修改配置文件
        modify_room_cfg(task_config["room_cfg_path"], usd_path, task_type)
        modify_task_mp(task_config["task_mp_path"], target_object_name, task_type, finger_grasp_mode, pre_grasp_offset, left_pre_grasp_offset, right_post_release_offset)
        
        # ⭐ 修改配置后等待确保文件系统同步
        log("等待配置文件同步...")
        time.sleep(2)
        
        # 2. 运行数据采集
        if not run_data_collection(data_collect_args):
            log(f"物体 {target_object_name} 数据采集失败", level="ERROR")
            return False
        
        # 2.1 等待进程完全退出
        if auto_cleanup:
            wait_for_process_exit(timeout=30)
            
            # 2.2 清理 IsaacSim 进程
            cleanup_isaac_sim()
            
            # ⭐ 清理后额外等待，确保资源完全释放
            log("等待资源完全释放...")
            time.sleep(5)
        
        if skip_conversion:
            log("跳过 Zarr 转换步骤（--skip-conversion 已启用）")
            return True
        
        # 3. 查找最新数据文件夹
        h5_dir = find_latest_data_folder(target_object_name, object_name_map, task_type)
        if not h5_dir:
            log(f"物体 {target_object_name} 未找到数据文件夹，跳过转换步骤", level="WARNING")
            return False
        
        # 4. 运行 Zarr 转换
        if not run_zarr_conversion(h5_dir, zarr_dir, zarr_convert_args):
            log(f"物体 {target_object_name} Zarr 转换失败", level="ERROR")
            return False
        
        log(f"物体 {target_object_name} 处理完成！")
        return True
        
    except Exception as e:
        log(f"处理物体 {target_object_name} 时出错: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 恢复备份文件
        restore_file(task_config["room_cfg_path"], room_cfg_backup)
        restore_file(task_config["task_mp_path"], task_mp_backup)
        
        # ⭐ 重要：恢复后立即清理缓存
        # 防止下次运行时使用采集时的旧缓存
        log("清理配置文件的 Python 缓存（恢复后）...")
        clear_python_cache(task_config["room_cfg_path"])
        clear_python_cache(task_config["task_mp_path"])
        
        # ⭐ 恢复配置后等待文件系统同步
        time.sleep(1)
        
        # 最终清理：确保所有进程都被清理
        if auto_cleanup:
            cleanup_isaac_sim()
            
            # ⭐ 最终清理后额外等待
            log("最终清理完成，等待环境稳定...")
            time.sleep(3)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="通用批量数据采集脚本")
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径 (YAML 格式)")
    parser.add_argument("--task", type=str, required=True, choices=["grasp", "pick_place", "handover"],
                        help="任务类型: grasp, pick_place 或 handover")
    parser.add_argument("--objects", type=str, nargs="+",
                        help="指定要处理的物体名称（不指定则处理全部）")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="不自动清理 IsaacSim 进程（调试用）")
    parser.add_argument("--wait-time", type=int, default=20,
                        help="物体之间的等待时间（秒），默认20秒")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="跳过 Zarr 转换步骤")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将要处理的物体，不实际执行")
    parser.add_argument("--no-headless", action="store_true",
                        help="不使用无头模式（显示GUI界面）")
    args = parser.parse_args()
    
    # 获取任务配置
    task_config_dict = TASK_CONFIGS[args.task]
    task_config_dict["task_type"] = args.task
    
    # 加载配置
    config = load_config_from_yaml(args.config)
    if not config:
        log("无法加载配置文件，退出", level="ERROR")
        sys.exit(1)
    
    objects_config = config.get("objects", [])
    
    # 过滤指定的物体
    if args.objects:
        objects_config = [obj for obj in objects_config if obj["target_object_name"] in args.objects]
    
    if not objects_config:
        log("没有要处理的物体", level="WARNING")
        sys.exit(0)
    
    log(f"任务类型: {args.task}")
    log(f"将处理 {len(objects_config)} 个物体:")
    for obj in objects_config:
        log(f"  - {obj['target_object_name']}")
    
    if args.dry_run:
        log("Dry run 模式，不执行实际操作")
        sys.exit(0)
    
    # 读取数据采集参数
    dc_config = config.get("data_collection", {})
    data_collect_args = [
        "--task", task_config_dict["task_name"],
        "--num_envs", str(dc_config.get("num_envs", 10)),
        "--seed", str(dc_config.get("seed", 17)),
        "--scene", task_config_dict["scene_name"],
        "--enable_cameras",
        "--enable_output",
        "--enable_random",
        "--enable_eval",
        "--async_reset",
        "--sample_step", str(dc_config.get("sample_step", 4)),
        "--max_episode", str(dc_config.get("max_episode", 500)),
    ]
    
    # 处理 headless 模式：命令行参数优先于配置文件
    use_headless = dc_config.get("headless", True) and not args.no_headless
    if use_headless:
        data_collect_args.append("--headless")
    
    log(f"数据采集参数: {' '.join(data_collect_args)}")
    
    # 读取 Zarr 转换参数
    zc_config = config.get("zarr_conversion", {})
    zarr_convert_args = [
        "--mode", zc_config.get("mode", "rgb"),
    ]
    if zc_config.get("with_mask", True):
        zarr_convert_args.append("--with_mask")
    if zc_config.get("with_depth", True):
        zarr_convert_args.append("--with_depth")
    if zc_config.get("with_normals", True):
        zarr_convert_args.append("--with_normals")
    
    zarr_dir = Path(zc_config.get("zarr_dir", str(WORKSPACE_ROOT / "data/zarr_final")))
    
    # 构建物体名称映射
    object_name_map = {}
    for obj in objects_config:
        if "chinese_name" in obj:
            object_name_map[obj["target_object_name"]] = obj["chinese_name"]
    
    log(f"Zarr 转换参数: {' '.join(zarr_convert_args)}")
    log(f"Zarr 输出目录: {zarr_dir}")
    
    # 批量处理
    auto_cleanup = not args.no_cleanup
    results = []
    
    for i, config_item in enumerate(objects_config):
        success = process_object(
            config_item, i, len(objects_config),
            task_config_dict, data_collect_args, zarr_dir, zarr_convert_args,
            object_name_map, args.skip_conversion, auto_cleanup
        )
        results.append({
            "object": config_item["target_object_name"],
            "success": success
        })
        
        completed = len(results)
        log(f"进度: {completed}/{len(objects_config)} 完成")
        
        if completed < len(objects_config):
            log(f"等待 {args.wait_time} 秒后继续下一个物体...")
            time.sleep(args.wait_time)
            
            if auto_cleanup:
                log("⚠️  开始处理下一个物体前，再次彻底清理进程...")
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
    log("批量数据采集完成！")
    log("=" * 60)
    log(f"任务类型: {args.task}")
    log(f"总计: {len(results)} 个物体")
    success_count = sum(1 for r in results if r["success"])
    log(f"成功: {success_count}")
    log(f"失败: {len(results) - success_count}")
    
    log("\n详细结果:")
    for r in results:
        status = "✓ 成功" if r["success"] else "✗ 失败"
        log(f"  {r['object']}: {status}")
    
    if success_count < len(results):
        log("\n失败的物体:")
        for r in results:
            if not r["success"]:
                log(f"  - {r['object']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n用户中断，退出程序", level="WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"程序异常: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

