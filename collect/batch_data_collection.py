#!/usr/bin/env python3
"""
批量数据采集脚本 (支持配置文件版本)
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

# 文件路径
ROOM_CFG_PATH = WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp/scenes/room_cfg.py"
GRASP_MP_PATH = WORKSPACE_ROOT / "psilab/source/psilab_tasks/psilab_tasks/imitation_learning/grasp/grasp_mp.py"
PLAY_SCRIPT_PATH = WORKSPACE_ROOT / "psilab/scripts_psi/workflows/motion_planning/play.py"
ZARR_UTILS_PATH = WORKSPACE_ROOT / "psilab/source/psilab/psilab/utils/zarr_utils.py"

# 默认配置文件路径
DEFAULT_CONFIG_FILE = WORKSPACE_ROOT / "objects_config.yaml"

# 默认数据采集参数
DEFAULT_DATA_COLLECT_ARGS = [
    "--task", "Psi-MP-Grasp-v1",
    "--num_envs", "30",
    "--seed", "17",
    "--scene", "room_cfg:PSI_DC_Grasp_CFG",
    "--enable_cameras",
    "--enable_output",
    "--enable_random",
    "--enable_eval",
    "--async_reset",
    "--sample_step", "4",
    "--max_episode", "500",
    "--headless",  # 无头模式，不显示 GUI
]

# 默认数据转换参数
DEFAULT_ZARR_DIR = WORKSPACE_ROOT / "data/zarr_point_cloud"
DEFAULT_ZARR_CONVERT_ARGS = [
    "--mode", "rgb",
    "--task_type", "single_hand",
    "--max_episodes", "50",
    "--with_mask",
    "--with_depth",
    "--with_normals"
]

# 默认物体名称映射（中英文）
DEFAULT_OBJECT_NAME_MAP = {
    "glass_beaker_100ml": "100ml玻璃烧杯",
    "glass_beaker_250ml": "250ml玻璃烧杯",
    "glass_beaker_50ml": "50ml玻璃烧杯",
    "glass_beaker_500ml": "500ml玻璃烧杯",
    "mortar": "坩埚",
    "funnel_stand": "漏斗架",
    "brown_reagent_bottle_large": "棕色试剂瓶(大)",
    "clear_reagent_bottle_large": "透明试剂瓶(大)",
}

# 默认物体配置列表
DEFAULT_OBJECTS_CONFIG = [
    {
        "target_object_name": "glass_beaker_100ml",
        "usd_path": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
    },
    {
        "target_object_name": "glass_beaker_250ml",
        "usd_path": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_250ml/Beaker004.usd",
    },
    {
        "target_object_name": "glass_beaker_50ml",
        "usd_path": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_50ml/Beaker002.usd",
    },
    {
        "target_object_name": "glass_beaker_500ml",
        "usd_path": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_500ml/Beaker005.usd",
    },
    {
        "target_object_name": "mortar",
        "usd_path": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/mortar/Mortar001.usd",
    },
]


# ========== 工具函数 ==========
def log(message, level="INFO"):
    """打印日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def clear_python_cache(py_file_path):
    """清理 Python 缓存文件 (.pyc 和 __pycache__)
    
    这是关键步骤！Python 会缓存编译后的 .pyc 文件，
    即使修改了 .py 文件，旧的缓存可能仍然被使用。
    """
    py_file = Path(py_file_path)
    
    # 1. 清理同目录下的 __pycache__
    pycache_dir = py_file.parent / "__pycache__"
    if pycache_dir.exists():
        module_name = py_file.stem
        # 查找所有相关的 .pyc 文件
        cache_files = list(pycache_dir.glob(f"{module_name}.*.pyc"))
        for pyc_file in cache_files:
            try:
                pyc_file.unlink()
                log(f"  ✓ 已删除缓存: {pyc_file.name}")
            except Exception as e:
                log(f"  ✗ 删除缓存失败 {pyc_file.name}: {e}", level="WARNING")
        
        if cache_files:
            log(f"清理了 {len(cache_files)} 个缓存文件")
    
    # 2. 清理父目录的 __pycache__（如果是包的一部分）
    parent_pycache = py_file.parent.parent / "__pycache__"
    if parent_pycache.exists():
        # 清理与这个模块相关的缓存
        parent_module_name = py_file.parent.name
        for pyc_file in parent_pycache.glob(f"{parent_module_name}.*.pyc"):
            try:
                pyc_file.unlink()
            except Exception:
                pass


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
        log(f"加载配置文件失败: {e}，使用默认配置", level="WARNING")
        return None


def backup_file(file_path):
    """备份文件"""
    backup_path = f"{file_path}.backup_{int(time.time())}"
    subprocess.run(["cp", str(file_path), backup_path], check=True)
    log(f"已备份文件: {backup_path}")
    return backup_path


def restore_file(file_path, backup_path):
    """恢复文件"""
    subprocess.run(["mv", str(backup_path), str(file_path)], check=True)
    log(f"已恢复文件: {file_path}")


def cleanup_isaac_sim():
    """清理 IsaacSim 相关进程"""
    log("清理 IsaacSim 进程...")
    
    try:
        # 查找并终止 Isaac Sim 相关进程
        # 常见的进程名包括: isaac-sim, python (运行 play.py), omniverse
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


def modify_room_cfg(usd_path):
    """修改 room_cfg.py 中的 usd_path（只修改 rigid_objects_cfg 中的 bottle）"""
    log(f"修改 room_cfg.py 中的 usd_path: {usd_path}")
    
    with open(ROOM_CFG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    in_psi_dc_grasp_cfg = False
    in_rigid_objects = False
    in_bottle_spawn = False
    usd_path_added = False
    
    for i, line in enumerate(lines):
        # 检测是否进入 PSI_DC_Grasp_CFG 块（通过查找这个特定的配置块）
        if 'PSI_DC_Grasp_CFG' in line and '=' in line and 'replace' in line:
            in_psi_dc_grasp_cfg = True
            new_lines.append(line)
            continue
        
        # 只在 PSI_DC_Grasp_CFG 块中处理
        if in_psi_dc_grasp_cfg:
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
                # 如果是 usd_path 行
                if 'usd_path' in line and '=' in line:
                    stripped = line.strip()
                    indent = len(line) - len(line.lstrip())
                    
                    # 如果还没有添加新的 usd_path
                    if not usd_path_added:
                        # 添加新的 usd_path（取消注释）
                        new_lines.append(' ' * indent + f'usd_path="{usd_path}",')
                        usd_path_added = True
                    else:
                        # 将其他所有 usd_path 行注释掉
                        if not stripped.startswith('#'):
                            new_lines.append(' ' * indent + '# ' + stripped)
                        else:
                            new_lines.append(line)
                else:
                    new_lines.append(line)
                    
                    # 检查是否结束 bottle 的配置
                    # 遇到 scale= 或 rigid_props= 表示 spawn 块结束
                    if 'scale=' in line or 'rigid_props=' in line:
                        in_bottle_spawn = False
                        usd_path_added = False
            else:
                new_lines.append(line)
                
                # 检查是否结束 rigid_objects_cfg 块
                if in_rigid_objects and line.strip() == '},':
                    in_rigid_objects = False
                
                # 检查是否结束 PSI_DC_Grasp_CFG 块
                if in_psi_dc_grasp_cfg and line.strip() == ')' and not in_rigid_objects:
                    in_psi_dc_grasp_cfg = False
        else:
            new_lines.append(line)
    
    modified_content = '\n'.join(new_lines)
    
    with open(ROOM_CFG_PATH, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    log("room_cfg.py 修改完成")
    
    # ⭐ 关键：清理 Python 缓存
    log("清理 room_cfg.py 的 Python 缓存...")
    clear_python_cache(ROOM_CFG_PATH)


def modify_grasp_mp(target_object_name, finger_grasp_mode=None, pre_grasp_offset=None):
    """修改 grasp_mp.py 中的 TARGET_OBJECT_NAME, finger_grasp_mode 和 pre_grasp_offset"""
    log(f"修改 grasp_mp.py 中的 TARGET_OBJECT_NAME: {target_object_name}")
    if finger_grasp_mode:
        log(f"  设置 finger_grasp_mode: {finger_grasp_mode}")
    if pre_grasp_offset:
        log(f"  设置 pre_grasp_offset: {pre_grasp_offset}")
    
    with open(GRASP_MP_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式替换 TARGET_OBJECT_NAME 的值
    lines = content.split('\n')
    new_lines = []
    replaced_global = False
    replaced_class_attr = False
    replaced_finger_mode = False
    replaced_pre_grasp_x = False
    replaced_pre_grasp_y = False
    replaced_pre_grasp_height = False
    
    for i, line in enumerate(lines):
        # 处理全局变量 TARGET_OBJECT_NAME (在文件顶部，没有缩进或很少缩进)
        if 'TARGET_OBJECT_NAME' in line and '=' in line and 'TASK_TYPE' not in line and 'target_object_name:' not in line:
            stripped = line.strip()
            # 检查是否是全局变量（缩进很少）
            indent_count = len(line) - len(line.lstrip())
            if indent_count == 0:  # 全局变量
                if not replaced_global and not stripped.startswith('#'):
                    # 第一个非注释的全局 TARGET_OBJECT_NAME 行
                    new_lines.append(f'TARGET_OBJECT_NAME = "{target_object_name}"  # 目标物体名称')
                    replaced_global = True
                else:
                    # 注释掉其他的
                    if not stripped.startswith('#'):
                        new_lines.append('# ' + line)
                    else:
                        new_lines.append(line)
            else:
                # 保持其他缩进的行不变
                new_lines.append(line)
        # 处理类属性 target_object_name (带缩进，可能被注释)
        elif 'target_object_name:' in line and 'str' in line:
            stripped = line.strip()
            # 获取原始缩进
            original_indent = len(line) - len(line.lstrip())
            
            # 无论是否被注释，都取消注释并设置
            if not replaced_class_attr:
                # 保持原始缩进（通常是4个空格）
                if original_indent == 0:
                    original_indent = 4  # 如果没有缩进，使用默认的4个空格
                new_lines.append(' ' * original_indent + 'target_object_name: str = TARGET_OBJECT_NAME')
                replaced_class_attr = True
            else:
                # 注释掉其他的
                if not stripped.startswith('#'):
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    new_lines.append(line)
        # 处理 finger_grasp_mode
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
        # 处理 pre_grasp_offset
        elif pre_grasp_offset and 'pre_grasp_x_offset:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not replaced_pre_grasp_x and not stripped.startswith('#'):
                new_lines.append(' ' * original_indent + f'pre_grasp_x_offset: float = {pre_grasp_offset[0]}')
                replaced_pre_grasp_x = True
            else:
                if not stripped.startswith('#'):
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    new_lines.append(line)
        elif pre_grasp_offset and 'pre_grasp_y_offset:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not replaced_pre_grasp_y and not stripped.startswith('#'):
                new_lines.append(' ' * original_indent + f'pre_grasp_y_offset: float = {pre_grasp_offset[1]}')
                replaced_pre_grasp_y = True
            else:
                if not stripped.startswith('#'):
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    new_lines.append(line)
        elif pre_grasp_offset and 'pre_grasp_height:' in line and 'float' in line and '=' in line:
            stripped = line.strip()
            original_indent = len(line) - len(line.lstrip())
            
            if not replaced_pre_grasp_height and not stripped.startswith('#'):
                new_lines.append(' ' * original_indent + f'pre_grasp_height: float = {pre_grasp_offset[2]}')
                replaced_pre_grasp_height = True
            else:
                if not stripped.startswith('#'):
                    new_lines.append(' ' * original_indent + '# ' + stripped)
                else:
                    new_lines.append(line)
        else:
            new_lines.append(line)
    
    modified_content = '\n'.join(new_lines)
    
    with open(GRASP_MP_PATH, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    log("grasp_mp.py 修改完成")
    
    # ⭐ 关键：清理 Python 缓存
    log("清理 grasp_mp.py 的 Python 缓存...")
    clear_python_cache(GRASP_MP_PATH)


def run_data_collection(args):
    """运行数据采集"""
    log("开始数据采集...")
    
    cmd = [sys.executable, str(PLAY_SCRIPT_PATH)] + args
    log(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 使用 Popen 启动进程，这样可以更好地控制
        process = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 实时输出日志并检测成功标记
        success_marker = False
        cleanup_started = False
        output_lines = []
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')
                output_lines.append(line)
                
                # 检测成功完成的标记
                if '已达到目标成功次数' in line or '🎉' in line or '最终成功率' in line:
                    success_marker = True
                    log("✅ 检测到成功标记，数据采集完成", level="INFO")
                
                # 检测清理开始的标记
                if 'Replicator:Annotators' in line or 'Replicator:Core' in line:
                    cleanup_started = True
                
                # 如果检测到成功标记且清理已开始，等待一小段时间后强制终止
                if success_marker and cleanup_started:
                    log("检测到进程正在清理，等待 5 秒后强制终止...", level="INFO")
                    time.sleep(5)
                    
                    # 检查进程是否还活着
                    if process.poll() is None:
                        log("进程仍在运行，强制终止", level="WARNING")
                        process.terminate()  # 先尝试温和终止
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()  # 强制终止
                            process.wait()
                    break
        
        # 如果循环正常结束（进程自然退出）
        if process.poll() is None:
            # 进程还在运行，等待退出
            try:
                returncode = process.wait(timeout=10)
                log(f"数据采集进程退出，返回码: {returncode}")
            except subprocess.TimeoutExpired:
                log("数据采集进程超时，强制终止", level="WARNING")
                process.kill()
                process.wait()
                returncode = -9
        else:
            returncode = process.returncode
            log(f"数据采集进程已退出，返回码: {returncode}")
        
        # 判断是否成功
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
        # 即使出现异常，也尝试检查输出
        if 'success_marker' in locals() and success_marker:
            log("虽然出现异常，但检测到成功标记，认为采集成功", level="WARNING")
            return True
        return False


def find_latest_data_folder(target_object_name, object_name_map):
    """查找最新生成的数据文件夹"""
    log(f"查找最新的数据文件夹: {target_object_name}")
    
    chinese_name = object_name_map.get(target_object_name, target_object_name)
    
    # 查找数据文件夹
    data_base_path = WORKSPACE_ROOT / "data/motion_plan/grasp"
    
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


def process_object(config, index, total, data_collect_args, zarr_dir, zarr_convert_args, object_name_map, skip_conversion=False, auto_cleanup=True):
    """处理单个物体的数据采集流程"""
    target_object_name = config["target_object_name"]
    usd_path = config["usd_path"]
    finger_grasp_mode = config.get("finger_grasp_mode")  # 可选
    pre_grasp_offset = config.get("pre_grasp_offset")    # 可选
    
    log(f"{'='*60}")
    log(f"处理物体 [{index+1}/{total}]: {target_object_name}")
    log(f"USD 路径: {usd_path}")
    if finger_grasp_mode:
        log(f"手指抓取模式: {finger_grasp_mode}")
    if pre_grasp_offset:
        log(f"预抓取偏移: {pre_grasp_offset}")
    log(f"{'='*60}")
    
    # ⭐ 重要：在开始前先清理所有残留进程
    if auto_cleanup and index > 0:  # 不是第一个物体
        log("🧹 开始前清理所有残留进程...")
        cleanup_isaac_sim()
        time.sleep(3)
    
    # 备份文件
    room_cfg_backup = backup_file(ROOM_CFG_PATH)
    grasp_mp_backup = backup_file(GRASP_MP_PATH)
    
    try:
        # 1. 修改配置文件
        modify_room_cfg(usd_path)
        modify_grasp_mp(target_object_name, finger_grasp_mode, pre_grasp_offset)
        
        # ⭐ 修改配置后等待确保文件系统同步
        log("等待配置文件同步...")
        time.sleep(2)
        
        # 2. 运行数据采集
        if not run_data_collection(data_collect_args):
            log(f"物体 {target_object_name} 数据采集失败，跳过后续步骤", level="ERROR")
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
        h5_dir = find_latest_data_folder(target_object_name, object_name_map)
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
        restore_file(ROOM_CFG_PATH, room_cfg_backup)
        restore_file(GRASP_MP_PATH, grasp_mp_backup)
        
        # ⭐ 重要：恢复后立即清理缓存
        # 防止下次运行时使用采集时的旧缓存
        log("清理配置文件的 Python 缓存（恢复后）...")
        clear_python_cache(ROOM_CFG_PATH)
        clear_python_cache(GRASP_MP_PATH)
        
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
    parser = argparse.ArgumentParser(description="批量数据采集脚本")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_FILE),
                        help="配置文件路径 (YAML 格式)")
    parser.add_argument("--objects", type=str, nargs="+",
                        help="指定要处理的物体名称（不指定则处理全部）")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="跳过 Zarr 转换步骤")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示将要处理的物体，不实际执行")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="不自动清理 IsaacSim 进程（调试用）")
    parser.add_argument("--wait-time", type=int, default=10,
                        help="物体之间的等待时间（秒），默认10秒")
    parser.add_argument("--no-headless", action="store_true",
                        help="禁用无头模式，显示 GUI（默认使用无头模式）")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config_from_yaml(args.config)
    
    if config:
        # 从配置文件读取
        objects_config = config.get("objects", DEFAULT_OBJECTS_CONFIG)
        
        # 读取数据采集参数
        dc_config = config.get("data_collection", {})
        data_collect_args = [
            "--task", "Psi-MP-Grasp-v1",
            "--num_envs", str(dc_config.get("num_envs", 30)),
            "--seed", str(dc_config.get("seed", 17)),
            "--scene", "room_cfg:PSI_DC_Grasp_CFG",
            "--enable_cameras",
            "--enable_output",
            "--enable_random",
            "--enable_eval",
            "--async_reset",
            "--sample_step", str(dc_config.get("sample_step", 4)),
            "--max_episode", str(dc_config.get("max_episode", 500)),
        ]
        
        # 添加无头模式参数
        # 优先级：命令行参数 > 配置文件 > 默认值（True）
        use_headless = not args.no_headless and dc_config.get("headless", True)
        if use_headless:
            data_collect_args.append("--headless")
            log("使用无头模式（headless）运行 IsaacSim")
        else:
            log("使用 GUI 模式运行 IsaacSim")
        
        # 读取 Zarr 转换参数
        zc_config = config.get("zarr_conversion", {})
        zarr_convert_args = [
            "--mode", zc_config.get("mode", "rgb"),
            "--task_type", zc_config.get("task_type", "auto"),
            "--max_episodes", str(zc_config.get("max_episodes", 50)),
        ]
        if zc_config.get("with_mask", True):
            zarr_convert_args.append("--with_mask")
        if zc_config.get("with_depth", True):
            zarr_convert_args.append("--with_depth")
        if zc_config.get("with_normals", True):
            zarr_convert_args.append("--with_normals")
        if zc_config.get("with_pointcloud", False):
            zarr_convert_args.append("--with_pointcloud")
            num_points = zc_config.get("num_points", 2048)
            zarr_convert_args.extend(["--num_points", str(num_points)])
        
        zarr_dir = Path(zc_config.get("zarr_dir", str(DEFAULT_ZARR_DIR)))
        
        # 构建物体名称映射
        object_name_map = DEFAULT_OBJECT_NAME_MAP.copy()
        for obj in objects_config:
            if "chinese_name" in obj:
                object_name_map[obj["target_object_name"]] = obj["chinese_name"]
    else:
        # 使用默认配置
        objects_config = DEFAULT_OBJECTS_CONFIG
        data_collect_args = DEFAULT_DATA_COLLECT_ARGS
        zarr_convert_args = DEFAULT_ZARR_CONVERT_ARGS
        zarr_dir = DEFAULT_ZARR_DIR
        object_name_map = DEFAULT_OBJECT_NAME_MAP
    
    # 过滤要处理的物体
    if args.objects:
        objects_config = [obj for obj in objects_config 
                          if obj["target_object_name"] in args.objects]
        if not objects_config:
            log(f"未找到指定的物体: {args.objects}", level="ERROR")
            return
    
    log("="*60)
    log("批量数据采集脚本启动")
    log(f"总共需要处理 {len(objects_config)} 个物体")
    log("="*60)
    
    # Dry run 模式
    if args.dry_run:
        log("Dry-run 模式：仅显示将要处理的物体")
        for i, obj in enumerate(objects_config):
            log(f"  [{i+1}] {obj['target_object_name']}: {obj['usd_path']}")
        return
    
    results = []
    
    auto_cleanup = not args.no_cleanup
    
    for i, config_item in enumerate(objects_config):
        success = process_object(
            config_item, i, len(objects_config),
            data_collect_args, zarr_dir, zarr_convert_args,
            object_name_map, args.skip_conversion, auto_cleanup
        )
        results.append({
            "object": config_item["target_object_name"],
            "success": success
        })
        
        # 打印进度
        completed = i + 1
        log(f"进度: {completed}/{len(objects_config)} 完成")
        
        if completed < len(objects_config):
            log(f"等待 {args.wait_time} 秒后继续下一个物体...")
            time.sleep(args.wait_time)
            
            # 在开始下一个物体前，再次确保所有进程都已清理
            if auto_cleanup:
                log("⚠️  开始处理下一个物体前，再次彻底清理进程...")
                cleanup_isaac_sim()
                time.sleep(5)  # 增加到 5 秒确保清理彻底
                
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
    
    # 打印总结
    log("="*60)
    log("批量数据采集完成！")
    log("="*60)
    
    success_count = sum(1 for r in results if r["success"])
    log(f"成功: {success_count}/{len(results)}")
    
    log("\n详细结果:")
    for r in results:
        status = "✓ 成功" if r["success"] else "✗ 失败"
        log(f"  {r['object']}: {status}")


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
