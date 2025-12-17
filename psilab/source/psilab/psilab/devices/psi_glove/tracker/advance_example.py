#!/usr/bin/env python3
"""
在宿主机上运行 SDK - 自动配置并打印位置数据
"""

import os
import sys
import json
import tempfile
import shutil

# 设置 SteamVR 路径
STEAMVR_PATH = "/home/admin01/.local/share/Steam/steamapps/common/SteamVR"

def print_separator():
    print("=" * 70)

def print_subseparator():
    print("-" * 70)

def setup_openvr_config():
    """设置 OpenVR 配置 - 使用snap配置或创建临时配置"""
    # 优先使用 snap 的配置
    snap_config = os.path.expanduser("~/snap/steam/common/.config")
    if os.path.exists(os.path.join(snap_config, "openvr/openvrpaths.vrpath")):
        os.environ['XDG_CONFIG_HOME'] = snap_config
        print(f"        使用 snap 配置: {snap_config}")
        return None
    
    # 如果snap配置不存在，创建临时配置
    tmp_home = tempfile.mkdtemp(prefix="steamvr_sdk_")
    tmp_config_dir = os.path.join(tmp_home, ".config", "openvr")
    os.makedirs(os.path.join(tmp_config_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(tmp_config_dir, "logs"), exist_ok=True)
    
    # 创建配置文件
    config = {
        "config": [os.path.join(tmp_config_dir, "config")],
        "external_drivers": None,
        "jsonid": "vrpathreg",
        "log": [os.path.join(tmp_config_dir, "logs")],
        "runtime": [STEAMVR_PATH],
        "version": 1
    }
    
    config_file = os.path.join(tmp_config_dir, "openvrpaths.vrpath")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent='\t')
    
    # 设置环境变量
    os.environ['HOME'] = tmp_home
    print(f"        临时配置目录: {tmp_home}")
    
    return tmp_home

def main():
    print_separator()
    print("  SteamVR Tracker SDK - 宿主机位置数据打印")
    print_separator()
    print()
    
    # 设置配置
    print("[准备] 配置 OpenVR 环境...")
    tmp_home = setup_openvr_config()
    print(f"        SteamVR 路径: {STEAMVR_PATH}")
    print()
    
    # 导入依赖
    import openvr
    import numpy as np
    from scipy.spatial.transform import Rotation
    import time
    
    # 初始化 OpenVR
    print("[1/4] 初始化 OpenVR 连接...")
    try:
        openvr.init(openvr.VRApplication_Utility)
        print("      ✓ OpenVR 初始化成功")
    except Exception as e:
        print(f"      ✗ OpenVR 初始化失败: {type(e).__name__}")
        print()
        print("解决方法:")
        print("  1. 在 Steam 客户端中启动 SteamVR:")
        print("     打开 Steam → 库 → SteamVR → 运行")
        print()
        print("  2. 或使用命令启动:")
        print("     steam steam://rungameid/250820 &")
        print()
        shutil.rmtree(tmp_home, ignore_errors=True)
        return 1
    
    print()
    
    # 获取 VRSystem
    try:
        vr_system = openvr.VRSystem()
        print("[2/4] VRSystem 接口获取成功")
    except Exception as e:
        print(f"[2/4] ✗ 无法获取 VRSystem: {e}")
        openvr.shutdown()
        shutil.rmtree(tmp_home, ignore_errors=True)
        return 1
    
    print()
    
    # 扫描 Tracker
    print("[3/4] 扫描 Tracker 设备...")
    trackers = []
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if not vr_system.isTrackedDeviceConnected(i):
            continue
        
        device_class = vr_system.getTrackedDeviceClass(i)
        if device_class == openvr.TrackedDeviceClass_GenericTracker:
            try:
                serial = vr_system.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
                model = vr_system.getStringTrackedDeviceProperty(i, openvr.Prop_ModelNumber_String)
                trackers.append({'index': i, 'serial': serial, 'model': model})
                print(f"      ✓ 发现 Tracker: {serial}")
                print(f"        型号: {model}")
            except:
                pass
    
    print()
    
    if not trackers:
        print("      ✗ 未发现任何 Tracker")
        print()
        print("请检查:")
        print("  - Tracker 已开机（长按按钮直到蓝灯亮）")
        print("  - 基站已开机（绿灯稳定）")
        print("  - Tracker 已在 SteamVR 中配对")
        openvr.shutdown()
        shutil.rmtree(tmp_home, ignore_errors=True)
        return 1
    
    print(f"      共发现 {len(trackers)} 个 Tracker")
    print()
    
    # 实时打印位置数据
    print("[4/4] 实时打印位置数据...")
    print()
    print_separator()
    print()
    
    try:
        frame_count = 0
        while True:
            frame_count += 1
            
            # 获取所有设备的 pose
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            
            # 清屏效果（移动光标）
            if frame_count > 1:
                print("\033[2J\033[H", end="")  # 清屏并移动到顶部
            
            print(f"📊 帧 #{frame_count}  |  ⏰ 时间: {time.strftime('%H:%M:%S')}")
            print_separator()
            print()
            
            # 处理每个 Tracker
            for tracker in trackers:
                idx = tracker['index']
                serial = tracker['serial']
                model = tracker['model']
                
                print(f"📍 Tracker: {serial}")
                print(f"   型号: {model}")
                
                if not poses[idx].bPoseIsValid:
                    print(f"   状态: ❌ 无效位置（可能被遮挡）")
                    print()
                    print_subseparator()
                    print()
                    continue
                
                mat = poses[idx].mDeviceToAbsoluteTracking
                
                # 提取位置
                x = mat[0][3]
                y = mat[1][3]
                z = mat[2][3]
                
                # 提取旋转矩阵
                rot_matrix = np.array([
                    [mat[0][0], mat[0][1], mat[0][2]],
                    [mat[1][0], mat[1][1], mat[1][2]],
                    [mat[2][0], mat[2][1], mat[2][2]]
                ])
                
                # 转换为四元数和欧拉角
                rotation = Rotation.from_matrix(rot_matrix)
                quat = rotation.as_quat()
                euler = rotation.as_euler('xyz', degrees=True)
                
                # 提取速度
                vel = poses[idx].vVelocity
                vx, vy, vz = vel[0], vel[1], vel[2]
                speed = np.sqrt(vx**2 + vy**2 + vz**2)
                
                # 打印数据
                print()
                print(f"   📏 位置 (米):")
                print(f"      X: {x:+8.4f} m")
                print(f"      Y: {y:+8.4f} m")
                print(f"      Z: {z:+8.4f} m")
                print()
                print(f"   🔄 姿态 (欧拉角):")
                print(f"      Roll:  {euler[0]:+8.2f}°")
                print(f"      Pitch: {euler[1]:+8.2f}°")
                print(f"      Yaw:   {euler[2]:+8.2f}°")
                print()
                print(f"   🎯 姿态 (四元数):")
                print(f"      X: {quat[0]:+7.4f}")
                print(f"      Y: {quat[1]:+7.4f}")
                print(f"      Z: {quat[2]:+7.4f}")
                print(f"      W: {quat[3]:+7.4f}")
                print()
                print(f"   ⚡ 速度:")
                print(f"      VX: {vx:+7.4f} m/s")
                print(f"      VY: {vy:+7.4f} m/s")
                print(f"      VZ: {vz:+7.4f} m/s")
                print(f"      速率: {speed:6.4f} m/s")
                print()
                print(f"   ✅ 状态: 正常跟踪")
                print()
                print_subseparator()
                print()
            
            print()
            print("💡 按 Ctrl+C 停止...")
            print()
            
            # 更新频率：10 Hz
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print()
        print()
        print_separator()
        print("  ⏹  已停止数据采集")
        print_separator()
        print()
    except Exception as e:
        print()
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        openvr.shutdown()
        if tmp_home:
            shutil.rmtree(tmp_home, ignore_errors=True)
            print("✓ 临时文件已清理")
        print("✓ OpenVR 已关闭")
        print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

