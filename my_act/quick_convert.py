#!/usr/bin/env python3
"""
快速转换命令生成器
Quick command generator for Zarr to HDF5 conversion
"""

import sys
import os

def print_banner():
    print("=" * 70)
    print("  Zarr → HDF5 转换工具 - 快速命令生成器")
    print("  Zarr to HDF5 Conversion Tool - Quick Command Generator")
    print("=" * 70)
    print()

def generate_commands():
    print_banner()
    
    print("📋 常用命令模板 / Common Command Templates")
    print("-" * 70)
    print()
    
    # Command 1: Basic conversion
    print("1️⃣  基本转换 (单个Zarr文件)")
    print("   Basic Conversion (Single Zarr File)")
    print()
    print("   python convert_zarr_to_hdf5.py \\")
    print("     --zarr_path \"data/zarr/YOUR_FILE.zarr\" \\")
    print("     --output_dir \"dataset/YOUR_OUTPUT\" \\")
    print("     --camera_names head_camera chest_camera third_camera \\")
    print("     --duplicate_arms")
    print()
    
    # Command 2: Verify
    print("2️⃣  验证转换结果")
    print("   Verify Conversion Results")
    print()
    print("   python verify_converted_data.py \\")
    print("     --dataset_dir \"dataset/YOUR_OUTPUT\" \\")
    print("     --verbose")
    print()
    
    # Command 3: Visualize
    print("3️⃣  可视化数据")
    print("   Visualize Data")
    print()
    print("   python inspect_episode.py \\")
    print("     dataset/YOUR_OUTPUT/episode_0.hdf5 \\")
    print("     --frame 0")
    print()
    
    # Command 4: Batch conversion
    print("4️⃣  批量转换 (整个目录)")
    print("   Batch Conversion (Entire Directory)")
    print()
    print("   ./batch_convert_zarr.sh \\")
    print("     --zarr_dir \"data/zarr/YOUR_DIR\" \\")
    print("     --output_base \"dataset\" \\")
    print("     --camera_names \"head_camera chest_camera third_camera\"")
    print()
    
    print("=" * 70)
    print("💡 提示 / Tips:")
    print("   - 使用 --duplicate_arms 将7维单臂数据转为14维双臂格式")
    print("     Use --duplicate_arms to convert 7-dim single-arm to 14-dim dual-arm")
    print()
    print("   - 转换后务必运行验证脚本检查数据完整性")
    print("     Always run verification script after conversion")
    print()
    print("   - 可用 inspect_episode.py 可视化检查转换结果")
    print("     Use inspect_episode.py to visually check conversion results")
    print("=" * 70)
    print()

def interactive_generator():
    """交互式命令生成器"""
    print_banner()
    print("🔧 交互式命令生成 / Interactive Command Generation")
    print("-" * 70)
    print()
    
    try:
        # Get zarr path
        zarr_path = input("Zarr文件路径 (Zarr file path): ").strip()
        if not zarr_path:
            print("❌ 路径不能为空 / Path cannot be empty")
            return
        
        # Get output dir
        zarr_basename = os.path.basename(zarr_path.rstrip('/')).replace('.zarr', '')
        default_output = f"dataset/grasp_{zarr_basename}"
        output_dir = input(f"输出目录 (Output dir) [{default_output}]: ").strip()
        if not output_dir:
            output_dir = default_output
        
        # Get camera names
        default_cameras = "head_camera chest_camera third_camera"
        cameras = input(f"相机名称 (Camera names) [{default_cameras}]: ").strip()
        if not cameras:
            cameras = default_cameras
        
        # Duplicate arms
        duplicate = input("是否转换为双臂格式? (Duplicate to dual-arm?) [Y/n]: ").strip().lower()
        duplicate_flag = "" if duplicate == 'n' else "--duplicate_arms"
        
        print()
        print("=" * 70)
        print("✨ 生成的命令 / Generated Commands")
        print("=" * 70)
        print()
        
        # Conversion command
        print("# 1. 转换命令 / Conversion Command")
        print(f"python convert_zarr_to_hdf5.py \\")
        print(f"  --zarr_path \"{zarr_path}\" \\")
        print(f"  --output_dir \"{output_dir}\" \\")
        print(f"  --camera_names {cameras}", end="")
        if duplicate_flag:
            print(f" \\")
            print(f"  {duplicate_flag}")
        else:
            print()
        print()
        
        # Verification command
        print("# 2. 验证命令 / Verification Command")
        print(f"python verify_converted_data.py \\")
        print(f"  --dataset_dir \"{output_dir}\" \\")
        print(f"  --verbose")
        print()
        
        # Visualization command
        print("# 3. 可视化命令 / Visualization Command")
        print(f"python inspect_episode.py \\")
        print(f"  {output_dir}/episode_0.hdf5 \\")
        print(f"  --frame 0")
        print()
        
        print("=" * 70)
        
        # Ask to run
        run_now = input("\n是否立即运行转换? (Run conversion now?) [Y/n]: ").strip().lower()
        if run_now != 'n':
            cmd = f"python convert_zarr_to_hdf5.py --zarr_path \"{zarr_path}\" --output_dir \"{output_dir}\" --camera_names {cameras}"
            if duplicate_flag:
                cmd += f" {duplicate_flag}"
            print(f"\n🚀 Running: {cmd}\n")
            os.system(cmd)
            
            # Ask to verify
            verify_now = input("\n是否运行验证? (Run verification?) [Y/n]: ").strip().lower()
            if verify_now != 'n':
                verify_cmd = f"python verify_converted_data.py --dataset_dir \"{output_dir}\""
                print(f"\n🔍 Running: {verify_cmd}\n")
                os.system(verify_cmd)
    
    except KeyboardInterrupt:
        print("\n\n❌ 取消 / Cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误 / Error: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_generator()
    else:
        generate_commands()
        print()
        print("💡 运行交互式生成器 / Run interactive generator:")
        print("   python quick_convert.py --interactive")
        print()

if __name__ == '__main__':
    main()


