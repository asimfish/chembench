#!/usr/bin/env python3
"""
批量转换所有 Grasp 数据集为 DP3 格式

Usage:
    python scripts/batch_convert_grasp_to_dp3.py \
        --input_dir /path/to/grasp/datasets \
        --output_dir /path/to/dp3/output
"""

import argparse
import zarr
from pathlib import Path
from termcolor import cprint
import sys

# 添加 scripts 目录到路径
sys.path.append(str(Path(__file__).parent))
from convert_grasp_to_dp3 import convert_grasp_data


def batch_convert(input_dir: str, output_dir: str, dry_run: bool = False):
    """
    批量转换 Grasp 数据集
    
    Args:
        input_dir: 输入目录（包含多个 .zarr 文件）
        output_dir: 输出目录
        dry_run: 如果为 True，只列出要转换的文件，不实际转换
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        cprint(f'❌ Error: Input directory does not exist: {input_path}', 'red')
        return
    
    # 查找所有 .zarr 文件
    zarr_files = list(input_path.glob('*.zarr'))
    
    if len(zarr_files) == 0:
        cprint(f'❌ Error: No .zarr files found in {input_path}', 'red')
        return
    
    cprint(f'\n{"="*60}', 'cyan')
    cprint(f'Batch Convert Grasp Data to DP3 Format', 'cyan')
    cprint(f'{"="*60}\n', 'cyan')
    
    cprint(f'Input directory: {input_path}', 'yellow')
    cprint(f'Output directory: {output_path}', 'yellow')
    cprint(f'Found {len(zarr_files)} .zarr files\n', 'yellow')
    
    # 列出所有文件
    for i, zarr_file in enumerate(zarr_files, 1):
        print(f'  {i:2d}. {zarr_file.name}')
    
    if dry_run:
        cprint(f'\n[Dry Run] Skipping actual conversion.', 'yellow')
        return
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 批量转换
    success_count = 0
    failed_files = []
    
    for i, zarr_file in enumerate(zarr_files, 1):
        cprint(f'\n[{i}/{len(zarr_files)}] Converting {zarr_file.name}...', 'cyan')
        
        # 生成输出文件名
        output_file = output_path / zarr_file.name
        
        try:
            convert_grasp_data(str(zarr_file), str(output_file))
            success_count += 1
            cprint(f'✅ [{i}/{len(zarr_files)}] Success: {zarr_file.name}', 'green')
        except Exception as e:
            failed_files.append((zarr_file.name, str(e)))
            cprint(f'❌ [{i}/{len(zarr_files)}] Failed: {zarr_file.name}', 'red')
            cprint(f'   Error: {str(e)}', 'red')
    
    # 总结
    cprint(f'\n{"="*60}', 'cyan')
    cprint(f'Batch Conversion Summary', 'cyan')
    cprint(f'{"="*60}\n', 'cyan')
    
    cprint(f'Total files: {len(zarr_files)}', 'yellow')
    cprint(f'✅ Successful: {success_count}', 'green')
    cprint(f'❌ Failed: {len(failed_files)}', 'red')
    
    if failed_files:
        cprint(f'\nFailed files:', 'red')
        for filename, error in failed_files:
            print(f'  - {filename}')
            print(f'    Error: {error}')
    
    if success_count == len(zarr_files):
        cprint(f'\n🎉 All files converted successfully!', 'green')
    
    cprint(f'\nOutput directory: {output_path}', 'yellow')


def main():
    parser = argparse.ArgumentParser(
        description='批量转换 Grasp 数据集为 DP3 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出要转换的文件（dry run）
  python batch_convert_grasp_to_dp3.py \\
      --input_dir data/zarr_point_cloud/motion_plan/grasp \\
      --output_dir data/dp3/grasp \\
      --dry_run
  
  # 实际转换
  python batch_convert_grasp_to_dp3.py \\
      --input_dir data/zarr_point_cloud/motion_plan/grasp \\
      --output_dir data/dp3/grasp
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='输入目录（包含多个 .zarr 文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='只列出要转换的文件，不实际转换'
    )
    
    args = parser.parse_args()
    
    batch_convert(args.input_dir, args.output_dir, args.dry_run)


if __name__ == '__main__':
    main()




