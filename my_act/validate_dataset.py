"""
验证转换后的HDF5数据格式是否正确
"""
import h5py
import os
import sys
import numpy as np


def validate_hdf5_dataset(dataset_dir, num_episodes=None):
    """验证HDF5数据集格式是否符合ACT要求"""
    
    print("=" * 60)
    print("验证HDF5数据集格式")
    print("=" * 60)
    
    # 检查目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"❌ 错误：目录不存在: {dataset_dir}")
        return False
    
    # 查找所有episode文件
    episode_files = sorted([f for f in os.listdir(dataset_dir) 
                           if f.startswith('episode_') and f.endswith('.hdf5')])
    
    if not episode_files:
        print(f"❌ 错误：未找到任何episode文件")
        return False
    
    if num_episodes is None:
        num_episodes = len(episode_files)
    
    print(f"\n✅ 找到 {len(episode_files)} 个episode文件")
    print(f"验证前 {num_episodes} 个文件...\n")
    
    all_valid = True
    action_dims = set()
    state_dims = set()
    
    for i in range(min(num_episodes, len(episode_files))):
        episode_file = episode_files[i]
        filepath = os.path.join(dataset_dir, episode_file)
        
        try:
            with h5py.File(filepath, 'r') as f:
                # 检查必需的字段
                required_fields = [
                    'action',
                    'observations/qpos',
                    'observations/qvel',
                ]
                
                missing_fields = []
                for field in required_fields:
                    if field not in f:
                        missing_fields.append(field)
                
                if missing_fields:
                    print(f"❌ {episode_file}: 缺少字段 {missing_fields}")
                    all_valid = False
                    continue
                
                # 检查数据形状
                action_shape = f['action'].shape
                qpos_shape = f['observations/qpos'].shape
                qvel_shape = f['observations/qvel'].shape
                
                action_dims.add(action_shape[1])
                state_dims.add(qpos_shape[1])
                
                # 检查是否有images
                has_images = 'observations/images' in f
                image_info = ""
                if has_images:
                    image_keys = list(f['observations/images'].keys())
                    image_info = f", 相机: {image_keys}"
                
                # 检查sim属性
                is_sim = f.attrs.get('sim', None)
                
                if i < 3 or i == num_episodes - 1:  # 显示前3个和最后一个
                    print(f"✅ {episode_file}:")
                    print(f"   - action: {action_shape}")
                    print(f"   - qpos: {qpos_shape}")
                    print(f"   - qvel: {qvel_shape}")
                    print(f"   - sim: {is_sim}{image_info}")
                
        except Exception as e:
            print(f"❌ {episode_file}: 读取错误 - {e}")
            all_valid = False
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    if all_valid:
        print("✅ 所有文件格式正确！")
    else:
        print("❌ 部分文件存在问题")
    
    if len(action_dims) == 1 and len(state_dims) == 1:
        action_dim = list(action_dims)[0]
        state_dim = list(state_dims)[0]
        print(f"\n📊 数据维度:")
        print(f"   - 动作维度: {action_dim}")
        print(f"   - 状态维度: {state_dim}")
        
        if action_dim == 14 and state_dim == 14:
            print("   ℹ️  标准双臂配置（14-DOF）")
        else:
            print(f"   ⚠️  非标准配置，需要修补模型")
            print(f"   💡 训练脚本会自动处理")
    else:
        print(f"\n⚠️  警告：不同episode的维度不一致")
        print(f"   - 动作维度: {action_dims}")
        print(f"   - 状态维度: {state_dims}")
    
    print("\n" + "=" * 60)
    
    return all_valid


def main():
    import argparse
    parser = argparse.ArgumentParser(description='验证HDF5数据集格式')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='HDF5数据集目录')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='验证的episode数量（默认：全部）')
    
    args = parser.parse_args()
    
    success = validate_hdf5_dataset(args.dataset_dir, args.num_episodes)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

