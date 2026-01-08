# Zarr到HDF5数据转换工具集

这个工具集用于将Zarr格式的机器人演示数据转换为ACT (Action Chunking Transformer)训练所需的HDF5格式。

## 📁 文件说明

### 核心脚本

1. **`convert_zarr_to_hdf5.py`** - 主转换脚本
   - 将单个Zarr数据集转换为多个HDF5 episode文件
   - 支持单臂到双臂格式的自动转换
   - 支持多相机视角

2. **`verify_converted_data.py`** - 数据验证脚本
   - 验证转换后的HDF5文件结构
   - 检查数据完整性和类型
   - 生成统计报告

3. **`inspect_episode.py`** - 数据可视化脚本
   - 可视化单个episode的图像和轨迹
   - 支持episode对比
   - 生成可视化图表

4. **`batch_convert_zarr.sh`** - 批量转换脚本
   - 批量处理目录中的所有Zarr文件
   - 自动验证转换结果
   - 生成汇总报告

### 文档

- **`CONVERSION_GUIDE.md`** - 详细转换指南
- **`README_CONVERSION.md`** - 本文件，快速入门指南

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install h5py zarr numpy tqdm matplotlib
```

### 2. 转换单个数据集

```bash
python convert_zarr_to_hdf5.py \
  --zarr_path "data/zarr/data/grasp/part1/100ml玻璃烧杯.zarr" \
  --output_dir "dataset/grasp_100ml_beaker" \
  --camera_names head_camera chest_camera third_camera \
  --duplicate_arms
```

### 3. 验证转换结果

```bash
python verify_converted_data.py \
  --dataset_dir "dataset/grasp_100ml_beaker" \
  --verbose
```

### 4. 可视化数据

```bash
python inspect_episode.py \
  dataset/grasp_100ml_beaker/episode_0.hdf5 \
  --frame 0
```

## 📊 转换示例结果

```
Opening Zarr dataset: data/zarr/data/grasp/part1/100ml玻璃烧杯.zarr
Found 50 episodes

Dataset information:
  Action shape: (2622, 13)
  Arm position shape: (2622, 7)
  Arm velocity shape: (2622, 7)
  head_camera RGB shape: (2622, 224, 224, 3)
  chest_camera RGB shape: (2622, 224, 224, 3)
  third_camera RGB shape: (2622, 224, 224, 3)

Converting episodes: 100%|██████████| 50/50 [00:05<00:00, 8.50it/s]

Conversion complete! 50 episodes saved to dataset/grasp_100ml_beaker

Dataset statistics:
  Action dimension: 14
  State dimension (qpos): 14
  Episode length: 53 timesteps
  Camera names: ['head_camera', 'chest_camera', 'third_camera']
  Image shape: (224, 224, 3)

✓ All episodes are valid!
```

## 🔧 高级用法

### 批量转换整个目录

```bash
./batch_convert_zarr.sh \
  --zarr_dir "data/zarr/data/grasp/part1" \
  --output_base "dataset" \
  --camera_names "head_camera chest_camera third_camera"
```

### 对比两个episodes

```bash
python inspect_episode.py \
  dataset/grasp_100ml_beaker/episode_0.hdf5 \
  --compare dataset/sim_transfer_cube_scripted/episode_0.hdf5
```

### 保持单臂格式（不转换为双臂）

```bash
python convert_zarr_to_hdf5.py \
  --zarr_path "data/zarr/data/grasp/part1/100ml玻璃烧杯.zarr" \
  --output_dir "dataset/grasp_100ml_beaker_single_arm" \
  --camera_names head_camera chest_camera third_camera
  # 不添加 --duplicate_arms 标志
```

## 📋 数据格式对比

### 输入 (Zarr)
```
data/zarr/data/grasp/part1/100ml玻璃烧杯.zarr/
├── data/
│   ├── action (2622, 13)
│   ├── arm2_pos (2622, 7)
│   ├── arm2_vel (2622, 7)
│   └── [camera]_rgb (2622, 224, 224, 3)
└── meta/
    └── episode_ends [50个episode的结束索引]
```

### 输出 (HDF5)
```
dataset/grasp_100ml_beaker/
├── episode_0.hdf5
│   ├── action (53, 14)
│   └── observations/
│       ├── qpos (53, 14)
│       ├── qvel (53, 14)
│       └── images/
│           ├── head_camera (53, 224, 224, 3)
│           ├── chest_camera (53, 224, 224, 3)
│           └── third_camera (53, 224, 224, 3)
├── episode_1.hdf5
├── ...
└── episode_49.hdf5
```

## ✅ 验证清单

转换后请确认：

- [x] 所有episodes都显示"✓ Valid"
- [x] Action维度正确 (14维双臂 或 13维单臂)
- [x] State维度正确 (14维双臂 或 7维单臂)
- [x] 图像数据类型为uint8
- [x] 状态/动作数据类型为float32
- [x] 相机名称正确
- [x] Episode数量正确
- [x] 无NaN或Inf值

## 🐛 常见问题

### Q: 转换后维度不匹配怎么办？

**A**: 使用`--duplicate_arms`标志将7维单臂数据转换为14维双臂格式。

### Q: 某些相机数据缺失？

**A**: 使用`--camera_names`参数指定实际存在的相机。检查Zarr文件中的相机名称：
```bash
python -c "import zarr; z=zarr.open('your.zarr', 'r'); print(list(z['data'].keys()))"
```

### Q: 内存不足？

**A**: 转换脚本使用chunking和流式处理，应该不会有内存问题。如果仍然有问题，可以分批处理episodes。

### Q: 图像尺寸需要统一吗？

**A**: 不同数据集可以有不同的图像尺寸，但同一数据集内的所有episodes必须使用相同的图像尺寸。

## 📈 性能优化

- **HDF5 Chunking**: 图像使用`(1, H, W, 3)`的chunking策略，优化随机访问
- **数据类型**: 使用float32而非float64，减少50%存储空间
- **压缩**: 可以在HDF5创建时添加compression参数（需要修改脚本）

## 🔗 相关工具

使用转换后的数据：

1. **训练ACT模型**:
   ```bash
   python imitate_episodes.py \
     --task_name grasp_100ml_beaker \
     --ckpt_dir ckpts/grasp \
     --policy_class ACT \
     --batch_size 8 \
     --num_epochs 2000
   ```

2. **可视化episodes**:
   ```bash
   python visualize_episodes.py \
     --dataset_dir dataset/grasp_100ml_beaker \
     --episode_idx 0
   ```

3. **验证数据加载**:
   ```bash
   python validate_dataset.py \
     --dataset_dir dataset/grasp_100ml_beaker
   ```

## 📝 注意事项

1. **备份原始数据**: 转换前请确保Zarr原始数据有备份
2. **磁盘空间**: HDF5文件通常比Zarr稍大，确保有足够空间
3. **数据一致性**: 确保所有episodes来自同一任务/配置
4. **相机标定**: 如果使用多相机，确保相机标定数据一致

## 🤝 贡献

如果发现问题或有改进建议，请：
1. 检查`CONVERSION_GUIDE.md`中的详细文档
2. 运行验证脚本确认问题
3. 提供详细的错误信息和数据格式

## 📚 更多信息

- 详细转换指南: `CONVERSION_GUIDE.md`
- ACT项目主页: [mobile-aloha](https://github.com/MarkFzp/mobile-aloha)
- HDF5文档: [h5py.org](https://docs.h5py.org/)
- Zarr文档: [zarr.readthedocs.io](https://zarr.readthedocs.io/)

---

**版本**: 1.0  
**更新日期**: 2026-01-06  
**兼容性**: Python 3.7+, h5py 3.0+, zarr 2.10+


