# 从Zarr数据集训练ACT模型

本指南说明如何将您的Zarr数据集转换为HDF5格式并训练ACT策略。

## 快速开始

### 方式1：使用Shell脚本（推荐）

```bash
# 编辑 train_grasp.sh 设置您的路径
# 然后运行：
bash train_grasp.sh
```

### 方式2：手动命令

```bash
python3 train_from_zarr.py \
    --zarr_path /share_data/liyufeng/code/chembench/data/final_real/data/grasp/part1/100ml玻璃烧杯.zarr \
    --dataset_dir ./data/grasp_100ml_beaker \
    --ckpt_dir ./ckpts/grasp_100ml_beaker \
    --camera_names head_camera chest_camera \
    --batch_size 8 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200
```

## 脚本说明

### 主要文件

1. **convert_zarr_to_hdf5.py** - 将Zarr转换为HDF5格式
2. **train_from_zarr.py** - 完整的训练流程（转换+训练）
3. **train_grasp.sh** - 预配置的训练脚本
4. **patch_detr_for_custom_dims.py** - 自动修改模型以支持您的数据维度
5. **TRAINING_GUIDE.md** - 详细的英文指南

### 您的数据集信息

- **路径**: `/share_data/liyufeng/code/chembench/data/final_real/data/grasp/part1/100ml玻璃烧杯.zarr`
- **回合数**: 50个回合
- **总时间步**: 2622步
- **动作维度**: 13
- **状态维度**: 7（机械臂关节位置）
- **可用相机**: head_camera, chest_camera, third_camera
- **图像尺寸**: 224×224×3

## 重要说明

### 关于维度适配

原始ACT实现是为双臂机器人设计的（state_dim=14，即2个手臂×7自由度）。
您的数据集维度不同（state_dim=7, action_dim=13）。

**训练脚本会自动修改模型代码**以支持您的维度。备份文件会自动保存为 `detr/models/detr_vae.py.backup`。

如需恢复原始代码：
```bash
python3 patch_detr_for_custom_dims.py restore
```

## 训练流程

脚本会自动执行以下步骤：

### 步骤1：数据转换
- 读取Zarr格式数据
- 按回合分割
- 转换为HDF5格式（ACT需要）
- 保存到 `--dataset_dir` 指定的目录

### 步骤2：模型适配
- 检测数据维度
- 如需要，自动修改模型代码

### 步骤3：数据加载
- 加载HDF5数据
- 计算归一化统计量
- 创建训练/验证集（8:2分割）

### 步骤4：训练
- 初始化ACT模型
- 训练指定轮数
- 每100轮保存检查点
- 保存最佳模型

## 参数说明

### 数据参数
- `--zarr_path`: Zarr数据集路径
- `--dataset_dir`: HDF5文件保存/加载目录
- `--skip_conversion`: 如果HDF5文件已存在，跳过转换
- `--camera_names`: 使用的相机列表（如：head_camera chest_camera）

### 训练参数
- `--ckpt_dir`: 模型检查点保存目录
- `--batch_size`: 批大小（默认：8）
- `--num_epochs`: 训练轮数（默认：2000）
- `--lr`: 学习率（默认：1e-5）
- `--seed`: 随机种子（默认：0）

### ACT模型参数
- `--kl_weight`: KL散度权重（默认：10）
- `--chunk_size`: 动作块大小（默认：100）
- `--hidden_dim`: 隐藏层维度（默认：512）
- `--dim_feedforward`: 前馈层维度（默认：3200）

## 训练建议

1. **初次训练**：使用默认参数训练2000轮
2. **真实机器人数据**：至少训练5000轮，或在损失平稳后继续训练3-4倍时间
3. **策略抖动**：如果策略执行抖动或暂停，需要训练更长时间
4. **相机选择**：使用2-3个相机效果最好，更多相机需要更多显存
5. **显存不足**：降低batch_size（尝试4或2）

## 监控训练

脚本会：
- 每轮打印训练和验证损失
- 每100轮保存检查点
- 根据验证损失保存最佳检查点
- 生成训练曲线图

## 输出文件

训练后会生成：
```
ckpts/grasp_100ml_beaker/
├── policy_best.ckpt                    # 最佳模型（用于评估）
├── policy_last.ckpt                    # 最后一轮模型
├── policy_epoch_X_seed_0.ckpt          # 周期性检查点
├── dataset_stats.pkl                   # 归一化统计量
└── train_val_loss_seed_0.png           # 训练曲线
```

## 使用训练好的模型

```python
import torch
import pickle
from policy import ACTPolicy

# 加载模型
ckpt_path = './ckpts/grasp_100ml_beaker/policy_best.ckpt'
stats_path = './ckpts/grasp_100ml_beaker/dataset_stats.pkl'

# 加载统计量
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

# 创建并加载策略
policy_config = {
    'lr': 1e-5,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['head_camera', 'chest_camera'],
    'state_dim': 7,      # 您的状态维度
    'action_dim': 13,    # 您的动作维度
}

policy = ACTPolicy(policy_config)
policy.load_state_dict(torch.load(ckpt_path))
policy.cuda()
policy.eval()

# 使用策略进行推理
# qpos: 当前关节位置（已预处理）
# image: 当前相机观测（已预处理）
# actions = policy(qpos, image)
```

## 常见问题

### 显存不足
- 降低 `--batch_size`（尝试4或2）
- 使用更少的相机
- 降低图像分辨率（需修改转换脚本）

### 损失不下降
- 训练更多轮（尝试5000+）
- 检查数据质量
- 确保相机能捕获相关信息

### 转换失败
- 检查Zarr数据集结构是否匹配预期格式
- 验证所有相机名称在数据集中存在
- 确保episode_ends元数据正确

### 模型维度错误
- 训练脚本应该自动处理维度适配
- 如果失败，手动运行：`python3 patch_detr_for_custom_dims.py`
- 或手动编辑 `detr/models/detr_vae.py`

## 数据格式

### 输入：Zarr格式
```
dataset.zarr/
├── meta/
│   └── episode_ends          # 每个回合的结束索引
└── data/
    ├── action                # 形状: (总时间步, 动作维度)
    ├── arm2_pos              # 形状: (总时间步, 7)
    ├── arm2_vel              # 形状: (总时间步, 7)
    ├── head_camera_rgb       # 形状: (总时间步, H, W, 3)
    ├── chest_camera_rgb      # 形状: (总时间步, H, W, 3)
    └── third_camera_rgb      # 形状: (总时间步, H, W, 3)
```

### 输出：HDF5格式
```
episode_0.hdf5
├── action                    # 形状: (回合长度, 动作维度)
└── observations/
    ├── qpos                  # 形状: (回合长度, 7)
    ├── qvel                  # 形状: (回合长度, 7)
    └── images/
        ├── head_camera       # 形状: (回合长度, H, W, 3)
        └── chest_camera      # 形状: (回合长度, H, W, 3)
```

## 参考资料

- ACT论文: https://arxiv.org/abs/2304.13705
- ACT调优技巧: https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit
- 原始ACT代码库: https://github.com/tonyzhaozh/act


