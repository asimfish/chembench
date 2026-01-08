# PSILab Diffusion Policy 快速启动指南

## 🚀 5分钟快速上手

### 第1步: 准备数据集

确保你的数据集在正确的位置：
```bash
# 默认路径
/share_data/liyufeng/code/chembench/data/psilab/demo.zarr

# 或者修改配置文件中的路径
# 编辑 psi_dp/config/task/psilab.yaml
# 修改 dataset_path 参数
```

### 第2步: 验证配置

```bash
# 运行验证脚本
bash validate_psilab.sh
```

如果验证失败，根据提示修复问题。

### 第3步: 激活环境

```bash
conda activate psilab
```

### 第4步: 开始训练

**前台运行** (推荐首次使用):
```bash
bash run_train_psilab.sh
```

**后台运行** (推荐长时间训练):
```bash
bash run_train_psilab.sh bg
```

就这么简单！🎉

---

## 📋 常见任务

### 切换观测模式

**使用RGB模式** (3通道):
```bash
bash configure_psilab.sh rgb
bash run_train_psilab.sh
```

**使用RGB+Mask模式** (4通道):
```bash
bash configure_psilab.sh rgbm
bash run_train_psilab.sh
```

**使用Normal+Depth模式** (4通道):
```bash
bash configure_psilab.sh nd
bash run_train_psilab.sh
```

**使用RGB+Normal+Depth模式** (7通道):
```bash
bash configure_psilab.sh rgbnd
bash run_train_psilab.sh
```

### 修改数据集路径

```bash
# 方法1: 直接编辑配置文件
nano psi_dp/config/task/psilab.yaml

# 修改这一行:
# dataset_path: "/your/new/path/demo.zarr"
```

```bash
# 方法2: 使用命令行覆盖（无需修改文件）
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name train_diffusion_transformer_psilab_workspace \
    task=psilab \
    task.dataset_path="/your/new/path/demo.zarr"
```

### 修改GPU设备

```bash
# 方法1: 编辑脚本
nano run_train_psilab.sh
# 修改: GPU_ID=0  改为  GPU_ID=1

# 方法2: 直接指定
CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-name train_diffusion_transformer_psilab_workspace \
    task=psilab
```

### 调整批次大小

```bash
# 编辑训练配置
nano psi_dp/config/train_diffusion_transformer_psilab_workspace.yaml

# 修改这一行:
# dataloader:
#   batch_size: 64  # 改为 32 或 128

# 或使用命令行覆盖
python train.py \
    --config-name train_diffusion_transformer_psilab_workspace \
    task=psilab \
    dataloader.batch_size=32
```

### 从checkpoint恢复训练

```bash
# 编辑训练配置
nano psi_dp/config/train_diffusion_transformer_psilab_workspace.yaml

# 修改这两行:
# training:
#   resume: True
#   lastest_ckpt_path: "data/outputs/psilab/.../checkpoints/latest.ckpt"

# 或使用命令行
python train.py \
    --config-name train_diffusion_transformer_psilab_workspace \
    task=psilab \
    training.resume=True \
    training.lastest_ckpt_path="path/to/checkpoint.ckpt"
```

### 查看训练日志

**实时查看后台日志**:
```bash
# 找到最新的日志文件
ls -lt logs/train_psilab_*/gpu0.log | head -1

# 实时查看
tail -f logs/train_psilab_TIMESTAMP/gpu0.log
```

**监控GPU使用**:
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

**查看WandB**:
```bash
# 在浏览器中打开
# https://wandb.ai/your-username/DP_PSILab_psilab
```

### 停止训练

**前台训练**:
```
按 Ctrl+C
```

**后台训练**:
```bash
# 找到进程ID
cat logs/train_psilab_TIMESTAMP/pid.txt

# 停止进程
kill <PID>

# 或者强制停止
kill -9 <PID>
```

---

## 🔧 高级配置

### 启用速度观测

```bash
# 1. 编辑任务配置
nano psi_dp/config/task/psilab.yaml

# 2. 修改 obs_config
# obs_config:
#   use_velocity: true

# 3. 取消注释 shape_meta 中的速度观测部分
# arm2_vel:
#   shape: [7]
#   type: low_dim 
#   horizon: ${n_obs_steps}
# hand2_vel:
#   shape: [6]
#   type: low_dim
#   horizon: ${n_obs_steps}

# 4. 更新 obs_keys
# obs_keys: ["chest_camera_rgb", "head_camera_rgb", "arm2_pos", "arm2_vel", "hand2_pos", "hand2_vel", ...]
```

### 启用第三人称相机

```bash
# 1. 编辑任务配置
nano psi_dp/config/task/psilab.yaml

# 2. 修改 obs_config
# obs_config:
#   use_third_camera: true

# 3. 取消注释 shape_meta 中的第三人称相机
# third_camera_rgb:
#   shape: ${task.image_shape}
#   type: ${task.obs_config.obs_mode}
#   horizon: ${n_obs_steps}

# 4. 更新 obs_keys
# obs_keys: ["chest_camera_rgb", "head_camera_rgb", "third_camera_rgb", ...]
```

### 修改模型架构

```bash
# 编辑训练配置
nano psi_dp/config/train_diffusion_transformer_psilab_workspace.yaml

# 修改这些参数:
# n_emb: 768        # 嵌入维度: 512, 768, 1024
# n_layer: 7        # Transformer层数: 4, 7, 12
# n_head: 8         # 注意力头数: 4, 8, 16
# n_action_steps: 8 # 动作步数: 4, 8, 16
```

### 使用不同的视觉编码器

```bash
# 编辑训练配置
nano psi_dp/config/train_diffusion_transformer_psilab_workspace.yaml

# 修改 model_name:
# policy:
#   obs_encoder:
#     model_name: 'vit_small_r26_s32_224'  # ViT Small (当前)
#     # 其他选项:
#     # 'vit_base_r26_s32_224'              # ViT Base (更大)
#     # 'resnet50'                          # ResNet50
#     # 'efficientnet_b3'                   # EfficientNet B3
```

### 冻结视觉编码器

```bash
# 编辑训练配置
nano psi_dp/config/train_diffusion_transformer_psilab_workspace.yaml

# 修改这两个参数:
# policy:
#   obs_encoder:
#     frozen: True    # 冻结编码器权重
#
# training:
#   freeze_encoder: True
```

---

## 📊 实验管理

### 组织多个实验

推荐的实验命名规范：

```bash
# 修改 object_name
nano psi_dp/config/task/psilab.yaml

# 示例命名:
# object_name: "exp001_rgb_baseline"
# object_name: "exp002_rgbm_with_mask"
# object_name: "exp003_nd_normal_depth"
# object_name: "exp004_rgbnd_full"
```

输出会自动组织到：
```
data/outputs/psilab/
├── exp001_rgb_baseline/
│   └── 20251231_100000_n50_rgb/
├── exp002_rgbm_with_mask/
│   └── 20251231_110000_n50_rgbm/
├── exp003_nd_normal_depth/
│   └── 20251231_120000_n50_nd/
└── exp004_rgbnd_full/
    └── 20251231_130000_n50_rgbnd/
```

### 批量运行实验

```bash
# 创建批量实验脚本
cat > run_experiments.sh << 'EOF'
#!/bin/bash
set -e

# 实验1: RGB模式
bash configure_psilab.sh rgb
sed -i 's/object_name: ".*"/object_name: "exp001_rgb_baseline"/' psi_dp/config/task/psilab.yaml
bash run_train_psilab.sh bg
sleep 5

# 实验2: RGBM模式
bash configure_psilab.sh rgbm
sed -i 's/object_name: ".*"/object_name: "exp002_rgbm_with_mask"/' psi_dp/config/task/psilab.yaml
bash run_train_psilab.sh bg
sleep 5

# 实验3: ND模式
bash configure_psilab.sh nd
sed -i 's/object_name: ".*"/object_name: "exp003_nd_normal_depth"/' psi_dp/config/task/psilab.yaml
bash run_train_psilab.sh bg
sleep 5

echo "所有实验已启动！"
EOF

chmod +x run_experiments.sh
./run_experiments.sh
```

---

## 🐛 常见问题

### Q: 数据集路径错误
```
FileNotFoundError: demo.zarr
```
**A**: 运行 `bash validate_psilab.sh` 检查路径，或修改 `psi_dp/config/task/psilab.yaml` 中的 `dataset_path`。

### Q: 通道数不匹配
```
RuntimeError: Expected 3 channels, got 4
```
**A**: 运行 `bash configure_psilab.sh <obs_mode>` 自动修复配置。

### Q: GPU显存不足
```
CUDA out of memory
```
**A**: 减小批次大小：
```bash
nano psi_dp/config/train_diffusion_transformer_psilab_workspace.yaml
# batch_size: 64 改为 32 或 16
```

### Q: WandB登录失败
```
wandb: ERROR API key not found
```
**A**: 
```bash
wandb login
# 输入你的API key (从 https://wandb.ai/authorize 获取)
```

### Q: Conda环境不存在
```
CondaEnvironmentNotFoundError: psilab
```
**A**: 
```bash
bash install_on_psilab.sh
```

### Q: 训练很慢
**A**: 
1. 检查是否在GPU上运行：`nvidia-smi`
2. 增加 `num_workers`：`dataloader.num_workers: 16`
3. 启用 `persistent_workers: True`

### Q: 如何选择最佳checkpoint？
**A**: 
- 自动保存top-5最佳模型（按 `train_loss` 排序）
- 查看 `data/outputs/.../checkpoints/` 目录
- 文件名包含 `train_loss` 值，选择最小的

---

## 📚 更多资源

- **详细文档**: [README_PSILAB.md](README_PSILAB.md)
- **配置验证**: `bash validate_psilab.sh`
- **快速配置**: `bash configure_psilab.sh <mode>`
- **官方文档**: https://diffusion-policy.cs.columbia.edu/

---

## 💡 最佳实践

1. **首次运行**: 先用小批次、少epoch测试配置是否正确
2. **数据检查**: 确保数据集格式正确，包含所需的观测和动作
3. **实验记录**: 使用有意义的 `object_name` 标识不同实验
4. **定期备份**: checkpoint会自动保存，但建议定期备份重要模型
5. **监控训练**: 使用WandB实时监控loss和metrics
6. **资源管理**: 长时间训练使用后台模式，并设置合理的checkpoint频率

---

**祝训练顺利！** 🚀

