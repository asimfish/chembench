# 📦 Isaac Sim 物体 BBox 获取与存储 - 完整方案

本文件夹包含在 Isaac Sim 中获取和存储物体 Bounding Box 的完整工具和文档。

---

## 🎯 功能特性

✅ **3D BBox**: 从 USD Prim 提取世界坐标系边界框  
✅ **2D BBox**: 从分割 Mask 自动计算屏幕空间边界框  
✅ **多相机支持**: 自动处理头部、胸部、第三相机  
✅ **HDF5 存储**: 集成到现有数据采集流程  
✅ **Zarr 转换**: 自动转换到训练数据格式  
✅ **可视化工具**: 验证 BBox 准确性  
✅ **单元测试**: 确保功能正常  

---

## 📁 文件结构

```
chembench/
├── psilab/source/psilab/psilab/utils/
│   └── bbox_utils.py                    # 🔧 核心工具库
├── docs/
│   ├── BBOX_USAGE_GUIDE.md              # 📖 完整使用指南
│   ├── BBOX_QUICK_REF.md                # ⚡ 快速参考卡片
│   ├── bbox_integration_example.py      # 💻 集成示例代码
│   └── BBOX_README.md                   # 📋 本文件
└── test_bbox_extraction.py              # 🧪 单元测试
```

---

## 🚀 快速开始（5分钟）

### Step 1: 查看快速参考

```bash
cat docs/BBOX_QUICK_REF.md
```

### Step 2: 运行测试

```bash
python test_bbox_extraction.py
```

### Step 3: 集成到你的任务

参考 `docs/bbox_integration_example.py`

---

## 📚 文档说明

### 1. 📖 [BBOX_USAGE_GUIDE.md](./BBOX_USAGE_GUIDE.md)

**适合人群**: 首次使用或需要深入了解

**内容**:
- 功能概述
- BBox 类型详解（3D/2D）
- 快速开始示例
- 完整集成步骤
  - 在 `grasp_mp.py` 中添加
  - 在 `zarr_utils.py` 中转换
  - 在 `analyze_zarr.py` 中可视化
- 常见问题与解决方案
- 性能优化建议

**预计阅读时间**: 15-20 分钟

---

### 2. ⚡ [BBOX_QUICK_REF.md](./BBOX_QUICK_REF.md)

**适合人群**: 已熟悉功能，需要快速查询

**内容**:
- 3步快速上手
- 数据格式速查
- 代码片段速查
- 实用技巧
- 常见陷阱

**预计阅读时间**: 2-3 分钟

---

### 3. 💻 [bbox_integration_example.py](./bbox_integration_example.py)

**适合人群**: 开发者，需要具体实现代码

**内容**:
- `grasp_mp.py` 完整集成代码
- `zarr_utils.py` 转换代码
- `analyze_zarr.py` 可视化代码
- 带注释的完整示例

**用法**: 复制粘贴到对应文件

---

## 🔧 核心 API

### BBoxExtractor 类

```python
from psilab.utils.bbox_utils import BBoxExtractor

# 初始化
extractor = BBoxExtractor(device="cuda:0")

# 方法 1: 获取 3D BBox（需要 Isaac Sim 环境）
bbox_3d = BBoxExtractor.get_3d_bbox_from_prim("/World/envs/env_0/bottle")
# 返回: {'center': [x,y,z], 'extent': [w,h,d], 'min': [...], 'max': [...], 'corners': [...]}

# 方法 2: 从 Mask 获取 2D BBox（推荐！）
bbox_2d = BBoxExtractor.get_bbox_from_mask(mask)
# 返回: {'x_min': ..., 'y_min': ..., 'x_max': ..., 'y_max': ..., 'width': ..., 'height': ..., 'center': [x,y]}

# 方法 3: 从 3D 投影到 2D（高级用法）
bbox_2d = BBoxExtractor.get_2d_bbox_from_3d(bbox_3d, view_mat, proj_mat, 640, 480)
```

### 辅助函数

```python
from psilab.utils.bbox_utils import add_bbox_to_h5, convert_bbox_to_zarr_format

# 保存到 HDF5
add_bbox_to_h5(h5_group, bbox_3d, "target_bbox_3d")

# 转换为 Zarr 格式
bbox_array = convert_bbox_to_zarr_format(bbox_list, num_frames)
```

---

## 🧪 测试

### 运行测试套件

```bash
python test_bbox_extraction.py
```

### 测试覆盖

| 测试项 | 说明 | 状态 |
|--------|------|------|
| Mask → 2D BBox | 从分割掩码提取边界框 | ✅ |
| 空 Mask 处理 | 无物体时返回 None | ✅ |
| 3D → 2D 投影 | 投影矩阵计算 | ✅ |
| 数据格式转换 | 转换为 Zarr 格式 | ✅ |
| 3D BBox 结构 | 验证数据结构完整性 | ✅ |

---

## 📊 数据流程图

```
┌─────────────────┐
│  Isaac Sim      │
│  运行采集任务    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  grasp_mp.py    │
│  ├─ 初始化 BBoxExtractor
│  ├─ _record_data(): 记录每帧 BBox
│  └─ _write_data_to_file(): 写入 HDF5
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HDF5 文件      │
│  ├─ rigid_objects/target_bbox_3d (N, 7)
│  ├─ cameras/head_camera_bbox_2d (N, 6)
│  ├─ cameras/chest_camera_bbox_2d (N, 6)
│  └─ cameras/third_camera_bbox_2d (N, 6)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  zarr_utils.py  │
│  convert_rgb_based(): 转换到 Zarr
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Zarr 文件      │
│  ├─ data/target_bbox_3d
│  ├─ data/target_bbox_center
│  ├─ data/target_bbox_extent
│  ├─ data/head_camera_bbox_2d
│  └─ ...
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  训练脚本        │
│  使用 BBox 数据  │
└─────────────────┘
```

---

## 💡 使用场景

### 场景 1: 视觉-语言模型训练

```python
# 需要: RGB + 2D BBox
bbox_2d = BBoxExtractor.get_bbox_from_mask(mask)
dataset_entry = {
    'image': rgb_frame,
    'bbox': [bbox_2d['x_min'], bbox_2d['y_min'], 
             bbox_2d['x_max'], bbox_2d['y_max']],
    'label': 'bottle'
}
```

### 场景 2: 3D 物体定位

```python
# 需要: 3D BBox
bbox_3d = BBoxExtractor.get_3d_bbox_from_prim(prim_path)
object_pose = {
    'position': bbox_3d['center'],
    'size': bbox_3d['extent']
}
```

### 场景 3: 目标检测训练

```python
# 需要: RGB + 多个 2D BBox
for camera_name in ['head_camera', 'chest_camera', 'third_camera']:
    bbox = extract_bbox_from_camera(camera_name)
    detection_data[camera_name] = {
        'image': rgb,
        'bbox': bbox,
        'class': 'target_object'
    }
```

---

## ⚠️ 注意事项

### ✅ 推荐做法

1. **优先使用 Mask 提取 2D BBox**: 简单可靠，不需要相机矩阵
2. **检查 None 返回值**: 物体可能不在视野内或 Prim 不存在
3. **使用 float32**: 节省存储空间，精度足够
4. **批处理优化**: 对于静态物体，只获取一次 BBox

### ❌ 常见错误

1. **错误的 Prim 路径**: 必须包含完整路径，如 `/World/envs/env_0/bottle`
2. **忘记清空缓冲区**: 在 `_write_data_to_file` 后要清空
3. **相机矩阵获取复杂**: 对于 2D BBox，推荐从 Mask 提取
4. **数据类型不匹配**: 统一使用 `np.float32`

---

## 🐛 故障排查

### 问题 1: `get_3d_bbox_from_prim` 返回 None

**可能原因**:
- Prim 路径错误
- 物体不在场景中
- 物体没有几何体

**解决**:
```python
import omni.isaac.core.utils.prims as prim_utils
prim = prim_utils.get_prim_at_path(prim_path)
print(f"Prim exists: {prim is not None}")
print(f"Prim type: {prim.GetTypeName() if prim else 'N/A'}")
```

### 问题 2: Mask 提取的 BBox 不准确

**可能原因**:
- Mask 质量差
- 分割不完整

**解决**:
```python
# 使用形态学操作改善 mask
import cv2
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
bbox = BBoxExtractor.get_bbox_from_mask(mask_clean)
```

### 问题 3: 2D BBox 超出图像边界

**解决**: 代码已自动处理，使用 `np.clip` 裁剪到图像范围

---

## 🔄 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2025-12-31 | 初始版本，支持 3D/2D BBox 提取和存储 |

---

## 📞 支持与反馈

- **问题反馈**: 在项目中创建 Issue
- **功能建议**: 提交 Pull Request
- **技术支持**: 参考文档或咨询团队

---

## 📖 相关文档

- [Isaac Sim USD API](https://docs.omniverse.nvidia.com/py/isaacsim/)
- [UsdGeom.BBoxCache](https://graphics.pixar.com/usd/docs/api/class_usd_geom___bbox_cache.html)
- [数据采集流程](../collect/README.md)
- [Zarr 转换指南](../psilab/source/psilab/psilab/utils/zarr_utils.py)

---

## 🎓 学习路径

### 初学者 (0-30分钟)
1. 阅读 `BBOX_QUICK_REF.md` (5分钟)
2. 运行 `test_bbox_extraction.py` (5分钟)
3. 查看 `bbox_integration_example.py` (10分钟)
4. 尝试集成到自己的任务 (10分钟)

### 进阶用户 (30-60分钟)
1. 深入阅读 `BBOX_USAGE_GUIDE.md` (20分钟)
2. 理解数据流程和存储格式 (10分钟)
3. 实现自定义可视化工具 (20分钟)
4. 优化性能和存储空间 (10分钟)

---

**最后更新**: 2025-12-31  
**维护者**: PsiRobot Team  
**许可证**: 与项目主许可证相同




