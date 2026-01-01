# 🎯 BBox 获取与存储 - 快速参考

## 📦 文件清单

| 文件 | 说明 |
|------|------|
| `psilab/source/psilab/psilab/utils/bbox_utils.py` | BBox 提取核心工具 |
| `docs/BBOX_USAGE_GUIDE.md` | 完整使用指南 |
| `docs/bbox_integration_example.py` | 集成示例代码 |
| `test_bbox_extraction.py` | 单元测试套件 |

---

## 🚀 快速上手（3步）

### 1️⃣ 导入工具

```python
from psilab.utils.bbox_utils import BBoxExtractor
```

### 2️⃣ 获取 BBox

```python
# 方法 A: 从 USD Prim 获取 3D BBox（需要 Isaac Sim）
bbox_3d = BBoxExtractor.get_3d_bbox_from_prim("/World/envs/env_0/bottle")

# 方法 B: 从 Mask 获取 2D BBox（推荐！）
bbox_2d = BBoxExtractor.get_bbox_from_mask(mask)
```

### 3️⃣ 存储到 HDF5

```python
# 在 _record_data() 中记录
self._bbox_buffer.append(bbox_3d['center'])

# 在 _write_data_to_file() 中写入
h5_file.create_dataset("rigid_objects/target_bbox_3d", data=bbox_array)
```

---

## 📊 数据格式

### 3D BBox (7维)

```python
[center_x, center_y, center_z, width, height, depth, valid]
# 示例: [1.0, 2.0, 0.5, 0.1, 0.1, 0.2, 1.0]
```

### 2D BBox (6维)

```python
[x_min, y_min, x_max, y_max, width, height]
# 示例: [50, 30, 200, 150, 150, 120]
```

---

## 🔧 在 grasp_mp.py 中集成

### A. 初始化（`__init__`）

```python
from psilab.utils.bbox_utils import BBoxExtractor

self.bbox_extractor = BBoxExtractor(device=self.device)
self._bbox_3d_buffer = []
```

### B. 记录数据（`_record_data`）

```python
# 3D BBox
target_path = f"/World/envs/env_{env_id}/{self._target_object_name}"
bbox_3d = BBoxExtractor.get_3d_bbox_from_prim(target_path)
if bbox_3d:
    data = np.concatenate([bbox_3d['center'], bbox_3d['extent'], [1.0]])
else:
    data = np.zeros(7)
self._bbox_3d_buffer.append(data)

# 2D BBox (从 mask)
mask = camera.data.output[env_id]["instance_segmentation_fast"].cpu().numpy()
bbox_2d = BBoxExtractor.get_bbox_from_mask(mask)
if bbox_2d:
    data = [bbox_2d['x_min'], bbox_2d['y_min'], bbox_2d['x_max'], 
            bbox_2d['y_max'], bbox_2d['width'], bbox_2d['height']]
```

### C. 写入文件（`_write_data_to_file`）

```python
# 3D BBox
bbox_array = np.array(self._bbox_3d_buffer)
h5_file.create_dataset("rigid_objects/target_bbox_3d", data=bbox_array)

# 2D BBox
h5_file["cameras"].create_dataset("head_camera_bbox_2d", data=bbox_2d_array)

# 清空
self._bbox_3d_buffer.clear()
```

---

## 📦 在 zarr_utils.py 中转换

### 在 `convert_rgb_based()` 中添加：

```python
# 3D BBox
if "rigid_objects/target_bbox_3d" in h5_file:
    episode['target_bbox_3d'] = np.array(h5_file["rigid_objects/target_bbox_3d"])
    episode['target_bbox_center'] = episode['target_bbox_3d'][:, :3]
    episode['target_bbox_extent'] = episode['target_bbox_3d'][:, 3:6]

# 2D BBox
if "cameras/head_camera_bbox_2d" in h5_file:
    episode['head_camera_bbox_2d'] = np.array(h5_file["cameras/head_camera_bbox_2d"])
```

---

## 👁️ 可视化验证

### 方法 1: 使用 analyze_zarr.py

在 `save_image_samples()` 中添加：

```python
# 读取 RGB 和 BBox
rgb = data_group['head_camera_rgb'][idx]
bbox = data_group['head_camera_bbox_2d'][idx]

# 绘制边界框
x_min, y_min, x_max, y_max = bbox[:4].astype(int)
cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.imwrite(f'frame_with_bbox_{idx}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
```

### 方法 2: 独立脚本

```python
import zarr
import cv2
import numpy as np

root = zarr.open("data.zarr", 'r')
rgb = root['data']['head_camera_rgb'][10]
bbox = root['data']['head_camera_bbox_2d'][10]

x_min, y_min, x_max, y_max = bbox[:4].astype(int)
cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.imshow('BBox', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
```

---

## ⚡ 实用技巧

### ✅ 推荐做法

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| **2D BBox** | 从 Mask 提取 | 简单、可靠、无需相机矩阵 |
| **3D BBox** | 从 Prim 提取 | 精确、独立于视角 |
| **多相机** | 统一处理 | 代码复用性好 |
| **存储格式** | float32 | 节省空间，精度足够 |

### ⚠️ 常见陷阱

```python
# ❌ 错误：直接用物体名称
bbox = get_3d_bbox_from_prim("bottle")  

# ✅ 正确：使用完整路径
bbox = get_3d_bbox_from_prim("/World/envs/env_0/bottle")

# ❌ 错误：不检查 None
bbox_data = bbox['center']  # 可能报错

# ✅ 正确：先检查
if bbox is not None:
    bbox_data = bbox['center']
else:
    bbox_data = np.zeros(3)
```

---

## 🧪 测试与验证

### 运行单元测试

```bash
python test_bbox_extraction.py
```

### 预期输出

```
🧪 BBox Utils 测试套件
============================================================
✅ 通过 - 从 Mask 提取 BBox
✅ 通过 - 空 Mask 处理
✅ 通过 - 3D 到 2D 投影
✅ 通过 - 数据格式转换
✅ 通过 - 3D BBox 结构
============================================================
总计: 5/5 测试通过
🎉 所有测试通过！
```

---

## 📚 延伸阅读

- **完整指南**: `docs/BBOX_USAGE_GUIDE.md`
- **集成示例**: `docs/bbox_integration_example.py`
- **API 文档**: `psilab/source/psilab/psilab/utils/bbox_utils.py`

---

## 💡 示例：完整工作流

```python
# 1. 数据采集 (grasp_mp.py)
bbox_3d = BBoxExtractor.get_3d_bbox_from_prim(prim_path)
self._bbox_buffer.append(bbox_3d)

# 2. 写入 HDF5
h5_file.create_dataset("rigid_objects/target_bbox_3d", data=bbox_array)

# 3. 转换 Zarr (zarr_utils.py)
episode['target_bbox_3d'] = h5_file["rigid_objects/target_bbox_3d"][:]

# 4. 使用训练 (training script)
bbox_center = batch['target_bbox_center']  # (B, 3)
bbox_extent = batch['target_bbox_extent']  # (B, 3)
```

---

**版本**: v1.0 | **更新**: 2025-12-31 | **测试状态**: ✅ 通过




