"""
grasp_mp.py 中集成 BBox 提取的完整示例

将以下代码片段添加到对应位置
"""

# ============================================================
# 1. 在文件开头添加导入
# ============================================================
from psilab.utils.bbox_utils import BBoxExtractor


# ============================================================
# 2. 在 GraspBottleEnv.__init__() 中初始化
# ============================================================
def __init__(self, cfg: GraspBottleEnvCfg, render_mode: str | None = None, **kwargs):
    # ... 现有代码 ...
    
    # 初始化 BBox 提取器
    self.bbox_extractor = BBoxExtractor(device=self.device)
    self._bbox_3d_buffer = []  # 存储每帧的 3D BBox
    self._bbox_2d_buffers = {
        'head_camera': [],
        'chest_camera': [],
        'third_camera': [],
    }  # 存储每帧的 2D BBox (可选)


# ============================================================
# 3. 在 _record_data() 中记录 BBox
# ============================================================
def _record_data(self, env_id: int):
    """记录数据到缓冲区"""
    # ... 现有的相机、机器人数据记录 ...
    
    # ==================== 新增: 记录 3D BBox ====================
    # 获取目标物体的 prim 路径
    # 注意: self._target_object_name 应该是物体在场景中的名称
    # 例如: "bottle", "mortar", "glass_beaker_100ml" 等
    target_prim_path = f"/World/envs/env_{env_id}/{self._target_object_name}"
    
    # 提取 3D BBox
    bbox_3d = BBoxExtractor.get_3d_bbox_from_prim(target_prim_path)
    
    if bbox_3d is not None:
        # 存储为 7 维数组: center(3) + extent(3) + valid(1)
        bbox_data = np.concatenate([
            bbox_3d['center'],
            bbox_3d['extent'],
            [1.0]  # valid flag (1.0 表示有效)
        ]).astype(np.float32)
    else:
        # 如果获取失败，存储全零数组
        bbox_data = np.zeros(7, dtype=np.float32)
        print(f"Warning: Failed to get 3D BBox for {target_prim_path}")
    
    self._bbox_3d_buffer.append(bbox_data)
    
    # ==================== 可选: 记录 2D BBox ====================
    # 方法 A: 从 mask 提取（推荐，简单可靠）
    for camera_name in ["head_camera", "chest_camera", "third_camera"]:
        if camera_name not in self.scene:
            continue
            
        camera = self.scene[camera_name]
        camera_data = camera.data.output[env_id]
        
        # 获取分割 mask
        mask_key = "instance_segmentation_fast"
        if mask_key in camera_data:
            mask = camera_data[mask_key].cpu().numpy()
            
            # 从 mask 提取 BBox
            bbox_2d = BBoxExtractor.get_bbox_from_mask(mask)
            
            if bbox_2d is not None:
                # 存储为 6 维数组: [x_min, y_min, x_max, y_max, width, height]
                bbox_2d_data = np.array([
                    bbox_2d['x_min'],
                    bbox_2d['y_min'],
                    bbox_2d['x_max'],
                    bbox_2d['y_max'],
                    bbox_2d['width'],
                    bbox_2d['height'],
                ], dtype=np.float32)
            else:
                bbox_2d_data = np.zeros(6, dtype=np.float32)
            
            self._bbox_2d_buffers[camera_name].append(bbox_2d_data)


# ============================================================
# 4. 在 _write_data_to_file() 中写入 HDF5
# ============================================================
def _write_data_to_file(self, env_id: int):
    """将缓冲区数据写入 HDF5 文件"""
    # ... 现有的写入逻辑 ...
    
    # ==================== 新增: 写入 3D BBox ====================
    if len(self._bbox_3d_buffer) > 0:
        bbox_3d_array = np.array(self._bbox_3d_buffer)
        
        # 写入完整 BBox 数据
        h5_file.create_dataset(
            "rigid_objects/target_bbox_3d",
            data=bbox_3d_array,
            dtype=np.float32
        )
        
        # 可选: 分别存储中心和尺寸（便于后续使用）
        h5_file.create_dataset(
            "rigid_objects/target_bbox_center",
            data=bbox_3d_array[:, :3],  # [x, y, z]
            dtype=np.float32
        )
        h5_file.create_dataset(
            "rigid_objects/target_bbox_extent",
            data=bbox_3d_array[:, 3:6],  # [width, height, depth]
            dtype=np.float32
        )
        
        print(f"  Saved 3D BBox data: {bbox_3d_array.shape}")
    
    # ==================== 可选: 写入 2D BBox ====================
    # 创建相机组（如果不存在）
    if "cameras" not in h5_file:
        cameras_group = h5_file.create_group("cameras")
    else:
        cameras_group = h5_file["cameras"]
    
    # 写入每个相机的 2D BBox
    for camera_name, bbox_buffer in self._bbox_2d_buffers.items():
        if len(bbox_buffer) > 0:
            bbox_2d_array = np.array(bbox_buffer)
            
            cameras_group.create_dataset(
                f"{camera_name}_bbox_2d",
                data=bbox_2d_array,
                dtype=np.float32
            )
            
            print(f"  Saved 2D BBox for {camera_name}: {bbox_2d_array.shape}")
    
    # ==================== 清空缓冲区 ====================
    self._bbox_3d_buffer.clear()
    for camera_name in self._bbox_2d_buffers:
        self._bbox_2d_buffers[camera_name].clear()


# ============================================================
# 5. 在 zarr_utils.py 的 convert_rgb_based() 中添加
# ============================================================
def convert_rgb_based(h5_file, h5_temp, ...):
    """RGB 模式转换"""
    episode = dict()
    
    # ... 现有的数据转换 ...
    
    # ==================== 新增: 处理 BBox 数据 ====================
    
    # 1. 处理 3D BBox
    if "rigid_objects/target_bbox_3d" in h5_file:
        bbox_3d_data = np.array(h5_file["rigid_objects/target_bbox_3d"])
        episode['target_bbox_3d'] = bbox_3d_data
        
        # 分离存储（便于训练时使用）
        episode['target_bbox_center'] = bbox_3d_data[:, :3]  # [x, y, z]
        episode['target_bbox_extent'] = bbox_3d_data[:, 3:6]  # [w, h, d]
        episode['target_bbox_valid'] = bbox_3d_data[:, 6:7]   # valid flag
        
        print(f"  Loaded 3D BBox: {bbox_3d_data.shape}")
    
    # 2. 处理 2D BBox
    if "cameras" in h5_file:
        cameras_group = h5_file["cameras"]
        
        for camera_name in ["head_camera", "chest_camera", "third_camera"]:
            bbox_key = f"{camera_name}_bbox_2d"
            if bbox_key in cameras_group:
                bbox_2d_data = np.array(cameras_group[bbox_key])
                episode[bbox_key] = bbox_2d_data
                print(f"  Loaded 2D BBox for {camera_name}: {bbox_2d_data.shape}")
    
    return episode


# ============================================================
# 6. 在 analyze_zarr.py 中添加 BBox 可视化
# ============================================================
def visualize_bbox_in_analyze_zarr(data_group, output_dir, num_samples=5):
    """在 analyze_zarr.py 中添加 BBox 可视化"""
    import cv2
    
    # 可视化 2D BBox
    for camera_name in ['head_camera', 'chest_camera', 'third_camera']:
        rgb_key = f'{camera_name}_rgb'
        bbox_key = f'{camera_name}_bbox_2d'
        
        if rgb_key in data_group and bbox_key in data_group:
            rgb_data = data_group[rgb_key][:]
            bbox_data = data_group[bbox_key][:]
            
            bbox_dir = os.path.join(output_dir, f'{camera_name}_with_bbox')
            if not os.path.exists(bbox_dir):
                os.makedirs(bbox_dir)
            
            total_frames = rgb_data.shape[0]
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            
            print(f"Saving {num_samples} RGB+BBox samples for {camera_name}...")
            
            for idx in sample_indices:
                frame = rgb_data[idx].copy()
                bbox = bbox_data[idx]
                
                # 绘制边界框
                x_min, y_min, x_max, y_max = bbox[:4].astype(int)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                             (0, 255, 0), 2)
                
                # 添加标签
                label = f"BBox: {bbox[4]:.0f}x{bbox[5]:.0f}"
                cv2.putText(frame, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 保存
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(bbox_dir, f'frame_{idx:04d}.png')
                cv2.imwrite(save_path, frame_bgr)
            
            print(f"  Saved to {bbox_dir}/")


# ============================================================
# 完整示例：集成所有步骤
# ============================================================
"""
总结：完整的 BBox 集成流程

1. grasp_mp.py:
   - __init__: 初始化 BBoxExtractor 和缓冲区
   - _record_data: 每帧记录 3D 和 2D BBox
   - _write_data_to_file: 写入 HDF5

2. zarr_utils.py:
   - convert_rgb_based: 转换 BBox 数据到 Zarr

3. analyze_zarr.py:
   - 添加 BBox 可视化功能

4. 数据结构:
   HDF5:
     rigid_objects/target_bbox_3d: (N, 7) [cx, cy, cz, w, h, d, valid]
     cameras/head_camera_bbox_2d: (N, 6) [x_min, y_min, x_max, y_max, w, h]
     cameras/chest_camera_bbox_2d: (N, 6)
     cameras/third_camera_bbox_2d: (N, 6)
   
   Zarr:
     data/target_bbox_3d: (N, 7)
     data/target_bbox_center: (N, 3)
     data/target_bbox_extent: (N, 3)
     data/head_camera_bbox_2d: (N, 6)
     data/chest_camera_bbox_2d: (N, 6)
     data/third_camera_bbox_2d: (N, 6)

5. 验证:
   python test_bbox_extraction.py  # 单元测试
   python analyze_zarr.py --path /path/to/data.zarr  # 可视化验证
"""

