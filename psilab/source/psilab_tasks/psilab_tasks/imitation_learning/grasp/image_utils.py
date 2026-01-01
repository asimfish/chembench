import torch
import torch.nn.functional as F

def process_depth(depth: torch.Tensor, image_size: tuple = (224, 224)) -> torch.Tensor:
    """
    处理 Depth 图像
    
    存储格式: float32
    
    输入: depth [B, H, W] 或 [B, H, W, 1]
    输出: [B, 1, H, W] float32 [0, 1] (归一化深度 + Resize)
    """
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)  # [B, H, W, 1]
    
    # 归一化: 0.2 - 1.8m
    near = 0.2
    far = 1.8
    depth_normalized = (depth - near) / (far - near)
    depth_normalized = torch.clamp(depth_normalized, 0, 1)
    
    # 保持无效值为 0 (如果原始值为 0 或负数)
    mask = depth > 0
    depth_normalized = depth_normalized * mask.float()
    
    # [B, H, W, 1] -> [B, 1, H, W]
    depth_normalized = depth_normalized.permute(0, 3, 1, 2)
    
    # Resize (nearest)
    depth_resized = F.interpolate(
        depth_normalized,
        size=image_size,
        mode='nearest'
    )
    
    return depth_resized

def process_normals(normals: torch.Tensor, image_size: tuple = (224, 224)) -> torch.Tensor:
    """
    处理 Normal 图像
    
    输入: normals [B, H, W, 3] float32 [-1, 1]
    输出: [B, 3, H, W] float32 [0, 1] (Resize)
    """
    # 归一化到 [0, 1]
    normals_norm = (normals + 1.0) / 2.0
    
    # [B, H, W, 3] -> [B, 3, H, W]
    normals_norm = normals_norm.permute(0, 3, 1, 2)
    
    # Resize (bilinear)
    normals_resized = F.interpolate(
        normals_norm,
        size=image_size,
        mode='bilinear',
        align_corners=False
    )
    
    return normals_resized

def process_rgb(rgb: torch.Tensor, image_size: tuple = (224, 224)) -> torch.Tensor:
    """
    处理 RGB 图像
    
    输入: rgb [B, H, W, 3] uint8 or float
    输出: [B, 3, H, W] float32 [0, 1] (Resize)
    """
    # 确保是 float
    if rgb.dtype == torch.uint8:
        rgb = rgb.float()
        
    # 如果还没有归一化，则归一化 (根据 max 值判断，简单起见如果 > 1 则除以 255)
    # 注意：这里假设如果输入是 float 且 > 1，则需要归一化。
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    # [B, H, W, 3] -> [B, 3, H, W]
    rgb = rgb.permute(0, 3, 1, 2)
    
    # Resize (bilinear)
    rgb_resized = F.interpolate(
        rgb,
        size=image_size,
        mode='bilinear',
        align_corners=False
    )
    
    return rgb_resized

def process_mask(mask: torch.Tensor, image_size: tuple = (224, 224)) -> torch.Tensor:
    """
    处理 Mask 图像
    
    输入: mask [B, H, W] or [B, H, W, 1]
    输出: [B, 1, H, W] float32 [0, 1] (Resize + Binarize)
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    
    mask = mask.float()
    
    # 归一化 (如果需要)
    if mask.max() > 1.0:
        mask = mask / mask.max()

    # [B, H, W, 1] -> [B, 1, H, W]
    mask = mask.permute(0, 3, 1, 2)
    
    # Resize (nearest)
    mask_resized = F.interpolate(
        mask,
        size=image_size,
        mode='nearest'
    )
    
    # Ensure binary
    mask_resized = (mask_resized > 0.5).float()
    
    return mask_resized

def process_batch_image_multimodal(
    rgb: torch.Tensor = None, 
    mask: torch.Tensor = None, 
    depth: torch.Tensor = None, 
    normal: torch.Tensor = None, 
    obs_mode: str = "rgbm",
    image_size: tuple = (224, 224)
) -> torch.Tensor:
    """
    处理多模态图像 batch
    
    Args:
        rgb: [B, H, W, 3]
        mask: [B, H, W] or [B, H, W, 1]
        depth: [B, H, W] or [B, H, W, 1]
        normal: [B, H, W, 3]
        obs_mode: ["rgb", "rgbm", "nd", "rgbnd", "rgb_masked", "rgb_masked_rgb"]
        image_size: (H, W)
        
    Returns:
        [B, C, H, W] tensor
    """
    
    # 确保所有输入都是 torch tensor
    if rgb is not None and not isinstance(rgb, torch.Tensor):
        rgb = torch.tensor(rgb)
    if mask is not None and not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    if depth is not None and not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth)
    if normal is not None and not isinstance(normal, torch.Tensor):
        normal = torch.tensor(normal)

    # 处理逻辑
    img = None
    
    if obs_mode == "rgb":
        # 3通道 RGB
        img = process_rgb(rgb, image_size) # [B, 3, H, W]
        
    elif obs_mode == "rgbm":
        # 4通道 RGB + Mask
        img_rgb = process_rgb(rgb, image_size) # [B, 3, H, W]
        img_mask = process_mask(mask, image_size) # [B, 1, H, W]
        img = torch.cat([img_rgb, img_mask], dim=1) # [B, 4, H, W]
        
    elif obs_mode == "rgb_masked":
        # 3通道 RGB Masked (RGB * Mask，只保留物体区域，背景置黑)
        img_rgb = process_rgb(rgb, image_size) # [B, 3, H, W]
        img_mask = process_mask(mask, image_size) # [B, 1, H, W]
        img = img_rgb * img_mask # [B, 3, H, W]，Mask为0的地方RGB也变为0
        
    elif obs_mode == "rgb_masked_rgb":
        # 6通道 RGB + RGB*Mask（原始RGB + 背景置黑RGB）
        img_rgb = process_rgb(rgb, image_size) # [B, 3, H, W]
        img_mask = process_mask(mask, image_size) # [B, 1, H, W]
        img_rgb_masked = img_rgb * img_mask # [B, 3, H, W]，背景置黑

        img = torch.cat([img_rgb, img_rgb_masked], dim=1) # [B, 6, H, W]
        # img = torch.cat([img_rgb_masked, img_rgb_masked], dim=1) # [B, 6, H, W]
    elif obs_mode == "nd":
        # 4通道 Normal + Depth
        img_normal = process_normals(normal, image_size) # [B, 3, H, W]
        img_depth = process_depth(depth, image_size) # [B, 1, H, W]
        img = torch.cat([img_normal, img_depth], dim=1) # [B, 4, H, W]
        
    elif obs_mode == "rgbnd":
        # 7通道 RGB + Normal + Depth
        img_rgb = process_rgb(rgb, image_size) # [B, 3, H, W]
        img_normal = process_normals(normal, image_size) # [B, 3, H, W]
        img_depth = process_depth(depth, image_size) # [B, 1, H, W]
        img = torch.cat([img_rgb, img_normal, img_depth], dim=1) # [B, 7, H, W]
        
    else:
        raise ValueError(f"Unknown obs_mode: {obs_mode}")
        
    return img.float()
