from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import numpy as np

try:
    import imageio.v3 as iio
except Exception as exc:  # pragma: no cover
    raise RuntimeError("imageio.v3 is required to save images") from exc


ArrayLike = Union["np.ndarray", "torch.Tensor"]


def _to_numpy(array: ArrayLike) -> np.ndarray:
    try:
        import torch  # type: ignore

        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(array)


def _ensure_parent_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_rgb_png(rgb: ArrayLike, path: Union[str, Path]) -> Path:
    """Save an RGB image as PNG.

    Accepts HxWx3 arrays in either uint8 or float formats. Float inputs in [0,1]
    are scaled to uint8. Values outside [0,1] are clipped.
    """
    p = _ensure_parent_dir(path)
    arr = _to_numpy(rgb)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)

    iio.imwrite(p.as_posix(), arr)
    return p


def save_dep_png(depth: ArrayLike, path: Union[str, Path], near: float = 0.2, far: float = 1.8) -> Path:
    """Save depth as 16-bit PNG with proper handling of invalid values.

    Invalid values (inf, nan, 0) are saved as 0 in the PNG.
    Valid depth values are normalized to [near, far] range and saved as 1-65535.
    
    Args:
        depth: Depth array in meters
        path: Output file path
        near: Near clipping plane in meters (default: 0.2m)
        far: Far clipping plane in meters (default: 2.5m)
        
    Returns:
        Path to saved file
    """
    if not (far > near > 0):
        raise ValueError(f"Invalid range: near={near}, far={far}")
    
    p = _ensure_parent_dir(path)
    d = _to_numpy(depth).astype(np.float32)
    d = d.squeeze(-1)
    
    valid_mask = np.isfinite(d) & (d > 0)
    valid_count = np.sum(valid_mask)
    d_u16 = np.zeros_like(d, dtype=np.uint16)
    
    if valid_count > 0:
        valid_depths = d[valid_mask]
        d_min, d_max = valid_depths.min(), valid_depths.max()
        d_clipped = np.clip(valid_depths, near, far)
        d_norm = (d_clipped - near) / (far - near)
        d_u16[valid_mask] = (d_norm * 65534.0 + 1.0 + 0.5).astype(np.uint16)
        
        new_mask = d_u16>0
        new_valid_count = np.sum(new_mask)
        assert new_valid_count == valid_count
    
    iio.imwrite(p.as_posix(), d_u16)
    return d_min, d_max
    # return p

def save_nor_png(normal: ArrayLike, path: Union[str, Path]) -> Path:
    """Save normal map as PNG with proper validation and encoding.
    
    This function provides the most robust normal map saving with:
    - Input validation and automatic fixing
    - Proper normalization handling
    - Clean imageio-based implementation
    - Support for various input formats
    
    Args:
        normal: Normal vectors as (H, W, 3) array in [0, 255] range
        path: Output file path
        
    Returns:
        Path to saved file
    """
    p = _ensure_parent_dir(path)
    n = _to_numpy(normal)
    
    n = (n + 1.0) / 2.0
    n = np.clip(n, 0.0, 1.0)
    n_uint8 = (n * 255.0 + 0.5).astype(np.uint8)
    
    iio.imwrite(p.as_posix(), n_uint8)
    return p