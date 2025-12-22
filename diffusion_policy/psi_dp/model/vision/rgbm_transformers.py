import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class RGBMColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def adjust_brightness(self, rgb, factor):
        # brightness在[-1,1]范围内的调整
        return rgb * factor

    def adjust_contrast(self, rgb, factor):
        # 计算每个通道的平均值
        mean = rgb.mean(dim=[-1, -2], keepdim=True)
        # 在[-1,1]范围内调整对比度
        return (rgb - mean) * factor + mean

    def adjust_saturation(self, rgb, factor):
        # 转换到HSV空间
        # 注意：输入是[-1,1]范围
        rgb_norm = (rgb + 1) / 2  # 转到[0,1]
        
        # 计算灰度图
        gray = rgb_norm.mean(dim=1, keepdim=True)
        
        # 调整饱和度
        adjusted = rgb_norm * factor + gray * (1 - factor)
        
        # 转回[-1,1]
        return adjusted * 2 - 1

    def adjust_hue(self, rgb, factor):
        # 转到[0,1]范围
        rgb_norm = (rgb + 1) / 2

        # RGB转HSV
        r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
        max_rgb, _ = torch.max(rgb_norm, dim=1)
        min_rgb, _ = torch.min(rgb_norm, dim=1)
        diff = max_rgb - min_rgb

        # 计算色相
        hue = torch.zeros_like(max_rgb)
        mask = diff != 0
        
        # R是最大值
        mask_r = mask & (max_rgb == r)
        hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r])) % 360

        # G是最大值
        mask_g = mask & (max_rgb == g)
        hue[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120

        # B是最大值
        mask_b = mask & (max_rgb == b)
        hue[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240

        # 调整色相
        hue = (hue + factor * 360) % 360

        # HSV转回RGB (简化版本)
        h_prime = hue / 60
        x = diff * (1 - torch.abs(h_prime % 2 - 1))
        
        rgb_adjusted = torch.zeros_like(rgb_norm)
        
        # 根据色相区间设置RGB值
        idx = (h_prime < 1).float()
        rgb_adjusted[:, 0] += idx * max_rgb
        rgb_adjusted[:, 1] += idx * x
        
        idx = ((h_prime >= 1) & (h_prime < 2)).float()
        rgb_adjusted[:, 0] += idx * x
        rgb_adjusted[:, 1] += idx * max_rgb
        
        # 转回[-1,1]范围
        return rgb_adjusted * 2 - 1

    def forward(self, img):
        rgb = img[:, :3]  # (B,3,H,W)
        mask = img[:, 3:]  # (B,1,H,W)

        # 随机生成变换因子
        if self.brightness > 0:
            factor = 1.0 + torch.rand(1).item() * self.brightness * 2 - self.brightness
            rgb = self.adjust_brightness(rgb, factor)

        if self.contrast > 0:
            factor = 1.0 + torch.rand(1).item() * self.contrast * 2 - self.contrast
            rgb = self.adjust_contrast(rgb, factor)

        if self.saturation > 0:
            factor = 1.0 + torch.rand(1).item() * self.saturation * 2 - self.saturation
            rgb = self.adjust_saturation(rgb, factor)

        if self.hue > 0:
            factor = torch.rand(1).item() * self.hue * 2 - self.hue
            rgb = self.adjust_hue(rgb, factor)

        # 确保输出在[-1,1]范围内
        rgb = torch.clamp(rgb, -1, 1)

        return torch.cat([rgb, mask], dim=1)

class RGBMRandomCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        # img: (B,C,H,W)
        _, _, h, w = img.shape
        
        # 获取随机裁剪的位置
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        
        # 裁剪
        img = img[:, :, top:top+self.size, left:left+self.size]
        return img

class RGBMResize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        # img: (B,C,H,W)
        rgb = img[:, :3]  # (B,3,H,W)
        mask = img[:, 3:]  # (B,1,H,W)
        
        # 调整大小
        rgb = TF.resize(rgb, [self.size, self.size], antialias=True)
        mask = TF.resize(mask, [self.size, self.size], antialias=False)
        
        return torch.cat([rgb, mask], dim=1)

class RGBMRandomRotation(nn.Module):
    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees

    def forward(self, img):
        # img: (B,C,H,W)
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        rgb = img[:, :3]  # (B,3,H,W)
        mask = img[:, 3:]  # (B,1,H,W)
        
        # 旋转
        rgb = TF.rotate(rgb, angle)
        mask = TF.rotate(mask, angle)
        
        return torch.cat([rgb, mask], dim=1)

class RGBRandomRotation(nn.Module):
    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees

    def forward(self, img):
        # img: (B,3,H,W)
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        # 直接旋转RGB图像
        return TF.rotate(img, angle)

class RGBColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def adjust_brightness(self, rgb, factor):
        return rgb * factor

    def adjust_contrast(self, rgb, factor):
        mean = rgb.mean(dim=[-1, -2], keepdim=True)
        return (rgb - mean) * factor + mean

    def adjust_saturation(self, rgb, factor):
        # 转到[0,1]范围
        rgb_norm = (rgb + 1) / 2
        
        # 计算灰度图
        gray = rgb_norm.mean(dim=1, keepdim=True)
        
        # 调整饱和度
        adjusted = rgb_norm * factor + gray * (1 - factor)
        
        # 转回[-1,1]
        return adjusted * 2 - 1

    def adjust_hue(self, rgb, factor):
        # 转到[0,1]范围
        rgb_norm = (rgb + 1) / 2

        # RGB转HSV
        r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
        max_rgb, _ = torch.max(rgb_norm, dim=1)
        min_rgb, _ = torch.min(rgb_norm, dim=1)
        diff = max_rgb - min_rgb

        # 计算色相
        hue = torch.zeros_like(max_rgb)
        mask = diff != 0
        
        mask_r = mask & (max_rgb == r)
        hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r])) % 360

        mask_g = mask & (max_rgb == g)
        hue[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120

        mask_b = mask & (max_rgb == b)
        hue[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240

        # 调整色相
        hue = (hue + factor * 360) % 360

        # HSV转回RGB
        h_prime = hue / 60
        x = diff * (1 - torch.abs(h_prime % 2 - 1))
        
        rgb_adjusted = torch.zeros_like(rgb_norm)
        
        idx = (h_prime < 1).float()
        rgb_adjusted[:, 0] += idx * max_rgb
        rgb_adjusted[:, 1] += idx * x
        
        idx = ((h_prime >= 1) & (h_prime < 2)).float()
        rgb_adjusted[:, 0] += idx * x
        rgb_adjusted[:, 1] += idx * max_rgb
        
        return rgb_adjusted * 2 - 1

    def forward(self, img):
        # img: (B,3,H,W)
        rgb = img

        # 随机生成变换因子
        if self.brightness > 0:
            factor = 1.0 + torch.rand(1).item() * self.brightness * 2 - self.brightness
            rgb = self.adjust_brightness(rgb, factor)

        if self.contrast > 0:
            factor = 1.0 + torch.rand(1).item() * self.contrast * 2 - self.contrast
            rgb = self.adjust_contrast(rgb, factor)

        if self.saturation > 0:
            factor = 1.0 + torch.rand(1).item() * self.saturation * 2 - self.saturation
            rgb = self.adjust_saturation(rgb, factor)

        if self.hue > 0:
            factor = torch.rand(1).item() * self.hue * 2 - self.hue
            rgb = self.adjust_hue(rgb, factor)

        return torch.clamp(rgb, -1, 1)


class RGBRandomCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        # img: (B,3,H,W)
        _, _, h, w = img.shape
        
        # 获取随机裁剪的位置
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        
        # 裁剪
        return img[:, :, top:top+self.size, left:left+self.size]