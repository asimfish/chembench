import copy

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
# Feng Yunduo,Start

# from diffusion_policy.model.vision.rgbm_transformers import RGBMResize, RGBMRandomCrop, RGBMColorJitter, RGBMRandomRotation, RGBColorJitter, RGBRandomRotation

from psi_dp.model.vision.rgbm_transformers import RGBMResize, RGBMRandomCrop, RGBMColorJitter, RGBMRandomRotation, RGBColorJitter, RGBRandomRotation
# Feng Yunduo,Start

from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


# 多通道图像变换类 (支持 nd 和 rgbnd 模式)
class MultiChannelRandomCrop(nn.Module):
    """支持任意通道数的随机裁剪"""
    def __init__(self, size):
        super().__init__()
        self.size = size
        
    def forward(self, x):
        # x: [B, C, H, W]
        _, _, h, w = x.shape
        if self.size < h and self.size < w:
            i = torch.randint(0, h - self.size + 1, size=(1,)).item()
            j = torch.randint(0, w - self.size + 1, size=(1,)).item()
            return x[:, :, i:i+self.size, j:j+self.size]
        return x


class MultiChannelResize(nn.Module):
    """支持任意通道数的缩放"""
    def __init__(self, size):
        super().__init__()
        self.size = size
        
    def forward(self, x):
        return F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)


class MultiChannelRandomRotation(nn.Module):
    """支持任意通道数的随机旋转"""
    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees
        
    def forward(self, x):
        angle = torch.FloatTensor(1).uniform_(self.degrees[0], self.degrees[1]).item()
        return torchvision.transforms.functional.rotate(x, angle)
    

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str='vit_base_patch16_clip_224.openai',
            global_pool: str='',
            transforms: list=None,
            n_emb: int=768,
            pretrained: bool=False,
            frozen: bool=False,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            feature_aggregation: str=None,
            downsample_ratio: int=32,
            use_mask_input: bool=False,
            in_channels: int=None,  # 新增：指定输入通道数，用于 nd/rgbnd 等模式
        ):
        """
        支持多种图像输入模式的观测编码器
        
        支持的图像类型:
            - rgb: 纯 RGB 3通道
            - rgbm: RGB + Mask 4通道  
            - rgb_masked: RGB * Mask 3通道 (背景置黑)
            - rgb_masked_rgb: RGB + RGB*Mask 6通道 (原始RGB + 背景置黑RGB)
            - nd: Normal + Depth 4通道
            - rgbnd: RGB + Normal + Depth 7通道
            
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        rgbm_keys = list()
        rgb_masked_keys = list()
        rgb_masked_rgb_keys = list()
        nd_keys = list()
        rgbnd_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()
        
        self.model_name = model_name

        # 创建 3 通道模型 (用于 rgb, rgb_masked)
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool,
            num_classes=0
        )

        # 创建 4 通道模型 (用于 rgbm 和 nd)
        model_4ch = None
        if use_mask_input or in_channels == 4:
            model_4ch = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,
                in_chans=4,
                num_classes=0
            )
        
        # 创建 6 通道模型 (用于 rgb_masked_rgb)
        model_6ch = None
        if in_channels == 6:
            model_6ch = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,
                in_chans=6,
                num_classes=0
            )
            
        # 创建 7 通道模型 (用于 rgbnd)
        model_7ch = None
        if in_channels == 7:
            model_7ch = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,
                in_chans=7,
                num_classes=0
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
            if model_4ch is not None:
                for param in model_4ch.parameters():
                    param.requires_grad = False
            if model_6ch is not None:
                for param in model_6ch.parameters():
                    param.requires_grad = False
            if model_7ch is not None:
                for param in model_7ch.parameters():
                    param.requires_grad = False
        
        feature_dim = None

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )

            if model_4ch is not None:
                model_4ch = replace_submodules(
                    root_module=model_4ch,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                        num_channels=x.num_features)
                )
            
            if model_6ch is not None:
                model_6ch = replace_submodules(
                    root_module=model_6ch,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                        num_channels=x.num_features)
                )
                
            if model_7ch is not None:
                model_7ch = replace_submodules(
                    root_module=model_7ch,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                        num_channels=x.num_features)
                )

        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            if self.feature_aggregation is None:
                pass
            elif self.feature_aggregation != 'cls':
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type in ['rgb', 'rgbm', 'rgb_masked', 'rgb_masked_rgb', 'nd', 'rgbnd']:
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        # 创建各种变换
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            
            # RGB 3通道变换
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True),
                RGBColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                RGBRandomRotation(degrees=(-5.0, 5.0))
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        # RGBM 4通道变换 (带颜色增强)
        rgbm_transforms = None
        if model_4ch is not None and transforms is not None:
            rgbm_transforms = torch.nn.Sequential(
                RGBMRandomCrop(size=int(image_shape[0] * ratio)),
                RGBMResize(size=image_shape[0]),
                RGBMRandomRotation(degrees=(-5.0, 5.0)),
                RGBMColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            )
            
        # ND 4通道变换 (不使用颜色增强，因为 normal 和 depth 不适合颜色抖动)
        nd_transforms = None
        if model_4ch is not None and transforms is not None:
            nd_transforms = torch.nn.Sequential(
                MultiChannelRandomCrop(size=int(image_shape[0] * ratio)),
                MultiChannelResize(size=image_shape[0]),
                MultiChannelRandomRotation(degrees=(-5.0, 5.0)),
            )
            
        # RGBND 7通道变换 (不使用颜色增强)
        rgbnd_transforms = None
        if model_7ch is not None and transforms is not None:
            rgbnd_transforms = torch.nn.Sequential(
                MultiChannelRandomCrop(size=int(image_shape[0] * ratio)),
                MultiChannelResize(size=image_shape[0]),
                MultiChannelRandomRotation(degrees=(-5.0, 5.0)),
            )
        
        # RGB_MASKED_RGB 6通道变换 (前3通道RGB使用颜色增强，后3通道RGB*Mask也使用颜色增强)
        rgb_masked_rgb_transforms = None
        if model_6ch is not None and transforms is not None:
            # 使用通用多通道变换，并保持前3和后3通道同步增强
            rgb_masked_rgb_transforms = torch.nn.Sequential(
                MultiChannelRandomCrop(size=int(image_shape[0] * ratio)),
                MultiChannelResize(size=image_shape[0]),
                MultiChannelRandomRotation(degrees=(-5.0, 5.0)),
                # 注意：颜色抖动需要特殊处理，对前3通道和后3通道分别应用
            )

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape

            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    logger.info(f"Creating RGB example tensor with shape: {example_img.shape}")
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj

                this_transform = transform
                key_transform_map[key] = this_transform

            elif type == 'rgb_masked':
                # RGB Masked 3通道 (与 RGB 处理相同，但使用单独的 key 列表)
                rgb_masked_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    logger.info(f"Creating RGB_MASKED example tensor with shape: {example_img.shape}")
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj

                this_transform = transform
                key_transform_map[key] = this_transform

            elif type == 'low_dim':
                dim = np.prod(shape)
                proj = nn.Identity()
                if dim != n_emb:
                    proj = nn.Linear(in_features=dim, out_features=n_emb)
                key_projection_map[key] = proj

                low_dim_keys.append(key)

            elif type == 'rgbm':
                rgbm_keys.append(key)

                this_model = model_4ch if share_rgb_model else copy.deepcopy(model_4ch)
                key_model_map[key] = this_model

                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    logger.info(f"Creating RGBM example tensor with shape: {example_img.shape}")
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]

                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj

                this_transform = rgbm_transforms if rgbm_transforms is not None else nn.Identity()
                key_transform_map[key] = this_transform
                
            elif type == 'nd':
                # Normal + Depth 4通道
                nd_keys.append(key)
                
                this_model = model_4ch if share_rgb_model else copy.deepcopy(model_4ch)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    logger.info(f"Creating ND example tensor with shape: {example_img.shape}")
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                    
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj
                
                this_transform = nd_transforms if nd_transforms is not None else nn.Identity()
                key_transform_map[key] = this_transform
                
            elif type == 'rgbnd':
                # RGB + Normal + Depth 7通道
                rgbnd_keys.append(key)
                
                this_model = model_7ch if share_rgb_model else copy.deepcopy(model_7ch)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    logger.info(f"Creating RGBND example tensor with shape: {example_img.shape}")
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                    
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj
                
                this_transform = rgbnd_transforms if rgbnd_transforms is not None else nn.Identity()
                key_transform_map[key] = this_transform
            
            elif type == 'rgb_masked_rgb':
                # RGB + RGB*Mask 6通道
                rgb_masked_rgb_keys.append(key)
                
                this_model = model_6ch if share_rgb_model else copy.deepcopy(model_6ch)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    logger.info(f"Creating RGB_MASKED_RGB example tensor with shape: {example_img.shape}")
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                    
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj
                
                this_transform = rgb_masked_rgb_transforms if rgb_masked_rgb_transforms is not None else nn.Identity()
                key_transform_map[key] = this_transform
                
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        rgbm_keys = sorted(rgbm_keys)
        rgb_masked_keys = sorted(rgb_masked_keys)
        rgb_masked_rgb_keys = sorted(rgb_masked_rgb_keys)
        nd_keys = sorted(nd_keys)
        rgbnd_keys = sorted(rgbnd_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.rgbm_keys = rgbm_keys
        self.rgb_masked_keys = rgb_masked_keys
        self.rgb_masked_rgb_keys = rgb_masked_rgb_keys
        self.nd_keys = nd_keys
        self.rgbnd_keys = rgbnd_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        logger.info(f"Image types - RGB: {rgb_keys}, RGBM: {rgbm_keys}, RGB_MASKED: {rgb_masked_keys}, RGB_MASKED_RGB: {rgb_masked_rgb_keys}, ND: {nd_keys}, RGBND: {rgbnd_keys}")

    def aggregate_feature(self, feature):
        # Return: B, N, C
        
        if self.model_name.startswith('vit'):
            # vit uses the CLS token
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]

            # or use all tokens
            assert self.feature_aggregation is None 
            return feature
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        else:
            assert self.feature_aggregation is None
            return feature
        
    def _process_image_keys(self, obs_dict, keys, batch_size):
        """通用的图像处理方法"""
        embeddings = []
        for key in keys:
            img = obs_dict[key]  # [B, T, C, H, W]
            B, T = img.shape[:2]
            assert B == batch_size            
            assert img.shape[2:] == self.key_shape_map[key]
            # Reshape to [B*T, C, H, W]
            img = img.reshape(B*T, *img.shape[2:])
            # Apply transform
            img_transformed = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img_transformed)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[key](feature)
            
            assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
            emb = emb.reshape(B, -1, self.n_emb)
            embeddings.append(emb)
        return embeddings
        
    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        embeddings.extend(self._process_image_keys(obs_dict, self.rgb_keys, batch_size))
        
        # process rgbm input
        embeddings.extend(self._process_image_keys(obs_dict, self.rgbm_keys, batch_size))
        
        # process rgb_masked input
        embeddings.extend(self._process_image_keys(obs_dict, self.rgb_masked_keys, batch_size))
        
        # process rgb_masked_rgb input (RGB + RGB*Mask 6通道)
        embeddings.extend(self._process_image_keys(obs_dict, self.rgb_masked_rgb_keys, batch_size))
        
        # process nd input (normal + depth)
        embeddings.extend(self._process_image_keys(obs_dict, self.nd_keys, batch_size))
        
        # process rgbnd input (rgb + normal + depth)
        embeddings.extend(self._process_image_keys(obs_dict, self.rgbnd_keys, batch_size))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B,T,-1)
            emb = self.key_projection_map[key](data)
            assert emb.shape[-1] == self.n_emb
            embeddings.append(emb)
        
        # concatenate all features along t
        result = torch.cat(embeddings, dim=1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1

        return example_output.shape



class LowDimObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            n_emb: int=768,
        ):
        """
        只处理低维输入: B,T,D
        参数:
            shape_meta (dict): 输入形状的元数据
            n_emb (int): 输出嵌入维度
        """
        super().__init__()
        
        low_dim_keys = list()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()

        # 处理shape_meta中的每个观测
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            
            if type != 'low_dim':
                continue
                
            key_shape_map[key] = shape
            low_dim_keys.append(key)
            
            # 计算输入维度并创建投影层
            dim = np.prod(shape)
            proj = nn.Identity()
            if dim != n_emb:
                proj = nn.Linear(in_features=dim, out_features=n_emb)
            key_projection_map[key] = proj

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_projection_map = key_projection_map
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # 处理所有low_dim输入
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B, T, -1)
            emb = self.key_projection_map[key](data)
            assert emb.shape[-1] == self.n_emb
            embeddings.append(emb)
        
        # 沿时间维度连接所有特征
        result = torch.cat(embeddings, dim=1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            if key not in self.low_dim_keys:
                continue
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        return example_output.shape


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_diffusion_transformer_umi_workspace')
        OmegaConf.resolve(cfg)

    shape_meta = cfg.task.shape_meta
    encoder = TransformerObsEncoder(
        shape_meta=shape_meta
    )
