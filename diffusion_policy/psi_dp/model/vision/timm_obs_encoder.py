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
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        rgbm_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()
        

        # import json
        # path2cfg = "/home/admin01/Downloads/config.json"
        # # path2mdl = r'path\to\model.safetensors'
        # with open(path2cfg, "r", encoding="utf-8") as reader:
        #     text = reader.read()
        #     cfg_dict = json.loads(text)


        # assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )
        # print("--------------------------")
        # print("model:",model)
        # input()
        # #修改以后的代码：FYD
        # model_path = "/home/admin01/桌面/Work/00-DiffusionPolicy/model/model.safetensors"
        # model  = timm.create_model(
        #     model_name,
        #     global_pool=global_pool, # '' means no pooling
        #     num_classes=0,
        #     pretrained=True, 
        #     pretrained_cfg={'file': model_path}
        #     )
        # model.load_state_dict(torch.load(model_path))

        self.model_name = model_name

        if use_mask_input:
            mask_model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool, # '' means no pooling
                in_chans=4,
                num_classes=0            # remove classification layer
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
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

            if use_mask_input:
                mask_model = replace_submodules(
                    root_module=mask_model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                        num_channels=x.num_features)
                )

        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation is None:
                # Use all tokens from ViT
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
            if type == 'rgb' or type == 'rgbm':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
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

        if use_mask_input:
            mask_transforms = [
                RGBMRandomCrop(size=int(image_shape[0] * ratio)),
                RGBMResize(size=image_shape[0]),
                RGBMRandomRotation(degrees=(-5.0, 5.0)),
                RGBMColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            ]
            mask_transforms = nn.Identity() if mask_transforms is None else torch.nn.Sequential(*mask_transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            # print(f"\nProcessing key: {key}")
            # print(f"Shape from meta: {shape}")
            # print(f"Type: {type}")
            key_shape_map[key] = shape

            if type == 'rgb':
                rgb_keys.append(key)
                # print(f"Added to rgb_keys: {key}")

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    print(f"Creating example tensor with shape: {example_img.shape}")
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

                this_model = mask_model if share_rgb_model else copy.deepcopy(mask_model)
                key_model_map[key] = this_model

                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]

                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj

                this_transform = mask_transforms
                key_transform_map[key] = this_transform
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        rgbm_keys = sorted(rgbm_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.rgbm_keys = rgbm_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

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
        
    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            # Old
            # assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[key](feature)
            assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
            emb = emb.reshape(B,-1,self.n_emb)
            embeddings.append(emb)

        # process rgb input
        for key in self.rgbm_keys:
            img = obs_dict[key]  # [B, T, C, H, W]
            B, T = img.shape[:2]
            assert B == batch_size            
            assert img.shape[2:] == self.key_shape_map[key]
            # Reshape to [B*T, C, H, W]
            img = img.reshape(B*T, *img.shape[2:])
            # Apply transform on entire RGBM image
            img_transformed = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img_transformed)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[key](feature)
            
            assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
            emb = emb.reshape(B, -1, self.n_emb)
            embeddings.append(emb)

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
