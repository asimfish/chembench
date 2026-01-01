# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0


import torch
import torch.nn.functional as F
import dill
import torch
import hydra
import sys
# sys.path.insert(0, '/home/psibot/dp_new')  # 添加 psi_dp 所在目录
sys.path.insert(0, '/home/psibot/chembench/diffusion_policy')  # 添加 psi_dp 所在目录

def load_diffusion_policy_2(checkpoint_path, device='cpu'):
    # 加载 checkpoint 字典
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location=device)
    
    # 从配置创建模型
    cfg = payload['cfg']
    model = hydra.utils.instantiate(cfg.policy)
    
    # 加载模型权重（通常存储在 'model' 或 'ema_model' key 下）
    if 'model' in payload['state_dicts']:
        model.load_state_dict(payload['state_dicts']['model'])
    elif 'ema_model' in payload['state_dicts']:
        model.load_state_dict(payload['state_dicts']['ema_model'])
    
    # 如果有 normalizer，也需要加载
    normalizer_path = checkpoint_path.replace('checkpoints/', '').rsplit('/', 1)[0] + '/normalizer.pkl'
    # 或者从 pickles 中恢复
    
    model.eval().to(device)
    return model

def load_diffusion_policy(checkpoint_path, device='cpu'):
    #
    model = torch.load(checkpoint_path, map_location=torch.device(device))
    model.eval().to(device)
    return model

def process_image(img:torch.Tensor):
    
    # 只保留RGB通道
    img = img[...,:3].float()  
    # 调整通道顺序 [H, W, C] -> [C, H, W]
    # img = img.permute(0, 3, 1, 2)  
    # img = img.permute(2, 1, 0)  # 调整通道顺序
    img = img.permute(2, 0, 1)  # 调整通道顺序

    # 归一化到[0,1]并调整尺寸
    img = F.interpolate(
        img.unsqueeze(0) / 255.0,  # 添加batch维度并归一化
        size=(224, 224),           # 调整到模型期望的尺寸
        mode='bilinear',
        align_corners=False
    )

    return img

def process_batch_image(imgs: torch.Tensor, mask_imgs: torch.Tensor = None, with_mask: bool = False):
    """
    处理批量图像，支持 RGB 或 RGBM 模式
    
    Args:
        imgs: RGB 图像张量 [N, H, W, C]
        mask_imgs: mask 图像张量 [N, H, W] 或 [N, H, W, C]，仅在 with_mask=True 时使用
        with_mask: 是否将 mask 通道拼接到 RGB 后（变成 4 通道 RGBM）
        
    Returns:
        处理后的图像张量 [N, C, 224, 224]，C=3(RGB) 或 C=4(RGBM)
    """
    batch_size = imgs.shape[0]
    num_channels = 4 if with_mask else 3
    images_processed = torch.zeros((batch_size, num_channels, 224, 224), 
                                    dtype=torch.float32, device=imgs.device)

    for i in range(batch_size):
        img = imgs[i, ...]
        # 只保留 RGB 通道
        img = img[..., :3].float()  
        # 调整通道顺序 [H, W, C] -> [C, H, W]
        img = img.permute(2, 0, 1)

        # 归一化到 [0,1] 并调整尺寸
        img = F.interpolate(
            img.unsqueeze(0) / 255.0,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        if with_mask and mask_imgs is not None:
            # 处理 mask
            mask = mask_imgs[i, ...]
            if len(mask.shape) == 3:
                # 如果 mask 是多通道的，取第一个通道
                mask = mask[:, :, 0] if mask.shape[2] >= 1 else mask
            # 转为二值 mask（非零值为 1.0）
            mask_binary = (mask > 0).float()
            # 调整尺寸
            mask_resized = F.interpolate(
                mask_binary.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode='nearest'
            )

            import matplotlib.pyplot as plt
            # head_camera_mask = self._robot.tiled_cameras["head_camera"].data.output["instance_segmentation_fast"][0,:,:,:].cpu().numpy()
            # head_camera_mask = self._robot.tiled_cameras["chest_camera"].data.output["instance_segmentation_fast"][0,:,:,:].cpu().numpy()
            # head_camera_depth = self._robot.tiled_cameras["chest_camera"].data.output["depth"][0,:,:,:].cpu().numpy()
            # head_camera_normal = self._robot.tiled_cameras["chest_camera"].data.output["normals"][0,:,:,:].cpu().numpy()
        
            # plt.figure(figsize=(12, 6))
            # # 展示RGB图
            # plt.subplot(1, 2, 1)
            # plt.title("RGB")
            # plt.imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
            # plt.axis('off')
            # # 展示Mask图
            # plt.subplot(1, 2, 2)
            # plt.title("Mask")
            # plt.imshow(mask_resized.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
            # plt.axis('off')
            # plt.show()
            # import time
            # time.sleep(0.1)
            
            # 拼接 RGB 和 Mask 为 4 通道
            images_processed[i, :3, :, :] = img.squeeze(0)
            images_processed[i, 3:4, :, :] = mask_resized.squeeze(0)
        else:
            images_processed[i, :3, :, :] = img.squeeze(0)

    return images_processed


# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

# import os
# import sys
# import yaml  # 用于读取配置文件
# import dill  # 用于加载模型
# import hydra  # 用于加载diffusion policy
# import torch
# import ssl
# import requests
# import torch.nn.functional as F

# # Add Diffusion Policy Project Path
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# # sys.path.append("/home/admin01/Work/02-DiffusionPolicy/diffusion_policy") 
# sys.path.append("/home/psibot/dp_new") 
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.policy.base_image_policy import BaseImagePolicy
# from psi_dp.workspace.train_diffusion_transformer_timm_workspace import TrainDiffusionTransformerTimmWorkspace

# # sys.path.append("/home/psibot/dp_new/diffusion_policy") 


# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.policy.base_image_policy import BaseImagePolicy
# from psi_dp.workspace.train_diffusion_transformer_timm_workspace import TrainDiffusionTransformerTimmWorkspace



# # 1.load policy （base policy and res policy）
# def load_diffusion_policy(checkpoint_path):

#     ssl._create_default_https_context = ssl._create_unverified_context

#     from requests.packages.urllib3.exceptions import InsecureRequestWarning
#     requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

#     # 修改 huggingface_hub 的设置
#     # import osct
#     os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
#     ckpt_path = checkpoint_path
#     payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
#     cfg = payload['cfg']

#     # 打印目标类路径以进行调试
#     print(f"Target class path: {cfg._target_}")

#     # 如果需要，手动修改目标类路径
#     # cfg._target_ = "../../../DexDiffusionPolicy/diffusion_policy/workspace/train_diffusion_transformer_timm_workspace.TrainDiffusionTransformerTimmWorkspace"

#     cls = hydra.utils.get_class(cfg._target_)
#     workspace = cls(cfg)
#     workspace: BaseWorkspace
#     workspace.load_payload(payload, exclude_keys=None, include_keys=None)

#     if 'diffusion' in cfg.name:
#         # diffusion model
#         policy: BaseImagePolicy
#         policy = workspace.model
#         # if cfg.training.use_ema:
#         #     policy = workspace.ema_model

#         device = torch.device('cuda')
#         policy.eval().to(device)

#         # set inference params
#         policy.num_inference_steps = 16 # DDIM inference iterations
#         policy.n_action_steps = 8
#     return policy


# def process_image(img:torch.Tensor):
    
#     # 只保留RGB通道
#     img = img[...,:3].float()  
#     # 调整通道顺序 [H, W, C] -> [C, H, W]
#     # img = img.permute(0, 3, 1, 2)  
#     # img = img.permute(2, 1, 0)  # 调整通道顺序
#     img = img.permute(2, 0, 1)  # 调整通道顺序

#     # 归一化到[0,1]并调整尺寸
#     img = F.interpolate(
#         img.unsqueeze(0) / 255.0,  # 添加batch维度并归一化
#         size=(224, 224),           # 调整到模型期望的尺寸
#         mode='bilinear',
#         align_corners=False
#     )

#     return img

# def process_batch_image(imgs:torch.Tensor):
    
#     images_proccessed = torch.tensor([],dtype=torch.uint8,device= imgs.device)

#     for i in range(imgs.shape[0]):
#         img = imgs[i,...]
#         # 只保留RGB通道
#         img = img[...,:3].float()  
#         # 调整通道顺序 [H, W, C] -> [C, H, W]
#         # img = img.permute(0, 3, 1, 2)  
#         # img = img.permute(2, 1, 0)  # 调整通道顺序
#         img = img.permute(2, 0, 1)  # 调整通道顺序

#         # 归一化到[0,1]并调整尺寸
#         img = F.interpolate(
#             img.unsqueeze(0) / 255.0,  # 添加batch维度并归一化
#             size=(224, 224),           # 调整到模型期望的尺寸
#             mode='bilinear',
#             align_corners=False
#         )

#         images_proccessed = torch.cat((images_proccessed,img),dim=0)

#     return images_proccessed