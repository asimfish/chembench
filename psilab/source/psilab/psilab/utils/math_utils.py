# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-11-10
# Vesion: 1.0


import torch


def compute_angle_vector_plane(vector: torch.Tensor, plane_normal_vector: torch.Tensor)->torch.Tensor:
    """
    Compute angle between vector and plane
    """

    # normalize vector, plane_normal_vector is already normalized
    vector_norm = vector / torch.norm(vector, dim=-1, keepdim=True)  

    # compute the dot product between the vector and the plane normal vector
    dot_product = torch.bmm(vector_norm.unsqueeze(1), plane_normal_vector.unsqueeze(2)).squeeze()  
    
    # clamp the dot product to avoid numerical issues with acos
    dot_product_clamped = torch.clamp(dot_product, -1.0, 1.0)
    
    # compute the angle between the vector and the plane normal vector
    angle_normal = torch.acos(dot_product_clamped)  
    
    # compute the angle between the vector and the plane
    angle = torch.pi / 2 - angle_normal  
    
    return angle

def normalize_v1(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor)->torch.Tensor:
    """
    normalize [lower,upper] to [-1,1]
    """
    return 2.0 * (value-lower) / (upper - lower) - 1.0

def normalize_v2(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor)->torch.Tensor:
    """
    normalize [lower,upper] to [0,1]
    """
    return (value-lower) / (upper - lower)

def unnormalize_v1(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor)->torch.Tensor:
    """
    normallize [-1,1]->[lower,upper]
    """
    return (0.5 * (value + 1.0) * (upper - lower) + lower)

def unnormalize_v2(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor)->torch.Tensor:
    """
    normallize [0,1]->[lower,upper]
    """
    return value * (upper-lower) + lower

def clamp(value: torch.Tensor, value_min: torch.Tensor, value_max: torch.Tensor)->torch.Tensor:
    return torch.max(torch.min(value, value_max), value_min)
