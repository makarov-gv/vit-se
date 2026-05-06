
"""
Predefined models of Vision Transformer with Sparse Encoder (ViT/SE). The same configurations as demonstrated in 
https://arxiv.org/abs/2010.11929, weights are fully compatible too.

Modified from torchvision (https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py).
Originally licensed under BSD 3-Clause License.
"""
from .vit_se import *
from .mlp import *

from typing import Any, Optional

from torchvision.models.vision_transformer import \
    ViT_B_16_Weights, ViT_B_32_Weights, \
    ViT_L_16_Weights, ViT_L_32_Weights, \
    ViT_H_14_Weights


def _load_pretrained(model: VisionTransformer, state_dict: dict[str, torch.Tensor]) -> None:
    target_classes = model.num_classes
    head_weight = state_dict.get('heads.head.weight')
    if head_weight is not None and head_weight.shape[0] != target_classes:
        state_dict = state_dict.copy()
        state_dict.pop('heads.head.weight', None)
        state_dict.pop('heads.head.bias', None)
        model.load_state_dict(state_dict, strict=False)
        return

    model.load_state_dict(state_dict)


def vit_se_b_16(pretrained: bool = True, weights: Optional[str] = None, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a VIT/SE-B/16 based on original VIT-B/16 architecture as per https://arxiv.org/abs/2010.11929. Bias in 
    convolutional embedding dimension projection is removed if present.
    """
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs
    )

    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        weights = ViT_B_16_Weights.verify(weights)
        state_dict = weights.get_state_dict(check_hash=True)
    else:
        state_dict = torch.load(weights)

    if 'conv_proj.bias' in state_dict.keys():
        state_dict.pop('conv_proj.bias')
    _load_pretrained(model, state_dict)

    return model


def vit_se_b_32(pretrained: bool = True, weights: Optional[str] = None, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a VIT/SE-B/32 based on original VIT-B/32 architecture as per https://arxiv.org/abs/2010.11929. Bias in 
    convolutional embedding dimension projection is removed if present.
    """
    model = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs
    )

    if pretrained:
        weights = ViT_B_32_Weights.DEFAULT
        weights = ViT_B_32_Weights.verify(weights)
        state_dict = weights.get_state_dict(check_hash=True)
    else:
        state_dict = torch.load(weights)

    if 'conv_proj.bias' in state_dict.keys():
        state_dict.pop('conv_proj.bias')
    _load_pretrained(model, state_dict)

    return model


def vit_se_l_16(pretrained: bool = True, weights: Optional[str] = None, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a VIT/SE-L/16 based on original VIT-L/16 architecture as per https://arxiv.org/abs/2010.11929. Bias in 
    convolutional embedding dimension projection is removed if present.
    """
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs
    )

    if pretrained:
        weights = ViT_L_16_Weights.DEFAULT
        weights = ViT_L_16_Weights.verify(weights)
        state_dict = weights.get_state_dict(check_hash=True)
    else:
        state_dict = torch.load(weights)

    if 'conv_proj.bias' in state_dict.keys():
        state_dict.pop('conv_proj.bias')
    _load_pretrained(model, state_dict)

    return model


def vit_se_l_32(pretrained: bool = True, weights: Optional[str] = None, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a VIT/SE-L/32 based on original VIT-L/32 architecture as per https://arxiv.org/abs/2010.11929. Bias in 
    convolutional embedding dimension projection is removed if present.
    """
    model = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs
    )

    if pretrained:
        weights = ViT_L_32_Weights.DEFAULT
        weights = ViT_L_32_Weights.verify(weights)
        state_dict = weights.get_state_dict(check_hash=True)
    else:
        state_dict = torch.load(weights)

    if 'conv_proj.bias' in state_dict.keys():
        state_dict.pop('conv_proj.bias')
    _load_pretrained(model, state_dict)

    return model


def vit_se_h_14(pretrained: bool = True, weights: Optional[str] = None, **kwargs: Any) -> VisionTransformer:
    """
    Constructs a VIT/SE-H/14 based on original VIT-H/14 architecture as per https://arxiv.org/abs/2010.11929. Bias in 
    convolutional embedding dimension projection is removed if present.
    """
    model = VisionTransformer(
        image_size=518,
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        **kwargs
    )

    if pretrained:
        weights = ViT_H_14_Weights.DEFAULT
        weights = ViT_H_14_Weights.verify(weights)
        state_dict = weights.get_state_dict(check_hash=True)
    else:
        state_dict = torch.load(weights)

    if 'conv_proj.bias' in state_dict.keys():
        state_dict.pop('conv_proj.bias')
    _load_pretrained(model, state_dict)

    return model
