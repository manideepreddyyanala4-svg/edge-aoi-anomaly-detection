from typing import Tuple, Optional

import torch
from torchvision import transforms
from PIL import Image

from src.config import Config


def build_image_transform(config: Config) -> transforms.Compose:
    """
    Standard preprocessing for both train and test images:
    Resize -> CenterCrop -> ToTensor -> Normalize
    """
    return transforms.Compose(
        [
            transforms.Resize((config.resize_size, config.resize_size)),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
        ]
    )


def build_mask_transform(config: Config) -> transforms.Compose:
    """
    Mask preprocessing for pixel-level evaluation.
    Keep masks binary and aligned with the image pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((config.resize_size, config.resize_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
        ]
    )


def preprocess_image(image: Image.Image, config: Config) -> torch.Tensor:
    """
    Preprocess a PIL RGB image into a normalized tensor.
    Returns tensor with shape [3, H, W].
    """
    transform = build_image_transform(config)
    return transform(image)


def preprocess_mask(mask: Image.Image, config: Config) -> torch.Tensor:
    """
    Preprocess a PIL mask into a tensor.
    Returns tensor with shape [1, H, W].
    """
    transform = build_mask_transform(config)
    mask_tensor = transform(mask)
    mask_tensor = (mask_tensor > 0.5).float()
    return mask_tensor


def tensor_to_image_range(x: torch.Tensor) -> torch.Tensor:
    """
    Utility to bring normalized tensors back to a visible 0-1 range if needed.
    This is only for visualization, not model input.
    """
    return torch.clamp(x, 0.0, 1.0)


def denormalize_image(tensor: torch.Tensor, config: Config) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization.
    Input: [3, H, W]
    Output: [3, H, W] in approx 0-1 range
    """
    mean = torch.tensor(config.imagenet_mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(config.imagenet_std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def preprocess_batch(images: Tuple[Image.Image, ...], config: Config) -> torch.Tensor:
    """
    Preprocess a batch of PIL images into a single tensor [N, 3, H, W].
    Useful for memory-bank building and inference acceleration.
    """
    transform = build_image_transform(config)
    tensors = [transform(img) for img in images]
    return torch.stack(tensors, dim=0)
