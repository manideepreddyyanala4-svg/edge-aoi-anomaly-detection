from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.config import Config


def _get_weights(backbone_name: str, use_pretrained: bool):
    """
    Return torchvision weights enum if pretrained is requested.
    Falls back to None if pretrained is False.
    """
    if not use_pretrained:
        return None

    backbone_name = backbone_name.lower()
    if backbone_name == "wide_resnet50_2":
        return models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
    if backbone_name == "resnet18":
        return models.ResNet18_Weights.IMAGENET1K_V1

    raise ValueError(f"Unsupported backbone: {backbone_name}")


def build_backbone(config: Config) -> nn.Module:
    """
    Build the feature backbone and remove the classification head.
    """
    backbone_name = config.backbone.lower()
    weights = _get_weights(backbone_name, config.use_pretrained)

    if backbone_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(weights=weights)
    elif backbone_name == "resnet18":
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {config.backbone}")

    model.fc = nn.Identity()
    return model


class FeatureExtractor(nn.Module):
    """
    Backbone wrapper that captures layer2 and layer3 features
    using forward hooks.

    Output:
        - layer2: [B, C2, H2, W2]
        - layer3: [B, C3, H3, W3]
        - fused : [B, C2 + C3, H3, W3] after pooling layer2 -> layer3 size
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config)

        self._features: Dict[str, torch.Tensor] = {}
        self._hooks = []

        self._register_hooks()

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def _hook_fn(self, name: str):
        def fn(module, input, output):
            self._features[name] = output
        return fn

    def _register_hooks(self) -> None:
        """
        Attach hooks to layer2 and layer3.
        """
        if not hasattr(self.backbone, "layer2") or not hasattr(self.backbone, "layer3"):
            raise AttributeError("Backbone does not expose layer2/layer3 as expected.")

        self._hooks.append(self.backbone.layer2.register_forward_hook(self._hook_fn("layer2")))
        self._hooks.append(self.backbone.layer3.register_forward_hook(self._hook_fn("layer3")))

    def remove_hooks(self) -> None:
        """Remove registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass that returns intermediate features.
        """
        self._features = {}

        _ = self.backbone.conv1(x)
        _ = self.backbone.bn1(_)
        _ = self.backbone.relu(_)
        _ = self.backbone.maxpool(_)

        _ = self.backbone.layer1(_)
        _ = self.backbone.layer2(_)
        _ = self.backbone.layer3(_)

        return self._features

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convenience wrapper for inference usage.
        """
        self.backbone.eval()
        return self.forward(x)

    @torch.no_grad()
    def extract_fused_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract layer2 and layer3, pool layer2 to match layer3 spatial dimensions,
        concatenate along channel axis, and L2-normalize along the channel axis.

        Returns:
            fused feature map of shape [B, C2 + C3, H3, W3]
        """
        feats = self.extract_features(x)
        if "layer2" not in feats or "layer3" not in feats:
            raise RuntimeError("Failed to capture layer2/layer3 features.")

        layer2 = feats["layer2"]
        layer3 = feats["layer3"]

        if layer2.shape[-2:] != layer3.shape[-2:]:
            layer2 = F.adaptive_avg_pool2d(layer2, output_size=layer3.shape[-2:])

        fused = torch.cat([layer2, layer3], dim=1)

        fused = F.normalize(fused, p=2, dim=1)
        return fused

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device


def fuse_patch_embeddings(fused_map: torch.Tensor) -> torch.Tensor:
    """
    Convert fused feature map [B, C, H, W] into patch embeddings [B*H*W, C].
    """
    if fused_map.dim() != 4:
        raise ValueError("Expected a 4D tensor [B, C, H, W].")
    b, c, h, w = fused_map.shape
    patches = fused_map.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
    return patches


def get_feature_grid_size(x: torch.Tensor, extractor: FeatureExtractor) -> Tuple[int, int]:
    """
    Utility to get the spatial grid size of the fused feature map.
    """
    with torch.no_grad():
        fused = extractor.extract_fused_feature_map(x)
    return fused.shape[-2], fused.shape[-1]
