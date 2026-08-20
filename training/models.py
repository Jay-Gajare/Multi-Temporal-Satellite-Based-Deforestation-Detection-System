"""
Model architectures for binary deforestation classification.

All models share the same interface:
  - Input: (B, C, 64, 64) where C = in_channels
  - Output: (B,) raw logits
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


def _make_resnet(backbone: str, in_channels: int, pretrained: bool, dropout: float) -> nn.Module:
    """Build a ResNet variant adapted for arbitrary input channels."""
    factory = getattr(models, backbone)
    weights = "IMAGENET1K_V1" if pretrained else None
    net = factory(weights=weights)

    # Replace first conv to accept in_channels instead of 3
    old_conv = net.conv1
    net.conv1 = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    if pretrained and in_channels != 3:
        # Initialize new channels by averaging pretrained weights
        with torch.no_grad():
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
            net.conv1.weight = nn.Parameter(
                mean_w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
            )

    # Replace classifier head
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return net


def _make_efficientnet(in_channels: int, pretrained: bool, dropout: float) -> nn.Module:
    """Build EfficientNet-B0 adapted for arbitrary input channels."""
    weights = "IMAGENET1K_V1" if pretrained else None
    net = models.efficientnet_b0(weights=weights)

    # Replace first conv
    old_conv = net.features[0][0]
    net.features[0][0] = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    if pretrained and in_channels != 3:
        with torch.no_grad():
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
            net.features[0][0].weight = nn.Parameter(
                mean_w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
            )

    # Replace classifier
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 1),
    )
    return net


def build_model(
    architecture: str,
    in_channels: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build a classification model.

    Parameters
    ----------
    architecture : str
        One of: resnet18, resnet50, efficientnet_b0
    in_channels : int
        Number of input channels (9, 108, etc.)
    pretrained : bool
        Use ImageNet pretrained weights
    dropout : float
        Dropout rate in classifier head

    Returns
    -------
    nn.Module
        Model outputting raw logits (B,)
    """
    arch = architecture.lower().replace("-", "_")

    if arch in ("resnet18", "resnet50"):
        net = _make_resnet(arch, in_channels, pretrained, dropout)
    elif arch in ("efficientnet_b0", "efficientnetb0"):
        net = _make_efficientnet(in_channels, pretrained, dropout)
    else:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. "
            f"Choose from: resnet18, resnet50, efficientnet_b0"
        )

    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    logger.info("Built %s — %.2fM params, in_channels=%d", architecture, n_params, in_channels)
    return net
