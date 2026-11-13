"""模型构建入口与通用辅助函数。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ourcnn import OurCNN
from .resnet import resnet18_cifar, resnet18_imagenet, resnet_forward_features

HEAD_PREFIX = "fc."  # 所有模型的分类层都命名为 fc


def build_model(mcfg: dict) -> nn.Module:
    arch = mcfg["arch"]
    if arch == "ourcnn":
        model = OurCNN(
            num_classes=mcfg["num_classes"],
            attention=mcfg.get("attention", "none"),
            se_reduction=mcfg.get("se_reduction", 16),
            batchnorm=mcfg.get("batchnorm", True),
            dropout=mcfg.get("dropout", 0.3),
        )
    elif arch == "resnet18_cifar":
        model = resnet18_cifar(mcfg["num_classes"])
    elif arch == "resnet18_imagenet":
        model = resnet18_imagenet(mcfg["num_classes"], pretrained=mcfg.get("pretrained", False))
    else:
        raise ValueError(f"未知模型: {arch}")
    if mcfg.get("freeze_backbone", False):
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(HEAD_PREFIX)
    return model


def set_train_mode(model: nn.Module, freeze_backbone: bool) -> None:
    """进入训练模式；冻结主干时主干保持 eval，使 BN 的 running mean/var 不被更新。"""
    model.train()
    if freeze_backbone:
        for name, child in model.named_children():
            if name != HEAD_PREFIX.rstrip("."):
                child.eval()


def forward_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    return resnet_forward_features(model, x)


def get_module(model: nn.Module, dotted: str) -> nn.Module:
    module = model
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
