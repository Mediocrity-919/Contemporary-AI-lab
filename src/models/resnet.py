"""ResNet-18 的两种用法（均基于 torchvision 实现）。

- resnet18_cifar：为 32×32 输入调整的版本。首层改为 3×3、stride 1，去掉最大池化，
  使 layer1–4 的特征图为 32/16/8/4，避免小图被过早下采样。从头训练。
- resnet18_imagenet：标准结构（7×7 stride 2 首层 + 最大池化），可加载 ImageNet 预训练权重；
  分类层替换为 512→10。配合 224 输入使用（特征图 56/28/14/7）。

两者都把 avgpool 换成 GlobalAvgPool（数值等价，但反向传播是确定性的）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet, ResNet18_Weights, resnet18


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


def resnet18_cifar(num_classes: int = 10) -> ResNet:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    model.avgpool = GlobalAvgPool()
    model.cam_layer = "layer3"  # 8×8；layer4 只有 4×4
    return model


def resnet18_imagenet(num_classes: int = 10, pretrained: bool = False) -> ResNet:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.avgpool = GlobalAvgPool()
    model.cam_layer = "layer4"  # 224 输入时为 7×7
    return model


def resnet_forward_features(model: ResNet, x: torch.Tensor) -> torch.Tensor:
    """ResNet 分类层之前的 512 维特征。"""
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    return torch.flatten(model.avgpool(x), 1)
