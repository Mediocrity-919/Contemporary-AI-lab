"""注意力模块（自行实现）。

- SEBlock：Squeeze-and-Excitation（Hu et al., 2018）。全局平均池化得到每个通道的描述，
  经两层全连接 + Sigmoid 得到 0–1 的通道权重，再按通道缩放特征图。
- CBAM：通道注意力（平均 + 最大池化共享 MLP）后接空间注意力（7×7 卷积），
  Woo et al., 2018。属于 P2 扩展。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, min_hidden: int = 8):
        super().__init__()
        hidden = max(channels // reduction, min_hidden)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.last_weights = None  # 最近一次前向的通道权重 [N, C]，供可解释性分析读取

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.mean(dim=(2, 3))  # 用 mean 代替 AdaptiveAvgPool2d：反向传播是确定性的
        weights = torch.sigmoid(self.fc2(torch.relu(self.fc1(squeeze))))
        self.last_weights = weights.detach()
        return x * weights[:, :, None, None]


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, min_hidden: int = 8, kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, min_hidden)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.last_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = torch.sigmoid(self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3))))
        self.last_weights = channel.detach()
        x = x * channel[:, :, None, None]
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(pooled))


def make_attention(kind: str, channels: int, reduction: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "se":
        return SEBlock(channels, reduction)
    if kind == "cbam":
        return CBAM(channels, reduction)
    raise ValueError(f"未知注意力类型: {kind}")
