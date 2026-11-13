"""自定义 CNN（OurCNN），结构见 plan.md 第 4 节。

输入 3×32×32
  stage1: [Conv3×3-BN-ReLU]×2, 64 通道  (+注意力) → MaxPool → 16×16
  stage2: [Conv3×3-BN-ReLU]×2, 128 通道 (+注意力) → MaxPool → 8×8
  stage3: [Conv3×3-BN-ReLU]×2, 256 通道 (+注意力)            ← Grad-CAM 目标层（8×8）
          → MaxPool → 4×4 → 全局平均池化 → Dropout → FC(256→10)

设计考虑：
- 3×3 小卷积堆叠（VGG 思路）：两层 3×3 的感受野等于一层 5×5，但参数更少、非线性更多。
- BN 加速收敛并有一定正则作用；卷积后接 BN 时卷积不需要 bias。
- 用全局平均池化代替大的全连接层，参数量约 1.15M，过拟合风险更小。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import make_attention


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, batchnorm: bool = True):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=not batchnorm),
            nn.BatchNorm2d(cout) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )


class Stage(nn.Module):
    def __init__(self, cin: int, cout: int, attention: str, reduction: int, batchnorm: bool):
        super().__init__()
        self.conv1 = ConvBNReLU(cin, cout, batchnorm)
        self.conv2 = ConvBNReLU(cout, cout, batchnorm)
        self.attn = make_attention(attention, cout, reduction)

    def forward(self, x):
        return self.attn(self.conv2(self.conv1(x)))


class OurCNN(nn.Module):
    cam_layer = "stage3"  # Grad-CAM 默认目标层

    def __init__(self, num_classes=10, widths=(64, 128, 256), attention="none", se_reduction=16,
                 batchnorm=True, dropout=0.3):
        super().__init__()
        c1, c2, c3 = widths
        self.stage1 = Stage(3, c1, attention, se_reduction, batchnorm)
        self.stage2 = Stage(c1, c2, attention, se_reduction, batchnorm)
        self.stage3 = Stage(c2, c3, attention, se_reduction, batchnorm)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回分类层之前的 256 维特征（t-SNE 使用）。"""
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.pool(self.stage3(x))
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))
