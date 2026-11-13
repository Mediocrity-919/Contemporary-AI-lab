"""损失函数（自行实现，统一使用“软标签交叉熵”）。

- ce：普通交叉熵，目标是 one-hot 分布。
- ls：标签平滑（Szegedy et al., 2016）。目标分布为 (1-ε)·one-hot + ε/K，
  防止模型把 logit 推得过大、过度自信。
- 知识蒸馏（Hinton et al., 2015）：
      L = (1-α)·L_hard + α·T²·KL(p_teacher^T ‖ p_student^T)
  其中 p^T = softmax(z/T)；乘 T² 使软目标项的梯度量级不随 T 变化。
- Mixup / CutMix：在 batch 上混合图像与软标签（P2 扩展）。

在 fp32 中计算损失，避免 AMP 下 log_softmax 的精度问题；
手写的软标签交叉熵在 CUDA 上反向传播是确定性的。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def make_targets(labels: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    onehot = F.one_hot(labels, num_classes).float()
    return onehot * (1.0 - smoothing) + smoothing / num_classes


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = temperature
    log_p_s = F.log_softmax(student_logits.float() / t, dim=1)
    log_p_t = F.log_softmax(teacher_logits.float() / t, dim=1)
    return (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=1).mean() * t * t


class TrainingLoss:
    """根据配置组合训练目标。"""

    def __init__(self, lcfg: dict, num_classes: int):
        self.num_classes = num_classes
        self.smoothing = lcfg.get("smoothing", 0.1) if lcfg["type"] == "ls" else 0.0
        if lcfg["type"] not in ("ce", "ls"):
            raise ValueError(f"未知损失类型: {lcfg['type']}")
        kd = lcfg.get("kd") or {}
        self.kd_alpha = float(kd.get("alpha", 0.0))
        self.kd_temperature = float(kd.get("T", 1.0))

    def targets(self, labels: torch.Tensor) -> torch.Tensor:
        return make_targets(labels, self.num_classes, self.smoothing)

    def __call__(self, logits, targets, teacher_logits=None) -> torch.Tensor:
        hard = soft_cross_entropy(logits, targets)
        if teacher_logits is None or self.kd_alpha == 0.0:
            return hard
        soft = kd_kl(logits, teacher_logits, self.kd_temperature)
        return (1.0 - self.kd_alpha) * hard + self.kd_alpha * soft


def apply_mix(x: torch.Tensor, targets: torch.Tensor, kind: str, alpha: float, generator: torch.Generator):
    """Mixup：按 λ~Beta(α,α) 线性混合两张图；CutMix：把另一张图的矩形块贴进来，λ 取实际面积比。"""
    if kind == "none":
        return x, targets
    lam = float(np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=generator))).beta(alpha, alpha))
    perm = torch.randperm(x.size(0), generator=generator).to(x.device)
    if kind == "mixup":
        x = lam * x + (1 - lam) * x[perm]
    elif kind == "cutmix":
        h, w = x.shape[-2:]
        cut_h, cut_w = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
        cy = int(torch.randint(0, h, (1,), generator=generator))
        cx = int(torch.randint(0, w, (1,), generator=generator))
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"未知混合方式: {kind}")
    return x, lam * targets + (1 - lam) * targets[perm]
