"""数据读取、增强与批处理。

数据来自 prepare_data.py 生成的 data/{train,val,test}.pt（uint8，NCHW，RGB，0–255）。

- 训练集：在 DataLoader worker 中逐张增强。每张图的随机数种子由 (运行种子, epoch, 样本索引)
  唯一确定，因此增强结果与 worker 数量无关，中断后续训也会得到相同的增强序列。
- 验证集 / 测试集 / 训练子集：不做增强，一次性归一化后放在 GPU 上按批推理。
- 224 输入的模型：先在 32×32 上完成增强和归一化，再在 GPU 上双线性放大
  （逐通道归一化是线性变换，与双线性插值可交换，顺序不影响结果）。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info
from torchvision.transforms import v2

from .utils import ROOT

# 只用训练集（45,000 张）计算的统计量，见 plan.md 第 1 节。
CIFAR_MEAN = (0.4913, 0.4819, 0.4461)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
# torchvision ResNet-18 IMAGENET1K_V1 权重对应的归一化参数。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLEAN_SUBSET_SEED = 2026  # 固定的“干净训练子集”抽样种子，与运行种子无关


def norm_stats(name: str):
    if name == "cifar":
        return CIFAR_MEAN, CIFAR_STD
    if name == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    raise ValueError(f"未知归一化方式: {name}")


def load_split(split: str, root: str | Path = "data"):
    """返回 (images uint8 [N,3,32,32], labels int64 [N], class_names)。"""
    path = (ROOT / root / f"{split}.pt") if not Path(root).is_absolute() else Path(root) / f"{split}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["images"], payload["labels"], payload["class_names"]


def build_train_transform(aug: str, norm: str) -> v2.Compose:
    """三档增强（参数事先固定，不在验证集上调）：
    none   ：只做归一化
    basic  ：RandomCrop(32, padding=4) + 水平翻转
    strong ：RandAugment(2, 9) → basic → 归一化 → RandomErasing(p=0.25)
    """
    mean, std = norm_stats(norm)
    ops = []
    if aug == "strong":
        ops.append(v2.RandAugment(num_ops=2, magnitude=9))
    if aug in ("basic", "strong"):
        ops += [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
    elif aug != "none":
        raise ValueError(f"未知增强档位: {aug}")
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    if aug == "strong":
        ops.append(v2.RandomErasing(p=0.25, value=0))
    return v2.Compose(ops)


def normalize_images(images_uint8: torch.Tensor, norm: str) -> torch.Tensor:
    """uint8 [N,3,H,W] → 归一化后的 float32。"""
    mean, std = norm_stats(norm)
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (images_uint8.float() / 255.0 - mean) / std


def denormalize(x: torch.Tensor, norm: str) -> torch.Tensor:
    """归一化后的张量 → [0,1] 像素（用于可视化）。"""
    mean, std = norm_stats(norm)
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def renormalize(x: torch.Tensor, src: str, dst: str) -> torch.Tensor:
    """在两种归一化之间转换（知识蒸馏时教师与学生的预处理可能不同）。"""
    if src == dst:
        return x
    s_mean, s_std = norm_stats(src)
    d_mean, d_std = norm_stats(dst)
    as_t = lambda v: torch.tensor(v, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * as_t(s_std) + as_t(s_mean) - as_t(d_mean)) / as_t(d_std)


def resize_batch(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def sample_seed(run_seed: int, epoch: int, index: int) -> int:
    """每张训练图在每个 epoch 的增强种子；index < 1e5、epoch < 1e3，保证互不相同。"""
    return (run_seed * 1000 + epoch) * 100_000 + index


class AugmentedDataset(Dataset):
    """按 (样本索引, epoch) 取图，并用由二者决定的种子做增强。"""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform, run_seed: int):
        self.images, self.labels = images, labels
        self.transform, self.run_seed = transform, run_seed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        index, epoch = key
        seed = sample_seed(self.run_seed, epoch, index)
        if get_worker_info() is None:
            # 主进程中取数时不能打乱全局随机状态（会影响 Dropout 等）。
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                image = self.transform(self.images[index])
        else:
            torch.default_generator.manual_seed(seed)
            image = self.transform(self.images[index])
        return image, self.labels[index]


class EpochSampler(Sampler):
    """每个 epoch 的打乱顺序只由 (运行种子, epoch) 决定；产出 (索引, epoch)。"""

    def __init__(self, n: int, run_seed: int):
        self.n, self.run_seed, self.epoch = n, run_seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.run_seed * 100_003 + self.epoch)
        for index in torch.randperm(self.n, generator=generator).tolist():
            yield index, self.epoch


def make_train_loader(cfg: dict, run_seed: int):
    dcfg, scfg = cfg["data"], cfg.get("system", {})
    images, labels, _ = load_split("train", dcfg["root"])
    dataset = AugmentedDataset(images, labels, build_train_transform(dcfg["aug"], dcfg["norm"]), run_seed)
    sampler = EpochSampler(len(dataset), run_seed)
    workers = int(scfg.get("num_workers", 6))
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        drop_last=False,
    )
    return loader, sampler


def stratified_subset(labels: torch.Tensor, per_class: int, seed: int = CLEAN_SUBSET_SEED) -> torch.Tensor:
    """每类随机取 per_class 个索引（固定种子），返回排序后的索引。"""
    generator = torch.Generator().manual_seed(seed)
    parts = []
    for c in torch.unique(labels).tolist():
        idx = torch.where(labels == c)[0]
        parts.append(idx[torch.randperm(len(idx), generator=generator)[:per_class]])
    return torch.cat(parts).sort().values


class EvalSet:
    """不做增强的评估集合：归一化后的图像常驻显存，按批产出。"""

    def __init__(self, images_uint8: torch.Tensor, labels: torch.Tensor, norm: str, device, indices=None):
        if indices is not None:
            images_uint8, labels = images_uint8[indices], labels[indices]
        self.x = normalize_images(images_uint8, norm).to(device)
        self.y = labels.to(device)
        self.indices = indices

    def __len__(self):
        return len(self.y)

    def batches(self, batch_size: int):
        for start in range(0, len(self.y), batch_size):
            yield self.x[start:start + batch_size], self.y[start:start + batch_size]


def make_eval_set(split: str, cfg: dict, device, per_class: int | None = None) -> EvalSet:
    images, labels, _ = load_split(split, cfg["data"]["root"])
    indices = stratified_subset(labels, per_class) if per_class else None
    return EvalSet(images, labels, cfg["data"]["norm"], device, indices)


def class_names(root: str = "data") -> list[str]:
    return load_split("val", root)[2]
