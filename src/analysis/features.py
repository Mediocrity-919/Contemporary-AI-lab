"""特征层面的可解释性分析（plan.md 6.3 节）。

    python -m src.analysis.features tsne     configs/cnn/F_final.yaml:OurCNN configs/transfer/E7_ft.yaml:FT
    python -m src.analysis.features filters  configs/cnn/F_final.yaml:OurCNN
    python -m src.analysis.features fmaps    configs/cnn/F_final.yaml:OurCNN
    python -m src.analysis.features se       configs/cnn/E2_C3_se.yaml:OurCNN+SE

- t-SNE：测试集按固定种子每类抽 200 张，取分类层之前的特征降到 2 维。
- 第一层卷积核：把 3×3×3 的卷积核按 RGB 显示。
- 中间特征图：各 stage 输出中激活最强的若干通道。
- SE 通道权重：每个 stage 中各类别的平均通道权重，以及类别之间的余弦相似度。
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from .. import data as D
from ..evaluate import load_trained
from ..models import forward_features
from ..models.attention import SEBlock
from .common import Runs, parse_spec, save_fig

SAMPLE_SEED = 0


def _load(spec):
    runs = Runs.load(*parse_spec(spec))
    seed = runs.representative_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained(runs.cfg, runs.dirs[seed], "best", device)
    return runs, seed, model, device


@torch.no_grad()
def tsne(specs, per_class=200):
    images, labels, names = D.load_split("test")
    idx = D.stratified_subset(labels, per_class, seed=SAMPLE_SEED)
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 5.6), squeeze=False)
    for ax, spec in zip(axes[0], specs):
        runs, seed, model, device = _load(spec)
        x = D.normalize_images(images[idx], runs.cfg["data"]["norm"]).to(device)
        feats = torch.cat([forward_features(model, D.resize_batch(x[i:i + 250], runs.cfg["data"]["input_size"])).float()
                           for i in range(0, len(x), 250)]).cpu().numpy()
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=SAMPLE_SEED).fit_transform(feats)
        for c in range(10):
            m = labels[idx].numpy() == c
            ax.scatter(emb[m, 0], emb[m, 1], s=4, label=names[c], alpha=0.7)
        ax.set(title=f"{runs.label}（seed {seed}，特征维度 {feats.shape[1]}）", xticks=[], yticks=[])
    axes[0, -1].legend(markerscale=3, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(f"倒数第二层特征的 t-SNE（测试集每类 {per_class} 张）", fontsize=10)
    save_fig(fig, "tsne_" + "_".join(Runs.load(*parse_spec(s)).label for s in specs) + ".png")


def filters(spec):
    runs, seed, model, _ = _load(spec)
    first = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    w = first.weight.detach().float().cpu()
    w = (w - w.amin(dim=(1, 2, 3), keepdim=True)) / (w.amax(dim=(1, 2, 3), keepdim=True) - w.amin(dim=(1, 2, 3), keepdim=True) + 1e-8)
    n = w.shape[0]
    cols = 16
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 0.6, np.ceil(n / cols) * 0.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), w):
        ax.imshow(k.permute(1, 2, 0).numpy(), interpolation="nearest")
    fig.suptitle(f"{runs.label} 第一层卷积核（{tuple(first.weight.shape)}，逐个归一化后按 RGB 显示）", fontsize=9)
    save_fig(fig, f"{runs.label}_first_layer_filters.png")


@torch.no_grad()
def feature_maps(spec, n_images=3, n_channels=8):
    runs, seed, model, device = _load(spec)
    layers = ["stage1", "stage2", "stage3"] if hasattr(model, "stage1") else ["layer1", "layer2", "layer3", "layer4"]
    images, labels, names = D.load_split("test")
    idx = D.stratified_subset(labels, 1, seed=SAMPLE_SEED)[:n_images]
    outs = {}
    hooks = [getattr(model, name).register_forward_hook(lambda m, i, o, name=name: outs.__setitem__(name, o.float()))
             for name in layers]
    x = D.normalize_images(images[idx], runs.cfg["data"]["norm"]).to(device)
    model(D.resize_batch(x, runs.cfg["data"]["input_size"]))
    for h in hooks:
        h.remove()
    fig, axes = plt.subplots(n_images * len(layers), n_channels + 1,
                             figsize=((n_channels + 1) * 0.95, n_images * len(layers) * 1.0))
    for i, k in enumerate(idx.tolist()):
        for j, name in enumerate(layers):
            row = i * len(layers) + j
            fmap = outs[name][i]
            top = fmap.mean(dim=(1, 2)).topk(n_channels).indices
            axes[row, 0].imshow(images[k].permute(1, 2, 0))
            axes[row, 0].set_title(f"{names[labels[k]]} / {name}", fontsize=6)
            for c, ch in enumerate(top.tolist(), 1):
                axes[row, c].imshow(fmap[ch].cpu(), cmap="viridis")
                axes[row, c].set_title(f"ch{ch} {tuple(fmap.shape[1:])}", fontsize=5)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{runs.label} 各 stage 平均激活最强的 {n_channels} 个通道", fontsize=9)
    save_fig(fig, f"{runs.label}_feature_maps.png")


@torch.no_grad()
def se_weights(spec):
    runs, seed, model, device = _load(spec)
    blocks = [(name, m) for name, m in model.named_modules() if isinstance(m, SEBlock)]
    if not blocks:
        raise SystemExit("该模型没有 SE 模块")
    images, labels, names = D.load_split("test")
    es = D.EvalSet(images, labels, runs.cfg["data"]["norm"], device)
    collected = {name: [] for name, _ in blocks}
    for x, _ in es.batches(500):
        model(x)
        for name, m in blocks:
            collected[name].append(m.last_weights.float().cpu())
    fig, axes = plt.subplots(2, len(blocks), figsize=(5.5 * len(blocks), 7.5))
    for j, (name, _) in enumerate(blocks):
        w = torch.cat(collected[name])
        per_class = torch.stack([w[labels == c].mean(0) for c in range(10)]).numpy()
        order = np.argsort(per_class.std(0))[::-1]  # 按类别间差异从大到小排列通道
        im = axes[0, j].imshow(per_class[:, order], aspect="auto", cmap="magma")
        axes[0, j].set(title=f"{name}：各类平均通道权重\n（通道按类间标准差降序）", xlabel="通道", yticks=range(10))
        axes[0, j].set_yticklabels(names, fontsize=7)
        axes[0, j].grid(False)
        fig.colorbar(im, ax=axes[0, j], fraction=0.046)
        normed = per_class / np.linalg.norm(per_class, axis=1, keepdims=True)
        sim = normed @ normed.T
        im = axes[1, j].imshow(sim, cmap="viridis")
        axes[1, j].set(title=f"{name}：类别间余弦相似度（均值 {sim[~np.eye(10, dtype=bool)].mean():.3f}）")
        axes[1, j].set_xticks(range(10), names, rotation=60, fontsize=7)
        axes[1, j].set_yticks(range(10), names, fontsize=7)
        axes[1, j].grid(False)
        fig.colorbar(im, ax=axes[1, j], fraction=0.046)
    fig.suptitle(f"{runs.label} 的 SE 通道权重（测试集，seed {seed}）", fontsize=10)
    fig.tight_layout()
    save_fig(fig, f"{runs.label}_se_weights.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tsne")
    p.add_argument("specs", nargs="+")
    for cmd in ("filters", "fmaps", "se"):
        p = sub.add_parser(cmd)
        p.add_argument("spec")
    args = parser.parse_args()
    {"tsne": lambda: tsne(args.specs), "filters": lambda: filters(args.spec),
     "fmaps": lambda: feature_maps(args.spec), "se": lambda: se_weights(args.spec)}[args.cmd]()


if __name__ == "__main__":
    main()
