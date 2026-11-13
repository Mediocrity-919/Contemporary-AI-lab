"""A2：自定义扰动下的鲁棒性评估（plan.md 6.5 节；不是官方 CIFAR-10-C）。

    python -m src.analysis.robustness --name A2 configs/cnn/E1_C0_none.yaml:无增强 \
        configs/cnn/E1_C1_basic.yaml:基础增强 configs/cnn/E1_C2_strong.yaml:强增强 ...

- 4 种扰动 × 3 档强度，参数在评估前固定（见 CORRUPTIONS）。
- 扰动施加在 32×32 原始像素上，再量化回 8 位，之后才做各模型自己的预处理（224 模型再放大）。
- 随机噪声用固定种子生成，所有模型看到完全相同的扰动图片（记录 SHA256 作为证明）。
- 每个运行的结果缓存到运行目录下的 robustness.json，并在 outputs/test_log.csv 中留痕。
"""
from __future__ import annotations

import argparse
import hashlib
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.v2 import functional as TF

from .. import data as D
from ..engine import predict_logits
from ..evaluate import load_trained, log_test_use
from ..utils import load_json, save_json
from .common import Runs, parse_spec, save_fig, save_table

CORRUPTIONS = {
    "gaussian_noise": ("高斯噪声 σ", [0.04, 0.08, 0.12]),
    "gaussian_blur": ("高斯模糊 σ(像素)", [0.5, 1.0, 1.5]),
    "brightness": ("亮度 ×", [0.7, 0.5, 0.3]),
    "contrast": ("对比度 ×", [0.7, 0.5, 0.3]),
}
NOISE_SEED = 20261106


def corrupt(images: torch.Tensor, kind: str, level: float, severity: int) -> torch.Tensor:
    """uint8 [N,3,32,32] → 扰动后的 uint8。"""
    x = images.float() / 255.0
    if kind == "gaussian_noise":
        gen = torch.Generator().manual_seed(NOISE_SEED + severity)
        x = x + torch.randn(x.shape, generator=gen) * level
    elif kind == "gaussian_blur":
        k = 2 * math.ceil(3 * level) + 1
        x = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[level, level])
    elif kind == "brightness":
        x = x * level
    elif kind == "contrast":
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * level + mean
    else:
        raise ValueError(kind)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def corrupted_sets(images: torch.Tensor):
    """依次产出 (扰动名, 档位 1–3, 参数, uint8 图像, sha256)。"""
    for kind, (_, levels) in CORRUPTIONS.items():
        for sev, level in enumerate(levels, 1):
            imgs = corrupt(images, kind, level, sev)
            yield kind, sev, level, imgs, hashlib.sha256(imgs.numpy().tobytes()).hexdigest()[:16]


def evaluate_runs(runs: Runs, images, labels, device) -> dict:
    """返回 {seed: {"clean": acc, "kind/sev": acc}}，有缓存时直接读取。"""
    out = {}
    for seed in sorted(runs.dirs):
        cache = runs.dirs[seed] / "robustness.json"
        if cache.exists():
            out[seed] = load_json(cache)["acc"]
            continue
        model = load_trained(runs.cfg, runs.dirs[seed], "best", device)
        accs, hashes = {}, {}
        for kind, sev, _, imgs, digest in [("clean", 0, 0, images, "clean")] + list(corrupted_sets(images)):
            es = D.EvalSet(imgs, labels, runs.cfg["data"]["norm"], device)
            logits = predict_logits(model, es, runs.cfg)
            accs[kind if kind == "clean" else f"{kind}/{sev}"] = float((logits.argmax(1) == labels).float().mean())
            hashes[f"{kind}/{sev}"] = digest
        save_json({"acc": accs, "sha256": hashes, "noise_seed": NOISE_SEED}, cache)
        log_test_use(runs.cfg, seed, "best+robustness", {"accuracy": accs["clean"], "n": len(labels)}, False)
        out[seed] = accs
        print(f"  {runs.label} seed{seed}: clean {accs['clean']:.4f}")
    return out


def plot_examples(images):
    idx = [3, 10, 21, 30]
    fig, axes = plt.subplots(len(idx), 1 + 12, figsize=(14, 4.6))
    sets = list(corrupted_sets(images[idx]))
    for r, k in enumerate(idx):
        axes[r, 0].imshow(images[k].permute(1, 2, 0))
        axes[r, 0].set_title("原图" if r == 0 else "", fontsize=7)
        for c, (kind, sev, level, imgs, _) in enumerate(sets, 1):
            axes[r, c].imshow(imgs[r].permute(1, 2, 0))
            if r == 0:
                axes[r, c].set_title(f"{CORRUPTIONS[kind][0]}\n{level}", fontsize=6)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("自定义扰动示例（每种扰动三档：轻 / 中 / 重）", fontsize=10)
    save_fig(fig, "A2_corruption_examples.png")


def run(specs, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, _ = D.load_split("test")
    plot_examples(images)
    results = {}
    for spec in specs:
        runs = Runs.load(*parse_spec(spec))
        results[runs.label] = evaluate_runs(runs, images, labels, device)

    keys = [f"{k}/{s}" for k in CORRUPTIONS for s in (1, 2, 3)]
    head = "| 模型 | 干净 % | " + " | ".join(CORRUPTIONS[k][0].split(" ")[0] for k in CORRUPTIONS) + " | 平均扰动准确率 % | 相对下降 % |"
    lines = [head, "|---|" + "---:|" * (len(CORRUPTIONS) + 3)]
    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(4 * len(CORRUPTIONS), 3.8), sharey=True)
    for label, per_seed in results.items():
        seeds = sorted(per_seed)
        clean = np.array([per_seed[s]["clean"] for s in seeds])
        mca = np.array([np.mean([per_seed[s][k] for k in keys]) for s in seeds])
        cells = []
        for ax, kind in zip(axes, CORRUPTIONS):
            curve = np.array([[per_seed[s]["clean"]] + [per_seed[s][f"{kind}/{v}"] for v in (1, 2, 3)] for s in seeds])
            ax.errorbar(range(4), curve.mean(0) * 100, yerr=curve.std(0, ddof=1) * 100 if len(seeds) > 1 else None,
                        marker="o", ms=3, capsize=2, label=label)
            avg = curve[:, 1:].mean(1) * 100
            cells.append(f"{avg.mean():.2f}")
        drop = (1 - mca / clean) * 100
        lines.append(f"| {label} | {clean.mean() * 100:.2f} | " + " | ".join(cells)
                     + f" | {mca.mean() * 100:.2f} ± {mca.std(ddof=1) * 100 if len(seeds) > 1 else 0:.2f} | {drop.mean():.1f} |")
    for ax, kind in zip(axes, CORRUPTIONS):
        title, levels = CORRUPTIONS[kind]
        ax.set_xticks(range(4), ["干净"] + [str(v) for v in levels])
        ax.set(title=title, xlabel="强度")
    axes[0].set_ylabel("测试准确率 %")
    axes[-1].legend(fontsize=7)
    fig.suptitle("强度–准确率曲线（种子均值 ± 标准差）", fontsize=10)
    save_fig(fig, f"{name}_robustness.png")
    lines.append("\n注：各扰动列为三档强度的平均准确率（种子均值）；相对下降 = 1 − 平均扰动准确率 / 干净准确率。"
                 "这是自定义扰动评估，不是官方 CIFAR-10-C。")
    save_table(lines, f"{name}_robustness.md")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--name", default="A2")
    args = parser.parse_args()
    run(args.specs, args.name)


if __name__ == "__main__":
    main()
