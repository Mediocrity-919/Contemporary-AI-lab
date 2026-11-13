"""A1：置信度校准与温度缩放（plan.md 6.4 节）。

    python -m src.analysis.calibration --name A1 configs/cnn/E2_C3_se.yaml:CE configs/cnn/E3_C4_ls.yaml:LS \
        configs/transfer/E7_ft.yaml:FT

- 温度缩放：p = softmax(z / T)。在整个验证集上最小化 NLL 拟合 T（优化 log T 保证 T>0），在测试集上评估。
  T>0 不改变各类 logit 的排序，所以 top-1 准确率不变，只改变置信度。
- 指标：ECE（15 个等宽分箱）、NLL、Brier；可靠性图使用代表性种子。
- 稳定性检查：把验证集按固定种子随机分成两半，分别拟合 T，看 T 与测试集 ECE 的变化。
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import softmax

from ..metrics import brier, ece, nll, reliability_bins
from .common import Runs, parse_spec, pct, save_fig, save_table

HALF_SPLIT_SEED = 0


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def scores(logits, labels, t=1.0) -> dict:
    probs = softmax(logits / t, 1)
    return {"acc": float((probs.argmax(1) == labels).mean()), "ece": ece(probs, labels),
            "nll": nll(logits / t, labels), "brier": brier(probs, labels)}


def reliability_axis(ax, logits, labels, t, title):
    rows = reliability_bins(softmax(logits / t, 1), labels)
    centers = [(lo + hi) / 2 for lo, hi, *_ in rows]
    accs = [a for *_, a, _ in rows]
    counts = np.array([n for _, _, n, _, _ in rows])
    ax.bar(centers, np.nan_to_num(accs), width=1 / len(rows), edgecolor="k", lw=0.4, alpha=0.75, label="各分箱准确率")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="完美校准")
    ax2 = ax.twinx()
    ax2.plot(centers, counts / counts.sum(), "o-", ms=2.5, color="tab:red", lw=0.8)
    ax2.set_ylabel("样本占比", color="tab:red", fontsize=8)
    ax2.grid(False)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="置信度", ylabel="准确率", title=title)


def run(specs, name):
    table = ["| 模型 | 校准 | T | 测试准确率 % | ECE % | NLL | Brier |", "|---|---|---:|---:|---:|---:|---:|"]
    stability = ["| 模型 | 种子 | T（全验证集） | T（前一半） | T（后一半） | 测试 ECE %（全 / 前半 / 后半 拟合） |",
                 "|---|---:|---:|---:|---:|---|"]
    fig, axes = plt.subplots(2, len(specs), figsize=(4.2 * len(specs), 8), squeeze=False)
    for col, spec in enumerate(specs):
        runs = Runs.load(*parse_spec(spec))
        per = {"未校准": [], "温度缩放": []}
        temps = []
        for seed in sorted(runs.dirs):
            vl, vy = runs.logits(seed, "val")
            tl, ty = runs.logits(seed, "test")
            t = fit_temperature(vl, vy)
            temps.append(t)
            per["未校准"].append(scores(tl, ty))
            per["温度缩放"].append(scores(tl, ty, t))
            perm = np.random.default_rng(HALF_SPLIT_SEED).permutation(len(vy))
            halves = [perm[: len(vy) // 2], perm[len(vy) // 2:]]
            t_half = [fit_temperature(vl[h], vy[h]) for h in halves]
            eces = [scores(tl, ty, x)["ece"] * 100 for x in [t] + t_half]
            stability.append(f"| {runs.label} | {seed} | {t:.3f} | {t_half[0]:.3f} | {t_half[1]:.3f} | "
                             + " / ".join(f"{e:.2f}" for e in eces) + " |")
        t_str = f"{np.mean(temps):.3f} ± {np.std(temps, ddof=1) if len(temps) > 1 else 0:.3f}"
        for mode, rows in per.items():
            table.append(f"| {runs.label} | {mode} | {'1' if mode == '未校准' else t_str} | "
                         f"{pct(r['acc'] for r in rows)} | {pct(r['ece'] for r in rows)} | "
                         f"{np.mean([r['nll'] for r in rows]):.4f} | {np.mean([r['brier'] for r in rows]):.4f} |")
        seed = runs.representative_seed()
        tl, ty = runs.logits(seed, "test")
        t = fit_temperature(*runs.logits(seed, "val"))
        reliability_axis(axes[0, col], tl, ty, 1.0, f"{runs.label} 未校准\nECE {scores(tl, ty)['ece'] * 100:.2f}%")
        reliability_axis(axes[1, col], tl, ty, t, f"{runs.label} T={t:.2f}\nECE {scores(tl, ty, t)['ece'] * 100:.2f}%")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("可靠性图（测试集，代表性种子；红线为各置信度分箱的样本占比）", fontsize=10)
    fig.tight_layout()
    save_fig(fig, f"{name}_reliability.png")
    table.append("\n注：T 在整个验证集上拟合；温度缩放不改变 top-1 准确率。ECE 为 15 个等宽分箱。")
    save_table(table, f"{name}_calibration.md")
    save_table(stability, f"{name}_temperature_stability.md")
    print("\n".join(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--name", default="A1")
    args = parser.parse_args()
    run(args.specs, args.name)


if __name__ == "__main__":
    main()
