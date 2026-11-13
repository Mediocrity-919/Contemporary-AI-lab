"""报告用的基础图表。

    python -m src.analysis.figures curves --name E1 configs/cnn/E1_C0_none.yaml:无增强 configs/cnn/E1_C1_basic.yaml:基础增强
    python -m src.analysis.figures lr --name E4 configs/cnn/E4_A_constant.yaml:固定 configs/cnn/F_final.yaml:Cosine
    python -m src.analysis.figures confusion configs/cnn/F_final.yaml:OurCNN
    python -m src.analysis.figures errors configs/cnn/F_final.yaml:OurCNN
    python -m src.analysis.figures overlap configs/cnn/F_final.yaml:OurCNN configs/transfer/E7_ft.yaml:FT
    python -m src.analysis.figures aug

需要单个模型的图（混淆矩阵、误分类样本）时使用代表性种子（验证准确率中位数）。
"""
from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import softmax

from .. import data as D
from ..metrics import confusion_matrix
from .common import TABLE_DIR, Runs, parse_spec, save_fig, save_table, show_path

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _band(ax, x, ys, color, label, style="-"):
    ys = np.stack(ys)
    ax.plot(x, ys.mean(0), style, color=color, label=label, lw=1.6)
    if len(ys) > 1:
        ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)


def plot_curves(specs, name):
    """loss / accuracy 曲线：实线 = 验证集，虚线 = 干净训练子集（eval 模式、无增强）；阴影为种子间范围。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, spec in enumerate(specs):
        runs = Runs.load(*parse_spec(spec))
        if not runs.dirs:
            continue
        logs = [runs.log(s) for s in sorted(runs.dirs)]
        n = min(len(l["epoch"]) for l in logs)
        x = logs[0]["epoch"][:n]
        c = COLORS[i % len(COLORS)]
        for ax, key in zip(axes, ("loss", "acc")):
            _band(ax, x, [l[f"val_{key}"][:n] for l in logs], c, f"{runs.label}（验证）")
            _band(ax, x, [l[f"clean_train_{key}"][:n] for l in logs], c, f"{runs.label}（训练）", "--")
    axes[0].set(xlabel="epoch", ylabel="交叉熵（NLL）", title="Loss")
    axes[1].set(xlabel="epoch", ylabel="准确率", title="Accuracy")
    axes[1].legend(fontsize=8, ncol=1, loc="lower right")
    fig.suptitle("训练曲线（实线：验证集；虚线：5000 张训练子集，eval 模式、无增强；阴影：种子间最小–最大）", fontsize=10)
    save_fig(fig, f"{name}_curves.png")

    # 带增强的训练指标（训练时的 running 平均）单独一张图，便于看增强 / 正则带来的差距
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, spec in enumerate(specs):
        runs = Runs.load(*parse_spec(spec))
        if runs.dirs:
            logs = [runs.log(s) for s in sorted(runs.dirs)]
            n = min(len(l["epoch"]) for l in logs)
            _band(ax, logs[0]["epoch"][:n], [l["train_acc"][:n] for l in logs], COLORS[i % len(COLORS)], runs.label)
    ax.set(xlabel="epoch", ylabel="准确率", title="训练准确率（含增强与 Dropout，epoch 内平均）")
    ax.legend(fontsize=8)
    save_fig(fig, f"{name}_train_running_acc.png")


def plot_lr(specs, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, spec in enumerate(specs):
        runs = Runs.load(*parse_spec(spec))
        if not runs.dirs:
            continue
        log = runs.log(runs.representative_seed())
        c = COLORS[i % len(COLORS)]
        axes[0].plot(log["epoch"], log["lr"], color=c, label=runs.label)
        axes[1].plot(log["epoch"], log["val_acc"], color=c, label=runs.label)
        es = runs.metrics(runs.representative_seed())["early_stopping"]
        if es["triggered_epoch"]:
            axes[1].axvline(es["triggered_epoch"], color=c, ls=":", lw=1)
    axes[0].set(xlabel="epoch", ylabel="学习率（每轮开始时）", yscale="log", title="学习率")
    axes[1].set(xlabel="epoch", ylabel="验证准确率", title="验证准确率（点线：早停触发位置）")
    axes[0].legend(fontsize=8)
    save_fig(fig, f"{name}_lr.png")


def plot_confusion(spec, split="test"):
    runs = Runs.load(*parse_spec(spec))
    seed = runs.representative_seed()
    rep = runs.eval_report(seed, split)
    if rep is None:
        raise SystemExit(f"请先运行 python -m src.evaluate {runs.config_file} --split {split}")
    names = rep["class_names"]
    cm = np.array(rep["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), layout="constrained")
    for ax, mat, title, fmt in [(axes[0], cm, "数量", "d"), (axes[1], cm / cm.sum(1, keepdims=True), "按行归一化（召回率）", ".2f")]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(10), names, rotation=45, ha="right")
        ax.set_yticks(range(10), names)
        ax.set(xlabel="预测类别", ylabel="真实类别", title=title)
        ax.grid(False)
        thresh = mat.max() / 2
        for i in range(10):
            for j in range(10):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{runs.label} 混淆矩阵（{split}，seed {seed}，总体准确率 {rep['accuracy'] * 100:.2f}%）")
    save_fig(fig, f"{runs.label}_confusion_{split}.png")

    # 每类指标：所有种子的均值 ± 标准差
    reports = [runs.eval_report(s, split) for s in sorted(runs.dirs)]
    reports = [r for r in reports if r]
    metrics = ["accuracy", "precision", "f1"]
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.27
    for k, m in enumerate(metrics):
        vals = np.array([r["per_class"][m] for r in reports])
        ax.bar(np.arange(10) + (k - 1) * width, vals.mean(0), width, yerr=vals.std(0, ddof=1) if len(vals) > 1 else None,
               label={"accuracy": "准确率（召回率）", "precision": "精确率", "f1": "F1"}[m], capsize=2)
    ax.set_xticks(range(10), names)
    ax.set_ylim(max(0, np.array([r["per_class"]["accuracy"] for r in reports]).min() - 0.1), 1.0)
    ax.set(ylabel="数值", title=f"{runs.label} 每类指标（{split}，{len(reports)} 个种子均值 ± 标准差）")
    ax.legend(fontsize=8, ncol=3)
    save_fig(fig, f"{runs.label}_per_class_{split}.png")

    lines = ["| 类别 | 准确率（召回率）% | 精确率 % | F1 % |", "|---|---:|---:|---:|"]
    for c in range(10):
        cells = []
        for m in metrics:
            v = np.array([r["per_class"][m][c] for r in reports]) * 100
            cells.append(f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}")
        lines.append(f"| {names[c]} | " + " | ".join(cells) + " |")
    accs = np.array([r["accuracy"] for r in reports]) * 100
    lines.append(f"| **总体** | {accs.mean():.2f} ± {accs.std(ddof=1) if len(accs) > 1 else 0:.2f} | | |")
    save_table(lines, f"{runs.label}_per_class_{split}.md")


def _gallery(images, titles, path_name, ncols=10, title=None):
    n = len(images)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.45, nrows * 1.75))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_title(t, fontsize=6.5)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    save_fig(fig, path_name)


def plot_confusion_pair(spec_a, spec_b, split="test"):
    """两个模型（各自的代表性种子）的混淆矩阵并排显示，便于在报告中直接对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    labels = []
    for ax, spec in zip(axes, (spec_a, spec_b)):
        runs = Runs.load(*parse_spec(spec))
        seed = runs.representative_seed()
        rep = runs.eval_report(seed, split)
        if rep is None:
            raise SystemExit(f"请先运行 python -m src.evaluate {runs.config_file} --split {split}")
        names, cm = rep["class_names"], np.array(rep["confusion_matrix"])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(10), names, rotation=45, ha="right")
        ax.set_yticks(range(10), names)
        ax.set(xlabel="预测类别", ylabel="真实类别",
               title=f"{runs.label}（seed {seed}，准确率 {rep['accuracy'] * 100:.2f}%）")
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if cm[i, j] > cm[i].sum() / 2 else "black")
        labels.append(runs.label)
    fig.suptitle("测试集混淆矩阵（格内为数量，颜色为按行归一化的比例）")
    save_fig(fig, f"confusion_{labels[0]}_vs_{labels[1]}_{split}.png")


def plot_errors(spec, split="test", top_pairs=5, per_pair=8, n_conf=20):
    """混淆最多的类别对 + 置信度最高 / 最低的误分类，并导出待人工标注原因的清单。"""
    runs = Runs.load(*parse_spec(spec))
    seed = runs.representative_seed()
    logits, labels = runs.logits(seed, split)
    images, _, names = D.load_split(split, runs.cfg["data"]["root"])
    probs = softmax(logits, 1)
    preds, conf = probs.argmax(1), probs.max(1)
    wrong = np.where(preds != labels)[0]

    cm = confusion_matrix(labels, preds, 10)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    # 无序类别对（猫→狗 与 狗→猫 合并统计）
    pair_counts = {(i, j): off[i, j] + off[j, i] for i in range(10) for j in range(i + 1, 10)}
    pairs = sorted(pair_counts, key=pair_counts.get, reverse=True)[:top_pairs]
    lines = ["| 类别对 | 合计 | A→B | B→A |", "|---|---:|---:|---:|"]
    imgs, titles = [], []
    for i, j in pairs:
        lines.append(f"| {names[i]} ↔ {names[j]} | {pair_counts[(i, j)]} | {off[i, j]} | {off[j, i]} |")
        idx = [k for k in wrong if (labels[k], preds[k]) in ((i, j), (j, i))]
        idx = sorted(idx, key=lambda k: -conf[k])[:per_pair]
        for k in idx:
            imgs.append(images[k])
            titles.append(f"#{k} {names[labels[k]]}→{names[preds[k]]}\n{conf[k]:.2f}")
    save_table(lines, f"{runs.label}_top_confusions_{split}.md")
    _gallery(imgs, titles, f"{runs.label}_top_confusion_examples_{split}.png", ncols=per_pair,
             title=f"{runs.label}：混淆最多的 {top_pairs} 对类别（每行同一对，按置信度排序；真实→预测，置信度）")

    for kind, order in [("most_confident", -conf[wrong]), ("least_confident", conf[wrong])]:
        idx = wrong[np.argsort(order)[:n_conf]]
        _gallery([images[k] for k in idx],
                 [f"#{k} {names[labels[k]]}→{names[preds[k]]}\n{conf[k]:.2f}" for k in idx],
                 f"{runs.label}_errors_{kind}_{split}.png",
                 title=f"{runs.label}：置信度{'最高' if kind == 'most_confident' else '最低'}的 {n_conf} 张误分类图（真实→预测，置信度）")
        # 导出清单，人工填写原因：外观相似 / 目标小或遮挡 / 背景干扰 / 视角少见 / 标注可疑
        out = TABLE_DIR / f"{runs.label}_errors_{kind}_{split}.csv"
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["test_index", "true", "pred", "confidence", "true_prob", "reason"])
            for k in idx:
                writer.writerow([k, names[labels[k]], names[preds[k]], f"{conf[k]:.4f}", f"{probs[k, labels[k]]:.4f}", ""])
        print(f"[表] {show_path(out)}（reason 列待人工填写）")


def plot_overlap(spec_a, spec_b, split="test"):
    """两个模型（各取代表性种子）误分类的重叠情况。"""
    a, b = Runs.load(*parse_spec(spec_a)), Runs.load(*parse_spec(spec_b))
    la, y = a.logits(a.representative_seed(), split)
    lb, yb = b.logits(b.representative_seed(), split)
    assert np.array_equal(y, yb)
    wa, wb = la.argmax(1) != y, lb.argmax(1) != y
    names = D.class_names(a.cfg["data"]["root"])
    lines = [f"| 类别 | 两者都错 | 仅 {a.label} 错 | 仅 {b.label} 错 | 两者都对 |", "|---|---:|---:|---:|---:|"]
    for c in list(range(10)) + [None]:
        m = np.ones_like(y, bool) if c is None else (y == c)
        lines.append(f"| {'**合计**' if c is None else names[c]} | {(wa & wb & m).sum()} | {(wa & ~wb & m).sum()} | "
                     f"{(~wa & wb & m).sum()} | {(~wa & ~wb & m).sum()} |")
    save_table(lines, f"overlap_{a.label}_vs_{b.label}_{split}.md")
    print("\n".join(lines))


def plot_augmentations(n_images=6, n_samples=4):
    """每张图：原图，以及 basic、strong 各 n_samples 个随机增强结果。"""
    images, labels, names = D.load_split("train")
    idx = D.stratified_subset(labels, 1)[:n_images]
    rows, titles = [], []
    for k in idx.tolist():
        rows.append(images[k])
        titles.append(f"原图 {names[labels[k]]}")
        for aug in ("basic", "strong"):
            tf = D.build_train_transform(aug, "cifar")
            for e in range(n_samples):
                torch.manual_seed(D.sample_seed(0, e, k))
                rows.append((D.denormalize(tf(images[k])[None], "cifar")[0] * 255).to(torch.uint8))
                titles.append(f"{aug} #{e + 1}")
    _gallery(rows, titles, "augmentation_examples.png", ncols=1 + 2 * n_samples,
             title="数据增强示例（每行一张训练图：原图 | basic×4 | strong×4）")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("curves", "lr"):
        p = sub.add_parser(cmd)
        p.add_argument("--name", required=True)
        p.add_argument("specs", nargs="+")
    for cmd in ("confusion", "errors"):
        p = sub.add_parser(cmd)
        p.add_argument("spec")
        p.add_argument("--split", default="test", choices=["val", "test"])
    for cmd in ("overlap", "confusion-pair"):
        p = sub.add_parser(cmd)
        p.add_argument("spec_a")
        p.add_argument("spec_b")
    sub.add_parser("aug")
    args = parser.parse_args()
    if args.cmd == "curves":
        plot_curves(args.specs, args.name)
    elif args.cmd == "lr":
        plot_lr(args.specs, args.name)
    elif args.cmd == "confusion":
        plot_confusion(args.spec, args.split)
    elif args.cmd == "errors":
        plot_errors(args.spec, args.split)
    elif args.cmd == "overlap":
        plot_overlap(args.spec_a, args.spec_b)
    elif args.cmd == "confusion-pair":
        plot_confusion_pair(args.spec_a, args.spec_b)
    elif args.cmd == "aug":
        plot_augmentations()


if __name__ == "__main__":
    main()
