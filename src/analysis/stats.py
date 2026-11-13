"""事先确定的 McNemar 配对检验（plan.md 6.1 节）。

    python -m src.analysis.stats                      # 使用 configs/analysis/mcnemar_pairs.yaml

做法：
- 每一对模型按种子配对（种子 k 对种子 k），在同一批 10,000 张测试图上比较对错；
  不把多个种子的预测拼起来当作独立样本。
- b = A 对 B 错的样本数，c = A 错 B 对的样本数；精确检验 p = 二项检验(b, b+c, 0.5)，双侧。
- 每个种子内，对所有事先确定的比较对做 Holm 校正。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

from ..utils import ROOT, UnresolvedDecision
from .common import Runs, save_table

PAIRS_FILE = ROOT / "configs" / "analysis" / "mcnemar_pairs.yaml"


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"a_only": b, "b_only": c, "p": float(p)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm–Bonferroni 校正后的 p 值（保持原顺序）。"""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def run(pairs_file=PAIRS_FILE, split="test"):
    spec = yaml.safe_load(pairs_file.read_text(encoding="utf-8"))
    results = []  # (pair_label, seed, res)
    for pair in spec["pairs"]:
        try:
            a, b = Runs.load(pair["a"]), Runs.load(pair["b"])
        except UnresolvedDecision as e:
            print(f"[跳过] {pair['label']}：等待决策 `{e}`")
            continue
        for seed in sorted(set(a.dirs) & set(b.dirs)):
            try:
                la, y = a.logits(seed, split)
                lb, yb = b.logits(seed, split)
            except FileNotFoundError as e:
                print(f"[跳过] {pair['label']} seed{seed}：{e}")
                continue
            assert np.array_equal(y, yb)
            res = mcnemar_exact(la.argmax(1) == y, lb.argmax(1) == y)
            res.update(acc_a=float((la.argmax(1) == y).mean()), acc_b=float((lb.argmax(1) == y).mean()))
            results.append((pair["label"], seed, res))
    lines = ["| 比较（A vs B） | 种子 | A 准确率 % | B 准确率 % | 仅 A 对 | 仅 B 对 | p | Holm 校正 p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in sorted({s for _, s, _ in results}):
        group = [(l, r) for l, s, r in results if s == seed]
        adjusted = holm([r["p"] for _, r in group])
        for (label, r), p_adj in zip(group, adjusted):
            lines.append(f"| {label} | {seed} | {r['acc_a'] * 100:.2f} | {r['acc_b'] * 100:.2f} | {r['a_only']} | "
                         f"{r['b_only']} | {r['p']:.3g} | {p_adj:.3g} |")
    lines.append("\n注：每个种子内对全部比较对做 Holm 校正；检验只回答“这一对模型在测试集上的错误是否有系统性差异”。")
    save_table(lines, "mcnemar.md")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default=str(PAIRS_FILE))
    run(Path(parser.parse_args().pairs))


if __name__ == "__main__":
    main()
