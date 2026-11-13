"""E9：同一配置 3 个种子的集成（P2，plan.md 5.5 节）。

    python -m src.analysis.ensemble configs/cnn/F_final.yaml:OurCNN configs/transfer/E7_ft.yaml:FT

集成方式事先固定：各种子 softmax 概率取平均，不在验证集上调权重。
单模型结果与集成结果分开报告，并列出推理与存储开销的倍数。
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.special import softmax

from ..metrics import ece
from .common import Runs, parse_spec, pct, save_table


def run(specs):
    lines = ["| 模型 | 数据 | 单模型准确率 %（均值 ± 标准差） | 集成准确率 % | 单模型 ECE % | 集成 ECE % | 推理 / 存储开销 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for spec in specs:
        runs = Runs.load(*parse_spec(spec))
        seeds = sorted(runs.dirs)
        for split in ("val", "test"):
            pairs = [runs.logits(s, split) for s in seeds]
            labels = pairs[0][1]
            probs = [softmax(l, 1) for l, _ in pairs]
            single = [(p.argmax(1) == labels).mean() for p in probs]
            single_ece = [ece(p, labels) for p in probs]
            avg = np.mean(probs, 0)
            lines.append(f"| {runs.label} | {split} | {pct(single)} | {(avg.argmax(1) == labels).mean() * 100:.2f} | "
                         f"{pct(single_ece)} | {ece(avg, labels) * 100:.2f} | ×{len(seeds)} |")
    save_table(lines, "E9_ensemble.md")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+")
    run(parser.parse_args().specs)


if __name__ == "__main__":
    main()
