"""训练入口：一个配置 + 一个种子。

示例：
    python -m src.train --config configs/cnn/E1_C1_basic.yaml --seed 0
    python -m src.train --config configs/cnn/T_cnn.yaml --seed 100 --set train.lr=0.05 --set meta.name=T_cnn_lr0.05

已完成的运行（存在 metrics.json）直接跳过；存在 last.pt 时自动续训。
"""
from __future__ import annotations

import argparse

from .engine import fit
from .utils import load_config, parse_value, run_dir_for


def parse_overrides(items: list[str]) -> dict:
    overrides = {}
    for item in items or []:
        key, _, value = item.partition("=")
        overrides[key.strip()] = parse_value(value)
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置字段，可重复")
    parser.add_argument("--output-root", default=None, help="输出根目录（默认 outputs/runs）")
    parser.add_argument("--stop-after", type=int, default=None, help="仅测试用：训练 N 轮后退出，模拟中断")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    if args.output_root:
        overrides["system.output_root"] = args.output_root
    cfg = load_config(args.config, overrides)
    fit(cfg, args.seed, run_dir_for(cfg, args.seed), stop_after_epochs=args.stop_after)


if __name__ == "__main__":
    main()
