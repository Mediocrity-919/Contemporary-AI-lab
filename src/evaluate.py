"""在验证集或测试集上评估已训练完成的模型。

    python -m src.evaluate configs/cnn/F_final.yaml --split test
    python -m src.evaluate "configs/cnn/E*.yaml" --split test --ckpt best

每个运行目录下生成：
    {split}_{ckpt}_logits.npz      logits 与标签（后续所有分析都从这里读取，不重复推理）
    {split}_{ckpt}_metrics.json    准确率、Wilson 区间、NLL、ECE、每类指标、混淆矩阵

测试集规则（plan.md 第 2 节）：只有本脚本读取测试集；每次测试集评估都追加一行到
outputs/test_log.csv。已评估过的运行默认不再重复评估（--force 会重新评估，同样记录在案）。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .engine import predict_logits
from .metrics import full_report
from .models import build_model
from .run_experiments import expand_paths, plan_runs
from .utils import config_hash, is_complete, load_json, run_dir_for, save_json, test_log_path


def load_trained(cfg: dict, run_dir: Path, ckpt: str = "best", device=None) -> torch.nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = dict(cfg["model"], pretrained=False)  # 权重来自 checkpoint，不需要再下载预训练权重
    model = build_model(mcfg).to(device)
    state = torch.load(Path(run_dir) / f"{ckpt}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if cfg["train"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


def log_test_use(cfg, seed, ckpt, report, force):
    path = test_log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "config_file", "name", "hash", "seed", "ckpt", "accuracy", "n", "forced"])
        writer.writerow([dt.datetime.now().isoformat(timespec="seconds"), cfg["meta"].get("config_file"),
                         cfg["meta"]["name"], config_hash(cfg), seed, ckpt, f"{report['accuracy']:.6f}",
                         report["n"], int(force)])


def evaluate_run(cfg: dict, seed: int, split: str, ckpt: str = "best", force: bool = False, device=None) -> dict | None:
    run_dir = run_dir_for(cfg, seed)
    if not is_complete(run_dir):
        return None
    out_json = run_dir / f"{split}_{ckpt}_metrics.json"
    if out_json.exists() and not force:
        return {**load_json(out_json), "cached": True}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_set = D.make_eval_set(split, cfg, device)
    model = load_trained(cfg, run_dir, ckpt, device)
    logits = predict_logits(model, eval_set, cfg).numpy()
    labels = eval_set.y.cpu().numpy()
    report = full_report(logits, labels, D.class_names(cfg["data"]["root"]))
    report.update({"split": split, "ckpt": ckpt, "seed": seed, "hash": config_hash(cfg)})
    np.savez_compressed(run_dir / f"{split}_{ckpt}_logits.npz", logits=logits, labels=labels)
    save_json(report, out_json)
    if split == "test":
        log_test_use(cfg, seed, ckpt, report, force)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--ckpt", choices=["best", "es_best", "last"], default="best")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--force", action="store_true", help="重新评估已评估过的运行")
    args = parser.parse_args()

    for path in expand_paths(args.patterns):
        for name, _, cfg, seeds, missing in plan_runs(path, args.seeds):
            if cfg is None:
                print(f"[跳过] {name}：等待决策 `{missing}`")
                continue
            for seed in seeds:
                report = evaluate_run(cfg, seed, args.split, args.ckpt, args.force)
                if report is None:
                    status = "尚未训练完成"
                else:
                    status = f"acc={report['accuracy']:.4f}" + ("（已有结果，未重复评估）" if report.get("cached") else "")
                print(f"{name:<34} seed{seed:<4} {args.split}/{args.ckpt}: {status}")


if __name__ == "__main__":
    main()
