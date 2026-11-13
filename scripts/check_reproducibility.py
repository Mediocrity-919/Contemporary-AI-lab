"""复现性检查（plan.md 第 3 节）：用同一配置、同一种子各训练 2 轮，比较结果。

    python scripts/check_reproducibility.py

1. 确定性模式（train.deterministic=true，默认）跑两次：逐轮指标和模型参数应完全相同。
2. 确定性模式下先训练 1 轮后中断，再续训到 2 轮：应与不中断的结果完全相同（验证断点续训）。
3. 默认模式（cudnn.benchmark，非确定）跑两次：记录两次结果相差多少。

测试配置：OurCNN + SE + 强增强 + 标签平滑（覆盖最多的代码路径），每次约 1 分钟。
结果写入 outputs/checks/reproducibility.json。
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "checks" / "repro"
SETS = ["data.aug=strong", "model.attention=se", "loss.type=ls", "train.epochs=2", "meta.name=repro"]


def train(tag: str, deterministic: bool, stop_after: int | None = None) -> Path:
    cmd = [sys.executable, "-m", "src.train", "--config", "configs/base.yaml", "--seed", "0",
           "--output-root", f"outputs/checks/repro/{tag}"]
    for s in SETS:
        cmd += ["--set", s]
    cmd += ["--set", f"train.deterministic={'true' if deterministic else 'false'}"]
    if stop_after:
        cmd += ["--stop-after", str(stop_after)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return next((OUT / tag).glob("*/seed0"))


def read_log(run_dir: Path) -> list[dict]:
    with (run_dir / "log.csv").open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items() if k in ("train_loss", "train_acc", "val_loss", "val_acc",
                                                           "clean_train_loss")} for r in csv.DictReader(f)]


def max_param_diff(a: Path, b: Path) -> float:
    sa = torch.load(a / "last.pt", map_location="cpu", weights_only=False)["model"]
    sb = torch.load(b / "last.pt", map_location="cpu", weights_only=False)["model"]
    return max((sa[k].float() - sb[k].float()).abs().max().item() for k in sa)


def compare(a: Path, b: Path) -> dict:
    la, lb = read_log(a), read_log(b)
    return {
        "logs_identical": la == lb,
        "max_metric_diff": max(abs(x[k] - y[k]) for x, y in zip(la, lb) for k in x),
        "max_param_diff": max_param_diff(a, b),
        "val_acc": [[r["val_acc"] for r in la], [r["val_acc"] for r in lb]],
    }


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    det_a = train("det_a", True)
    det_b = train("det_b", True)
    train("det_resume", True, stop_after=1)
    det_resume = train("det_resume", True)          # 第二次调用从 last.pt 续训
    nd_a = train("nondet_a", False)
    nd_b = train("nondet_b", False)
    result = {
        "deterministic_repeat": compare(det_a, det_b),
        "deterministic_resume_vs_uninterrupted": compare(det_a, det_resume),
        "nondeterministic_repeat": compare(nd_a, nd_b),
    }
    path = ROOT / "outputs" / "checks" / "reproducibility.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    ok = result["deterministic_repeat"]["logs_identical"] and result["deterministic_resume_vs_uninterrupted"]["logs_identical"]
    print("\n结论：" + ("确定性模式可逐位复现，断点续训结果一致。" if ok else "存在差异，请检查！"))


if __name__ == "__main__":
    main()
