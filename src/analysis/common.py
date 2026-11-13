"""分析模块的公共工具：按配置文件找到各种子的运行结果、选代表性种子、统一绘图风格。"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils import ROOT, config_hash, is_complete, load_config, load_json, run_dir_for  # noqa: E402

REPORT_DIR = Path(os.environ.get("LAB3_REPORT_DIR", ROOT / "report"))  # 环境变量仅供隔离演练使用
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


def setup_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


setup_style()


@dataclass
class Runs:
    """一个配置文件（不含扫参）在各种子上的已完成运行。"""
    config_file: str
    cfg: dict
    label: str
    dirs: dict = field(default_factory=dict)  # seed -> run_dir

    @classmethod
    def load(cls, config_file: str, label: str | None = None, seeds=None, overrides: dict | None = None) -> "Runs":
        cfg = load_config(ROOT / config_file, overrides)
        seeds = seeds or cfg["meta"].get("seeds", [0, 1, 2])
        dirs = {s: run_dir_for(cfg, s) for s in seeds if is_complete(run_dir_for(cfg, s))}
        return cls(config_file, cfg, label or cfg["meta"]["name"], dirs)

    @property
    def hash(self) -> str:
        return config_hash(self.cfg)

    def metrics(self, seed) -> dict:
        return load_json(self.dirs[seed] / "metrics.json")

    def log(self, seed) -> dict[str, np.ndarray]:
        with (self.dirs[seed] / "log.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

    def timing(self, seed, upto: int | None = None) -> tuple[float, float]:
        """(每轮训练秒数的中位数, 总耗时分钟)。总耗时 = (训练 + 验证的每轮中位数) × 轮数，
        用中位数而不是实际墙钟时间，避免机器休眠、控制台被挂起等暂停把个别轮次拉长到几十分钟。
        upto 给定时只统计前 upto 轮（例如早停时刻）。"""
        log = self.log(seed)
        n = len(log["epoch"]) if upto is None else int(upto)
        train, evaluate = np.median(log["epoch_train_time"]), np.median(log["epoch_eval_time"])
        return float(train), float((train + evaluate) * n / 60)

    def val_accs(self) -> dict[int, float]:
        return {s: self.metrics(s)["best"]["val_acc"] for s in self.dirs}

    def eval_report(self, seed, split="test", ckpt="best") -> dict | None:
        path = self.dirs[seed] / f"{split}_{ckpt}_metrics.json"
        return load_json(path) if path.exists() else None

    def logits(self, seed, split="test", ckpt="best"):
        """返回 (logits, labels)；验证集 best 的 logits 在训练结束时已保存。"""
        name = "val_best_logits.npz" if (split, ckpt) == ("val", "best") else f"{split}_{ckpt}_logits.npz"
        path = self.dirs[seed] / name
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在；请先运行 python -m src.evaluate {self.config_file} --split {split}")
        data = np.load(path)
        return data["logits"], data["labels"]

    def representative_seed(self) -> int:
        """验证准确率处于中位数的种子（plan.md 第 2 节规则 7）。"""
        accs = self.val_accs()
        if not accs:
            raise RuntimeError(f"{self.config_file} 没有已完成的运行")
        ordered = sorted(accs, key=lambda s: (accs[s], s))
        return ordered[(len(ordered) - 1) // 2]


def show_path(path: Path) -> str:
    """打印用：项目内的路径显示为相对路径，项目外（例如临时目录）显示绝对路径。"""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def mean_std(values) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def pct(values) -> str:
    m, s = mean_std(values)
    return f"{m * 100:.2f} ± {s * 100:.2f}"


def save_fig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[图] {show_path(path)}")
    return path


def save_table(lines: list[str], name: str) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[表] {show_path(path)}")
    return path


def parse_spec(spec: str) -> tuple[str, str | None]:
    """命令行里的 “配置文件[:显示名]”。"""
    path, _, label = spec.partition(":")
    return path, label or None
