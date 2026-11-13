"""批量运行实验：展开配置文件的扫参变体 × 种子，已完成的自动跳过（相同哈希直接复用）。

示例：
    python -m src.run_experiments configs/cnn/E1_*.yaml            # 运行
    python -m src.run_experiments configs/cnn/E1_*.yaml --dry-run  # 只列出将要运行的内容
    python -m src.run_experiments "configs/**/*.yaml" --status     # 查看全部实验的完成情况

每个运行在独立子进程中执行；中途中断后重新执行同一命令即可从断点继续。
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

from .utils import (ROOT, UnresolvedDecision, _coerce, _load_raw, config_hash, expand_sweep, is_complete,
                    load_config, load_json, run_dir_for)


def expand_paths(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(ROOT / pattern) if not Path(pattern).is_absolute() else pattern,
                                   recursive=True))
        # 跳过公共配置（以 _ 开头）、base、决策文件和分析用的 YAML
        paths += [Path(m) for m in matches
                  if not Path(m).name.startswith("_") and Path(m).name not in ("base.yaml", "decisions.yaml")
                  and Path(m).parent.name != "analysis"]
    return list(dict.fromkeys(paths))


def plan_runs(path: Path, seeds_override=None):
    """返回 [(变体名, 覆盖字典, 配置或 None, 种子列表, 未确定的决策名或 None)]。"""
    raw = _coerce(_load_raw(path))
    seeds = seeds_override or raw.get("meta", {}).get("seeds", [0, 1, 2])
    out = []
    for name, overrides in expand_sweep(raw):
        try:
            cfg = load_config(path, overrides)
            out.append((name, overrides, cfg, seeds, None))
        except UnresolvedDecision as e:
            out.append((name, overrides, None, seeds, str(e)))
    return out


def format_value(v) -> str:
    return "null" if v is None else str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+", help="配置文件路径或通配符（相对项目根目录）")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="覆盖配置中的种子列表")
    parser.add_argument("--status", action="store_true", help="只显示完成情况")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要运行的内容")
    parser.add_argument("--keep-going", action="store_true", help="某个运行失败时继续后面的运行")
    args = parser.parse_args()

    paths = expand_paths(args.patterns)
    if not paths:
        sys.exit("没有匹配的配置文件")

    pending, failures = [], []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        for name, overrides, cfg, seeds, missing in plan_runs(path, args.seeds):
            if cfg is None:
                print(f"[等待决策] {rel} :: {name} 需要 decisions.yaml 中的 `{missing}`")
                continue
            done = []
            for seed in seeds:
                run_dir = run_dir_for(cfg, seed)
                if is_complete(run_dir):
                    m = load_json(run_dir / "metrics.json")
                    reused = "" if run_dir.parent.name.startswith(name + "-") else f"（复用 {run_dir.parent.name}）"
                    done.append(f"s{seed}={m['best']['val_acc']:.4f}{reused}")
                else:
                    pending.append((path, name, overrides, seed, cfg))
            print(f"[{len(done)}/{len(seeds)}] {name:<34} hash={config_hash(cfg)}  {'  '.join(done)}")

    if args.status or args.dry_run:
        print(f"\n待运行：{len(pending)} 个")
        for path, name, _, seed, _ in pending:
            print(f"  {name} seed{seed}  ({path.relative_to(ROOT).as_posix()})")
        return

    for i, (path, name, overrides, seed, cfg) in enumerate(pending, 1):
        if is_complete(run_dir_for(cfg, seed)):  # 本次批量中前面的运行可能已经产出相同哈希的结果
            print(f"\n=== [{i}/{len(pending)}] {name} seed{seed}：已由相同配置完成，跳过")
            continue
        print(f"\n=== [{i}/{len(pending)}] {name} seed{seed} ===", flush=True)
        cmd = [sys.executable, "-m", "src.train", "--config", str(path), "--seed", str(seed)]
        for key, value in overrides.items():
            cmd += ["--set", f"{key}={format_value(value)}"]
        start = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failures.append(f"{name} seed{seed}")
            print(f"!!! {name} seed{seed} 失败（返回码 {result.returncode}）", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"--- {name} seed{seed} 完成，用时 {(time.time() - start) / 60:.1f} 分钟", flush=True)

    if failures:
        sys.exit("失败的运行：" + "，".join(failures))


if __name__ == "__main__":
    main()
