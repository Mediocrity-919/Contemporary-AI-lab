"""按 plan.md 5.4 节事先确定的规则做决策，写入 configs/decisions.yaml，并把依据追加到
configs/decisions_log.md。规则写成代码，避免看到结果后临时改变标准。

    python -m src.select_decisions set epochs_32 100 --reason "预实验：验证准确率在第 9x 轮后趋于平稳"
    python -m src.select_decisions lr configs/cnn/T_cnn.yaml cnn_lr
    python -m src.select_decisions aug
    python -m src.select_decisions attention
    python -m src.select_decisions loss
    python -m src.select_decisions kd configs/ext/T_kd.yaml

已确定的决策默认不允许覆盖（--overwrite 可以覆盖，并在日志中注明）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import yaml

from .run_experiments import plan_runs
from .utils import (DECISIONS_FILE, ROOT, config_hash, get_by_path, load_config, load_decisions, load_json,
                    run_dir_for)

LOG_FILE = DECISIONS_FILE.parent / "decisions_log.md"
TIE_THRESHOLD = 0.001  # 0.1 个百分点：组件带来的验证集均值提升不超过该值时视为持平，选更简单的配置


def val_accs(cfg: dict, seeds) -> list[float]:
    accs = []
    for seed in seeds:
        path = run_dir_for(cfg, seed) / "metrics.json"
        if not path.exists():
            sys.exit(f"缺少结果：{cfg['meta']['name']} seed{seed}（hash {config_hash(cfg)}）")
        accs.append(load_json(path)["best"]["val_acc"])
    return accs


def write_decision(updates: dict, evidence: str, overwrite: bool) -> None:
    decisions = load_decisions()
    for key, value in updates.items():
        if key not in decisions:
            sys.exit(f"decisions.yaml 中没有键 `{key}`")
        if decisions[key] is not None and decisions[key] != value and not overwrite:
            sys.exit(f"`{key}` 已确定为 {decisions[key]}；如确需修改，加 --overwrite 并在 plan.md 变更记录中说明")
    old = {k: decisions[k] for k in updates}
    if all(old[k] == v for k, v in updates.items()):  # 重跑阶段时重复决策：结果相同，不重复记录
        print(evidence)
        print(f"\n决策未变（{', '.join(f'{k} = {v}' for k, v in updates.items())}），不重复写入。")
        return
    decisions.update(updates)
    DECISIONS_FILE.write_text(yaml.safe_dump(decisions, allow_unicode=True, sort_keys=False), encoding="utf-8")
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    changes = "，".join(f"`{k}`: {old[k]} → **{v}**" for k, v in updates.items())
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n## {stamp}  {changes}\n\n{evidence}\n")
    print(evidence)
    print(f"\n已写入 decisions.yaml：{changes}")


def fmt(accs) -> str:
    return f"{np.mean(accs) * 100:.2f} ± {np.std(accs, ddof=1) * 100 if len(accs) > 1 else 0:.2f}"


def decide_sweep(config_path: str, keys: list[str], overwrite: bool) -> None:
    """扫参：调参种子上验证集准确率最高的变体（规则 3）。

    最优点落在网格边缘时，按规则应沿该方向补 1 个点：如果配置里还没有标记 meta.edge_extended，
    就不写入决策、以返回码 2 退出；补点后（配置标记 edge_extended: true）即使仍在边缘也接受结果。
    """
    raw = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    extended = bool((raw.get("meta") or {}).get("edge_extended", False))
    runs = plan_runs(ROOT / config_path)
    rows = []
    for name, overrides, cfg, seeds, missing in runs:
        if cfg is None:
            sys.exit(f"{name} 等待决策 `{missing}`")
        rows.append((name, {k: get_by_path(cfg, k) for k in overrides if k != "meta.name"}, val_accs(cfg, seeds)[0]))
    best = max(rows, key=lambda r: r[2])
    lines = ["| 变体 | 取值 | 验证准确率 |", "|---|---|---:|"]
    lines += [f"| {n} | {', '.join(f'{k}={x}' for k, x in v.items())} | {a * 100:.2f}{' ← 最优' if n == best[0] else ''} |"
              for n, v, a in rows]
    swept = list(best[1])
    if len(keys) != len(swept):
        sys.exit(f"需要为 {swept} 依次提供决策键名")
    edge = []
    for key in swept:
        values = sorted({r[1][key] for r in rows})
        if len(values) > 2 and best[1][key] in (values[0], values[-1]):
            edge.append((key, best[1][key], "更大" if best[1][key] == values[-1] else "更小"))
    if edge and not extended:
        print(f"规则 3：扫参取验证集最优（{config_path}）\n\n" + "\n".join(lines))
        for key, value, direction in edge:
            print(f"\n**最优值在网格边缘**：`{key}` = {value}。按规则应向{direction}的方向再加 1 个点：")
            print(f"  在 {config_path} 的 meta.sweep 中加入新取值，并加上 meta.edge_extended: true，"
                  "在 plan.md 变更记录中注明，然后重跑该阶段。")
        print("\n未写入 decisions.yaml。")
        sys.exit(2)
    for key, value, _ in edge:
        lines.append(f"\n注：`{key}` 的最优值 {value} 仍在网格边缘；网格已按规则扩展过一次，接受该结果。")
    write_decision(dict(zip(keys, (best[1][k] for k in swept))), f"规则 3：扫参取验证集最优（{config_path}）\n\n"
                   + "\n".join(lines), overwrite)


def decide_aug(overwrite: bool) -> None:
    """规则 1：三档增强中 3 个种子验证集均值最高的一档。"""
    rows = []
    for level, file in [("none", "E1_C0_none"), ("basic", "E1_C1_basic"), ("strong", "E1_C2_strong")]:
        cfg = load_config(ROOT / f"configs/cnn/{file}.yaml")
        rows.append((level, val_accs(cfg, cfg["meta"]["seeds"])))
    best = max(rows, key=lambda r: np.mean(r[1]))
    table = ["| 增强 | 验证准确率 % (均值 ± 标准差) | 各种子 |", "|---|---:|---|"]
    table += [f"| {l} | {fmt(a)} | {', '.join(f'{x * 100:.2f}' for x in a)} |" for l, a in rows]
    write_decision({"aug_best": best[0]}, "规则 1：选 3 个种子验证集均值最高的增强档\n\n" + "\n".join(table), overwrite)


def decide_component(key: str, config_file: str, field: str, on_value: str, off_value: str, overwrite: bool) -> None:
    """规则 2：组件使验证集均值提升超过 0.1 个百分点才加入，否则视为持平、选更简单的配置。"""
    on_cfg = load_config(ROOT / config_file)
    off_cfg = load_config(ROOT / config_file, {field: off_value})
    seeds = on_cfg["meta"]["seeds"]
    on, off = val_accs(on_cfg, seeds), val_accs(off_cfg, seeds)
    delta = np.mean(on) - np.mean(off)
    choice = on_value if delta > TIE_THRESHOLD else off_value
    table = [f"| {field} | 验证准确率 % | 各种子 |", "|---|---:|---|",
             f"| {on_value} | {fmt(on)} | {', '.join(f'{x * 100:.2f}' for x in on)} |",
             f"| {off_value} | {fmt(off)} | {', '.join(f'{x * 100:.2f}' for x in off)} |",
             f"\n均值差 = {delta * 100:+.2f} 个百分点（阈值 +{TIE_THRESHOLD * 100:.1f}）→ 选 **{choice}**"]
    write_decision({key: choice}, f"规则 2：{config_file} 中 `{field}` 的取舍\n\n" + "\n".join(table), overwrite)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("set", help="手动决策（例如训练轮数）")
    p.add_argument("key")
    p.add_argument("value")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("lr", help="从学习率扫参中选最优")
    p.add_argument("config")
    p.add_argument("key")
    sub.add_parser("aug")
    sub.add_parser("attention")
    sub.add_parser("loss")
    p = sub.add_parser("kd", help="从蒸馏扫参中选 T 和 α")
    p.add_argument("config")
    args = parser.parse_args()

    if args.cmd == "set":
        value = yaml.safe_load(args.value)
        write_decision({args.key: value}, f"手动决策：{args.reason}", args.overwrite)
    elif args.cmd == "lr":
        decide_sweep(args.config, [args.key], args.overwrite)
    elif args.cmd == "kd":
        decide_sweep(args.config, ["kd_T", "kd_alpha"], args.overwrite)
    elif args.cmd == "aug":
        decide_aug(args.overwrite)
    elif args.cmd == "attention":
        decide_component("attention_best", "configs/cnn/E2_C3_se.yaml", "model.attention", "se", "none", args.overwrite)
    elif args.cmd == "loss":
        decide_component("loss_best", "configs/cnn/E3_C4_ls.yaml", "loss.type", "ls", "ce", args.overwrite)


if __name__ == "__main__":
    main()
