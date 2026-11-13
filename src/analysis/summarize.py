"""汇总多个配置的结果为 Markdown 表格（均值 ± 标准差，以及附录用的逐种子结果）。

    python -m src.analysis.summarize --name E1_aug configs/cnn/E1_C0_none.yaml:无增强 \
        configs/cnn/E1_C1_basic.yaml:基础增强 configs/cnn/E1_C2_strong.yaml:强增强
    python -m src.analysis.summarize --name E4_sched --es configs/cnn/F_final.yaml:B-Cosine ...

--es 额外列出影子早停的结果：触发轮次、所用轮数、耗时，以及 es_best.pt 的验证 / 测试准确率。
"""
from __future__ import annotations

import argparse

import numpy as np

from .common import Runs, mean_std, parse_spec, pct, save_table


def fmt_params(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 100_000 else f"{n:,}"


def summarize(specs: list[str], name: str, with_es: bool = False) -> None:
    head = ["配置", "种子", "验证准确率 %", "测试准确率 %", "测试 NLL", "最优轮", "实际轮数", "每轮训练 s", "总耗时 min", "可训练参数"]
    if with_es:
        head += ["早停触发轮", "早停所用轮数", "早停耗时 min", "早停验证 %", "早停测试 %"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    appendix = ["| 配置 | 种子 | 哈希 | 验证 % | 测试 % | 最优轮 |", "|---|---|---|---:|---:|---:|"]
    for spec in specs:
        path, label = parse_spec(spec)
        runs = Runs.load(path, label)
        if not runs.dirs:
            lines.append(f"| {runs.label} | 0 | （尚无结果） |" + " |" * (len(head) - 3))
            continue
        seeds = sorted(runs.dirs)
        ms = {s: runs.metrics(s) for s in seeds}
        tests = {s: runs.eval_report(s) for s in seeds}
        test_accs = [t["accuracy"] for t in tests.values() if t]
        test_nll = [t["nll"] for t in tests.values() if t]
        row = [
            runs.label, str(len(seeds)), pct(ms[s]["best"]["val_acc"] for s in seeds),
            pct(test_accs) if len(test_accs) == len(seeds) else "—",
            f"{np.mean(test_nll):.3f}" if len(test_nll) == len(seeds) else "—",
            f"{mean_std(ms[s]['best']['epoch'] for s in seeds)[0]:.0f}",
            f"{mean_std(ms[s]['epochs_run'] for s in seeds)[0]:.0f}",
            f"{mean_std(runs.timing(s)[0] for s in seeds)[0]:.1f}",
            f"{mean_std(runs.timing(s)[1] for s in seeds)[0]:.1f}",
            fmt_params(ms[seeds[0]]['params']['trainable']),
        ]
        if with_es:
            es = {s: ms[s]["early_stopping"] for s in seeds}
            es_tests = [runs.eval_report(s, ckpt="es_best") for s in seeds]
            triggered = [e["triggered_epoch"] for e in es.values() if e["triggered_epoch"] is not None]
            row += [
                f"{len(triggered)}/{len(seeds)} 次触发" + (f"，均值 {np.mean(triggered):.0f}" if triggered else ""),
                f"{mean_std(e['epochs_used'] for e in es.values())[0]:.0f}",
                f"{mean_std(runs.timing(s, es[s]['epochs_used'])[1] for s in seeds)[0]:.1f}",
                pct(e["val_acc"] for e in es.values()),
                pct(t["accuracy"] for t in es_tests) if all(es_tests) else "—",
            ]
        lines.append("| " + " | ".join(row) + " |")
        for s in seeds:
            test_str = f"{tests[s]['accuracy'] * 100:.2f}" if tests[s] else "—"
            appendix.append(f"| {runs.label} | {s} | {runs.hash} | {ms[s]['best']['val_acc'] * 100:.2f} | "
                            f"{test_str} | {ms[s]['best']['epoch']} |")
    note = ["", "注：均值 ± 样本标准差（种子数 n 见表），只反映训练随机性；测试准确率是 best.pt 的结果。",
            "耗时按每轮中位数估算（总耗时 = 每轮训练 + 验证的中位数 × 轮数），排除机器休眠等暂停的影响。"]
    save_table(lines + note, f"{name}.md")
    save_table(appendix, f"{name}_per_seed.md")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+", help="配置文件[:显示名]")
    parser.add_argument("--name", required=True, help="输出表格文件名（不含扩展名）")
    parser.add_argument("--es", action="store_true", help="附加影子早停的结果")
    args = parser.parse_args()
    summarize(args.specs, args.name, args.es)


if __name__ == "__main__":
    main()
