"""阶段 7 / 8：统一完成测试集评估和全部报告图表。

    python -m src.analysis.run_all          # P0 + P1：测试集评估、汇总表、曲线、混淆矩阵、误分类、
                                            #           校准、鲁棒性、Grad-CAM、特征分析、模型开销、McNemar
    python -m src.analysis.run_all --p2     # P2 扩展：CBAM / CutMix / 正则 / 调度扩展、蒸馏、集成

“最终成绩模型”按验证集均值选出（plan.md 5.4 规则 4），不看测试集：
  自定义 CNN 系列候选：F_final、E6_ourcnn；预训练系列候选：E7_ft、E7_lp、E7b_ft32。
某一步缺少结果时会跳过并在最后列出，不影响其他步骤。
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from ..evaluate import evaluate_run
from ..run_experiments import expand_paths, plan_runs
from ..utils import load_decisions
from . import calibration, ensemble, features, figures, gradcam, profile, robustness, stats, summarize
from .common import Runs, pct, save_table

CNN = "configs/cnn/"
P01_CONFIGS = ["configs/cnn/E*.yaml", "configs/cnn/F_final.yaml", "configs/arch/E6_*.yaml",
               "configs/transfer/E7*.yaml"]
P2_CONFIGS = ["configs/ext/X_*.yaml", "configs/ext/E8_kd.yaml"]
FAILED: list[str] = []


def step(title, fn, *args, **kwargs):
    print(f"\n==== {title}", flush=True)
    try:
        fn(*args, **kwargs)
    except (Exception, SystemExit) as e:  # 缺少结果等情况：记录后继续
        FAILED.append(f"{title}：{e}")
        traceback.print_exc(limit=1)


def evaluate_all(patterns, ckpts=("best",)):
    for path in expand_paths(patterns):
        for name, _, cfg, seeds, missing in plan_runs(path):
            if cfg is None:
                continue
            for seed in seeds:
                for ckpt in ckpts:
                    rep = evaluate_run(cfg, seed, "test", ckpt)
                    if rep and not rep.get("cached"):
                        print(f"  {name} seed{seed} {ckpt}: test acc {rep['accuracy']:.4f}")


def pick_headline(candidates: list[str]) -> str:
    """候选配置中验证集均值最高者（要求 3 个种子齐全）。"""
    scored = []
    for spec in candidates:
        path, _, label = spec.partition(":")
        try:
            runs = Runs.load(path, label or None)
        except Exception:
            continue
        accs = runs.val_accs()
        if len(accs) == 3:
            scored.append((np.mean(list(accs.values())), spec))
    if not scored:
        raise RuntimeError(f"候选配置都没有完整的 3 个种子：{candidates}")
    best = max(scored)
    print(f"  按验证集均值选出：{best[1]}（{best[0] * 100:.2f}%）；候选：" +
          "，".join(f"{s.split(':')[-1]} {a * 100:.2f}" for a, s in scored))
    return best[1]


def safe_pick(candidates: list[str]) -> str:
    """选不出来时（结果不全）退回第一个候选，并记为未完成。"""
    try:
        return pick_headline(candidates)
    except RuntimeError as e:
        FAILED.append(f"最终成绩模型选择：{e}；暂用 {candidates[0]}")
        return candidates[0]


def headline_table(cnn_spec, pre_spec):
    lines = ["| 系列 | 选中的配置 | 验证准确率 % | 测试准确率 %（3 个种子） | 代表性种子测试准确率与 Wilson 95% 区间 |",
             "|---|---|---:|---:|---|"]
    for family, spec in [("自定义 CNN", cnn_spec), ("ImageNet 预训练", pre_spec)]:
        path, _, label = spec.partition(":")
        runs = Runs.load(path, label)
        seeds = sorted(runs.dirs)
        rep = runs.eval_report(runs.representative_seed())
        lo, hi = rep["accuracy_wilson95"]
        lines.append(f"| {family} | {runs.label}（{path}） | {pct(runs.val_accs().values())} | "
                     f"{pct(runs.eval_report(s)['accuracy'] for s in seeds)} | "
                     f"seed {runs.representative_seed()}: {rep['accuracy'] * 100:.2f} [{lo * 100:.2f}, {hi * 100:.2f}] |")
    save_table(lines, "headline.md")
    print("\n".join(lines))


def run_p01():
    d = load_decisions()
    step("测试集评估（best 与 es_best）", evaluate_all, P01_CONFIGS, ("best", "es_best"))
    cnn = safe_pick([f"{CNN}F_final.yaml:OurCNN-F", "configs/arch/E6_ourcnn.yaml:OurCNN-E6"])
    pre = safe_pick(["configs/transfer/E7_ft.yaml:FT", "configs/transfer/E7_lp.yaml:LP",
                     "configs/transfer/E7b_ft32.yaml:FT32"])
    step("最终成绩模型", headline_table, cnn, pre)

    c0, c1, c2 = f"{CNN}E1_C0_none.yaml:C0-无增强", f"{CNN}E1_C1_basic.yaml:C1-基础增强", f"{CNN}E1_C2_strong.yaml:C2-强增强"
    c3, c4 = f"{CNN}E2_C3_se.yaml:C3-加SE", f"{CNN}E3_C4_ls.yaml:C4-标签平滑"
    f_spec = f"{CNN}F_final.yaml:F"
    step("汇总：E1 增强", summarize.summarize, [c0, c1, c2], "E1_aug")
    step("汇总：逐项加组件", summarize.summarize, [c0, c1, c2, c3, c4, f_spec], "E1_E3_ladder")
    step("汇总：E4 调度与早停", summarize.summarize,
         [f"{CNN}E4_A_constant.yaml:A-固定", f"{CNN}F_final.yaml:B-Cosine", f"{CNN}E4_C_plateau.yaml:C-Plateau",
          f"{CNN}E4_D_plateau_es_check.yaml:D-真实早停检查"], "E4_schedule", True)
    step("汇总：E5 逐个去掉组件", summarize.summarize,
         [f_spec, f"{CNN}E5_minus_aug.yaml:F−增强", f"{CNN}E5_minus_attention.yaml:F−注意力",
          f"{CNN}E5_minus_ls.yaml:F−标签平滑"], "E5_ablation")
    e6 = ["configs/arch/E6_ourcnn.yaml:OurCNN", "configs/arch/E6_resnet.yaml:ResNet18-CIFAR"]
    e7 = ["configs/transfer/E7_ft.yaml:FT", "configs/transfer/E7_s224.yaml:S224", "configs/transfer/E7_lp.yaml:LP",
          "configs/transfer/E7b_ft32.yaml:FT32"]
    step("汇总：E6 架构", summarize.summarize, e6, "E6_arch")
    step("汇总：E7 迁移学习", summarize.summarize, e7, "E7_transfer")

    step("曲线：E1", figures.plot_curves, [c0, c1, c2], "E1")
    step("曲线：E2/E3", figures.plot_curves, [c3, c4, f_spec], "E2_E3")
    step("曲线：E4 学习率", figures.plot_lr, [f"{CNN}E4_A_constant.yaml:A-固定", f"{CNN}F_final.yaml:B-Cosine",
                                          f"{CNN}E4_C_plateau.yaml:C-Plateau"], "E4")
    step("曲线：E6", figures.plot_curves, e6, "E6")
    step("曲线：E7", figures.plot_curves, e7, "E7")
    step("增强示例", figures.plot_augmentations)
    for spec in (cnn, pre):
        step(f"混淆矩阵 {spec}", figures.plot_confusion, spec)
        step(f"误分类 {spec}", figures.plot_errors, spec)
    step("误分类重叠", figures.plot_overlap, cnn, pre)
    step("混淆矩阵并排", figures.plot_confusion_pair, cnn, pre)

    # 校准：F 与“只改损失”的对照（F 用 ls 时对照是 E5_minus_ls，否则是 E3_C4_ls）
    ls_pair = ([f"{CNN}E5_minus_ls.yaml:CE", f"{CNN}F_final.yaml:LS"] if d.get("loss_best") == "ls"
               else [f"{CNN}F_final.yaml:CE", f"{CNN}E3_C4_ls.yaml:LS"])
    step("A1 校准", calibration.run, ls_pair + [pre], "A1")
    step("A2 鲁棒性", robustness.run, [c0, c1, c2, f_spec, e6[1], pre], "A2")
    for spec in (cnn, pre):
        step(f"Grad-CAM 样例 {spec}", gradcam.gallery, spec)
    step("A3 Grad-CAM 遮挡检查", gradcam.deletion, [cnn, e6[1], pre], "A3")
    step("t-SNE", features.tsne, [cnn, e6[1], pre])
    for spec in (cnn, pre):
        step(f"第一层卷积核 {spec}", features.filters, spec)
    step("中间特征图", features.feature_maps, cnn)
    step("SE 通道权重", features.se_weights, c3)
    step("模型开销", profile.run, e6 + e7, "E6_E7")
    step("McNemar", stats.run)


def run_p2():
    step("测试集评估（P2）", evaluate_all, P2_CONFIGS)
    f_spec = f"{CNN}F_final.yaml:F"
    ext = [f_spec, "configs/ext/X_cbam.yaml:CBAM", "configs/ext/X_cutmix.yaml:CutMix",
           "configs/ext/X_no_dropout.yaml:F−Dropout", "configs/ext/X_no_bn.yaml:F−BN",
           "configs/ext/X_step.yaml:StepLR", "configs/ext/X_onecycle.yaml:OneCycle"]
    step("汇总：P2 扩展", summarize.summarize, ext, "P2_ext")
    kd = [f_spec, "configs/arch/E6_resnet.yaml:教师ResNet18", "configs/ext/E8_kd.yaml:蒸馏学生"]
    step("汇总：E8 蒸馏", summarize.summarize, kd, "E8_kd")
    step("开销：E8 蒸馏", profile.run, kd, "E8_kd")
    step("曲线：E8", figures.plot_curves, [f_spec, "configs/ext/E8_kd.yaml:蒸馏学生"], "E8")
    step("Grad-CAM 随机化检查", gradcam.sanity, f_spec)
    step("E9 集成", ensemble.run, [f_spec, "configs/transfer/E7_ft.yaml:FT"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p2", action="store_true")
    args = parser.parse_args()
    run_p2() if args.p2 else run_p01()
    print("\n全部完成。" if not FAILED else "\n以下步骤未完成：\n  " + "\n  ".join(FAILED))


if __name__ == "__main__":
    main()
