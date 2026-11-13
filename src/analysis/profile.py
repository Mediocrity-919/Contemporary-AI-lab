"""模型开销对比：参数量、MACs、推理吞吐、checkpoint 大小、训练耗时（plan.md 5.3 节）。

    python -m src.analysis.profile --name E6_E7 configs/cnn/F_final.yaml:OurCNN configs/arch/E6_resnet.yaml:ResNet18-CIFAR ...

- MACs 由 thop 统计（乘加次数，只计卷积、全连接、BN 等标准层；SE 中的 mean / 乘法未计入，影响很小）。
- 推理吞吐：batch 256、fp16 AMP、channels_last，预热 10 次后计时 50 次，输入已在显存中（不含数据读取）。
- 计时请在接电源、高性能模式、GPU 无其他任务时进行。
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from thop import profile as thop_profile

from ..models import build_model, count_params
from .common import Runs, mean_std, parse_spec, save_table


@torch.no_grad()
def throughput(model, input_size, device, batch=256, warmup=30, iters=30, rounds=5) -> float:
    """推理吞吐（img/s）：充分预热后测 rounds 轮、每轮 iters 个 batch，取中位数，减少 GPU 频率爬升带来的偏差。"""
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    x = torch.randn(batch, 3, input_size, input_size, device=device).to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        speeds = []
        for _ in range(rounds):
            t = time.perf_counter()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
            speeds.append(batch * iters / (time.perf_counter() - t))
    return float(np.median(speeds))


def run(specs, name):
    device = torch.device("cuda")
    lines = ["| 模型 | 输入 | 参数量 | 可训练参数 | MACs | 推理吞吐 img/s | checkpoint MB | 每轮训练 s | 总训练 min | 测试准确率 % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for spec in specs:
        runs = Runs.load(*parse_spec(spec))
        mcfg = dict(runs.cfg["model"], pretrained=False)
        size = runs.cfg["data"]["input_size"]
        model = build_model(mcfg)
        params = count_params(model)
        macs, _ = thop_profile(copy.deepcopy(model), inputs=(torch.randn(1, 3, size, size),), verbose=False)
        ips = throughput(model, size, device)
        seeds = sorted(runs.dirs)
        if seeds:
            ckpt_mb = (runs.dirs[seeds[0]] / "best.pt").stat().st_size / 2**20
            epoch_s = mean_std(runs.timing(s)[0] for s in seeds)
            total_min = mean_std(runs.timing(s)[1] for s in seeds)
            tests = [runs.eval_report(s) for s in seeds]
            test = (f"{np.mean([t['accuracy'] for t in tests]) * 100:.2f} ± {np.std([t['accuracy'] for t in tests], ddof=1) * 100 if len(tests) > 1 else 0:.2f}"
                    if all(tests) else "—")
            timing = f"{ckpt_mb:.1f} | {epoch_s[0]:.1f} ± {epoch_s[1]:.1f} | {total_min[0]:.1f} | {test}"
        else:
            timing = "— | — | — | —"
        lines.append(f"| {runs.label} | {size} | {params['total'] / 1e6:.2f}M | {params['trainable'] / 1e6:.3f}M | "
                     f"{macs / 1e9:.3f}G | {ips:,.0f} | {timing} |")
    lines.append("\n注：推理吞吐为 RTX 4060 Laptop、batch 256、fp16 下的纯前向速度；训练耗时按每轮中位数估算，总耗时含每轮验证。")
    save_table(lines, f"{name}_profile.md")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.specs, args.name)


if __name__ == "__main__":
    main()
