"""Grad-CAM 可视化与定量检查（plan.md 6.3 节，自行实现）。

    python -m src.analysis.gradcam gallery  configs/cnn/F_final.yaml:OurCNN
    python -m src.analysis.gradcam deletion --name A3 configs/cnn/F_final.yaml:OurCNN configs/transfer/E7_ft.yaml:FT
    python -m src.analysis.gradcam sanity   configs/cnn/F_final.yaml:OurCNN

Grad-CAM（Selvaraju et al., 2017）：取目标卷积层的特征图 A^k 和目标类得分 y^c 对它的梯度，
  α_k = 梯度在空间上的平均，CAM = ReLU(Σ_k α_k A^k)，再上采样到 32×32 并归一化到 [0,1]。

定量检查（A3）：按 CAM 遮挡得分最高的 k% 像素，与遮挡同样面积的随机像素（重复 10 次）对比，
看目标类（模型在原图上的预测类）的概率下降多少。遮挡处填 0，即归一化后的均值颜色。
随机化检查（P2，Adebayo et al., 2018）：把模型权重换成随机初始化，若 CAM 几乎不变，
说明热力图主要反映输入边缘等结构，而不是模型学到的东西。
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .. import data as D
from ..evaluate import load_trained
from ..models import build_model, get_module
from .common import Runs, parse_spec, save_fig, save_table

SAMPLE_SEED = 0
FRACTIONS = [0.05, 0.10, 0.20, 0.30]
RANDOM_REPEATS = 10


class GradCAM:
    def __init__(self, model, layer_name: str, input_size: int):
        self.model = model.eval().requires_grad_(True)  # 冻结主干的模型也需要梯度流过目标层
        self.input_size = input_size
        self.layer = get_module(model, layer_name)
        self.handle = self.layer.register_forward_hook(self._hook)
        self.acts = self.grads = None

    def _hook(self, _module, _inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x32: torch.Tensor, target: torch.Tensor | None = None):
        """x32：归一化后的 32×32 输入。返回 (cam [N,32,32]，logits)。不用 AMP，保证梯度精度。"""
        x = D.resize_batch(x32, self.input_size)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        target = logits.argmax(1) if target is None else target
        logits.gather(1, target[:, None]).sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
        return cam.detach(), logits.detach()

    def remove(self):
        self.handle.remove()


def batched_cam(cam: GradCAM, x, target=None, bs=200):
    cams, logits = [], []
    for i in range(0, len(x), bs):
        c, l = cam(x[i:i + bs], None if target is None else target[i:i + bs])
        cams.append(c)
        logits.append(l)
    return torch.cat(cams), torch.cat(logits)


def overlay(ax, img_uint8, cam=None, title=""):
    ax.imshow(img_uint8.permute(1, 2, 0).numpy(), interpolation="bilinear")
    if cam is not None:
        ax.imshow(cam.cpu().numpy(), cmap="jet", alpha=0.45, vmin=0, vmax=1, interpolation="bilinear")
    ax.set_title(title, fontsize=6.5)
    ax.axis("off")


def gallery(spec):
    """正确样本（每类 1 张）与置信度最高的 10 张错误样本；错误样本同时画预测类和真实类的 CAM。"""
    runs = Runs.load(*parse_spec(spec))
    seed = runs.representative_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = runs.cfg
    model = load_trained(cfg, runs.dirs[seed], "best", device).to(memory_format=torch.contiguous_format)
    cam = GradCAM(model, model.cam_layer, cfg["data"]["input_size"])
    images, labels, names = D.load_split("test")
    logits_all, _ = runs.logits(seed, "test")
    probs = torch.softmax(torch.as_tensor(logits_all), 1)
    preds, conf = probs.argmax(1), probs.max(1).values
    rng = np.random.default_rng(SAMPLE_SEED)
    correct = [int(rng.choice(np.where((labels == c) & (preds == labels))[0])) for c in range(10)]
    wrong = np.where(preds != labels)[0]
    wrong = wrong[np.argsort(-conf[wrong].numpy())][:10].tolist()

    def norm(idx):
        return D.normalize_images(images[idx], cfg["data"]["norm"]).to(device)

    c_cam, _ = cam(norm(correct))
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.3))
    for j, k in enumerate(correct):
        overlay(axes[0, j], images[k], None, f"{names[labels[k]]}\n{conf[k]:.2f}")
        overlay(axes[1, j], images[k], c_cam[j])
    fig.suptitle(f"{runs.label}：预测正确的样本（每类 1 张，固定随机抽取）及 Grad-CAM（层 {model.cam_layer}）", fontsize=9)
    save_fig(fig, f"{runs.label}_gradcam_correct.png")

    w_pred, _ = cam(norm(wrong))
    w_true, _ = cam(norm(wrong), labels[wrong].to(device))
    fig, axes = plt.subplots(3, 10, figsize=(14, 4.9))
    for j, k in enumerate(wrong):
        overlay(axes[0, j], images[k], None, f"#{k}\n{names[labels[k]]}→{names[preds[k]]} {conf[k]:.2f}")
        overlay(axes[1, j], images[k], w_pred[j], f"预测类 {names[preds[k]]}")
        overlay(axes[2, j], images[k], w_true[j], f"真实类 {names[labels[k]]}")
    fig.suptitle(f"{runs.label}：置信度最高的 10 张误分类图（第 2 行：预测类的 CAM；第 3 行：真实类的 CAM）", fontsize=9)
    save_fig(fig, f"{runs.label}_gradcam_errors.png")
    cam.remove()


def sample_indices(labels, per_class=100):
    return D.stratified_subset(labels, per_class, seed=SAMPLE_SEED)


@torch.no_grad()
def _target_prob(model, x32, target, input_size):
    return torch.softmax(model(D.resize_batch(x32, input_size)), 1).gather(1, target[:, None])[:, 0]


def deletion_for_run(cfg, run_dir, images, labels, device) -> dict:
    """遮挡检查：遮住 CAM 最高的 k 个像素后，目标类概率下降多少。两种对照：
    - 打乱的 CAM（主对照）：用另一张图的 CAM 选同样多的像素。遮挡区域的形状、大小、连续程度与真 CAM 相同，
      只有位置与本图无关，因此“CAM 下降 > 打乱 CAM 下降”才说明 CAM 指出了对本图重要的位置；
    - 随机像素：逐像素独立随机。遮挡是分散的高频噪声，对模型的破坏方式与成块遮挡不同，与 CAM 不可直接比较，仅作参考。
    """
    model = load_trained(cfg, run_dir, "best", device).to(memory_format=torch.contiguous_format)
    size = cfg["data"]["input_size"]
    cam = GradCAM(model, model.cam_layer, size)
    x = D.normalize_images(images, cfg["data"]["norm"]).to(device)
    cams, logits = batched_cam(cam, x)
    cam.remove()
    target = logits.argmax(1)
    p0 = torch.cat([_target_prob(model, x[i:i + 200], target[i:i + 200], size) for i in range(0, len(x), 200)])
    gen = torch.Generator(device="cpu").manual_seed(SAMPLE_SEED)
    out = {"p0": p0.mean().item()}
    flat = cams.flatten(1)
    n_img, n_pix = flat.shape

    def masked_prob(indices):
        mask = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, indices, True).view(-1, 1, 32, 32)
        xm = x.masked_fill(mask, 0.0)
        return torch.cat([_target_prob(model, xm[i:i + 200], target[i:i + 200], size) for i in range(0, len(x), 200)])

    def random_donors():
        """随机错位：donor[i] 是另一张图（!= i）的下标。先随机排列，再整体错开 shift 位。"""
        order = torch.randperm(n_img, generator=gen)
        shift = int(torch.randint(1, n_img, (1,), generator=gen))
        donor = torch.empty_like(order)
        donor[order] = order.roll(shift)
        return donor.to(flat.device)

    for frac in FRACTIONS:
        k = int(round(frac * n_pix))
        top = flat.topk(k, dim=1).indices
        drop_cam = p0 - masked_prob(top)
        shuffled = [p0 - masked_prob(top[random_donors()]) for _ in range(RANDOM_REPEATS)]
        drop_shuf = torch.stack(shuffled).mean(0)
        drop_rand = torch.stack([p0 - masked_prob(torch.rand(flat.shape, generator=gen).to(device).topk(k, 1).indices)
                                 for _ in range(RANDOM_REPEATS)]).mean(0)
        out[f"{frac}"] = {"cam": drop_cam.mean().item(), "shuffled": drop_shuf.mean().item(),
                          "random": drop_rand.mean().item(),
                          "frac_cam_gt_shuffled": (drop_cam > drop_shuf).float().mean().item()}
    return out


def deletion(specs, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, _ = D.load_split("test")
    idx = sample_indices(labels)
    images, labels = images[idx], labels[idx]
    lines = ["| 模型 | 遮挡比例 | CAM 遮挡 | 打乱 CAM 遮挡（主对照） | 随机像素遮挡（参考） | CAM 下降 > 打乱 CAM 的图片比例 |",
             "|---|---:|---:|---:|---:|---:|"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, spec in enumerate(specs):
        runs = Runs.load(*parse_spec(spec))
        res = [deletion_for_run(runs.cfg, runs.dirs[s], images, labels, device) for s in sorted(runs.dirs)]
        curve = {key: np.array([[r[f"{f}"][key] for f in FRACTIONS] for r in res]) for key in ("cam", "shuffled", "random")}
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]
        xs = [f * 100 for f in FRACTIONS]
        ax.plot(xs, curve["cam"].mean(0), "o-", color=color, label=f"{runs.label} CAM")
        ax.plot(xs, curve["shuffled"].mean(0), "s--", color=color, label=f"{runs.label} 打乱 CAM")
        for j, f in enumerate(FRACTIONS):
            frac_gt = np.mean([r[f"{f}"]["frac_cam_gt_shuffled"] for r in res])
            lines.append(f"| {runs.label} | {f * 100:.0f}% | {curve['cam'][:, j].mean():.3f} | "
                         f"{curve['shuffled'][:, j].mean():.3f} | {curve['random'][:, j].mean():.3f} | {frac_gt * 100:.1f}% |")
    ax.set(xlabel="遮挡像素比例 %", ylabel="目标类概率平均下降", title="Grad-CAM 遮挡检查（1000 张测试图，种子均值）")
    ax.legend(fontsize=7)
    save_fig(fig, f"{name}_gradcam_deletion.png")
    lines.append("\n注：表中为目标类（模型在原图上的预测类）概率的平均下降。遮挡处填 0（归一化后的均值颜色）。"
                 "主对照“打乱 CAM”用另一张图的 CAM 选取同样多的像素，遮挡形状与真 CAM 相同、只有位置无关；"
                 "“随机像素”逐像素独立抽取，相当于高频噪声，破坏方式不同，与 CAM 不可直接比较，仅作参考。"
                 "两种对照均重复 10 次取平均。遮挡会使输入偏离训练分布，结果只作辅助证据。")
    save_table(lines, f"{name}_gradcam_deletion.md")
    print("\n".join(lines))


def sanity(spec, n=200):
    """模型参数随机化检查：训练后模型与随机初始化模型的 CAM 的 Spearman 相关。"""
    runs = Runs.load(*parse_spec(spec))
    seed = runs.representative_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, _ = D.load_split("test")
    idx = sample_indices(labels, n // 10)
    x = D.normalize_images(images[idx], runs.cfg["data"]["norm"]).to(device)
    trained = load_trained(runs.cfg, runs.dirs[seed], "best", device).to(memory_format=torch.contiguous_format)
    cam_t, logits = GradCAM(trained, trained.cam_layer, runs.cfg["data"]["input_size"])(x)
    torch.manual_seed(SAMPLE_SEED)
    random_model = build_model(dict(runs.cfg["model"], pretrained=False)).to(device)
    cam_r, _ = GradCAM(random_model, random_model.cam_layer, runs.cfg["data"]["input_size"])(x, logits.argmax(1))
    rho = [spearmanr(a.flatten().cpu(), b.flatten().cpu())[0] for a, b in zip(cam_t, cam_r)]
    rho = np.nan_to_num(np.array(rho))
    lines = ["| 模型 | 图片数 | Spearman ρ 均值 | 中位数 |", "|---|---:|---:|---:|",
             f"| {runs.label} | {len(rho)} | {rho.mean():.3f} | {np.median(rho):.3f} |",
             "\n注：比较训练后模型与随机初始化模型（同一目标类）的 CAM；相关性低说明 CAM 依赖于学到的参数。"]
    save_table(lines, f"{runs.label}_gradcam_sanity.md")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gallery")
    p.add_argument("spec")
    p = sub.add_parser("deletion")
    p.add_argument("specs", nargs="+")
    p.add_argument("--name", default="A3")
    p = sub.add_parser("sanity")
    p.add_argument("spec")
    args = parser.parse_args()
    if args.cmd == "gallery":
        gallery(args.spec)
    elif args.cmd == "deletion":
        deletion(args.specs, args.name)
    else:
        sanity(args.spec)


if __name__ == "__main__":
    main()
