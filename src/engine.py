"""训练与验证循环。

一次运行（一个配置 + 一个种子）的产物，均保存在运行目录下：
  config.yaml        展开后的完整配置（含哈希）
  env.json           软硬件环境
  log.csv            逐轮日志
  best.pt            验证集准确率最高的模型（并列时取验证 loss 更低者）
  es_best.pt         早停（影子模式）触发时的最优模型；未触发时与 best.pt 相同
  last.pt            最后一轮的完整状态（模型、优化器、调度器、随机数状态），用于断点续训
  val_best_logits.npz  best.pt 在验证集上的 logits（校准、选择决策等使用）
  metrics.json       运行完成的标志与摘要
"""
from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .losses import TrainingLoss, apply_mix
from .models import build_model, count_params, set_train_mode, HEAD_PREFIX
from .utils import (atomic_torch_save, config_hash, env_info, rng_state, save_json, save_yaml,
                    seed_everything, set_rng_state, load_config, run_dir_for)


# ----------------------------------------------------------------------------- 早停
class EarlyStopper:
    """连续 patience 轮验证集没有改进时触发。

    默认以“影子模式”运行：只记录触发轮次并保存当时的最优模型，训练照常进行到最大轮数。
    这样“开启早停”和“不开启早停”来自同一次训练，可直接配对比较（plan.md 5.2 节）。
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.triggered_epoch = None
        self.time_to_stop = None

    def step(self, improved: bool, epoch: int) -> bool:
        """返回本轮是否“首次触发”。"""
        self.counter = 0 if improved else self.counter + 1
        if self.triggered_epoch is None and self.counter >= self.patience:
            self.triggered_epoch = epoch
            return True
        return False

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__.update(state)


# ----------------------------------------------------------------------------- 优化器与调度
def build_optimizer(model, tcfg: dict) -> torch.optim.Optimizer:
    """SGD（Nesterov 动量）。分类层学习率 = lr × head_lr_mult；BN 参数和 bias 不做权重衰减。"""
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith(HEAD_PREFIX)
        no_decay = param.ndim <= 1
        key = (is_head, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": tcfg["lr"] * (tcfg.get("head_lr_mult", 1.0) if is_head else 1.0),
                "weight_decay": 0.0 if no_decay else tcfg["weight_decay"],
                "name": ("head" if is_head else "backbone") + ("_nodecay" if no_decay else ""),
            }
        groups[key]["params"].append(param)
    return torch.optim.SGD(list(groups.values()), lr=tcfg["lr"], momentum=tcfg["momentum"],
                           nesterov=tcfg["nesterov"])


def build_scheduler(optimizer, tcfg: dict, steps_per_epoch: int):
    """返回 (调度器, 步进方式)。步进方式：'iter' 每个 batch、'epoch' 每轮、'plateau' 按验证准确率、None 不调度。"""
    kind = tcfg["scheduler"]
    total = tcfg["epochs"] * steps_per_epoch
    if kind == "constant":
        return None, None
    if kind == "cosine":
        lam = lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lam), "iter"
    if kind == "onecycle":
        max_lr = [g["lr"] for g in optimizer.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total), "iter"
    if kind == "step":
        milestones = [int(round(f * tcfg["epochs"])) for f in tcfg["step_milestones"]]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=tcfg["step_gamma"]), "epoch"
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=tcfg["plateau_factor"], patience=tcfg["plateau_patience"]), "plateau"
    raise ValueError(f"未知调度器: {kind}")


# ----------------------------------------------------------------------------- 推理
def prepare_input(x: torch.Tensor, input_size: int, channels_last: bool) -> torch.Tensor:
    x = D.resize_batch(x, input_size)
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x


@torch.no_grad()
def predict_logits(model, eval_set: D.EvalSet, cfg: dict) -> torch.Tensor:
    """返回整个评估集合的 logits（float32，CPU）。"""
    model.eval()
    amp = cfg["train"]["amp"] and torch.cuda.is_available()
    outs = []
    for x, _ in eval_set.batches(cfg.get("system", {}).get("eval_batch_size", 500)):
        x = prepare_input(x, cfg["data"]["input_size"], cfg["train"]["channels_last"])
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            outs.append(model(x).float())
    return torch.cat(outs).cpu()


def nll_and_acc(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """评估指标统一用不带平滑的交叉熵（NLL），使 ce 与 ls 训练的模型可以直接比较。"""
    labels = labels.cpu()
    return F.cross_entropy(logits, labels).item(), (logits.argmax(1) == labels).float().mean().item()


# ----------------------------------------------------------------------------- 知识蒸馏教师
def load_teacher(kd: dict, device):
    """教师 = 指定配置下验证集准确率最高的那个种子的 best.pt；固定参数、eval 模式。"""
    from .utils import ROOT, completed_seeds, load_json

    tcfg = load_config(ROOT / kd["teacher_config"])
    runs = completed_seeds(tcfg, tcfg["meta"].get("seeds", [0, 1, 2]))
    if not runs:
        raise RuntimeError(f"教师模型尚未训练完成: {kd['teacher_config']}")
    seed = kd.get("teacher_seed", "best")
    if seed == "best":
        seed = max(runs, key=lambda s: load_json(runs[s] / "metrics.json")["best"]["val_acc"])
    teacher = build_model(tcfg["model"]).to(device)
    teacher.load_state_dict(torch.load(runs[seed] / "best.pt", map_location=device, weights_only=True)["model"])
    teacher.eval().requires_grad_(False)
    return teacher, tcfg, int(seed)


# ----------------------------------------------------------------------------- 主循环
LOG_FIELDS = ["epoch", "lr", "head_lr", "train_loss", "train_acc", "clean_train_loss", "clean_train_acc",
              "val_loss", "val_acc", "best_val_acc", "epoch_train_time", "epoch_eval_time", "elapsed"]


def _write_log(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fit(cfg: dict, seed: int, run_dir: Path | None = None, stop_after_epochs: int | None = None,
        verbose: bool = True) -> dict | None:
    """训练一个配置的一个种子。已完成则直接返回摘要；存在 last.pt 则从中断处续训。

    stop_after_epochs 仅用于测试断点续训：训练指定轮数后直接返回 None。
    """
    run_dir = Path(run_dir or run_dir_for(cfg, seed))
    if (run_dir / "metrics.json").exists():
        from .utils import load_json
        return load_json(run_dir / "metrics.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    tcfg, dcfg, mcfg = cfg["train"], cfg["data"], cfg["model"]
    log = print if verbose else (lambda *a, **k: None)

    seed_everything(seed, tcfg["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = tcfg["amp"] and device.type == "cuda"
    save_yaml({**cfg, "hash": config_hash(cfg), "seed": seed}, run_dir / "config.yaml")
    save_json(env_info(), run_dir / "env.json")

    # 数据
    train_loader, sampler = D.make_train_loader(cfg, seed)
    val_set = D.make_eval_set("val", cfg, device)
    clean_set = D.make_eval_set("train", cfg, device, per_class=dcfg["clean_train_subset"] // mcfg["num_classes"])

    # 模型、损失、优化器
    model = build_model(mcfg).to(device)
    if tcfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    criterion = TrainingLoss(cfg["loss"], mcfg["num_classes"])
    teacher = teacher_cfg = None
    if cfg["loss"].get("kd"):
        teacher, teacher_cfg, teacher_seed = load_teacher(cfg["loss"]["kd"], device)
        log(f"[kd] 教师 {cfg['loss']['kd']['teacher_config']} seed{teacher_seed}")
    optimizer = build_optimizer(model, tcfg)
    scheduler, sched_mode = build_scheduler(optimizer, tcfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    freeze = mcfg.get("freeze_backbone", False)

    # 状态（可从 last.pt 恢复）
    start_epoch, global_step, elapsed = 1, 0, 0.0
    history: list[dict] = []
    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}
    stopper = EarlyStopper(tcfg["es_patience"])
    last_path = run_dir / "last.pt"
    if last_path.exists():
        # 先加载到 CPU：随机数状态必须是 CPU 张量；模型和优化器的 load_state_dict 会自行搬到参数所在设备
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        stopper.load_state_dict(state["stopper"])
        start_epoch, global_step, elapsed = state["epoch"] + 1, state["global_step"], state["elapsed"]
        history, best = state["history"], state["best"]
        set_rng_state(state["rng"])
        log(f"[resume] 从第 {state['epoch']} 轮之后继续")

    params = count_params(model)
    log(f"[run] {cfg['meta']['name']} seed={seed} hash={config_hash(cfg)} params={params['total']:,} "
        f"trainable={params['trainable']:,} dir={run_dir}")
    max_steps = tcfg.get("max_steps")
    head_group = next((i for i, g in enumerate(optimizer.param_groups) if g["name"].startswith("head")), 0)
    finished_early = False

    diverged = None
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        sampler.set_epoch(epoch)
        set_train_mode(model, freeze)
        _sync()
        t0 = time.perf_counter()
        loss_sum = correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = prepare_input(x, dcfg["input_size"], tcfg["channels_last"])
            targets = criterion.targets(y)
            if dcfg.get("mix", "none") != "none":
                gen = torch.Generator().manual_seed(seed * 10_000_019 + global_step)
                x, targets = apply_mix(x, targets, dcfg["mix"], dcfg["mix_alpha"], gen)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        tx = D.renormalize(x, dcfg["norm"], teacher_cfg["data"]["norm"])
                        tx = prepare_input(tx, teacher_cfg["data"]["input_size"], tcfg["channels_last"])
                        teacher_logits = teacher(tx)
            loss = criterion(logits, targets, teacher_logits)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if sched_mode == "iter":
                scheduler.step()
            global_step += 1
            n = y.size(0)
            loss_sum += loss.detach() * n  # 在 GPU 上累加，避免每步同步
            correct += (logits.argmax(1) == y).sum()
            seen += n
            if max_steps and global_step >= max_steps:
                finished_early = True
                break
        _sync()
        train_time = time.perf_counter() - t0
        loss_sum, correct = float(loss_sum), float(correct)

        # 评估：验证集 + 固定的干净训练子集（eval 模式、无增强）
        t1 = time.perf_counter()
        val_loss, val_acc = nll_and_acc(predict_logits(model, val_set, cfg), val_set.y)
        clean_loss, clean_acc = nll_and_acc(predict_logits(model, clean_set, cfg), clean_set.y)
        eval_time = time.perf_counter() - t1
        elapsed += train_time + eval_time

        lr_now = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[head_group]["lr"]
        if sched_mode == "epoch":
            scheduler.step()
        elif sched_mode == "plateau":
            scheduler.step(val_acc)

        improved = val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"])
        if improved:
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
            atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                               "val_loss": val_loss}, run_dir / "best.pt")
        if stopper.step(improved, epoch):
            stopper.time_to_stop = elapsed
            shutil.copyfile(run_dir / "best.pt", run_dir / "es_best.pt")
            log(f"[early-stop] 第 {epoch} 轮触发（最优轮 {best['epoch']}）"
                + ("，停止训练" if tcfg["early_stopping"] else "，影子模式：继续训练"))

        row = {"epoch": epoch, "lr": lr_now, "head_lr": head_lr, "train_loss": loss_sum / max(seen, 1),
               "train_acc": correct / max(seen, 1), "clean_train_loss": clean_loss, "clean_train_acc": clean_acc,
               "val_loss": val_loss, "val_acc": val_acc, "best_val_acc": best["val_acc"],
               "epoch_train_time": train_time, "epoch_eval_time": eval_time, "elapsed": elapsed}
        history.append(row)
        _write_log(run_dir / "log.csv", history)
        log(f"[{epoch:3d}/{tcfg['epochs']}] lr={lr_now:.4g} train {row['train_loss']:.4f}/{row['train_acc']:.4f} "
            f"clean {clean_loss:.4f}/{clean_acc:.4f} val {val_loss:.4f}/{val_acc:.4f} "
            f"best {best['val_acc']:.4f}@{best['epoch']} ({train_time:.1f}s+{eval_time:.1f}s)", flush=True)

        atomic_torch_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(), "stopper": stopper.state_dict(), "epoch": epoch,
            "global_step": global_step, "elapsed": elapsed, "history": history, "best": best, "rng": rng_state(),
        }, last_path)

        if not math.isfinite(row["train_loss"]):
            # 训练发散（例如去掉 BN 后学习率过大）：记录并停止，不再白跑剩余轮数
            diverged = epoch
            log(f"[diverged] 第 {epoch} 轮训练损失为 {row['train_loss']}，判定为发散，停止训练")
            break
        if finished_early or (tcfg["early_stopping"] and stopper.triggered_epoch is not None):
            break
        if stop_after_epochs is not None and epoch - start_epoch + 1 >= stop_after_epochs:
            return None

    # 收尾：影子早停未触发时，es_best 等于 best
    if stopper.triggered_epoch is None:
        shutil.copyfile(run_dir / "best.pt", run_dir / "es_best.pt")
    best_state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_state["model"])
    val_logits = predict_logits(model, val_set, cfg)
    np.savez_compressed(run_dir / "val_best_logits.npz", logits=val_logits.numpy(), labels=val_set.y.cpu().numpy())

    es_epoch = stopper.triggered_epoch
    es_best_epoch = torch.load(run_dir / "es_best.pt", map_location="cpu", weights_only=True)["epoch"]
    summary = {
        "name": cfg["meta"]["name"], "config_file": cfg["meta"].get("config_file"), "hash": config_hash(cfg),
        "seed": seed, "params": params, "epochs_run": len(history),
        "best": best,
        "last": {k: history[-1][k] for k in ("epoch", "val_acc", "val_loss", "clean_train_acc", "train_acc")},
        "early_stopping": {
            "mode": "real" if tcfg["early_stopping"] else "shadow", "patience": tcfg["es_patience"],
            "triggered_epoch": es_epoch, "best_epoch": es_best_epoch,
            "val_acc": history[es_best_epoch - 1]["val_acc"],
            "epochs_used": es_epoch if es_epoch is not None else len(history),
            "time_to_stop": stopper.time_to_stop if es_epoch is not None else elapsed,
        },
        "time": {"total": elapsed, "train": sum(h["epoch_train_time"] for h in history),
                 "mean_epoch_train": float(np.mean([h["epoch_train_time"] for h in history]))},
        "stopped_by_max_steps": finished_early,
        "diverged_epoch": diverged,
    }
    save_json(summary, run_dir / "metrics.json")
    log(f"[done] best val {best['val_acc']:.4f} @ epoch {best['epoch']}；用时 {elapsed / 60:.1f} 分钟")
    return summary
