"""训练引擎：早停、优化器分组、学习率调度、小批量过拟合、Grad-CAM。"""
import math

import pytest
import torch

from src import data as D
from src.analysis.gradcam import GradCAM
from src.engine import EarlyStopper, build_optimizer, build_scheduler
from src.losses import TrainingLoss
from src.models import build_model

TCFG = {"lr": 0.1, "head_lr_mult": 10.0, "weight_decay": 5e-4, "momentum": 0.9, "nesterov": True,
        "epochs": 10, "scheduler": "cosine", "plateau_factor": 0.1, "plateau_patience": 2,
        "step_milestones": [0.5, 0.75], "step_gamma": 0.1}
MCFG = {"arch": "ourcnn", "num_classes": 10, "attention": "se", "se_reduction": 16, "batchnorm": True, "dropout": 0.3}


def test_early_stopper():
    es = EarlyStopper(patience=3)
    improved = [True, False, True, False, False, False, False, True]
    fired = [es.step(imp, epoch) for epoch, imp in enumerate(improved, 1)]
    assert fired == [False, False, False, False, False, True, False, False]
    assert es.triggered_epoch == 6


def test_optimizer_groups():
    model = build_model(MCFG)
    opt = build_optimizer(model, TCFG)
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(1.0 if g["name"].startswith("head") else 0.1)
        assert g["weight_decay"] == (0.0 if g["name"].endswith("nodecay") else 5e-4)
        if g["name"].endswith("nodecay"):
            assert all(p.ndim <= 1 for p in g["params"])
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == sum(p.numel() for p in model.parameters())


def test_cosine_and_plateau():
    model = build_model(MCFG)
    opt = build_optimizer(model, TCFG)
    sched, mode = build_scheduler(opt, TCFG, steps_per_epoch=5)
    assert mode == "iter"
    lrs = []
    for _ in range(50):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[24] == pytest.approx(0.1 * 0.5 * (1 + math.cos(math.pi * 25 / 50)))
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12)

    opt = build_optimizer(model, TCFG)
    sched, mode = build_scheduler(opt, dict(TCFG, scheduler="plateau"), 5)
    assert mode == "plateau"
    for acc in [0.5, 0.6, 0.6, 0.6, 0.6]:
        sched.step(acc)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_overfit_small_batch():
    """64 张真实图片、无增强、无 Dropout：模型应能把训练准确率推到 100%。"""
    images, labels, _ = D.load_split("val")
    x = D.normalize_images(images[:64], "cifar").cuda()
    y = labels[:64].cuda()
    torch.manual_seed(0)
    model = build_model(dict(MCFG, dropout=0.0)).cuda()
    opt = build_optimizer(model, dict(TCFG, lr=0.05, head_lr_mult=1.0, weight_decay=0.0))
    loss_fn = TrainingLoss({"type": "ce"}, 10)
    for _ in range(150):
        loss = loss_fn(model(x), loss_fn.targets(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    assert (model(x).argmax(1) == y).float().mean().item() == 1.0
    assert loss.item() < 0.05


def test_gradcam_shapes_and_range():
    model = build_model(MCFG).eval()
    cam, logits = GradCAM(model, model.cam_layer, 32)(torch.randn(4, 3, 32, 32))
    assert cam.shape == (4, 32, 32) and logits.shape == (4, 10)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6
    frozen = build_model({**MCFG, "arch": "resnet18_imagenet", "attention": "none", "freeze_backbone": True,
                          "pretrained": False}).eval()
    cam, _ = GradCAM(frozen, frozen.cam_layer, 224)(torch.randn(2, 3, 32, 32))
    assert cam.shape == (2, 32, 32)
