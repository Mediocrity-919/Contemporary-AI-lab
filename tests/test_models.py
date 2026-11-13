"""模型：前向形状、参数量、特征维度、Grad-CAM 目标层、冻结主干时 BN 不更新。"""
import pytest
import torch

from src.engine import build_optimizer
from src.models import build_model, count_params, forward_features, get_module, set_train_mode

BASE = {"num_classes": 10, "attention": "none", "se_reduction": 16, "batchnorm": True, "dropout": 0.3,
        "pretrained": False, "freeze_backbone": False}


@pytest.mark.parametrize("arch,attention,size,feat", [
    ("ourcnn", "none", 32, 256), ("ourcnn", "se", 32, 256), ("ourcnn", "cbam", 32, 256),
    ("resnet18_cifar", "none", 32, 512), ("resnet18_imagenet", "none", 224, 512), ("resnet18_imagenet", "none", 32, 512),
])
def test_forward_shapes(arch, attention, size, feat):
    model = build_model({**BASE, "arch": arch, "attention": attention}).eval()
    x = torch.randn(2, 3, size, size)
    assert model(x).shape == (2, 10)
    assert forward_features(model, x).shape == (2, feat)
    get_module(model, model.cam_layer)


def test_ourcnn_without_bn():
    model = build_model({**BASE, "arch": "ourcnn", "batchnorm": False})
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_param_counts():
    n = count_params(build_model({**BASE, "arch": "ourcnn"}))["total"]
    n_se = count_params(build_model({**BASE, "arch": "ourcnn", "attention": "se"}))["total"]
    assert 1.1e6 < n < 1.2e6
    assert 0 < n_se - n < 0.05 * n   # SE 只增加很少的参数
    assert 11.0e6 < count_params(build_model({**BASE, "arch": "resnet18_cifar"}))["total"] < 11.3e6


def test_frozen_backbone_keeps_bn_statistics():
    """冻结主干：只有 fc 可训练；训练模式下主干仍为 eval，BN 的 running mean/var 不变。"""
    model = build_model({**BASE, "arch": "resnet18_imagenet", "freeze_backbone": True})
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["fc.weight", "fc.bias"]
    bn_before = {k: v.clone() for k, v in model.state_dict().items() if "running" in k or "num_batches" in k}
    fc_before = model.fc.weight.detach().clone()
    opt = build_optimizer(model, {"lr": 0.1, "head_lr_mult": 1.0, "weight_decay": 1e-4, "momentum": 0.9,
                                  "nesterov": True})
    set_train_mode(model, freeze_backbone=True)
    assert model.fc.training and not model.layer1.training and not model.bn1.training
    for _ in range(2):
        loss = model(torch.randn(4, 3, 64, 64)).logsumexp(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = model.state_dict()
    assert all(torch.equal(bn_before[k], after[k]) for k in bn_before)
    assert not torch.equal(fc_before, model.fc.weight)


def test_unfrozen_model_updates_bn_statistics():
    model = build_model({**BASE, "arch": "ourcnn"})
    before = model.stage1.conv1[1].running_mean.clone()
    set_train_mode(model, freeze_backbone=False)
    model(torch.randn(4, 3, 32, 32))
    assert not torch.equal(before, model.stage1.conv1[1].running_mean)
