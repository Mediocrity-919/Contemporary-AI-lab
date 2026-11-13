"""配置系统：继承、占位符、哈希与去重。"""
import pytest

from src.run_experiments import expand_paths
from src.utils import (ROOT, UnresolvedDecision, _coerce, config_hash, expand_sweep, load_config, load_variants,
                       resolve_placeholders)


def test_inherit_and_placeholders_unresolved():
    cfg = load_config(ROOT / "configs/cnn/E1_C2_strong.yaml", resolve=False)
    assert cfg["data"]["aug"] == "strong"
    assert cfg["train"]["lr"] == "${cnn_lr}"
    assert cfg["train"]["batch_size"] == 128  # 来自 base.yaml
    with pytest.raises(UnresolvedDecision):
        resolve_placeholders(cfg, {"cnn_lr": None})


def test_scientific_notation_is_float():
    assert _coerce({"a": "1e-4", "b": "5.0e-4", "c": "abc", "d": 3}) == {"a": 1e-4, "b": 5e-4, "c": "abc", "d": 3}


def test_hash_ignores_meta_and_system(fake_decisions):
    a = load_config(ROOT / "configs/cnn/E1_C1_basic.yaml")
    b = load_config(ROOT / "configs/cnn/E1_C1_basic.yaml", {"meta.name": "x", "system.num_workers": 0})
    c = load_config(ROOT / "configs/cnn/E1_C1_basic.yaml", {"train.lr": 0.1})
    assert config_hash(a) == config_hash(b) != config_hash(c)


def test_identical_configs_share_hash(fake_decisions):
    """F 与它在 E1 中对应的配置完全相同时，应得到相同哈希（从而复用结果）。"""
    h = lambda p, o=None: config_hash(load_config(ROOT / p, o))
    assert h("configs/cnn/F_final.yaml") == h("configs/cnn/E1_C1_basic.yaml")
    assert h("configs/cnn/E5_minus_ls.yaml") == h("configs/cnn/F_final.yaml")          # F 本来就用 ce
    assert h("configs/cnn/E5_minus_attention.yaml") == h("configs/cnn/F_final.yaml")   # F 本来就没有注意力
    assert h("configs/cnn/P0_pilot_R1.yaml") == h("configs/cnn/T_cnn.yaml", {"train.lr": 0.05})
    assert h("configs/arch/E6_ourcnn.yaml") == h("configs/cnn/F_final.yaml")           # cnn_lr_F == cnn_lr
    fake_decisions["attention_best"] = "se"
    assert h("configs/cnn/F_final.yaml") == h("configs/cnn/E2_C3_se.yaml")
    assert h("configs/cnn/E2_C3_se.yaml", {"model.attention": "none"}) == h("configs/cnn/E1_C1_basic.yaml")


def test_sweep_expansion(fake_decisions):
    variants = load_variants(ROOT / "configs/ext/T_kd.yaml")
    assert len(variants) == 4
    assert {(v["loss"]["kd"]["T"], v["loss"]["kd"]["alpha"]) for v in variants} == {(2.0, 0.5), (2.0, 0.9), (4.0, 0.5), (4.0, 0.9)}
    assert expand_sweep({"meta": {"name": "a"}}) == [("a", {})]


def test_all_configs_resolve(fake_decisions):
    paths = expand_paths(["configs/**/*.yaml"])
    assert len(paths) > 30
    for p in paths:
        for cfg in load_variants(p):
            assert cfg["model"]["arch"] in ("ourcnn", "resnet18_cifar", "resnet18_imagenet")
            assert cfg["data"]["input_size"] in (32, 224)
            if cfg["model"]["arch"] == "resnet18_imagenet":
                assert cfg["data"]["norm"] == "imagenet"   # 预训练模型及其随机初始化对照都用 ImageNet 归一化
            if cfg["model"]["freeze_backbone"]:
                assert cfg["model"]["pretrained"]
