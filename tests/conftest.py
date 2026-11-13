import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 用于测试“配置去重”的一组假想决策（不读取真实的 configs/decisions.yaml）
FAKE_DECISIONS = {
    "epochs_32": 100, "epochs_224": 15, "cnn_lr": 0.05, "aug_best": "basic", "attention_best": "none",
    "loss_best": "ce", "cnn_lr_F": 0.05, "resnet_lr": 0.1, "ft_lr": 0.003, "s224_lr": 0.05, "lp_lr": 0.1,
    "ft32_lr": 0.003, "kd_T": 4.0, "kd_alpha": 0.9,
}


@pytest.fixture
def fake_decisions(monkeypatch):
    from src import utils

    decisions = dict(FAKE_DECISIONS)
    monkeypatch.setattr(utils, "load_decisions", lambda path=None: decisions)
    return decisions
