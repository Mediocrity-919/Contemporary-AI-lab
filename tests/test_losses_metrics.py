"""损失函数、评估指标、温度缩放、McNemar / Holm。"""
import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.calibration import fit_temperature
from src.analysis.stats import holm, mcnemar_exact
from src.losses import TrainingLoss, apply_mix, kd_kl, make_targets, soft_cross_entropy
from src.metrics import confusion_matrix, ece, full_report, per_class_report, wilson_ci


def test_label_smoothing_matches_pytorch():
    logits, y = torch.randn(32, 10), torch.randint(0, 10, (32,))
    t = make_targets(y, 10, 0.1)
    assert torch.allclose(t.sum(1), torch.ones(32))
    assert torch.allclose(soft_cross_entropy(logits, t), F.cross_entropy(logits, y, label_smoothing=0.1), atol=1e-6)
    assert torch.allclose(soft_cross_entropy(logits, make_targets(y, 10)), F.cross_entropy(logits, y), atol=1e-6)


def test_kd_terms():
    s, t = torch.randn(16, 10), torch.randn(16, 10)
    assert kd_kl(s, s, 4.0).abs() < 1e-6 and kd_kl(s, t, 4.0) > 0
    loss = TrainingLoss({"type": "ce", "kd": {"T": 4.0, "alpha": 0.9}}, 10)
    y = torch.randint(0, 10, (16,))
    expected = 0.1 * F.cross_entropy(s, y) + 0.9 * kd_kl(s, t, 4.0)
    assert torch.allclose(loss(s, loss.targets(y), t), expected, atol=1e-6)


def test_mix_targets():
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    for kind in ("mixup", "cutmix"):
        xm, t = apply_mix(x, make_targets(y, 10), kind, 1.0, torch.Generator().manual_seed(0))
        assert xm.shape == x.shape and torch.allclose(t.sum(1), torch.ones(8))


def test_confusion_and_per_class():
    labels = np.array([0, 0, 1, 1, 2])
    preds = np.array([0, 1, 1, 1, 0])
    cm = confusion_matrix(labels, preds, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    rep = per_class_report(cm)
    assert rep["accuracy"] == [0.5, 1.0, 0.0]
    assert np.isclose(rep["precision"][1], 2 / 3)


def test_ece_and_report():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(probs, np.array([0, 1])) == 0.0
    assert np.isclose(ece(np.array([[0.9, 0.1]] * 10), np.array([0] * 5 + [1] * 5)), 0.4)
    rep = full_report(np.log(np.array([[0.9, 0.1], [0.2, 0.8]])), np.array([0, 1]), ["a", "b"])
    assert rep["accuracy"] == 1.0 and rep["confusion_matrix"] == [[1, 0], [0, 1]]


def test_wilson():
    lo, hi = wilson_ci(9000, 10000)
    assert lo < 0.9 < hi and 0.005 < (hi - lo) / 2 < 0.007   # 约 ±0.59 个百分点


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(5000, 10)) * 2
    labels = np.array([rng.choice(10, p=np.exp(l) / np.exp(l).sum()) for l in true_logits])
    t = fit_temperature(true_logits * 3.0, labels)   # 过度自信 3 倍 → T≈3
    assert 2.6 < t < 3.4


def test_mcnemar_and_holm():
    a = np.array([True] * 30 + [False] * 10 + [True] * 60)
    b = np.array([False] * 30 + [True] * 10 + [True] * 60)
    res = mcnemar_exact(a, b)
    assert (res["a_only"], res["b_only"]) == (30, 10) and res["p"] < 0.01
    assert mcnemar_exact(a, a)["p"] == 1.0
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
