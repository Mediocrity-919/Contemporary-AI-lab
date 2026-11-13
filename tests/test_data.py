"""数据管线：形状、增强的确定性、采样器、分层子集、与 worker 数无关。"""
import torch
from torch.utils.data import DataLoader

from src import data as D


def _small():
    images, labels, names = D.load_split("val")
    return images[:64], labels[:64], names


def test_splits_shapes():
    for split, n in [("train", 45000), ("val", 5000), ("test", 10000)]:
        images, labels, names = D.load_split(split)
        assert images.shape == (n, 3, 32, 32) and images.dtype == torch.uint8
        assert torch.bincount(labels, minlength=10).tolist() == [n // 10] * 10
        assert len(names) == 10


def test_transforms_shape_and_dtype():
    images, _, _ = _small()
    for aug in ("none", "basic", "strong"):
        for norm in ("cifar", "imagenet"):
            out = D.build_train_transform(aug, norm)(images[0])
            assert out.shape == (3, 32, 32) and out.dtype == torch.float32


def test_none_aug_equals_eval_normalization():
    images, _, _ = _small()
    a = D.build_train_transform("none", "cifar")(images[0])
    b = D.normalize_images(images[:1], "cifar")[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_augmentation_is_deterministic_per_sample_and_epoch():
    images, labels, _ = _small()
    ds = D.AugmentedDataset(images, labels, D.build_train_transform("strong", "cifar"), run_seed=0)
    assert torch.equal(ds[(5, 1)][0], ds[(5, 1)][0])
    assert not torch.equal(ds[(5, 1)][0], ds[(5, 2)][0])
    ds2 = D.AugmentedDataset(images, labels, D.build_train_transform("strong", "cifar"), run_seed=1)
    assert not torch.equal(ds[(5, 1)][0], ds2[(5, 1)][0])


def test_main_process_augmentation_does_not_touch_global_rng():
    images, labels, _ = _small()
    ds = D.AugmentedDataset(images, labels, D.build_train_transform("strong", "cifar"), run_seed=0)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    ds[(0, 1)]
    assert torch.equal(torch.rand(3), expected)


def test_sampler_is_deterministic_permutation():
    s = D.EpochSampler(100, run_seed=3)
    s.set_epoch(1)
    a = list(s)
    s.set_epoch(1)
    assert a == list(s)
    assert sorted(i for i, _ in a) == list(range(100)) and all(e == 1 for _, e in a)
    s.set_epoch(2)
    assert [i for i, _ in s] != [i for i, _ in a]


def test_stratified_subset():
    _, labels, _ = D.load_split("train")
    idx = D.stratified_subset(labels, 500)
    assert len(idx) == 5000 and len(set(idx.tolist())) == 5000
    assert torch.bincount(labels[idx], minlength=10).tolist() == [500] * 10
    assert torch.equal(idx, D.stratified_subset(labels, 500))


def test_result_independent_of_num_workers():
    images, labels, _ = _small()
    ds = D.AugmentedDataset(images, labels, D.build_train_transform("strong", "cifar"), run_seed=7)
    batches = []
    for workers in (0, 2):
        sampler = D.EpochSampler(len(ds), 7)
        sampler.set_epoch(4)
        loader = DataLoader(ds, batch_size=16, sampler=sampler, num_workers=workers)
        batches.append(torch.cat([x for x, _ in loader]))
    assert torch.equal(batches[0], batches[1])


def test_renormalize_and_resize():
    images, _, _ = _small()
    x_c = D.normalize_images(images[:4], "cifar")
    x_i = D.normalize_images(images[:4], "imagenet")
    assert torch.allclose(D.renormalize(x_c, "cifar", "imagenet"), x_i, atol=1e-5)
    assert D.resize_batch(x_c, 224).shape == (4, 3, 224, 224)
    assert D.resize_batch(x_c, 32) is x_c
