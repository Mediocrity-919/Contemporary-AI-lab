"""Download CIFAR-10 and save reproducible, stratified PyTorch splits."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torchvision.datasets import CIFAR10


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_indices(labels, seed):
    """Reserve 500 images per class for validation, without replacement."""
    generator = torch.Generator().manual_seed(seed)
    train_parts, val_parts = [], []
    for class_id in range(10):
        indices = torch.where(labels == class_id)[0]
        assert len(indices) == 5000
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val_parts.append(indices[:500])
        train_parts.append(indices[500:])
    train_indices = torch.cat(train_parts)
    val_indices = torch.cat(val_parts)
    train_indices = train_indices[
        torch.randperm(len(train_indices), generator=generator)
    ]
    val_indices = val_indices[
        torch.randperm(len(val_indices), generator=generator)
    ]
    # Sorting the combined indices proves complete coverage without overlap.
    combined = torch.cat([train_indices, val_indices]).sort().values
    assert torch.equal(combined, torch.arange(50000))
    return train_indices, val_indices


def write_and_verify_split(data_dir, name, dataset, indices, source, seed):
    # Store RGB pixels unchanged as uint8 in PyTorch's NCHW layout.
    images = torch.from_numpy(dataset.data[indices.numpy()])
    images = images.permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(dataset.targets, dtype=torch.int64)[indices]
    payload = {
        "images": images,
        "labels": labels,
        "source_indices": indices,
        "source_split": source,
        "class_names": dataset.classes,
        "split": name,
        "seed": seed,
        "format_version": 1,
    }
    output = data_dir / f"{name}.pt"
    temporary = output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)

    # Read back the saved bytes and check against the original CIFAR-10 arrays.
    loaded = torch.load(temporary, map_location="cpu", weights_only=True)
    assert loaded["images"].dtype == torch.uint8
    assert loaded["labels"].dtype == torch.int64
    assert tuple(loaded["images"].shape) == (len(indices), 3, 32, 32)
    assert torch.equal(loaded["images"], images)
    assert torch.equal(loaded["labels"], labels)
    assert torch.equal(loaded["source_indices"], indices)
    assert loaded["source_split"] == source
    assert loaded["class_names"] == dataset.classes
    counts = torch.bincount(loaded["labels"], minlength=10)
    expected_per_class = {"train": 4500, "val": 500, "test": 1000}[name]
    assert counts.tolist() == [expected_per_class] * 10
    temporary.replace(output)
    print(
        f"Verified {name}: {len(indices)} images, "
        f"{expected_per_class} per class -> {output}",
        flush=True,
    )
    return {
        "file": output.name,
        "samples": len(indices),
        "shape": list(images.shape),
        "dtype": "uint8",
        "pixel_range": [0, 255],
        "source_split": source,
        "samples_per_class": counts.tolist(),
        "size_bytes": output.stat().st_size,
        "sha256": file_digest(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch {torch.__version__}; torchvision {torchvision.__version__}")
    print(f"Downloading/checking official CIFAR-10 in {raw_dir}", flush=True)
    official_train = CIFAR10(root=str(raw_dir), train=True, download=True)
    official_test = CIFAR10(root=str(raw_dir), train=False, download=True)
    assert len(official_train) == 50000
    assert len(official_test) == 10000
    assert official_train.classes == official_test.classes

    archive = raw_dir / CIFAR10.filename
    archive_md5 = file_digest(archive, "md5") if archive.exists() else None
    if archive_md5 is not None and archive_md5 != CIFAR10.tgz_md5:
        raise ValueError("Downloaded archive does not match official MD5")

    labels = torch.tensor(official_train.targets, dtype=torch.int64)
    train_indices, val_indices = stratified_indices(labels, args.seed)
    manifest = {
        "dataset": "CIFAR-10",
        "source_url": CIFAR10.url,
        "source_page": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "archive_md5": archive_md5,
        "expected_archive_md5": CIFAR10.tgz_md5,
        "seed": args.seed,
        "split_method": "Per-class random 90/10 split of official training set",
        "official_test_preserved": True,
        "train_val_source_indices_disjoint": True,
        "train_val_source_indices_cover_all_50000": True,
        "class_names": official_train.classes,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "splits": {},
    }
    for name, dataset, indices, source in [
        ("train", official_train, train_indices, "official_train"),
        ("val", official_train, val_indices, "official_train"),
        ("test", official_test, torch.arange(10000), "official_test"),
    ]:
        manifest["splits"][name] = write_and_verify_split(
            data_dir, name, dataset, indices, source, args.seed
        )

    (data_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("All splits saved and verified. No training has been performed.", flush=True)


if __name__ == "__main__":
    main()
