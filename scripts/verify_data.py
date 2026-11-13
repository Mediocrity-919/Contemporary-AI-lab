"""校验 data/ 下的数据集：官方压缩包 MD5、三个 .pt 的 SHA256、与原始 batch 逐张一致、
划分不重叠、类别均衡、三份数据之间没有完全相同的图片；并输出训练集的通道均值和标准差。

    python scripts/verify_data.py [data 目录，默认 ./data]
"""
import hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
import torch

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data"
raw = root / "raw" / "cifar-10-batches-py"
man = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))

def digest(p, alg):
    h = hashlib.new(alg)
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

print("archive md5 ok:", digest(root / "raw" / "cifar-10-python.tar.gz", "md5") == man["expected_archive_md5"])
for s in ["train", "val", "test"]:
    print(s, "sha256 ok:", digest(root / f"{s}.pt", "sha256") == man["splits"][s]["sha256"])

def load_batch(name):
    with open(raw / name, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"])

xs, ys = zip(*[load_batch(f"data_batch_{i}") for i in range(1, 6)])
tr_x, tr_y = np.concatenate(xs), np.concatenate(ys)
te_x, te_y = load_batch("test_batch")

splits = {s: torch.load(root / f"{s}.pt", map_location="cpu", weights_only=True) for s in ["train", "val", "test"]}
for s, d in splits.items():
    src_x, src_y = (te_x, te_y) if d["source_split"] == "official_test" else (tr_x, tr_y)
    idx = d["source_indices"].numpy()
    print(s, d["images"].shape, d["images"].dtype,
          "pixels match raw:", np.array_equal(d["images"].numpy(), src_x[idx]),
          "labels match raw:", np.array_equal(d["labels"].numpy(), src_y[idx]),
          "per-class:", torch.bincount(d["labels"], minlength=10).tolist())

tr_i, va_i = set(splits["train"]["source_indices"].tolist()), set(splits["val"]["source_indices"].tolist())
print("train/val index overlap:", len(tr_i & va_i), "union:", len(tr_i | va_i))

# exact pixel duplicates across splits (hash each image)
def hashes(x):
    return [hashlib.md5(a.tobytes()).hexdigest() for a in x]
H = {s: hashes(splits[s]["images"].numpy()) for s in splits}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    print(f"exact duplicate images {a}&{b}:", len(set(H[a]) & set(H[b])))
for s in splits:
    print(f"exact duplicates within {s}:", len(H[s]) - len(set(H[s])))

# channel stats from TRAIN split only (for normalization)
x = splits["train"]["images"].float() / 255.0
print("train mean:", [round(v, 4) for v in x.mean(dim=(0, 2, 3)).tolist()])
print("train std :", [round(v, 4) for v in x.std(dim=(0, 2, 3)).tolist()])
