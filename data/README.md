# CIFAR-10 数据与划分

使用 torchvision.datasets.CIFAR10 从官网下载原始数据。官方训练集按类别分层，
以随机种子 42 划分为训练集和验证集；官方测试集完整保留，顺序不变。

| 文件 | 图片数 | 每类图片数 | 来源 |
| --- | ---: | ---: | --- |
| train.pt | 45,000 | 4,500 | 官方训练集 |
| val.pt | 5,000 | 500 | 官方训练集 |
| test.pt | 10,000 | 1,000 | 官方测试集 |

`raw/` 保留官方压缩包和解压后的数据。`split_manifest.json` 记录随机种子、
库版本、各类数量、来源地址和文件校验值。根目录的 `prepare_data.py` 可重新生成划分：

```bat
python prepare_data.py --seed 42
```

## 文件内容

三个 `.pt` 文件均为字典，可使用 `torch.load(..., weights_only=True)` 安全加载：

- `images`：`torch.uint8`，形状 `(N, 3, 32, 32)`，RGB，像素值 0 到 255。
- `labels`：`torch.int64`，形状 `(N,)`，类别编号 0 到 9。
- `class_names`：按标签编号排列的 10 个英文类别名称。
- `source_indices`：图片在对应官方数据集中的原始索引。
- `source_split`：`official_train` 或 `official_test`。
- `split`：`train`、`val` 或 `test`。
- `seed`：划分使用的随机种子。
- `format_version`：保存格式版本，当前为 1。

`source_indices` 必须与 `source_split` 一起理解：官方训练集和测试集分别从 0 编号。
训练集和验证集的来源索引不重叠，合并后恰好覆盖官方 50,000 个训练样本。
这里保证的是原始样本索引不重叠，并未修改或去重官方数据中的图片内容。

## 加载示例

在实验根目录运行：

```python
from pathlib import Path
import torch

data_dir = Path("data")
train = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=True)
print(train["images"].shape)  # torch.Size([45000, 3, 32, 32])
print(train["labels"].shape)  # torch.Size([45000])
print(train["class_names"])

# 单张原始图片转为模型可用的浮点像素值；正式训练时还需预处理。
image = train["images"][0].float() / 255.0
label = train["labels"][0]
```

图片尚未归一化、增强或放大，便于后续公平比较不同预处理方法。
随机增强仅用于训练集；验证集和测试集使用固定预处理。
现有图片已经是 Tensor，不能直接对其调用旧版 transforms.ToTensor()；
可以使用支持 Tensor 的变换，或先转换为 PIL 图片。
模型选择和调参依据验证集；实验方案确定后才使用测试集评估。

## 数据来源

- 官方页面：https://www.cs.toronto.edu/~kriz/cifar.html
- Python 压缩包官方 MD5：`c58f30108f718f92721af3b95e74349a`
- 数据集文献：Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009。
