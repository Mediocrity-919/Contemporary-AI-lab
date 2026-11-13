# CIFAR-10 图像分类实验（当代人工智能 作业三）

自定义 CNN 与 ResNet-18（从头训练 / ImageNet 预训练微调）在 CIFAR-10 上的分类实验，
包括数据增强、SE 注意力、标签平滑、学习率调度与早停、架构比较、迁移学习、
校准、鲁棒性和 Grad-CAM 可解释性分析。实验设计及全部变更记录见 [plan.md](plan.md)。

- **实验报告**：[report.md](report.md) / [report.pdf](report.pdf)
- **测试集准确率**（3 个种子）：自定义 CNN（OurCNN-F）93.59 ± 0.13%，蒸馏学生 94.52 ± 0.02%，
  ImageNet 预训练 ResNet-18 微调 96.06 ± 0.23%
- 仓库不包含数据文件和训练产物（`data/*.pt`、`outputs/`，约 8GB），可按下面的步骤重新生成；
  报告用到的全部图表在 `report/figures/`、`report/tables/`。

## 1. 环境

| 项目 | 版本 |
|---|---|
| Python | 3.10.20（Anaconda 环境 `pytorch_env`） |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| 其他 | numpy 2.2、scipy 1.15、scikit-learn 1.7、matplotlib 3.10、PyYAML 6.0、thop 2.2、pytest 9.2 |
| GPU | NVIDIA GeForce RTX 4060 Laptop（8GB） |

```bash
conda activate pytorch_env
pip install -r requirements.txt     # PyTorch 请按本机 CUDA 版本安装对应的 wheel
```

以下命令都在 `lab3/` 目录下执行。没有激活环境时，可以把 `python` 换成
`D:\Anaconda_envs\envs\pytorch_env\python.exe`。

## 2. 数据

```bash
python prepare_data.py --seed 42     # 下载官方 CIFAR-10，按类别分层划分 45000 / 5000，保留官方测试集
python scripts/verify_data.py        # 校验 MD5 / SHA256、逐张比对、划分不重叠、无重复图片
```

| 文件 | 图片数 | 用途 |
|---|---:|---|
| `data/train.pt` | 45,000（每类 4,500） | 训练 |
| `data/val.pt` | 5,000（每类 500） | 选 checkpoint、调参、方案决策、拟合温度 |
| `data/test.pt` | 10,000（每类 1,000） | 配置冻结后的最终评估 |

详见 [data/README.md](data/README.md)。

## 3. 目录结构

```
configs/          实验配置（YAML，用 inherit 继承，只写差异）
  base.yaml         全部默认值及说明
  decisions.yaml    按规则做出的决策（学习率、增强档等），由 src/select_decisions.py 写入
  decisions_log.md  每个决策的依据
  cnn/ arch/ transfer/ ext/   各组实验；以 _ 开头的是公共配置
  analysis/mcnemar_pairs.yaml 事先确定的显著性检验比较对
src/
  data.py           数据读取、三档增强、确定性采样、GPU 放大
  models/           OurCNN、SE / CBAM、ResNet-18 两种版本
  losses.py         交叉熵 / 标签平滑 / 知识蒸馏 / Mixup / CutMix
  engine.py         训练循环（AMP、调度器、影子早停、断点续训）
  train.py          单次训练入口
  run_experiments.py  批量运行（扫参、多种子、自动跳过已完成）
  select_decisions.py 事先确定的决策规则
  evaluate.py       验证集 / 测试集评估（测试集使用会留痕）
  metrics.py        准确率、每类指标、混淆矩阵、NLL、ECE、Brier、Wilson 区间
  analysis/         汇总表、曲线、混淆矩阵、误分类、校准、鲁棒性、Grad-CAM、t-SNE、开销、McNemar、集成
scripts/
  run_phase.ps1     按阶段运行全部实验（Windows PowerShell）
  verify_data.py    数据完整性校验
  check_reproducibility.py  确定性复现检查
tests/            pytest 单元测试（38 项）
report.md / report.pdf      实验报告
report/figures/   分析图（由 src.analysis 生成）
report/tables/    结果表（每个种子的结果、校准、鲁棒性、McNemar 等）
outputs/runs/     训练结果（每个配置 × 种子一个目录；不在仓库中）
```

## 4. 快速检查

```bash
python -m pytest tests -q                    # 单元测试，约 30 秒
python scripts/check_reproducibility.py      # 复现性检查，约 5 分钟
```

训练一个配置的一个种子：

```bash
python -m src.train --config configs/cnn/E1_C1_basic.yaml --seed 0
python -m src.train --config configs/base.yaml --seed 0 --set train.epochs=5 --set data.aug=strong
```

## 5. 完整实验流程

按 plan.md 第 8 节分阶段执行。每个阶段可以随时 `Ctrl+C` 中断；重新执行同一命令会跳过已完成的运行，
未完成的运行从中断的那一轮之后继续，结果与不中断时逐位相同。

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_phase.ps1 status   # 查看全部实验进度
```

| 阶段 | 命令参数 | 内容 | 训练次数 | 预计耗时 |
|---|---|---|---:|---:|
| 1 | `1` | 预实验：OurCNN 100 轮、预训练微调 20 轮（首次下载约 45MB 官方权重） | 2 | ~1.5 h |
| — | 手动 | 根据预实验曲线确定 `epochs_32`、`epochs_224`（见下） | — | — |
| 2 | `2` | OurCNN 学习率调参 → 自动写入 `cnn_lr` | 2–3 | ~1 h |
| 3 | `3` | E1 增强 → E2 SE → E3 标签平滑，每步按规则自动决策 | 15 | ~6 h |
| 4 | `4` | E4 调度与早停、E5 逐个去掉组件 | ≤13 | ~5 h |
| 5 | `5` | E6 架构比较：两种模型分别调参，再各跑 3 个种子 | 9–12 | ~8 h |
| 6 | `6` | E7 迁移学习：FT / S224 / LP / FT32 分别调参，再各跑 3 个种子 | 24 | ~10 h |
| 7 | `7` | 统一测试集评估 + 全部 P0/P1 分析图表 | — | ~0.5 h |
| 8 | `8` | P2 扩展（可选）：CBAM、CutMix、正则消融、StepLR/OneCycle、知识蒸馏、集成 | 25 | ~12 h |

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_phase.ps1 1
```

非 Windows 环境可以按 `scripts/run_phase.ps1` 中各阶段列出的命令逐条执行，例如阶段 2：

```bash
python -m src.run_experiments configs/cnn/T_cnn.yaml
python -m src.select_decisions lr configs/cnn/T_cnn.yaml cnn_lr
```

阶段 1 之后，根据 `report/figures/pilot_*_curves.png` 确定训练轮数，例如：

```bash
python -m src.select_decisions set epochs_32 100 --reason "预实验：验证准确率在第 90 轮后基本不再上升"
```

耗时按 RTX 4060 Laptop 实测速度估算：OurCNN 每轮约 14 秒，
ResNet-18 CIFAR 版约 36 秒，224 输入的 ResNet-18 全部微调约 180 秒。计时前请接上电源、使用高性能模式。

### 决策是怎么做出的

配置中写成 `"${key}"` 的字段从 `configs/decisions.yaml` 读取；值为 `null` 时该配置拒绝运行。
决策由 `src/select_decisions.py` 按 plan.md 5.4 节的规则计算（学习率取调参种子上验证集最优；
增强档取 3 个种子验证集均值最高；SE / 标签平滑只有在验证集均值提升超过 0.1 个百分点时才加入），
依据自动追加到 `configs/decisions_log.md`。已做出的决策默认不允许覆盖。

如果学习率的最优点落在网格边缘，脚本不写入决策、阶段随之停止（返回码 2）。这时按规则沿该方向补 1 个点：
在对应 `T_*.yaml` 的 `meta.sweep` 中加入新取值、加上 `meta.edge_extended: true`，在 plan.md 变更记录中注明，再重跑该阶段。
补过点后即使仍在边缘也接受结果。

### 相同配置自动复用

配置哈希只覆盖影响训练结果的字段（不含 `meta`、`system`）。例如最终配置 F 与 E1–E3 中的某个配置完全相同，
或者 F 本来就不用标签平滑时“F − 标签平滑”就等于 F，这些都会直接复用已有结果，不重复训练。

## 6. 输出文件

每个运行目录 `outputs/runs/<配置名>-<哈希>/seed<k>/`：

| 文件 | 内容 |
|---|---|
| `config.yaml` / `env.json` | 展开后的完整配置（含哈希、种子）/ 软硬件版本 |
| `log.csv` | 逐轮：学习率、训练 loss/acc（含增强）、干净训练子集 loss/acc、验证 loss/acc、耗时 |
| `best.pt` | 验证准确率最高的模型（并列取验证 loss 更低者） |
| `es_best.pt` | 早停（影子模式，patience 15）触发时的最优模型；未触发时同 best.pt |
| `last.pt` | 最后一轮的完整训练状态，用于断点续训 |
| `val_best_logits.npz` | best.pt 在验证集上的 logits |
| `metrics.json` | 运行摘要（最优轮、早停信息、耗时、参数量）；存在即表示运行完成 |
| `test_<ckpt>_logits.npz` / `test_<ckpt>_metrics.json` | 测试集评估结果（阶段 7 生成） |

评估指标中的 loss 一律是不带平滑的交叉熵（NLL），交叉熵和标签平滑训练的模型可以直接比较；
`log.csv` 里的 `train_loss` 是训练目标本身（带平滑 / 蒸馏项）。

## 7. 测试集使用规则

- 只有 `src/evaluate.py`（及阶段 7 的分析脚本）读取测试集，每次使用都追加记录到 `outputs/test_log.csv`。
- 所有选择（checkpoint、超参数、组件取舍、最终成绩模型）只依据验证集。
- 某个配置评估过测试集后不再修改；默认不会重复评估同一个运行。

## 8. 分析

阶段 7 / 8 的 `src.analysis.run_all` 会生成全部图表（`report/figures/`）和表格（`report/tables/`）。
各模块也可以单独调用，用法见各文件开头的说明，例如：

```bash
python -m src.evaluate "configs/cnn/E1_*.yaml" --split test
python -m src.analysis.summarize --name E1_aug configs/cnn/E1_C0_none.yaml:无增强 configs/cnn/E1_C2_strong.yaml:强增强
python -m src.analysis.figures confusion configs/cnn/F_final.yaml:OurCNN
python -m src.analysis.gradcam gallery configs/transfer/E7_ft.yaml:FT
```

## 9. 复现性

- 全部实验默认开启确定性模式（`train.deterministic: true`），实测只慢 3–5%。
- 训练增强的随机数由（运行种子, epoch, 样本索引）决定，与 DataLoader 进程数无关。
- `scripts/check_reproducibility.py` 的结果（`outputs/checks/reproducibility.json`）：
  同一种子跑两次，逐轮指标和全部参数完全相同；训练 1 轮后中断再续训，与不中断完全相同；
  关闭确定性模式时，两次运行的验证准确率最多相差 0.06 个百分点。
- 同一配置在不同 GPU、驱动或库版本上不保证逐位相同，但统计结论应一致。

## 10. 报告

实验报告为根目录的 [report.md](report.md)（PDF 版 [report.pdf](report.pdf)），其中的数字均来自 `report/tables/`。
