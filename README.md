# Lab1：新闻文本分类

代码位于 [`lab1` 分支](https://github.com/Mediocrity-919/Contemporary-AI-lab/tree/lab1)。比较 NB、LinearSVC、LR 和 PyTorch MLP，采用 TF-IDF 表示。

最终方案：LinearSVC，C=0.5，词50k＋字符100k，min_df=1。固定验证 accuracy **93.62%**、macro-F1 **0.9365**；同划分五折 accuracy **93.16% ± 0.65个百分点**、macro-F1 **0.9317**。

最终提交文件位于根目录：[`prediction.csv`](prediction.csv)（2457行，列 `id,target`）。

## 代码入口

| 文件 | 用途 |
|---|---|
| [`data-Project1/train_experiment.py`](data-Project1/train_experiment.py) | 四模型训练、验证与预测的核心代码，支持 CLI |
| [`data-Project1/prepare_data.py`](data-Project1/prepare_data.py) | 数据清洗与固定分层划分 |
| [`data-Project1/tools/tuning/run_tuning.py`](data-Project1/tools/tuning/run_tuning.py) | 分步调参、多种子、五折选模与全量重训 |
| [`data-Project1/tools/revision/revise_experiments.py`](data-Project1/tools/revision/revise_experiments.py) | 清洗和初期特征对照实验 |

## 运行

在项目根目录打开 Anaconda Prompt，使用 Python 3.10。已有实验环境可直接激活 `pytorch_env`；新环境先安装 `requirements.txt`。MLP 可用 CUDA 或 CPU，原实验使用 PyTorch 2.11.0+cu128。

课程原始数据不上传。请将 `train_data.csv` 和 `test_data_unlabeled.csv` 放入 `data-Project1/`。

```cmd
conda activate pytorch_env
python data-Project1/prepare_data.py --seed 42
python data-Project1/train_experiment.py --model SVM --batch reproduce_final --C 0.5 --word-features 50000 --char-features 100000 --char-weight 1 --min-df 1 --seed 42 --full-train --stage final_fulltrain
python -c "from pathlib import Path; import shutil; p=max(Path('experiments/predictions/reproduce_final').glob('*.csv'), key=lambda p:p.stat().st_mtime); shutil.copyfile(p, 'prediction.csv')"
```

训练输出保存在 `experiments/predictions/reproduce_final/` 和相应 runs 目录；最后一条命令将新预测复制为根目录的 `prediction.csv`。当前提交文件已使用全部7368条训练后生成，与归档结果一致。

需要完整重跑新版分步调参时，在清洗后依次执行：

```cmd
python data-Project1/tools/revision/revise_experiments.py
python data-Project1/tools/tuning/run_tuning.py
```

完整调参沿用原 `pytorch_env` 路径检查，耗时明显长于只复现最终模型。历史初次调参保留结果表，旧临时搜索入口未完整保留。

## 文件

- `data-Project1/tools/tuning/plot_history.py`、`plot_results.py`：实验参数曲线、MLP 训练曲线与五折结果绘图。
- `experiments/tables/`：历史及新版结果；`experiments/runs/20261008_tuning/`：少量选中配置、指标与 MLP 曲线证据。
- `实验报告.md`、`experiments/reports/20261014_report/`：报告、两页 PDF 和三张实验结果图。

五折结果为开发候选后的非嵌套比较；测试集无标签，未报告测试 accuracy。模型权重、原始/处理后数据、PDF 排版工具、缓存和临时核验文件均不上传。
