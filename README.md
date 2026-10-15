# Lab1：新闻文本分类

代码位于 [`lab1` 分支](https://github.com/Mediocrity-919/Contemporary-AI-lab/tree/lab1)。比较 NB、LinearSVC、LR 和 PyTorch MLP，采用 TF-IDF 表示。

最终方案：LinearSVC，C=0.5，词50k＋字符100k，min_df=1。固定验证 accuracy **93.62%**、macro-F1 **0.9365**；同划分五折 accuracy **93.16% ± 0.65个百分点**、macro-F1 **0.9317**。

## 运行

在项目根目录打开 Anaconda Prompt，使用 Python 3.10。已有实验环境可直接激活 `pytorch_env`；新环境先安装 `requirements.txt`。MLP 可用 CUDA 或 CPU，原实验使用 PyTorch 2.11.0+cu128。

课程原始数据不上传。请将 `train_data.csv` 和 `test_data_unlabeled.csv` 放入 `data-Project1/`。

```cmd
conda activate pytorch_env
python data-Project1/prepare_data.py --seed 42
python data-Project1/train_experiment.py --model SVM --batch reproduce_final --C 0.5 --word-features 50000 --char-features 100000 --char-weight 1 --min-df 1 --seed 42 --full-train --stage final_fulltrain
```

新预测保存于 `experiments/predictions/reproduce_final/`，同时保存模型与配置。仓库中的最终提交文件是 `data-Project1/predictions.csv`（2457行，列 `id,target`）；使用全部7368条训练后生成，与归档结果一致。

需要完整重跑新版分步调参时，在清洗后依次执行：

```cmd
python data-Project1/tools/revision/revise_experiments.py
python data-Project1/tools/tuning/run_tuning.py
```

完整调参沿用原 `pytorch_env` 路径检查，耗时明显长于只复现最终模型。历史初次调参保留结果表，旧临时搜索入口未完整保留。

## 文件

- `data-Project1/prepare_data.py`、`train_experiment.py`：清洗、四模型训练与评估，支持 CLI。
- `data-Project1/tools/`：新版调参、历史曲线绘制与两页 PDF 导出。
- `experiments/tables/`：历史及新版结果；`experiments/runs/20261008_tuning/`：少量选中配置、指标与 MLP 曲线证据。
- `实验报告.md`、`experiments/reports/20261014_report/`：报告、两页 PDF 和三张图。PDF 导出依赖 Windows 宋体/黑体：`python data-Project1/tools/report/build_two_page_report.py`。

五折结果为开发候选后的非嵌套比较；测试集无标签，未报告测试 accuracy。大模型、原始/处理后数据、缓存和临时核验文件均不上传。
