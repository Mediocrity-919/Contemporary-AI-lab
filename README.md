# Lab2 搜索算法

## 环境与运行
- Python 3.10 及以上 (在 3.14.2 上测试). 算法、测试与对照实验只用 Python 标准库，无需安装额外依赖.
- 在 lab2 根目录运行:
  - `python tests/test_all.py`: PDF 样例 + 边界情况测试, 全部通过时退出码为 0.
  - `python tests/experiments.py`: 四道题的基础对照实验 (节点数 + 计时).
  - `python tests/innovation.py`: 拓展实验 (大树访问顺序、大图启发函数/平局打破、距离矩阵缓存).
  - `python tests/order_check.py`: T4 在 H=4 时不同宝藏访问顺序的对照.
  - `python tests/verify_independent.py`: 用独立实现的 Dijkstra / 完整回传对拍 (随机迷宫、地形、博弈树).
  - `python src/alphabeta.py tree.json`: 读取 PDF 格式的 JSON 博弈树, 输出 MAX 的最优值 (不给文件时用 PDF 样例).
  - `python src/treasure_game.py`: 打印 T4 的 6x6 距离矩阵, 以及 H=2/4 下 Minimax 与 Alpha-Beta 的结果.
  - `python tests/report_data.py`: 复现 T1–T4 与 E1–E4 的完整对照实验，输出统计摘要；生成的 `results/report_data.json` 仅保存在本地，已由 Git 忽略.
- 实验报告: `report.md` 和 `report.pdf` (5 页).
- 中文输出乱码时加 `-X utf8`, 例如 `python -X utf8 tests/experiments.py`.

## 目录
- `src/astar_maze.py`: T1 迷宫 A*.
- `src/astar_weighted.py`: T2 加权地形 A* (T4 距离矩阵也用它).
- `src/alphabeta.py`: T3 Minimax / Alpha-Beta.
- `src/treasure_game.py`: T4 地图、博弈状态、距离矩阵与博弈搜索 (支持动态排序 `dynamic=True`、Minimax 状态缓存 `memo=True`).
- `tests/bench.py`: 计时工具 (perf_counter_ns, 预热后取中位数).
- `figures/`: 正式报告引用的四张插图.

## 接口
- `astar_maze(start, end, grid) -> List[Tuple[int,int]]`, 无解返回 `[]`.
- `astar_weighted(start, end, grid) -> int`, 无解返回 `-1`.
- `alphabeta(root) -> int`, 根为 MAX, 子节点从左到右访问.
- `solve_tree_json(data) -> int`: 接收 PDF 的 JSON 输入 (字符串或字典), 按 `is_maximizing_player` 判断 MAX/MIN.
- `best_first_move(H, use_ab=True) -> {'best_target', 'best_value'}`: 与 PDF 的输出格式一致.
- `solve_game(H) -> {'best_target', 'best_value', 'counter', 'matrix'}`: 额外返回计数器和距离矩阵, 供实验使用.
- 对照实验用的可选参数: `astar_maze(..., h_func, tie_break)`, `astar_weighted(..., h_scale, h_func, tie_break)`,
  `solve_game(H, use_ab, order, dynamic, memo, tmap)`.

## 规则与假设
- 起点或终点越界、在障碍上、地图为空: T1 返回 `[]`, T2 返回 `-1`.
- T2: 进入代价 = 地形值, S/A/B/T 格为 1, 终点格为 'E' 时进入代价为 0. 不支持负代价.
- T2 启发函数: m = 除 X、E 和起点格外的最小进入代价, d = 曼哈顿距离.
  终点为 'E' 时 h = m·max(d−1, 0) (最后一步免费); 终点为普通格时 h = m·d.
  地图含 0 代价格时 m = 0, h 退化为 0.
- T4: 宝藏价值按 PDF 第 14 页表格: T1=9, T2=6, T3=7, T4=7. 不可达的宝藏不能选.

## 统计口径
- A* `expanded`: 从优先队列弹出并关闭的节点数, 含终点.
- 博弈树 `visited`: 递归访问的节点数.
- `cutoffs`: alpha ≥ beta 的触发次数 (含在最后一个子节点之后才触发的情况).
- `effective_cutoffs`: 真正跳过了后续子节点的剪枝次数; `skipped_children`: 被跳过的子树个数.
