"""T2 加权地形 A*: 从 A 进入相邻格 B 的代价 = B 的地形权值.

代价规则:
- 数字格: 进入代价为该数字 (要求非负).
- S/A/B/T 格: 进入代价为 1.
- 终点格为 'E' 时进入代价为 0 (PDF: 最后一步没有代价).
- X 为障碍.

启发函数: d = 曼哈顿距离(p, end), m = 除 X、E 和起点格外所有格的最小进入代价.
  起点格不参与取最小值: 最优路径不会重新进入起点, 而起点 S 固定按 1 计,
  若把它算进来, 全图地形都 >=3 时 m 也只能是 1, 启发函数会变得很弱 (见报告 E2).
  进入起点的边可能比 m 便宜, 但起点第一个出队且 g=0 已是最优, 不会被重新打开, 不影响正确性.
- 终点为 'E': h(p) = m * max(d - 1, 0).
  任一 p->end 路径至少 d 步, 最后一步进入 E 代价 0, 其余 d-1 步每步 >= m,
  所以真实代价 >= m*(d-1) = h, 可采纳. 相邻两格 h 之差 <= m <= 边代价, 也满足一致性.
- 终点为普通格 (如 T4 中的宝藏 'T'): 最后一步也 >= m, 用 h(p) = m * d.
- 地图里有 0 代价格时 m = 0, h 退化为 0 (即 Dijkstra), 仍然正确.

平局打破: f 相同时优先 h 更小 (更接近终点), 可用 tie_break='g' 切回"优先 g 更小"做对照.
"""
import heapq
from typing import Callable, List, Optional, Tuple

Coord = Tuple[int, int]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def is_block(v) -> bool:
    return v == 'X'


def cell_cost(v) -> int:
    """进入格子 v 的代价 (不含终点 E 免费的特殊规则)."""
    if v in ('S', 'A', 'B', 'T'):
        return 1
    if v == 'E':
        return 0
    if is_block(v):
        raise ValueError('障碍格 X 不可进入')
    c = int(v)
    if c < 0:
        raise ValueError(f'不支持负代价地形: {v}')
    return c


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def min_terrain_cost(grid, skip=()) -> int:
    """除 X/E 和 skip 中的坐标外的最小进入代价.
    0 代价格会让结果为 0 (不能改成 1, 否则 h 会高估)."""
    m = None
    skip = set(skip)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if is_block(v) or v == 'E' or (r, c) in skip:
                continue
            cost = cell_cost(v)
            if m is None or cost < m:
                m = cost
    return m if m is not None else 0


def _valid_cell(grid, p: Coord) -> bool:
    r, c = p
    return 0 <= r < len(grid) and 0 <= c < len(grid[r]) and not is_block(grid[r][c])


def _search(start: Coord, end: Coord, grid, h_scale: Optional[int] = None,
            tie_break: str = 'h', h_func: Optional[Callable[[Coord], float]] = None,
            trace: Optional[list] = None):
    """加权 A* 搜索核心, 两个对外接口共用.
    h_func: 自定义启发函数 (覆盖默认 h), 用于对照实验, 例如不可采纳的放大启发函数.
    trace: 传入列表时按扩展顺序记录 (节点, g, h, f).
    返回 (cost, path, expanded); 无解时 cost=None, path=[]."""
    if not grid or not _valid_cell(grid, start) or not _valid_cell(grid, end):
        return None, [], 0
    if start == end:
        return 0, [start], 0
    if h_scale is None:
        h_scale = min_terrain_cost(grid, skip=(start,))
    end_free = grid[end[0]][end[1]] == 'E'

    def default_h(p: Coord) -> int:
        d = manhattan(p, end)
        return h_scale * (max(d - 1, 0) if end_free else d)

    h = h_func if h_func is not None else default_h

    def key(g: int, hv: int) -> int:
        # 第二排序键: 'h' 优先离终点近的; 'g' 优先已走代价小的 (旧行为)
        return hv if tie_break == 'h' else g

    h0 = h(start)
    tie = 0
    open_h = [(h0, key(0, h0), 0, tie, start)]
    g_best = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while open_h:
        _, _, g, _, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1  # 口径: 弹出并关闭的节点数, 含终点
        if trace is not None:
            trace.append((cur, g, h(cur), g + h(cur)))
        if cur == end:
            path = []
            p = cur
            while p is not None:
                path.append(p)
                p = parent[p]
            return g, path[::-1], expanded
        for dr, dc in DIRS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not _valid_cell(grid, nxt):
                continue
            step = 0 if (nxt == end and end_free) else cell_cost(grid[nxt[0]][nxt[1]])
            ng = g + step
            if nxt in g_best and ng >= g_best[nxt]:
                continue
            g_best[nxt] = ng
            parent[nxt] = cur
            tie += 1
            hn = h(nxt)
            heapq.heappush(open_h, (ng + hn, key(ng, hn), ng, tie, nxt))
    return None, [], expanded


def astar_weighted(start: Coord, end: Coord, grid, h_scale: Optional[int] = None,
                   count_expanded: bool = False, tie_break: str = 'h',
                   h_func=None, trace=None):
    """输入: start/end Tuple[int,int], grid 由数字/S/E/X 组成.
    输出: int 最小代价, 无解 (含起点/终点越界或在障碍上) 返回 -1.
    h_scale: None 表示用 min_terrain_cost; 传 0 即 h=0 基线. h_func/trace 见 _search."""
    cost, _, expanded = _search(start, end, grid, h_scale, tie_break, h_func, trace)
    cost = -1 if cost is None else cost
    return (cost, expanded) if count_expanded else cost


def astar_weighted_path(start: Coord, end: Coord, grid, h_func=None) -> Tuple[float, List[Coord]]:
    """返回 (cost, path), 供 T4 距离矩阵使用; 无解返回 (inf, [])."""
    cost, path, _ = _search(start, end, grid, h_func=h_func)
    return (float('inf'), []) if cost is None else (cost, path)
