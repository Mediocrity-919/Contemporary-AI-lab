"""T1 迷宫寻路: 非加权 A* (4方向, g=1, h=曼哈顿距离)."""
import heapq
from typing import List, Tuple

Coord = Tuple[int, int]

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_maze(start: Coord, end: Coord, grid: List[List[str]],
               h_func=manhattan, count_expanded: bool = False, trace=None,
               tie_break: str = 'g'):
    """输入: start/end Tuple[int,int], grid List[List[str]] 含 S/./X/E.
    输出: List[Tuple[int,int]] 起点到终点路径, 无解返回 [].
    trace: 传入列表时, 按扩展顺序记录 (节点, g, h, f), 用于报告展示搜索过程.
    tie_break: f 相同时的第二排序键. 'g' 优先 g 小的 (默认); 'h' 优先 h 小的, 即更靠近终点."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    def passable(r: int, c: int) -> bool:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        return grid[r][c] != 'X'

    # 空地图、起点/终点越界或在障碍上: 视为无解
    if not passable(*start) or not passable(*end):
        return ([], 0) if count_expanded else []
    if start == end:
        return [start] if not count_expanded else ([start], 0)
    # A*: open=(f, 第二排序键, g, tie, node)
    def key(g: int, hv: int) -> int:
        return hv if tie_break == 'h' else g

    open_h = []
    tie = 0
    h0 = h_func(start, end)
    heapq.heappush(open_h, (h0, key(0, h0), 0, tie, start))
    g_best = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while open_h:
        f, _, g, _, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1  # 口径: 弹出并关闭的节点数, 含终点
        if trace is not None:
            trace.append((cur, g, f - g, f))
        if cur == end:
            path = []
            p = cur
            while p is not None:
                path.append(p)
                p = parent[p]
            path.reverse()
            return (path, expanded) if count_expanded else path
        for dr, dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not passable(nr, nc):
                continue
            nxt = (nr, nc)
            ng = g + 1
            if nxt in g_best and ng >= g_best[nxt]:
                continue
            g_best[nxt] = ng
            parent[nxt] = cur
            tie += 1
            hn = h_func(nxt, end)
            heapq.heappush(open_h, (ng + hn, key(ng, hn), ng, tie, nxt))
    return ([], expanded) if count_expanded else []


def solve_t1_demo():
    start = (0, 0)
    end = (3, 4)
    grid = [
        ['S', '.', '.', 'X', '.'],
        ['.', 'X', '.', '.', '.'],
        ['.', 'X', '.', 'X', '.'],
        ['.', '.', '.', 'X', 'E'],
        ['X', 'X', '.', '.', '.'],
    ]
    return astar_maze(start, end, grid)
