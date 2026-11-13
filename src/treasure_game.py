"""T4 竞争宝藏分配: A* 距离矩阵 + Minimax/Alpha-Beta.
地图 7x7, A(0,0) B(6,6), T1(0,6)=9 T2(2,2)=6 T3(4,4)=7 T4(6,0)=7 (价值见 PDF 第 14 页表格).
规则: X 障碍, 数字=进入代价, A/B/T 格代价 1, 4 方向, 顺序 A->B->A->B.
Gain(p,Ti)=Vi-d(p,Ti), U=ScoreA-ScoreB. 预期 H=2->{T2,-1}, H=4->{T3,3}.

求解器 solve_game 支持三种搜索设置:
- use_ab: Alpha-Beta 剪枝 (False 为纯 Minimax).
- order / dynamic: 宝藏访问顺序. order 是固定顺序; dynamic=True 时每个节点按
  "当前玩家这一步的收益"从高到低排序 (MAX 和 MIN 都先试对自己最有利的宝藏).
- memo: 仅用于 Minimax 的状态缓存 (置换表). 同一状态 (双方位置, 剩余宝藏集合, 当前玩家, 步数)
  的"未来收益差增量"与已累计分数无关, 可以直接复用. Alpha-Beta 返回的可能只是上下界, 不能这样缓存.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from astar_weighted import astar_weighted_path

A_POS = (0, 0); B_POS = (6, 6)
TREASURES = {'T1': ((0, 6), 9), 'T2': ((2, 2), 6), 'T3': ((4, 4), 7), 'T4': ((6, 0), 7)}
ORDER = ['T1', 'T2', 'T3', 'T4']
POINTS = ['A', 'B', 'T1', 'T2', 'T3', 'T4']

GRID = [
    ['A', 1, 1, 'X', 2, 1, 'T'],
    [1, 'X', 6, 'X', 1, 'X', 1],
    [1, 1, 'T', 1, 1, 1, 1],
    ['X', 1, 'X', 8, 'X', 1, 'X'],
    [1, 1, 1, 1, 'T', 1, 1],
    [1, 'X', 9, 'X', 1, 'X', 1],
    ['T', 1, 1, 'X', 1, 1, 'B'],
]


class TreasureMap:
    """一张宝藏地图: 网格、双方起点、宝藏 {名字: (坐标, 价值)}, 以及缓存的距离矩阵."""

    def __init__(self, grid, a_pos, b_pos, treasures: Dict[str, Tuple[Tuple[int, int], int]]):
        self.grid = grid
        self.a_pos, self.b_pos = a_pos, b_pos
        self.treasures = dict(treasures)
        self.points = ['A', 'B'] + list(self.treasures)
        self.idx = {p: i for i, p in enumerate(self.points)}
        self._dm = None

    def pos_of(self, name):
        if name == 'A': return self.a_pos
        if name == 'B': return self.b_pos
        return self.treasures[name][0]

    def value_of(self, name) -> int:
        return self.treasures[name][1]

    def distance_matrix(self, use_cache=True) -> List[List[int]]:
        """n x n 矩阵, 顺序 [A, B, 宝藏...], 不可达记为 -1.
        use_cache=True 时只在第一次调用时跑 n(n-1) 次 A*, 之后返回缓存的副本."""
        if use_cache and self._dm is not None:
            return [row[:] for row in self._dm]
        n = len(self.points)
        mat = [[0] * n for _ in range(n)]
        for i, a in enumerate(self.points):
            for j, b in enumerate(self.points):
                if i == j:
                    continue
                c, _ = astar_weighted_path(self.pos_of(a), self.pos_of(b), self.grid)
                mat[i][j] = -1 if c == float('inf') else c
        self._dm = [row[:] for row in mat]
        return mat


DEFAULT_MAP = TreasureMap(GRID, A_POS, B_POS, TREASURES)


def pos_of(name):
    return DEFAULT_MAP.pos_of(name)


def distance_matrix(use_cache=True):
    """题目给定地图的 6x6 距离矩阵, 顺序 [A,B,T1..T4]."""
    return DEFAULT_MAP.distance_matrix(use_cache)


@dataclass(frozen=True)
class GameState:
    """博弈状态. 位置用点名表示 ('A'/'B' 为出发点, 'T1'.. 为已取宝藏所在格),
    对应坐标可用 TreasureMap.pos_of 查得."""
    a_pt: str                   # A 当前位置
    b_pt: str                   # B 当前位置
    remaining: Tuple[str, ...]  # 剩余宝藏 (按访问顺序排列)
    score_a: int                # A 累计收益
    score_b: int                # B 累计收益
    turn: int                   # 当前玩家: 0=A(MAX), 1=B(MIN)
    ply: int = 0                # 已走步数, 到 H 时停止

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b

    @property
    def mover(self) -> str:
        return self.a_pt if self.turn == 0 else self.b_pt

    def key(self):
        """缓存键: 与已累计分数无关的部分."""
        return (self.a_pt, self.b_pt, frozenset(self.remaining), self.turn, self.ply)


def initial_state(tmap: TreasureMap, order=None) -> GameState:
    return GameState('A', 'B', tuple(order if order else tmap.treasures), 0, 0, 0, 0)


def gain_of(tmap: TreasureMap, D, s: GameState, t: str) -> int:
    """当前玩家从所在位置去取宝藏 t 的收益 Gain = V - d."""
    return tmap.value_of(t) - D[tmap.idx[s.mover]][tmap.idx[t]]


def apply_move(tmap: TreasureMap, D, s: GameState, t: str) -> GameState:
    """当前玩家取宝藏 t, 返回新状态."""
    g = gain_of(tmap, D, s, t)
    rem = tuple(x for x in s.remaining if x != t)
    if s.turn == 0:
        return GameState(t, s.b_pt, rem, s.score_a + g, s.score_b, 1, s.ply + 1)
    return GameState(s.a_pt, t, rem, s.score_a, s.score_b + g, 0, s.ply + 1)


def legal_moves(tmap: TreasureMap, D, s: GameState, dynamic=False) -> List[str]:
    """可选宝藏 (不可达的不能选). dynamic=True 时按本步收益从高到低排 (稳定排序, 平局保持原顺序)."""
    cur = tmap.idx[s.mover]
    moves = [t for t in s.remaining if D[cur][tmap.idx[t]] != -1]
    if dynamic:
        moves.sort(key=lambda t: -gain_of(tmap, D, s, t))
    return moves


def solve_game(H: int, use_ab: bool = True, order=None, dynamic: bool = False,
               memo: bool = False, tmap: Optional[TreasureMap] = None):
    """H: 搜索深度 (总步数). order: 固定的宝藏访问顺序, 默认按地图中的顺序.
    返回 best_target / best_value / counter / matrix."""
    if memo and use_ab:
        raise ValueError('状态缓存只用于 Minimax; Alpha-Beta 的返回值可能是上下界')
    tmap = tmap or DEFAULT_MAP
    D = tmap.distance_matrix()
    counter = {'visited': 0, 'cutoffs': 0, 'effective_cutoffs': 0, 'skipped_children': 0,
               'memo_hits': 0}
    table = {}
    best_first = [None]

    def cutoff(n_moves, i):
        counter['cutoffs'] += 1
        if n_moves - i - 1 > 0:
            counter['effective_cutoffs'] += 1
            counter['skipped_children'] += n_moves - i - 1

    def recurse(s: GameState, alpha, beta, is_root=False):
        if memo and not is_root:
            k = s.key()
            if k in table:
                counter['memo_hits'] += 1
                return s.utility + table[k]
        counter['visited'] += 1
        moves = legal_moves(tmap, D, s, dynamic)
        if s.ply == H or not moves:
            return s.utility
        if s.turn == 0:  # A: MAX
            best = float('-inf')
            for i, t in enumerate(moves):
                val = recurse(apply_move(tmap, D, s, t), alpha, beta)
                if is_root and val > best:  # 用 > : 相等时保留先访问的宝藏
                    best_first[0] = t
                best = max(best, val)
                if use_ab:
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        cutoff(len(moves), i)
                        break
        else:  # B: MIN
            best = float('inf')
            for i, t in enumerate(moves):
                val = recurse(apply_move(tmap, D, s, t), alpha, beta)
                best = min(best, val)
                if use_ab:
                    beta = min(beta, best)
                    if alpha >= beta:
                        cutoff(len(moves), i)
                        break
        if memo:
            table[s.key()] = best - s.utility
        return best

    val = recurse(initial_state(tmap, order), float('-inf'), float('inf'), True)
    return {'best_target': best_first[0], 'best_value': val, 'counter': counter, 'matrix': D}


def best_first_move(H: int, use_ab: bool = True) -> Dict[str, object]:
    """PDF 输出格式: {"best_target": ..., "best_value": ...}."""
    r = solve_game(H, use_ab=use_ab)
    return {'best_target': r['best_target'], 'best_value': r['best_value']}


def minimax_first(H, order=None):
    r = solve_game(H, use_ab=False, order=order)
    return {'best_target': r['best_target'], 'best_value': r['best_value']}


def first_move_values(H: int, tmap: Optional[TreasureMap] = None) -> Dict[str, int]:
    """A 每个首步对应的精确 Minimax 值 (用于分析为什么选这个首步)."""
    tmap = tmap or DEFAULT_MAP
    D = tmap.distance_matrix()
    s0 = initial_state(tmap)
    out = {}
    for t in legal_moves(tmap, D, s0):
        out[t] = _minimax_line(tmap, D, apply_move(tmap, D, s0, t), H)[0]
    return out


def _minimax_line(tmap, D, s: GameState, H: int):
    """返回 (值, 最优对抗序列). 序列每项为 (玩家, 宝藏, 本步收益)."""
    moves = legal_moves(tmap, D, s)
    if s.ply == H or not moves:
        return s.utility, []
    best_val, best_line = None, None
    for t in moves:
        v, line = _minimax_line(tmap, D, apply_move(tmap, D, s, t), H)
        better = best_val is None or (v > best_val if s.turn == 0 else v < best_val)
        if better:
            best_val = v
            best_line = [('A' if s.turn == 0 else 'B', t, gain_of(tmap, D, s, t))] + line
    return best_val, best_line


def best_line(H: int, tmap: Optional[TreasureMap] = None):
    """从初始状态出发的最优对抗序列 (主变例)."""
    tmap = tmap or DEFAULT_MAP
    D = tmap.distance_matrix()
    return _minimax_line(tmap, D, initial_state(tmap), H)


if __name__ == '__main__':
    print('6x6 距离矩阵 (顺序 A, B, T1, T2, T3, T4):')
    for name, row in zip(POINTS, distance_matrix()):
        print(f'{name:>3}', row)
    for H in (2, 4):
        print(f'H = {H}  Minimax:', best_first_move(H, use_ab=False),
              ' Alpha-Beta:', best_first_move(H, use_ab=True))
