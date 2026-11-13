"""拓展实验: 大树访问顺序 / 大图上启发函数与平局打破对照 / 距离矩阵缓存."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from alphabeta import TreeNode, minimax_count, alphabeta_count
from astar_weighted import astar_weighted
import treasure_game
from treasure_game import solve_game, distance_matrix
from bench import bench, fmt

print('=== innovation 1: move ordering on bigger tree ===')
# 深度 4, 分支 3 的树, 叶子值固定
vals = [3,5,2,9,1,7,6,8,4,0,5,3,8,2,9,4,6,1,7,3,5,8,2,6,9,0,4]

def build(depth):
    it = iter(vals * 40)
    def rec(d):
        if d == 0: return TreeNode(value=next(it))
        return TreeNode(children=[rec(d - 1) for _ in range(3)])
    return rec(depth)

def rev(n):
    if not n.children: return TreeNode(value=n.value)
    return TreeNode(children=[rev(c) for c in reversed(n.children)])

for order_name, tree in [('L->R', build(4)), ('R->L', rev(build(4)))]:
    v1, n1 = minimax_count(tree, True)
    v2, c = alphabeta_count(tree, True, return_counter=True)
    print(f"{order_name}: minimax nodes={n1} ab nodes={c['visited']} value={v1} "
          f"saved={n1 - c['visited']} effective_cutoffs={c['effective_cutoffs']}")

print('=== innovation 2: heuristic & tie-breaking on 10x10 weighted map ===')
G = [[1] * 10 for _ in range(10)]
for r in range(10):
    for c in range(10):
        if (r + c) % 7 == 3: G[r][c] = 8
G[0][0] = 'S'; G[9][9] = 'E'
for name, hs, tb in [('h=0', 0, 'h'), ('admissible, tie=g', None, 'g'), ('admissible, tie=h', None, 'h')]:
    (cost, n), *t = bench(lambda: astar_weighted((0, 0), (9, 9), G, h_scale=hs, count_expanded=True, tie_break=tb))
    print(f'{name}: cost={cost} expanded={n} {fmt(*t)}')

print('=== innovation 3: distance-matrix cache effect (T4) ===')
m_cold, *t_cold = bench(lambda: distance_matrix(use_cache=False), repeat=50)
m_warm, *t_warm = bench(lambda: distance_matrix(use_cache=True), repeat=50)
print(f'no cache (30x A*): {fmt(*t_cold)}')
print(f'cached          : {fmt(*t_warm)}  same={m_cold == m_warm}')
for H in (2, 4):
    r = solve_game(H, use_ab=True)
    print(f"H={H}: {r['best_target']} {r['best_value']} visited={r['counter']['visited']}")
