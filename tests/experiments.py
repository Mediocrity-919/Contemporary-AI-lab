"""四道题的基础对照实验. 节点口径:
- A*: expanded = 弹出并关闭的节点数 (含终点).
- 博弈树: visited = 递归访问的节点数; cutoffs = alpha>=beta 触发次数;
  effective_cutoffs = 真正跳过了后续子节点的次数; skipped_children = 被跳过的子树个数.
计时: perf_counter_ns, 预热后重复 200 次取中位数."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from astar_maze import astar_maze, manhattan
from astar_weighted import astar_weighted
from alphabeta import build_sample_tree, minimax_count, alphabeta_count, TreeNode
from treasure_game import solve_game
from bench import bench, fmt

print('=== T1: h=0 vs manhattan ===')
grid1 = [['S','.','.','X','.'],['.','X','.','.','.'],['.','X','.','X','.'],['.','.','.','X','E'],['X','X','.','.','.']]
for name, h in [('h=0', lambda a, b: 0), ('manhattan', manhattan)]:
    (p, n), *t = bench(lambda: astar_maze((0, 0), (3, 4), grid1, h_func=h, count_expanded=True))
    print(f'{name}: expanded={n} moves={len(p) - 1} {fmt(*t)}')

print('=== T2: h=0 vs admissible h (tie-break by h / by g) ===')
grid2 = [['S',1,1,8,1],[1,8,3,1,1],[1,8,1,8,1],[1,1,1,8,1],[8,8,1,1,'E']]
for name, hs, tb in [('h=0', 0, 'h'), ('admissible, tie=h', None, 'h'), ('admissible, tie=g', None, 'g')]:
    (c, n), *t = bench(lambda: astar_weighted((0, 0), (4, 4), grid2, h_scale=hs, count_expanded=True, tie_break=tb))
    print(f'{name}: cost={c} expanded={n} {fmt(*t)}')

print('=== T3: minimax vs alphabeta + order ===')
v, n = minimax_count(build_sample_tree(), True)
print(f'minimax L->R: value={v} nodes={n}')

def rev(n):
    """递归反转所有子节点顺序."""
    if not n.children: return TreeNode(value=n.value)
    return TreeNode(children=[rev(c) for c in reversed(n.children)])

for nm, tree in [('L->R', build_sample_tree()), ('R->L', rev(build_sample_tree()))]:
    v, c = alphabeta_count(tree, True, return_counter=True)
    print(f'alphabeta {nm}: value={v} {c}')

print('=== T4: H=2/4 minimax vs ab ===')
for H in (2, 4):
    for ab in (False, True):
        r, *t = bench(lambda: solve_game(H, use_ab=ab))
        c = r['counter']
        print(f"H={H} {'ab' if ab else 'mm'}: best={r['best_target']} val={r['best_value']} "
              f"visited={c['visited']} cutoffs={c['cutoffs']} effective={c['effective_cutoffs']} "
              f"skipped={c['skipped_children']} {fmt(*t)}")
