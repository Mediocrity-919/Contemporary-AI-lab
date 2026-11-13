import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from astar_maze import astar_maze
from astar_weighted import astar_weighted
from alphabeta import build_sample_tree, alphabeta, minimax, solve_tree_json, SAMPLE_JSON
from treasure_game import solve_game, distance_matrix, best_first_move

fails = 0
def check(name, got, want):
    global fails
    ok = got == want
    print(('PASS' if ok else 'FAIL'), name, 'got=', got, 'want=', want)
    if not ok: fails += 1

# T1
grid1 = [['S','.','.','X','.'],['.','X','.','.','.'],['.','X','.','X','.'],['.','.','.','X','E'],['X','X','.','.','.']]
p = astar_maze((0,0),(3,4),grid1)
check('T1 reachable', (len(p)-1==7 and p[0]==(0,0) and p[-1]==(3,4)), True)
check('T1 exact', p, [(0,0),(0,1),(0,2),(1,2),(1,3),(1,4),(2,4),(3,4)])

# T2
grid2 = [['S',1,1,8,1],[1,8,3,1,1],[1,8,1,8,1],[1,1,1,8,1],[8,8,1,1,'E']]
check('T2 cost', astar_weighted((0,0),(4,4),grid2), 7)

# T3
root = build_sample_tree()
check('T3 alphabeta', alphabeta(root, is_max=True), 5)
check('T3 minimax', minimax(root, True), 5)
check('T3 PDF JSON input', solve_tree_json(SAMPLE_JSON), 5)
import json
check('T3 JSON string input', solve_tree_json(json.dumps(SAMPLE_JSON)), 5)
# is_maximizing_player 生效: 根改成 MIN 后结果应为 min(5, 2) = 2
min_root = json.loads(json.dumps(SAMPLE_JSON)); min_root['tree_structure']['is_maximizing_player'] = False
check('T3 JSON root is MIN', solve_tree_json(min_root), 2)

# T4
m = distance_matrix()
print('T4 matrix:')
for row in m: print(row)
r2 = solve_game(2, use_ab=True)
check('T4 H=2', (r2['best_target'], r2['best_value']), ('T2', -1))
r4 = solve_game(4, use_ab=True)
check('T4 H=4', (r4['best_target'], r4['best_value']), ('T3', 3))
r2m = solve_game(2, use_ab=False)
check('T4 minimax==ab H=2', (r2m['best_target'], r2m['best_value']), ('T2', -1))

r4m = solve_game(4, use_ab=False)
check('T4 minimax==ab H=4', (r4m['best_target'], r4m['best_value']), ('T3', 3))
check('T4 PDF output format H=2', best_first_move(2), {'best_target': 'T2', 'best_value': -1})
check('T4 PDF output format H=4', best_first_move(4), {'best_target': 'T3', 'best_value': 3})

# 边界情况
check('T1 no path', astar_maze((0,0),(0,2),[['S','X','E']]), [])
check('T1 start on wall', astar_maze((0,0),(1,1),[['X','.'],['.','E']]), [])
check('T1 start out of grid', astar_maze((5,5),(0,0),[['E','.']]), [])
check('T1 empty grid', astar_maze((0,0),(0,0),[]), [])
check('T2 no path', astar_weighted((0,0),(0,2),[['S','X','E']]), -1)
check('T2 start on wall', astar_weighted((0,0),(1,1),[['X',1],[1,'E']]), -1)
check('T2 empty grid', astar_weighted((0,0),(0,0),[]), -1)
check('T2 adjacent E is free', astar_weighted((0,0),(0,1),[['S','E']]), 0)
# 含 0 代价格: 最小代价 m=0, h 必须退化为 0, 否则会返回 7
grid0 = [['S',0,3,0],[8,3,3,1],[3,3,0,3],[8,8,0,'E']]
check('T2 zero-cost terrain', astar_weighted((0,0),(3,3),grid0), 6)
# 终点为普通格时不免费
check('T2 numeric end', astar_weighted((0,0),(0,2),[['S',2,3]]), 5)

print('FAILS=', fails)
sys.exit(1 if fails else 0)
