"""T4 H=4 访问顺序对照: 默认顺序 vs 重排顺序 (结果应一致, 访问节点数不同)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from treasure_game import solve_game
from bench import bench, fmt

for name, order in [('default  ', None), ('reordered', ['T3', 'T1', 'T2', 'T4'])]:
    r, *t = bench(lambda: solve_game(4, True, order=order))
    print(f"{name}: {r['best_target']} {r['best_value']} {r['counter']} {fmt(*t)}")
