"""生成实验报告用到的全部数据, 写入 results/report_data.json 并打印摘要.
所有随机实验使用固定种子, 重新运行得到相同的节点数 (计时会有小幅波动).
运行: python -X utf8 tests/report_data.py  (约 1 分钟)"""
import itertools, json, os, random, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from astar_maze import astar_maze, manhattan
from astar_weighted import astar_weighted, astar_weighted_path, min_terrain_cost
from alphabeta import TreeNode, alphabeta_count, minimax_count, build_sample_tree
import treasure_game as tg
from bench import bench

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = {}
H0 = lambda a, b: 0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- T1
GRID1 = [['S','.','.','X','.'],['.','X','.','.','.'],['.','X','.','X','.'],['.','.','.','X','E'],['X','X','.','.','.']]
t1 = {}
for name, h, tb in [('h0', H0, 'g'), ('manhattan', manhattan, 'g'), ('manhattan_tie_h', manhattan, 'h')]:
    tr = []
    path, n = astar_maze((0, 0), (3, 4), GRID1, h_func=h, count_expanded=True, trace=tr, tie_break=tb)
    _, med, lo, hi = bench(lambda: astar_maze((0, 0), (3, 4), GRID1, h_func=h, tie_break=tb), repeat=500)
    t1[name] = {'path': path, 'expanded': n, 'trace': tr, 'time_us': med}
OUT['t1'] = t1
log('T1', {k: (v['expanded'], round(v['time_us'], 1)) for k, v in t1.items()})

# ---------------------------------------------------------------- T2
GRID2 = [['S',1,1,8,1],[1,8,3,1,1],[1,8,1,8,1],[1,1,1,8,1],[8,8,1,1,'E']]
cost, path = astar_weighted_path((0, 0), (4, 4), GRID2)
steps = []
acc = 0
for p in path[1:]:
    v = GRID2[p[0]][p[1]]
    c = 0 if v == 'E' else int(v)
    acc += c
    steps.append({'cell': p, 'terrain': v, 'step': c, 'g': acc})
m2 = min_terrain_cost(GRID2, skip=[(0, 0)])
variants = {
    'h0': dict(h_scale=0),
    'admissible_tie_h': dict(),
    'admissible_tie_g': dict(tie_break='g'),
    'overestimate_m_d': dict(h_func=lambda p: m2 * manhattan(p, (4, 4)) if p != (4, 4) else 0),
    'weighted_w2': dict(h_func=lambda p: 2 * m2 * max(manhattan(p, (4, 4)) - 1, 0)),
}
t2 = {'cost': cost, 'path': path, 'steps': steps, 'm': m2, 'variants': {}}
for name, kw in variants.items():
    tr = []
    c, n = astar_weighted((0, 0), (4, 4), GRID2, count_expanded=True, trace=tr, **kw)
    t2['variants'][name] = {'cost': c, 'expanded': n, 'trace': tr}
OUT['t2'] = t2
log('T2', cost, {k: (v['cost'], v['expanded']) for k, v in t2['variants'].items()})

# ---------------------------------------------------------------- T3
def t3_tree(mask):
    """mask 的第 i 位为 1 表示把第 i 个内部节点的子节点反转 (max2 被两个 MIN 共用, 只反转一次)."""
    root = build_sample_tree()
    min1, min2 = root.children
    max1, max2 = min1.children
    max3 = min2.children[1]
    internal = [root, min1, min2, max1, max2, max3]
    for i, node in enumerate(internal):
        if mask >> i & 1:
            node.children = list(reversed(node.children))
    return root

names = ['root', 'min1', 'min2', 'max1', 'max2', 'max3']
rows = []
for mask in range(64):
    v, c = alphabeta_count(t3_tree(mask), True, return_counter=True)
    rows.append({'mask': mask, 'reversed': [names[i] for i in range(6) if mask >> i & 1],
                 'value': v, **c})
vis = [r['visited'] for r in rows]
OUT['t3'] = {
    'minimax_visited': minimax_count(build_sample_tree())[1],
    'default': rows[0],
    'all_orders': rows,
    'visited_min': min(vis), 'visited_max': max(vis),
    'hist': {str(k): vis.count(k) for k in sorted(set(vis))},
    'best_example': min(rows, key=lambda r: (r['visited'], len(r['reversed']))),
    'worst_example': max(rows, key=lambda r: (r['visited'], -len(r['reversed']))),
}
assert all(r['value'] == 5 for r in rows)
log('T3', OUT['t3']['minimax_visited'], OUT['t3']['hist'])

# ---------------------------------------------------------------- T4
t4 = {'matrix': tg.distance_matrix(), 'points': tg.POINTS, 'by_depth': {}, 'orders': {}}
for H in (1, 2, 3, 4):
    mm = tg.solve_game(H, use_ab=False)
    ab = tg.solve_game(H, use_ab=True)
    dy = tg.solve_game(H, use_ab=True, dynamic=True)
    me = tg.solve_game(H, use_ab=False, memo=True)
    val, line = tg.best_line(H)
    t4['by_depth'][H] = {
        'best_target': ab['best_target'], 'best_value': ab['best_value'],
        'mm': mm['counter'], 'ab': ab['counter'], 'ab_dynamic': dy['counter'],
        'mm_memo': me['counter'], 'dynamic_result': [dy['best_target'], dy['best_value']],
        'memo_result': [me['best_target'], me['best_value']],
        'first_move_values': tg.first_move_values(H), 'best_line': line,
        'time_us': {
            'mm': bench(lambda: tg.solve_game(H, use_ab=False), repeat=100)[1],
            'ab': bench(lambda: tg.solve_game(H, use_ab=True), repeat=100)[1],
            'ab_dynamic': bench(lambda: tg.solve_game(H, use_ab=True, dynamic=True), repeat=100)[1],
            'mm_memo': bench(lambda: tg.solve_game(H, use_ab=False, memo=True), repeat=100)[1],
        },
    }
    assert [mm['best_target'], mm['best_value']] == [ab['best_target'], ab['best_value']]
    assert dy['best_value'] == ab['best_value'] and me['best_value'] == mm['best_value']
for H in (2, 4):
    res = []
    for order in itertools.permutations(tg.ORDER):
        r = tg.solve_game(H, use_ab=True, order=list(order))
        res.append({'order': list(order), 'visited': r['counter']['visited'],
                    'target': r['best_target'], 'value': r['best_value']})
    v = [r['visited'] for r in res]
    t4['orders'][H] = {'all': res, 'min': min(v), 'max': max(v), 'mean': statistics.mean(v),
                       'default': res[0]['visited']}
_, cold, _, _ = bench(lambda: tg.distance_matrix(use_cache=False), repeat=50)
_, warm, _, _ = bench(lambda: tg.distance_matrix(use_cache=True), repeat=200)
t4['matrix_time_us'] = {'no_cache': cold, 'cached': warm}
OUT['t4'] = t4
for H, d in t4['by_depth'].items():
    log(f"T4 H={H}", d['best_target'], d['best_value'], 'mm', d['mm']['visited'], 'ab', d['ab']['visited'],
        'dyn', d['ab_dynamic']['visited'], 'memo', d['mm_memo']['visited'], d['first_move_values'])
log('T4 orders', {H: (o['min'], o['max'], round(o['mean'], 1), o['default']) for H, o in t4['orders'].items()})

# ---------------------------------------------------------------- E1 迷宫: 规模 / 障碍密度 / 陷阱地图
def random_maze(n, density, rnd):
    while True:
        g = [['X' if rnd.random() < density else '.' for _ in range(n)] for _ in range(n)]
        g[0][0], g[n - 1][n - 1] = 'S', 'E'
        if astar_maze((0, 0), (n - 1, n - 1), g, h_func=H0):
            return g

def maze_stats(n, density, seeds=30, seed0=2026):
    rnd = random.Random(seed0 + n * 100 + int(density * 100))
    e0, em, et, lens = [], [], [], []
    times = {'h0': [], 'manhattan': [], 'manhattan_tie_h': []}
    goal = (n - 1, n - 1)
    variants = {'h0': dict(h_func=H0), 'manhattan': dict(), 'manhattan_tie_h': dict(tie_break='h')}
    for _ in range(seeds):
        g = random_maze(n, density, rnd)
        p0, a = astar_maze((0, 0), goal, g, h_func=H0, count_expanded=True)
        p1, b = astar_maze((0, 0), goal, g, count_expanded=True)
        p2, c = astar_maze((0, 0), goal, g, count_expanded=True, tie_break='h')
        assert len(p0) == len(p1) == len(p2)
        e0.append(a); em.append(b); et.append(c); lens.append(len(p1) - 1)
        for key, kw in variants.items():  # 每张图每种设置测 5 次取中位数 (ms)
            times[key].append(bench(lambda: astar_maze((0, 0), goal, g, **kw), repeat=5, warmup=1)[1] / 1000)
    return {'n': n, 'density': density, 'h0': statistics.mean(e0), 'manhattan': statistics.mean(em),
            'manhattan_tie_h': statistics.mean(et),
            't_h0_ms': statistics.mean(times['h0']), 't_manhattan_ms': statistics.mean(times['manhattan']),
            't_tie_h_ms': statistics.mean(times['manhattan_tie_h']),
            'ratio': statistics.mean(b / a for a, b in zip(e0, em)), 'path_len': statistics.mean(lens),
            'free_cells': n * n * (1 - density)}

e1 = {'size': [maze_stats(n, 0.25) for n in (10, 20, 40, 80)],
      'density': [maze_stats(40, d) for d in (0.0, 0.1, 0.2, 0.3, 0.35)]}
# 陷阱地图: 终点在一面 U 形墙的背后, 曼哈顿距离会把搜索引进死胡同
n = 30
trap = [['.'] * n for _ in range(n)]
for r in range(5, 25):
    trap[r][20] = 'X'
for c in range(8, 21):
    trap[5][c] = 'X'; trap[24][c] = 'X'
trap[15][2], trap[15][27] = 'S', 'E'
e1['trap'] = {}
for name, h, tb in [('h0', H0, 'g'), ('manhattan', manhattan, 'g'), ('manhattan_tie_h', manhattan, 'h')]:
    p, k = astar_maze((15, 2), (15, 27), trap, h_func=h, count_expanded=True, tie_break=tb)
    t_ms = bench(lambda: astar_maze((15, 2), (15, 27), trap, h_func=h, tie_break=tb), repeat=20)[1] / 1000
    e1['trap'][name] = {'expanded': k, 'path_len': len(p) - 1, 'time_ms': t_ms}
e1['trap']['free_cells'] = sum(row.count('.') for row in trap) + 2
OUT['e1'] = e1
log('E1 size', [(s['n'], round(s['h0']), round(s['manhattan']), round(s['manhattan_tie_h']), round(s['ratio'], 2),
                 'ms', round(s['t_h0_ms'], 2), round(s['t_manhattan_ms'], 2), round(s['t_tie_h_ms'], 2)) for s in e1['size']])
log('E1 density', [(s['density'], round(s['h0']), round(s['manhattan']), round(s['manhattan_tie_h']), round(s['ratio'], 2),
                    'ms', round(s['t_h0_ms'], 2), round(s['t_manhattan_ms'], 2), round(s['t_tie_h_ms'], 2)) for s in e1['density']])
log('E1 trap', e1['trap'])

# ---------------------------------------------------------------- E2 加权地形: 地形分布 / 加权 A* / 旧启发函数
TERRAINS = {
    'plain': ([1, 3, 8], [0.7, 0.2, 0.1]),       # 平原为主, m=1
    'rough': ([3, 5, 8], [0.5, 0.3, 0.2]),       # 全部 >=3, m=3
    'uniform': (list(range(1, 10)), [1 / 9] * 9),  # 1..9 均匀
}

def random_terrain(n, kind, rnd, wall=0.15):
    vals, w = TERRAINS[kind]
    while True:
        g = [['X' if rnd.random() < wall else rnd.choices(vals, w)[0] for _ in range(n)] for _ in range(n)]
        g[0][0], g[n - 1][n - 1] = 'S', 'E'
        if astar_weighted((0, 0), (n - 1, n - 1), g, h_scale=0) != -1:
            return g

e2 = {'terrain': [], 'weighted': []}
for kind in TERRAINS:
    rnd = random.Random(7 + len(kind))
    e0, eh, eo = [], [], []
    for _ in range(30):
        g = random_terrain(30, kind, rnd)
        c0, a = astar_weighted((0, 0), (29, 29), g, h_scale=0, count_expanded=True)
        c1, b = astar_weighted((0, 0), (29, 29), g, count_expanded=True)
        c2, o = astar_weighted((0, 0), (29, 29), g, h_scale=min_terrain_cost(g), count_expanded=True)
        assert c0 == c1 == c2
        e0.append(a); eh.append(b); eo.append(o)
    e2['terrain'].append({'kind': kind, 'h0': statistics.mean(e0), 'h': statistics.mean(eh),
                          'h_old_m': statistics.mean(eo),
                          'ratio': statistics.mean(b / a for a, b in zip(e0, eh)),
                          'ratio_old_m': statistics.mean(o / a for a, o in zip(e0, eo))})
rnd = random.Random(99)
maps = [random_terrain(30, 'plain', rnd) for _ in range(30)]
opt = [astar_weighted((0, 0), (29, 29), g, h_scale=0) for g in maps]
for w in (1, 1.5, 2, 3, 5):
    exp, ratio, subopt = [], [], 0
    for g, c_opt in zip(maps, opt):
        m = min_terrain_cost(g, skip=[(0, 0)])
        hf = lambda p, m=m: w * m * max(manhattan(p, (29, 29)) - 1, 0)
        c, k = astar_weighted((0, 0), (29, 29), g, h_func=hf, count_expanded=True)
        exp.append(k); ratio.append(c / c_opt); subopt += c != c_opt
    e2['weighted'].append({'w': w, 'expanded': statistics.mean(exp), 'cost_ratio': statistics.mean(ratio),
                           'max_cost_ratio': max(ratio), 'suboptimal_maps': subopt})
# 旧启发函数 h=m*d (没有扣掉免费的最后一步) 在小图上是否出错
rnd = random.Random(5)
bad = total = 0
example = None
for _ in range(20000):
    R, C = rnd.randint(2, 5), rnd.randint(2, 5)
    g = [[rnd.choice(['X', 1, 1, 2, 3]) for _ in range(C)] for _ in range(R)]
    g[0][0], g[R - 1][C - 1] = 'S', 'E'
    c_opt = astar_weighted((0, 0), (R - 1, C - 1), g, h_scale=0)
    if c_opt == -1:
        continue
    total += 1
    m = min_terrain_cost(g, skip=[(0, 0)]); end = (R - 1, C - 1)
    c_old = astar_weighted((0, 0), end, g, h_func=lambda p: m * manhattan(p, end))
    if c_old != c_opt:
        bad += 1
        if example is None or len(g) * len(g[0]) < len(example['grid']) * len(example['grid'][0]):
            example = {'grid': g, 'optimal': c_opt, 'old_h_result': c_old}
e2['old_h'] = {'maps': total, 'wrong': bad, 'example': example}
# m 不含起点格后的正确性: 高代价地形 / 含 0 代价 / 终点为普通格, 与 h=0 (Dijkstra) 对比
rnd = random.Random(3)
bad2 = 0
for _ in range(5000):
    R, C = rnd.randint(2, 8), rnd.randint(2, 8)
    cells = ['X', 3, 4, 5, 8, 0] if rnd.random() < 0.1 else ['X', 3, 5, 8]
    g = [[rnd.choice(cells) for _ in range(C)] for _ in range(R)]
    g[0][0], g[R - 1][C - 1] = 'S', rnd.choice(['E', 3])
    bad2 += astar_weighted((0, 0), (R - 1, C - 1), g) != astar_weighted((0, 0), (R - 1, C - 1), g, h_scale=0)
e2['start_excluded_check'] = {'maps': 5000, 'mismatch': bad2}
OUT['e2'] = e2
log('E2 terrain', [(t['kind'], round(t['h0']), round(t['h_old_m']), round(t['h']), round(t['ratio'], 2)) for t in e2['terrain']])
log('E2 weighted', [(x['w'], round(x['expanded']), round(x['cost_ratio'], 3), x['suboptimal_maps']) for x in e2['weighted']])
log('E2 old h', bad, '/', total, example, 'start-excluded check mismatch', bad2)

# ---------------------------------------------------------------- E3 随机博弈树: 访问顺序
LEAVES = [0]

class Leaf(TreeNode):
    """被读取 value 时计数, 用来统计 Alpha-Beta 实际评估了多少个叶子."""
    def __init__(self, v):
        super().__init__(value=None)
        self._v = v

    @property
    def value(self):
        LEAVES[0] += 1
        return self._v

    @value.setter
    def value(self, _):
        pass

def rand_tree(b, d, rnd):
    if d == 0:
        return Leaf(rnd.randint(0, 99))
    return TreeNode(children=[rand_tree(b, d - 1, rnd) for _ in range(b)])

def exact(node, is_max):
    if node.is_leaf: return node._v
    vals = [exact(c, not is_max) for c in node.children]
    return max(vals) if is_max else min(vals)

def reorder(node, is_max, best_first):
    """按子节点的精确值排序: best_first=True 为理想顺序, False 为最差顺序."""
    if node.is_leaf: return node
    for c in node.children: reorder(c, not is_max, best_first)
    node.children.sort(key=lambda c: exact(c, not is_max), reverse=(is_max == best_first))
    return node

def leaves_ab(tree):
    LEAVES[0] = 0
    v, c = alphabeta_count(tree, True, return_counter=True)
    return v, c['visited'], LEAVES[0]

e3 = []
for b, d in [(3, 4), (3, 6), (4, 5), (5, 4)]:
    rnd = random.Random(b * 10 + d)
    acc = {'natural': [], 'best': [], 'worst': [], 'nat_leaves': [], 'best_leaves': [], 'worst_leaves': []}
    for _ in range(30):
        t = rand_tree(b, d, rnd)
        v0 = exact(t, True)
        for key in ('natural', 'best', 'worst'):
            if key != 'natural':
                reorder(t, True, key == 'best')
            v, n_vis, n_leaf = leaves_ab(t)
            assert v == v0
            acc[key].append(n_vis)
            acc[{'natural': 'nat_leaves', 'best': 'best_leaves', 'worst': 'worst_leaves'}[key]].append(n_leaf)
    full_nodes = sum(b ** i for i in range(d + 1))
    e3.append({'b': b, 'd': d, 'minimax_nodes': full_nodes, 'minimax_leaves': b ** d,
               'theory_best_leaves': b ** ((d + 1) // 2) + b ** (d // 2) - 1,
               **{k: statistics.mean(v) for k, v in acc.items()}})
OUT['e3'] = e3
log('E3', [(x['b'], x['d'], x['minimax_leaves'], round(x['nat_leaves']), round(x['best_leaves']),
            x['theory_best_leaves'], round(x['worst_leaves'])) for x in e3])

# ---------------------------------------------------------------- E4 更多宝藏: 剪枝/排序/缓存的扩展性
def random_treasure_map(k, rnd, n=9):
    while True:
        g = [['X' if rnd.random() < 0.18 else rnd.choices([1, 2, 3, 5], [0.6, 0.2, 0.1, 0.1])[0]
              for _ in range(n)] for _ in range(n)]
        g[0][0], g[n - 1][n - 1] = 'A', 'B'
        free = [(r, c) for r in range(n) for c in range(n) if g[r][c] != 'X' and (r, c) not in ((0, 0), (n - 1, n - 1))]
        cells = rnd.sample(free, k)
        tre = {}
        for i, p in enumerate(cells):
            g[p[0]][p[1]] = 'T'
            tre[f'T{i + 1}'] = (p, rnd.randint(6, 14))
        tm = tg.TreasureMap(g, (0, 0), (n - 1, n - 1), tre)
        if all(x != -1 for row in tm.distance_matrix() for x in row):
            return tm

e4 = []
for k in (4, 5, 6, 7):
    rnd = random.Random(400 + k)
    acc = {key: [] for key in ('mm', 'ab', 'ab_dynamic', 'mm_memo', 't_mm', 't_ab', 't_dyn', 't_memo')}
    same_first = 0
    for _ in range(15):
        tm = random_treasure_map(k, rnd)
        mm = tg.solve_game(k, use_ab=False, tmap=tm)
        ab = tg.solve_game(k, use_ab=True, tmap=tm)
        dy = tg.solve_game(k, use_ab=True, dynamic=True, tmap=tm)
        me = tg.solve_game(k, use_ab=False, memo=True, tmap=tm)
        assert mm['best_value'] == ab['best_value'] == dy['best_value'] == me['best_value']
        same_first += dy['best_target'] == ab['best_target']
        acc['mm'].append(mm['counter']['visited']); acc['ab'].append(ab['counter']['visited'])
        acc['ab_dynamic'].append(dy['counter']['visited']); acc['mm_memo'].append(me['counter']['visited'])
        rep = 3 if k >= 7 else 10
        acc['t_mm'].append(bench(lambda: tg.solve_game(k, use_ab=False, tmap=tm), repeat=rep, warmup=1)[1])
        acc['t_ab'].append(bench(lambda: tg.solve_game(k, use_ab=True, tmap=tm), repeat=rep, warmup=1)[1])
        acc['t_dyn'].append(bench(lambda: tg.solve_game(k, use_ab=True, dynamic=True, tmap=tm), repeat=rep, warmup=1)[1])
        acc['t_memo'].append(bench(lambda: tg.solve_game(k, use_ab=False, memo=True, tmap=tm), repeat=rep, warmup=1)[1])
    e4.append({'k': k, **{key: statistics.mean(v) for key, v in acc.items()}, 'maps': 15,
               'same_first_move': same_first})
    log('E4', k, {key: round(statistics.mean(v)) for key, v in acc.items()})
OUT['e4'] = e4

os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
with open(os.path.join(ROOT, 'results', 'report_data.json'), 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=list)
log('saved results/report_data.json')
