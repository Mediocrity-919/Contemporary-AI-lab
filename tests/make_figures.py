"""根据 results/report_data.json 生成报告插图, 输出到 figures/.
图片里不写编号, 编号和说明由报告中的图注给出.
运行: python -X utf8 tests/make_figures.py (需要 matplotlib; 先运行 report_data.py)."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(__file__), '..')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)
with open(os.path.join(ROOT, 'results', 'report_data.json'), encoding='utf-8') as f:
    D = json.load(f)

# 配色: 分类色按固定顺序使用; 文字用中性墨色, 不用系列色
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED, GRID, SURF, WALL = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#fcfcfb', '#3d3d3a'
SEQ = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor=SURF)
    plt.close(fig)
    print('saved figures/' + name)


def end_label(ax, x, y, text, color_dot, dy=0):
    """直接标注: 折线末端放一个同色小点 + 中性色文字."""
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords='offset points', va='center',
                color=INK2, fontsize=9)


# ---------------------------------------------------------------- 图1 T1 扩展顺序
GRID1 = [['S','.','.','X','.'],['.','X','.','.','.'],['.','X','.','X','.'],['.','.','.','X','E'],['X','X','.','.','.']]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
for ax, key, title in zip(axes, ['h0', 'manhattan', 'manhattan_tie_h'],
                          ['h = 0', '曼哈顿距离 (同 f 先取 g 小)', '曼哈顿距离 (同 f 先取 h 小)']):
    t = D['t1'][key]
    order = {tuple(n): i + 1 for i, (n, g, h, f) in enumerate(t['trace'])}
    total = len(order)
    ax.set_xlim(0, 5); ax.set_ylim(5, 0); ax.set_aspect('equal'); ax.grid(False)
    ax.set_xticks([i + 0.5 for i in range(5)], range(5)); ax.set_yticks([i + 0.5 for i in range(5)], range(5))
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for r in range(5):
        for c in range(5):
            if GRID1[r][c] == 'X':
                fc = WALL
            elif (r, c) in order:
                fc = SEQ(0.15 + 0.8 * order[(r, c)] / total)
            else:
                fc = '#f0efec'
            ax.add_patch(Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, fc=fc, ec='none', zorder=1))
            if (r, c) in order:
                txt_c = '#ffffff' if order[(r, c)] / total > 0.45 else INK
                ax.text(c + 0.5, r + 0.5, str(order[(r, c)]), ha='center', va='center', fontsize=9, color=txt_c, zorder=3)
    xs = [p[1] + 0.5 for p in t['path']]; ys = [p[0] + 0.5 for p in t['path']]
    ax.plot(xs, ys, color=S2, lw=2, alpha=0.75, zorder=2, solid_capstyle='round')
    ax.set_title(f'{title}\n扩展 {t["expanded"]} 个节点', fontsize=10)
fig.tight_layout()
save(fig, 'fig1_t1_expansion.png')

# ---------------------------------------------------------------- 图2 E1 迷宫规模
sz = D['e1']['size']
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ns = [s['n'] for s in sz]
for key, col, mk, lab in [('h0', S1, 'o', 'h=0'), ('manhattan', S2, 's', '曼哈顿'), ('manhattan_tie_h', S3, '^', '曼哈顿+同f先取h小')]:
    ys = [s[key] for s in sz]
    ax.plot(ns, ys, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5)
    end_label(ax, ns[-1], ys[-1], f'{lab}  {ys[-1]:.0f}', col)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xticks(ns, [f'{n}×{n}' for n in ns]); ax.minorticks_off()
ax.set_xlim(8, 260)
ax.set_xlabel('地图规模 (障碍密度 0.25, 每点 30 张图取平均)'); ax.set_ylabel('扩展节点数 (对数刻度)')
ax.set_title('扩展节点数随地图规模增长')
ax = axes[1]
dn = D['e1']['density']
ds = [s['density'] for s in dn]
for key, col, mk, lab in [('h0', S1, 'o', 'h=0'), ('manhattan', S2, 's', '曼哈顿'), ('manhattan_tie_h', S3, '^', '曼哈顿+同f先取h小')]:
    ys = [s[key] for s in dn]
    ax.plot(ds, ys, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5)
    end_label(ax, ds[-1], ys[-1], lab, col, dy={'h0': 0, 'manhattan': 9, 'manhattan_tie_h': -9}[key])
ax.set_xlim(-0.02, 0.5); ax.set_ylim(0, 1750)
ax.set_xlabel('障碍密度 (40×40 地图)'); ax.set_ylabel('扩展节点数')
ax.set_title('扩展节点数随障碍密度变化')
fig.tight_layout()
save(fig, 'fig2_e1_maze_scaling.png')

# ---------------------------------------------------------------- 图3 E2 加权 A* 权衡
wt = D['e2']['weighted']
ws = [str(x['w']) for x in wt]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
ax = axes[0]
b = ax.bar(ws, [x['expanded'] for x in wt], color=S1, width=0.55)
ax.bar_label(b, fmt='%.0f', color=INK2, fontsize=9, padding=2)
ax.set_xlabel('启发函数放大系数 w (w=1 为可采纳)'); ax.set_ylabel('平均扩展节点数')
ax.set_title('扩展节点数'); ax.grid(axis='x', visible=False)
ax = axes[1]
b = ax.bar(ws, [(x['cost_ratio'] - 1) * 100 for x in wt], color=S2, width=0.55)
ax.bar_label(b, labels=[f"{(x['cost_ratio'] - 1) * 100:.1f}%\n({x['suboptimal_maps']}/30 张次优)" for x in wt],
             color=INK2, fontsize=8.5, padding=2)
ax.set_ylim(0, 45)
ax.set_xlabel('启发函数放大系数 w'); ax.set_ylabel('平均代价超出最优 (%)')
ax.set_title('解的质量'); ax.grid(axis='x', visible=False)
fig.tight_layout()
save(fig, 'fig3_e2_weighted_astar.png')

# ---------------------------------------------------------------- 图4 T3 64 种顺序 + 图5 T4 24 种顺序
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={'width_ratios': [1, 1.6]})
ax = axes[0]
hist = D['t3']['hist']
ks = sorted(hist, key=int)
b = ax.bar(ks, [hist[k] for k in ks], color=S1, width=0.6)
ax.bar_label(b, color=INK2, fontsize=9, padding=2)
ax.set_xlabel('Alpha-Beta 访问节点数 (Minimax 为 15)'); ax.set_ylabel('顺序个数 (共 64 种)')
ax.set_title('T3: 64 种子节点顺序'); ax.grid(axis='x', visible=False)
ax.set_title('T3: 64 种子节点顺序 (默认左到右为 14)')
ax.set_ylim(0, max(hist.values()) * 1.2)
ax = axes[1]
od = sorted(D['t4']['orders']['4']['all'], key=lambda r: r['visited'])
labels = [''.join(t[1] for t in r['order']) for r in od]
cols = [S2 if r['order'] == ['T1', 'T2', 'T3', 'T4'] else S1 for r in od]
ax.bar(range(len(od)), [r['visited'] for r in od], color=cols, width=0.7)
ax.set_xticks(range(len(od)), labels, rotation=90, fontsize=8)
dyn = D['t4']['by_depth']['4']['ab_dynamic']['visited']
mm = D['t4']['by_depth']['4']['mm']['visited']
for y, txt in [(mm, f'Minimax {mm}'), (dyn, f'动态排序 {dyn}')]:
    ax.axhline(y, color=INK2, lw=1, ls=(0, (4, 3)), zorder=3)
    ax.text(-0.5, y + 1.2, txt, ha='left', va='bottom', color=INK2, fontsize=9, zorder=4,
            bbox=dict(fc=SURF, ec='none', pad=1))
ax.set_ylim(0, 75)
ax.set_xlabel('固定访问顺序 (例: 3124 表示 T3,T1,T2,T4); 橙色为默认顺序 T1..T4')
ax.set_ylabel('访问节点数'); ax.set_title('T4 (H=4): 24 种宝藏访问顺序'); ax.grid(axis='x', visible=False)
fig.tight_layout()
save(fig, 'fig4_order_effect.png')

# ---------------------------------------------------------------- 图5 E3 随机树叶子数
e3 = D['e3']
fig, ax = plt.subplots(figsize=(10.5, 3.8))
groups = [f"b={x['b']}, d={x['d']}" for x in e3]
series = [('minimax_leaves', 'Minimax (全部叶子)', '#c3c2b7'), ('worst_leaves', 'AB 最差顺序', S2),
          ('nat_leaves', 'AB 随机顺序', S1), ('best_leaves', 'AB 理想顺序', S3)]
wbar = 0.2
for i, (key, lab, col) in enumerate(series):
    xs = [g + (i - 1.5) * wbar for g in range(len(e3))]
    ys = [x[key] for x in e3]
    bars = ax.bar(xs, ys, width=wbar - 0.02, color=col, label=lab)
    ax.bar_label(bars, fmt='%.0f', fontsize=7.5, color=INK2, padding=1)
ax.set_xticks(range(len(e3)), groups)
ax.set_ylabel('评估的叶子数 (30 棵随机树平均)')
ax.legend(ncol=4, loc='upper left', fontsize=9)
ax.set_ylim(0, 1200); ax.grid(axis='x', visible=False)

fig.tight_layout()
save(fig, 'fig5_e3_random_trees.png')

# ---------------------------------------------------------------- 图6 E4 宝藏数扩展
e4 = D['e4']
ks = [x['k'] for x in e4]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
lines = [('mm', 'Minimax', S1, 'o'), ('mm_memo', 'Minimax+状态缓存', S4, 'D'),
         ('ab', 'Alpha-Beta', S2, 's'), ('ab_dynamic', 'AB+动态排序', S3, '^')]
for ax, prefix, ylab, title in [(axes[0], '', '访问节点数 (对数刻度)', '访问节点数'),
                                (axes[1], 't_', '耗时 µs (对数刻度, 中位数)', '耗时')]:
    for key, lab, col, mk in lines:
        k2 = key if not prefix else {'mm': 't_mm', 'mm_memo': 't_memo', 'ab': 't_ab', 'ab_dynamic': 't_dyn'}[key]
        ys = [x[k2] for x in e4]
        ax.plot(ks, ys, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5)
        end_label(ax, ks[-1], ys[-1], f'{lab}  {ys[-1]:.0f}', col)
    ax.set_yscale('log'); ax.set_xticks(ks); ax.set_xlim(3.8, 8.6); ax.minorticks_off()
    ax.set_xlabel('宝藏数 k (搜索到底, H=k; 每点 15 张随机 9×9 地图)'); ax.set_ylabel(ylab); ax.set_title(title)
fig.tight_layout()
save(fig, 'fig6_e4_treasure_scaling.png')
