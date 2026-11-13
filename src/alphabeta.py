"""T3 极大极小博弈树 + Alpha-Beta (根为MAX, 从左到右).

两种输入方式:
1. PDF 的 TreeNode 构建代码: alphabeta(root) -> int.
2. PDF 的 JSON 输入样例: solve_tree_json(json 字符串或字典) -> int,
   节点上的 is_maximizing_player 决定 MAX/MIN; 没有该字段时按层轮流.
命令行: python src/alphabeta.py tree.json (不给文件时读标准输入, 再不行用 PDF 样例).
"""
import json
import sys


class TreeNode:
    def __init__(self, value=None, children=None, is_max=None):
        self.value = value
        self.children = children if children is not None else []
        self.is_max = is_max  # None 表示由父节点决定 (与父节点相反)

    @property
    def is_leaf(self):
        return len(self.children) == 0


def minimax(node: TreeNode, is_max: bool, counter=None) -> int:
    if counter is not None:
        counter['visited'] += 1
    if node.is_max is not None:
        is_max = node.is_max
    if node.is_leaf:
        return node.value
    if is_max:
        best = float('-inf')
        for ch in node.children:
            best = max(best, minimax(ch, False, counter))
        return best
    else:
        best = float('inf')
        for ch in node.children:
            best = min(best, minimax(ch, True, counter))
        return best


def _record_cutoff(counter, n_children: int, i: int):
    """统计剪枝. cutoffs: alpha>=beta 触发次数 (含最后一个子节点后才触发的情况);
    effective_cutoffs: 真正跳过了后续子节点的次数; skipped_children: 被跳过的子树个数."""
    if counter is None:
        return
    skipped = n_children - i - 1
    counter['cutoffs'] = counter.get('cutoffs', 0) + 1
    if skipped > 0:
        counter['effective_cutoffs'] = counter.get('effective_cutoffs', 0) + 1
        counter['skipped_children'] = counter.get('skipped_children', 0) + skipped


def alphabeta(node: TreeNode, alpha=float('-inf'), beta=float('inf'),
              is_max: bool = True, counter=None) -> int:
    """PDF要求: 接收根节点, 返回MAX最优值. 默认根为MAX, 子节点从左到右访问."""
    if counter is not None:
        counter['visited'] += 1
    if node.is_max is not None:
        is_max = node.is_max
    if node.is_leaf:
        return node.value
    n = len(node.children)
    if is_max:
        best = float('-inf')
        for i, ch in enumerate(node.children):
            best = max(best, alphabeta(ch, alpha, beta, False, counter))
            alpha = max(alpha, best)
            if alpha >= beta:
                _record_cutoff(counter, n, i)
                break
        return best
    else:
        best = float('inf')
        for i, ch in enumerate(node.children):
            best = min(best, alphabeta(ch, alpha, beta, True, counter))
            beta = min(beta, best)
            if alpha >= beta:
                _record_cutoff(counter, n, i)
                break
        return best


def minimax_count(node: TreeNode, is_max: bool = True):
    c = {'visited': 0}
    v = minimax(node, is_max, c)
    return v, c['visited']


def alphabeta_count(node: TreeNode, is_max: bool = True, return_counter: bool = False):
    """返回 (value, visited); return_counter=True 时返回 (value, counter 字典)."""
    c = {'visited': 0, 'cutoffs': 0, 'effective_cutoffs': 0, 'skipped_children': 0}
    v = alphabeta(node, float('-inf'), float('inf'), is_max, c)
    return (v, c) if return_counter else (v, c['visited'])


def build_sample_tree() -> TreeNode:
    leaf1 = TreeNode(value=3); leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6); leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1); leaf6 = TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    # PDF原文 min2 = [max2, max3] (复用max2), 照抄
    min2 = TreeNode(children=[max2, max3])
    root = TreeNode(children=[min1, min2])
    return root


def tree_from_dict(d) -> TreeNode:
    """把 PDF 的 JSON 结构转成 TreeNode. 叶子: {"value": v};
    非叶子: {"is_maximizing_player": bool, "children": [...]}; 外层可包 {"tree_structure": ...}."""
    if 'tree_structure' in d:
        d = d['tree_structure']
    children = d.get('children') or []
    if not children:
        if 'value' not in d:
            raise ValueError(f'叶子节点缺少 value: {d}')
        return TreeNode(value=d['value'])
    return TreeNode(children=[tree_from_dict(c) for c in children],
                    is_max=d.get('is_maximizing_player'))


def solve_tree_json(data) -> int:
    """输入 PDF 格式的 JSON (字符串或字典), 返回根节点的最优值."""
    if isinstance(data, str):
        data = json.loads(data)
    root = tree_from_dict(data)
    return alphabeta(root, is_max=True if root.is_max is None else root.is_max)


SAMPLE_JSON = {"tree_structure": {"is_maximizing_player": True, "children": [
    {"is_maximizing_player": False, "children": [
        {"is_maximizing_player": True, "children": [{"value": 3}, {"value": 5}]},
        {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]}]},
    {"is_maximizing_player": False, "children": [
        {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
        {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 2}]}]}]}}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding='utf-8') as f:
            data = json.load(f)
    elif not sys.stdin.isatty() and (text := sys.stdin.read().strip()):
        data = json.loads(text)
    else:
        data = SAMPLE_JSON
    print(solve_tree_json(data))
