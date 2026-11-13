"""独立对拍: 用另外实现的 Dijkstra 和完整 Minimax 回传检查 src/ 中的算法 (不修改被测代码).

运行 (lab2 根目录): python -X utf8 tests/verify_independent.py
覆盖 1000 个随机迷宫、2000 张正权地形、500 棵随机博弈树、题目四的距离矩阵与 24 种访问顺序;
边界输入的结果单独记录在输出的 JSON 中.
"""
import heapq
import itertools
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar_maze import astar_maze
from astar_weighted import astar_weighted, astar_weighted_path
from alphabeta import TreeNode, alphabeta
import treasure_game as game


def reference_distance(start, end, grid, weighted=True):
    """参考 Dijkstra, 不依赖被测代码中的代价函数."""
    rows, cols = len(grid), len(grid[0])
    queue, best = [(0, start)], {start: 0}
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != best[current]:
            continue
        if current == end:
            return distance
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = current[0] + dr, current[1] + dc
            if not (0 <= r < rows and 0 <= c < cols) or grid[r][c] == "X":
                continue
            cell = grid[r][c]
            step = (0 if cell == "E" else 1 if cell in ("S", "A", "B", "T")
                    else int(cell)) if weighted else 1
            candidate = distance + step
            if candidate < best.get((r, c), float("inf")):
                best[r, c] = candidate
                heapq.heappush(queue, (candidate, (r, c)))
    return -1


def main():
    results = {"seed": 20261015}
    rng = random.Random(results["seed"])
    for label, zero in (("positive_weighted", False), ("zero_weighted_exploratory", True)):
        failures = []
        for case in range(2000):
            rows, cols = rng.randint(2, 7), rng.randint(2, 7)
            choices = [1, 1, 1, 3, 8, "X"] + ([0, 0] if zero else [])
            grid = [[rng.choice(choices) for _ in range(cols)] for _ in range(rows)]
            grid[0][0], grid[-1][-1] = "S", "E"
            end = rows - 1, cols - 1
            expected = reference_distance((0, 0), end, grid)
            got = astar_weighted((0, 0), end, grid)
            path_cost, _ = astar_weighted_path((0, 0), end, grid)
            path_cost = -1 if path_cost == float("inf") else path_cost
            if got != expected or path_cost != expected:
                failures.append({"case": case, "expected": expected, "got": got,
                                 "path_cost": path_cost, "grid": grid})
        results[label] = {"cases": 2000, "failures": len(failures),
                          "first_failure": failures[0] if failures else None}
        if not zero:
            assert not failures

    failures = 0
    for _ in range(1000):
        rows, cols = rng.randint(2, 8), rng.randint(2, 8)
        grid = [[rng.choice([".", ".", ".", "X"]) for _ in range(cols)] for _ in range(rows)]
        grid[0][0], grid[-1][-1] = "S", "E"
        end = rows - 1, cols - 1
        expected = reference_distance((0, 0), end, grid, weighted=False)
        path = astar_maze((0, 0), end, grid)
        got = len(path) - 1 if path else -1
        legal = not path or (
            path[0] == (0, 0) and path[-1] == end
            and all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
            and all(grid[r][c] != "X" for r, c in path)
        )
        failures += got != expected or not legal
    assert not failures
    results["maze"] = {"cases": 1000, "failures": failures}

    tree_rng = random.Random(20261015)

    def build_tree(depth):
        if depth == 0 or tree_rng.random() < 0.2:
            return TreeNode(value=tree_rng.randint(-100, 100))
        return TreeNode(children=[build_tree(depth - 1) for _ in range(tree_rng.randint(1, 4))])

    def tree_reference(node, maximizing=True):
        if not node.children:
            return node.value
        return (max if maximizing else min)(tree_reference(c, not maximizing) for c in node.children)

    for _ in range(500):
        root = build_tree(5)
        assert alphabeta(root) == tree_reference(root)
    results["alphabeta_random_trees"] = {"cases": 500, "failures": 0}

    matrix = [[reference_distance(game.pos_of(a), game.pos_of(b), game.GRID)
               for b in game.POINTS] for a in game.POINTS]
    assert matrix == game.distance_matrix()
    results["distance_matrix_matches_dijkstra"] = True
    results["treasure_orders"] = {}
    for depth in (2, 4):
        runs = []
        for order in itertools.permutations(game.ORDER):
            ab = game.solve_game(depth, order=list(order))
            mm = game.solve_game(depth, use_ab=False, order=list(order))
            assert (ab["best_target"], ab["best_value"]) == (mm["best_target"], mm["best_value"])
            runs.append({"order": list(order), "visited": ab["counter"]["visited"],
                         "target": ab["best_target"], "value": ab["best_value"]})
        results["treasure_orders"][str(depth)] = {
            "cases": len(runs), "failures": 0,
            "best": min(runs, key=lambda r: r["visited"]),
            "worst": max(runs, key=lambda r: r["visited"]),
        }

    free_goal_grid = [["S", "E"]]
    results["admissibility_counterexample"] = {
        "grid": free_goal_grid, "h_start": 1,
        "true_remaining_cost": reference_distance((0, 0), (0, 1), free_goal_grid),
    }

    original = game.astar_weighted_path
    calls = [0]

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    try:
        game.astar_weighted_path = counted
        per_call = []
        for _ in range(2):
            before = calls[0]
            game.distance_matrix()
            per_call.append(calls[0] - before)
        results["distance_matrix_astar_calls"] = per_call
    finally:
        game.astar_weighted_path = original

    results["boundary_exploratory"] = {}
    for name, function, args in (
        ("maze_no_path", astar_maze, ((0, 0), (0, 2), [["S", "X", "E"]])),
        ("weighted_no_path", astar_weighted, ((0, 0), (0, 2), [["S", "X", "E"]])),
        ("maze_empty", astar_maze, ((0, 0), (0, 0), [])),
        ("weighted_empty", astar_weighted, ((0, 0), (0, 0), [])),
        ("maze_outside_start", astar_maze, ((5, 0), (0, 1), [["S", "E"]])),
        ("weighted_wall_start", astar_weighted, ((0, 0), (0, 1), [["X", "E"]])),
    ):
        try:
            results["boundary_exploratory"][name] = function(*args)
        except Exception as exc:
            results["boundary_exploratory"][name] = type(exc).__name__ + ": " + str(exc)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
