"""计时工具: perf_counter_ns, 先预热再重复测量, 报告中位数 (us)."""
import statistics
from time import perf_counter_ns


def bench(fn, repeat: int = 200, warmup: int = 5):
    """返回 (fn 的结果, 中位数微秒, 最小微秒, 最大微秒)."""
    for _ in range(warmup):
        fn()
    times = []
    result = None
    for _ in range(repeat):
        t0 = perf_counter_ns()
        result = fn()
        times.append((perf_counter_ns() - t0) / 1000)
    return result, statistics.median(times), min(times), max(times)


def fmt(med, lo, hi) -> str:
    return f'median={med:.1f}us (min {lo:.1f}, max {hi:.1f})'
