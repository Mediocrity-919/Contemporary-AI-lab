"""通用工具：随机种子、配置加载（继承 / 覆盖 / 占位符 / 哈希）、运行目录与环境信息。

配置约定（详见 configs/base.yaml）：
- 每个实验 YAML 用 `inherit:` 继承另一个 YAML，只写不同的字段。
- 值写成 "${key}" 时，从 configs/decisions.yaml 读取；该决定尚未做出（值为 null）时拒绝运行。
- 配置哈希只覆盖影响训练结果的字段（不含 meta、system），完全相同的配置 + 种子会复用已有结果。
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import platform
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
# 以下环境变量只用于隔离测试（把输出和决策文件重定向到临时目录），正式实验不要设置
DECISIONS_FILE = Path(os.environ.get("LAB3_DECISIONS", ROOT / "configs" / "decisions.yaml"))
OUTPUT_ROOT_OVERRIDE = os.environ.get("LAB3_OUTPUT_ROOT")
EXCLUDED_FROM_HASH = ("meta", "system")
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z0-9_]+)\}$")


class UnresolvedDecision(RuntimeError):
    """配置引用了 decisions.yaml 中尚未确定的值。"""


# ----------------------------------------------------------------------------- 配置
def deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典，override 优先；返回新字典。"""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _load_raw(path: Path) -> dict:
    """读取 YAML 并递归展开 inherit 链（尚不解析占位符）。"""
    path = Path(path).resolve()
    raw = _read_yaml(path)
    parent = raw.pop("inherit", None)
    if parent:
        raw = deep_merge(_load_raw(path.parent / parent), raw)
    return raw


def _coerce(obj):
    """把形如 '1e-4' 的字符串转成数字（PyYAML 不把无小数点的科学计数法识别为浮点数）。"""
    if isinstance(obj, dict):
        return {k: _coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce(v) for v in obj]
    if isinstance(obj, str) and _NUMBER.match(obj):
        value = float(obj)
        return int(value) if value.is_integer() and "." not in obj and "e" not in obj.lower() else value
    return obj


def set_by_path(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def get_by_path(cfg: dict, dotted: str):
    node = cfg
    for key in dotted.split("."):
        node = node[key]
    return node


def parse_value(text: str):
    """解析命令行 --set 的值（YAML 语法）。"""
    return _coerce(yaml.safe_load(text))


def load_decisions(path: Path | None = None) -> dict:
    path = Path(path or DECISIONS_FILE)
    return _read_yaml(path) if path.exists() else {}


def resolve_placeholders(obj, decisions: dict):
    """把 "${key}" 替换为 decisions[key]；未确定的值抛出 UnresolvedDecision。"""
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, decisions) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(v, decisions) for v in obj]
    if isinstance(obj, str):
        match = _PLACEHOLDER.match(obj)
        if match:
            key = match.group(1)
            if decisions.get(key) is None:
                raise UnresolvedDecision(key)
            return _coerce(decisions[key])
    return obj


def load_config(path, overrides: dict | None = None, resolve: bool = True) -> dict:
    """读取实验配置：展开继承 → 应用覆盖 → 解析占位符。"""
    path = Path(path).resolve()
    cfg = _load_raw(path)
    for dotted, value in (overrides or {}).items():
        set_by_path(cfg, dotted, value)
    cfg = _coerce(cfg)
    if resolve:
        cfg = resolve_placeholders(cfg, load_decisions())
    cfg.setdefault("meta", {})["config_file"] = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    """只对影响训练结果的字段求哈希。"""
    payload = {k: v for k, v in cfg.items() if k not in EXCLUDED_FROM_HASH}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def expand_sweep(cfg_raw: dict) -> list[tuple[str, dict]]:
    """把 meta.sweep（{点路径: [取值...]}）展开为 [(变体名, 覆盖字典)]，多个键取笛卡尔积。"""
    name = cfg_raw.get("meta", {}).get("name", "run")
    sweep = cfg_raw.get("meta", {}).get("sweep") or {}
    if not sweep:
        return [(name, {})]
    keys = list(sweep)
    variants = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        overrides = dict(zip(keys, values))
        suffix = "_".join(f"{k.split('.')[-1]}{v}" for k, v in overrides.items())
        overrides["meta.name"] = f"{name}_{suffix}"
        variants.append((overrides["meta.name"], overrides))
    return variants


def load_variants(path) -> list[dict]:
    """读取一个配置文件的全部扫参变体（未确定的占位符会抛出 UnresolvedDecision）。"""
    raw = _coerce(_load_raw(Path(path)))
    return [load_config(path, overrides) for _, overrides in expand_sweep(raw)]


# ----------------------------------------------------------------------------- 运行目录
def output_root(cfg: dict) -> Path:
    return ROOT / (OUTPUT_ROOT_OVERRIDE or cfg.get("system", {}).get("output_root", "outputs/runs"))


def test_log_path(cfg: dict) -> Path:
    """测试集使用记录：与运行目录同级，例如 outputs/runs → outputs/test_log.csv。"""
    return output_root(cfg).parent / "test_log.csv"


def run_dir_for(cfg: dict, seed: int) -> Path:
    """运行目录：outputs/runs/<首次使用的配置名>-<哈希>/seed<k>。相同哈希复用同一目录。"""
    root = output_root(cfg)
    digest = config_hash(cfg)
    existing = sorted(root.glob(f"*-{digest}")) if root.exists() else []
    base = existing[0] if existing else root / f"{cfg['meta']['name']}-{digest}"
    return base / f"seed{seed}"


def is_complete(run_dir: Path) -> bool:
    return (Path(run_dir) / "metrics.json").exists()


def completed_seeds(cfg: dict, seeds) -> dict[int, Path]:
    out = {}
    for seed in seeds:
        run_dir = run_dir_for(cfg, seed)
        if is_complete(run_dir):
            out[seed] = run_dir
    return out


# ----------------------------------------------------------------------------- 随机性
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子；deterministic=True 时要求逐位可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ----------------------------------------------------------------------------- 杂项
def env_info() -> dict:
    import torchvision

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_yaml(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")


def atomic_torch_save(obj, path) -> None:
    """先写临时文件再替换，避免中断时留下损坏的 checkpoint。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
