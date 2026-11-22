"""Shared helpers: config, IO, seeding, async progress and query heuristics."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
from typing import Any, Iterable, List, Sequence

import numpy as np
import yaml


# ----------------------------------------------------------------------------- config
class Cfg(dict):
    """A dict with attribute access. Nested dicts are converted on load."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_cfg(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _to_cfg(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_cfg(v) for v in obj]
    return obj


def load_config(path: str, overrides: Sequence[str] | None = None) -> Cfg:
    """Load YAML config and apply ``key.sub=value`` overrides (values parsed as YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return _to_cfg(data)


def add_common_args(parser):
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config entries, e.g. --set vlm.n_perm=1 rerank.k_deep=8",
    )
    return parser


def work_path(cfg: Cfg, *parts: str) -> str:
    path = os.path.join(cfg.paths.work_dir, *parts)
    ensure_dir(os.path.dirname(path))
    return path


# ----------------------------------------------------------------------------- io
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------------- misc
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_of(key: str, dev_frac: float, calib_frac: float) -> str:
    """Deterministic train/dev/calib split from a string key."""
    u = (stable_hash(key) % 10_000) / 10_000.0
    if u < dev_frac:
        return "dev"
    if u < dev_frac + calib_frac:
        return "calib"
    return "train"


def softmax_np(x, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax_np(x, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


_TEMPORAL = re.compile(
    r"\b(then|before|after|while|first|finally|later|afterwards|begins?|starts?|ends?|followed|until|next|again)\b"
)
_NEGATION = re.compile(r"\b(not|no|without|never|nobody|nothing|none|empty)\b|n't\b")
_COUNT = re.compile(
    r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten|several|both|couple|pair|many|multiple|group)\b"
)


def tag_query(text: str) -> List[str]:
    """Cheap heuristic tags used only for error breakdowns (not part of the method)."""
    t = text.lower()
    tags = []
    if _TEMPORAL.search(t):
        tags.append("temporal")
    if _NEGATION.search(t):
        tags.append("negation")
    if _COUNT.search(t):
        tags.append("count")
    if len(t.split()) >= 15:
        tags.append("long")
    return tags or ["plain"]


async def gather_progress(coros: Iterable, desc: str | None = None) -> list:
    """asyncio.gather that keeps order and shows a tqdm bar."""
    from tqdm import tqdm

    coros = list(coros)
    if not coros:
        return []
    pbar = tqdm(total=len(coros), desc=desc, disable=desc is None)

    async def _wrap(c):
        result = await c
        pbar.update(1)
        return result

    try:
        return list(await asyncio.gather(*[_wrap(c) for c in coros]))
    finally:
        pbar.close()
