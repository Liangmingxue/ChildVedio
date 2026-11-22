"""Small local adapters for the existing repository implementation."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from . import ROOT
from cvr.utils import _to_cfg, load_config, read_json, read_jsonl, write_json, write_jsonl


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    part = path.with_name(path.name + ".part")
    write_json(str(part), value)
    part.replace(path)


def atomic_rows(path, rows):
    path = Path(path)
    part = path.with_name(path.name + ".part")
    write_jsonl(str(part), rows)
    part.replace(path)


def normalize_learning_rates(config):
    for name in ("warmup_lr", "lora_lr", "head_lr"):
        if name in config:
            config[name] = float(config[name])
    return config


def settings(config_path):
    config = load_config(str(resolve(config_path)))
    config.stage_config = str(resolve(config.stage_config))
    return normalize_learning_rates(config)


@lru_cache(None)
def module_from_file(relative):
    path = resolve(relative)
    name = "fanganC_reuse_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def legacy():
    return module_from_file("scripts/13_stage2_router_experiment.py")


def snapshot_config(run_dir):
    folder = Path(run_dir) / "snapshot"
    # Snapshots are serialized as JSON, including numbers in scientific notation.
    config = normalize_learning_rates(_to_cfg(read_json(folder / "experiment.yaml")))
    return config, _to_cfg(read_json(folder / "stage.yaml"))


def frozen_rows(run_dir):
    folder = Path(run_dir) / "snapshot"
    manifest = read_json(folder / "manifest.json")
    for name, checksum in manifest["frozen_sha256"].items():
        if digest(folder / name) != checksum:
            raise ValueError(f"Frozen snapshot was changed: {name}")
    for name, checksum in manifest["reused_code_sha256"].items():
        if digest(resolve(name)) != checksum:
            raise ValueError(f"Reused implementation changed since snapshot: {name}")
    return read_jsonl(folder / "router_data.jsonl"), read_jsonl(folder / "test.jsonl"), manifest


def require_local_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "fanganC"):
        raise ValueError("Run directories must be inside fanganC/")
    return path
