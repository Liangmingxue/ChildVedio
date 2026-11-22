"""Isolated candidate-aware intervention routing experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Reuse the project's existing dependency bundles without changing the environment.
for dependency in (ROOT / "data/models/router-python-deps", ROOT / "data/msrvtt/source/python-deps"):
    if dependency.is_dir():
        sys.path.insert(0, str(dependency))
