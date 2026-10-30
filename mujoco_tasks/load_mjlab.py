"""Initialise mjlab and register the Booster K1 tasks.

mjlab is imported lazily, so importing mujoco_tasks works without it installed.
Install it with:  pip install -e ".[mjlab]"
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_config(name: str = "k1") -> dict[str, Any]:
  """Read mujoco_tasks/config/<name>.yaml."""
  import yaml

  path = CONFIG_DIR / f"{name}.yaml"
  if not path.exists():
    raise FileNotFoundError(f"No such task config: {path}")
  with path.open() as f:
    return yaml.safe_load(f) or {}


def init_mjlab():
  """Import mjlab, with an actionable error if it is missing."""
  try:
    import mjlab
  except ImportError as e:
    raise ImportError(
      'mjlab is not installed. Run: pip install -e ".[mjlab]"'
    ) from e
  return mjlab


def load_tasks(config: str = "k1") -> list[str]:
  """Register the K1 tasks with mjlab and return their IDs."""
  init_mjlab()
  try:
    importlib.import_module("mujoco_tasks.tasks")  # registers on import
  except ValueError as e:
    if "already registered" in str(e):
      raise RuntimeError(
        "mjlab already has K1 tasks registered (an in-tree copy in your "
        "mjlab checkout?). Use an mjlab without them."
      ) from e
    raise

  from mjlab.tasks.registry import list_tasks

  registered = list_tasks()
  missing = [t for t in load_config(config).get("tasks", []) if t not in registered]
  if missing:
    raise RuntimeError(f"Tasks listed in config/{config}.yaml but not registered: {missing}")
  return [t for t in registered if t.endswith("Booster-K1")]


def make_env_cfg(task_id: str, play: bool = False):
  """Return a copy of a registered task's environment config."""
  load_tasks()
  from mjlab.tasks.registry import load_env_cfg

  return load_env_cfg(task_id, play=play)


def make_rl_cfg(task_id: str):
  """Return a copy of a registered task's RL runner config."""
  load_tasks()
  from mjlab.tasks.registry import load_rl_cfg

  return load_rl_cfg(task_id)
