"""Shared helpers for the train / evaluate / export scripts (no Isaac Sim imports --
this module must be importable before the simulation app is launched)."""

from __future__ import annotations

import glob
import os

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_agent_cfg(path: str) -> dict:
    """Load an rsl_rl agent config YAML (paths may be relative to the repo root)."""
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_by_dotted_path(obj: object, dotted_path: str, value) -> None:
    """Set a nested attribute, e.g. ``scene.num_envs`` or ``rewards.energy.weight``.

    Fails loudly on a bad path so a typo in an override YAML cannot silently train
    with default settings.
    """
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"Env cfg has no attribute '{dotted_path}'")
    setattr(obj, parts[-1], value)


def apply_env_overrides(env_cfg: object, overrides_path: str | None) -> None:
    """Apply a flat ``dotted.path: value`` YAML onto an env config instance."""
    if overrides_path is None:
        return
    if not os.path.isabs(overrides_path):
        overrides_path = os.path.join(REPO_ROOT, overrides_path)
    with open(overrides_path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}
    for dotted_path, value in overrides.items():
        set_by_dotted_path(env_cfg, dotted_path, value)


def resolve_checkpoint(
    log_root: str, load_run: str | None = None, checkpoint: str | None = None
) -> str:
    """Resolve a model checkpoint path.

    - ``checkpoint`` given: used directly (absolute or repo-relative).
    - else: pick ``load_run`` (or the lexicographically latest run dir, which is the
      newest since run names start with a timestamp) and its highest-numbered
      ``model_*.pt``.
    """
    if checkpoint is not None:
        return checkpoint if os.path.isabs(checkpoint) else os.path.join(REPO_ROOT, checkpoint)
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"No experiment logs at {log_root}")
    if load_run is None:
        runs = sorted(
            d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))
        )
        if not runs:
            raise FileNotFoundError(f"No runs inside {log_root}")
        load_run = runs[-1]
    run_dir = os.path.join(log_root, load_run)
    models = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not models:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {run_dir}")
    models.sort(key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    return models[-1]
