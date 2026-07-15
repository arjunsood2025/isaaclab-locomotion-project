"""Gym registration for the Go2 locomotion tasks.

Importing this package registers all tasks; the training/eval scripts import it once
and then use ``gym.make(task_id, cfg=...)``.
"""

import gymnasium as gym

from .locomotion_env import (
    Go2FlatEnvCfg,
    Go2FlatPlayEnvCfg,
    Go2RoughDREnvCfg,
    Go2RoughEnvCfg,
    Go2RoughPlayEnvCfg,
)
from .vision_env import Go2VisionEnvCfg, Go2VisionPlayEnvCfg

# task id -> env cfg class
TASK_CFGS = {
    "Go2-Flat-v0": Go2FlatEnvCfg,
    "Go2-Flat-Play-v0": Go2FlatPlayEnvCfg,
    "Go2-Rough-v0": Go2RoughEnvCfg,
    "Go2-Rough-Play-v0": Go2RoughPlayEnvCfg,
    "Go2-Rough-DR-v0": Go2RoughDREnvCfg,
    "Go2-Vision-v0": Go2VisionEnvCfg,
    "Go2-Vision-Play-v0": Go2VisionPlayEnvCfg,
}

for task_id, cfg_class in TASK_CFGS.items():
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": cfg_class},
    )


def get_env_cfg(task_id: str):
    """Instantiate the env config for a task id (raises with the valid ids on typo)."""
    if task_id not in TASK_CFGS:
        raise KeyError(
            f"Unknown task '{task_id}'. Available tasks: {sorted(TASK_CFGS)}"
        )
    return TASK_CFGS[task_id]()
