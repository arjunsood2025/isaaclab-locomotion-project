"""Custom domain-randomization event terms.

Most randomizations use Isaac Lab's built-in ``mdp`` events directly in the env config
(mass, friction, PD gains, joint friction/armature, pushes, external wrenches). This
module adds the ones that have no built-in equivalent.

Event modes (assigned in the config):
    startup  -- once per simulation start (e.g. per-env friction materials)
    reset    -- on every episode reset (e.g. motor strength, initial state)
    interval -- every N seconds during rollout (e.g. pushes)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_motor_strength(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    strength_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Scale each actuator's effort limit by a per-env, per-joint random factor.

    Models motor-to-motor manufacturing variance and battery-voltage sag: the same
    torque command produces less torque on a weak motor. Complements PD-gain
    randomization (which changes the response shape, not the ceiling).

    The nominal limits are cached on first call so repeated randomization scales from
    the true datasheet values instead of compounding.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    for actuator in asset.actuators.values():
        if not hasattr(actuator, "_nominal_effort_limit"):
            actuator._nominal_effort_limit = actuator.effort_limit.clone()
        scale = sample_uniform(
            strength_range[0],
            strength_range[1],
            (len(env_ids), actuator.effort_limit.shape[1]),
            device=asset.device,
        )
        actuator.effort_limit[env_ids] = (
            actuator._nominal_effort_limit[env_ids] * scale
        )


def push_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Perturb the base velocity by a random delta (an impulsive shove).

    Setting velocity directly (rather than applying a force over time) approximates an
    instantaneous impact -- the hardest perturbation class -- and is the standard
    push-recovery training signal for legged robots. Ranges are dict keys
    ``x, y, z, roll, pitch, yaw``; missing keys default to (0, 0).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.root_vel_w[env_ids].clone()
    ranges = [velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
    lo = torch.tensor([r[0] for r in ranges], device=asset.device)
    hi = torch.tensor([r[1] for r in ranges], device=asset.device)
    delta = sample_uniform(lo, hi, (len(env_ids), 6), device=asset.device)
    asset.write_root_velocity_to_sim(vel + delta, env_ids=env_ids)
