"""Procedural terrain generation and curriculum logic.

Terrain layout: the generator builds a grid of ``num_rows x num_cols`` terrain patches.
*Rows* encode difficulty (row 0 easiest, row N-1 hardest -- each sub-terrain type
interpolates its own difficulty parameter across rows) and *columns* cycle through the
terrain types. A robot's "terrain level" is simply the row its spawn origin sits on, so
promoting a robot = respawning it one row further.

Curriculum rule (game-inspired, from Rudin et al. 2022 "Learning to Walk in Minutes"):
    - promote if the robot walked more than half a terrain patch this episode
    - demote  if it covered less than half the distance its command implied
This is measured per-episode at reset time, so it costs nothing during rollout.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.terrains as terrain_gen
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Terrain generator config: a difficulty-interpolated grid of sub-terrain types
# (flat -> rough -> slopes -> boxes -> stairs -> gaps), 10 rows x 20 columns.
# ---------------------------------------------------------------------------

LOCOMOTION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),  # each terrain patch is 8m x 8m
    border_width=20.0,
    num_rows=10,  # 10 difficulty levels
    num_cols=20,  # 20 columns cycling through the sub-terrain types
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,  # steeper-than-this heightfield faces become vertical walls
    use_cache=False,
    curriculum=True,  # difficulty parameter interpolates across rows
    sub_terrains={
        # proportions sum to 1.0; flat patches are kept so the policy never forgets
        # the nominal gait while the curriculum pushes it onto harder terrain
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.02, 0.08),
            noise_step=0.02,
            border_width=0.25,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.025, 0.10),
            platform_width=2.0,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.05, 0.18),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.05, 0.18),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.05,
            gap_width_range=(0.05, 0.40),
            platform_width=2.0,
        ),
    },
)


# ---------------------------------------------------------------------------
# Curriculum terms (called by the CurriculumManager at env reset)
# ---------------------------------------------------------------------------


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Promote/demote robots across terrain difficulty rows based on distance walked.

    Returns the mean terrain level so it shows up as a training curve in TensorBoard
    (a healthy run shows a steady climb; a plateau usually means a reward term is
    fighting the curriculum).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")[env_ids]
    # distance covered this episode, measured from the assigned spawn origin
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    # promoted: crossed more than half of the 8m terrain patch
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # demoted: covered less than half of what the commanded speed implies over the
    # episode (i.e. the robot is clearly failing at this difficulty)
    move_down = (
        distance < torch.norm(command[:, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def command_vel_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    reward_term_name: str = "track_lin_vel_xy",
    max_speed: float = 1.5,
    step_size: float = 0.1,
    promotion_threshold: float = 0.8,
) -> torch.Tensor:
    """Widen the commanded forward-speed range once tracking is reliably good.

    Promotion rule: if the mean episodic tracking reward (normalized by its maximum
    possible value) exceeds ``promotion_threshold``, extend the x-velocity command
    range by ``step_size`` in both directions, capped at ``max_speed``.

    Note: reads the reward manager's episodic sums, which is a semi-private interface
    (`_episode_sums`) -- the standard pattern in Isaac Lab community tasks until a
    public accessor exists.
    """
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    # max achievable episodic sum = weight * dt * num_steps (kernel maxes out at 1.0)
    max_possible = reward_cfg.weight * env.max_episode_length_s
    normalized = episode_sums / max_possible

    cmd_term = env.command_manager.get_term("base_velocity")
    lo, hi = cmd_term.cfg.ranges.lin_vel_x
    if torch.mean(normalized) > promotion_threshold:
        lo = max(lo - step_size, -max_speed)
        hi = min(hi + step_size, max_speed)
        cmd_term.cfg.ranges.lin_vel_x = (lo, hi)
    return torch.tensor(hi, device=env.device)
