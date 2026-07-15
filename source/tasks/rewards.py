"""Custom reward terms for quadruped locomotion.

Every term follows the Isaac Lab manager-based convention: a function (or a
``ManagerTermBase`` subclass for stateful terms) that receives the environment as its
first argument and returns an *unweighted* per-env tensor of shape ``(num_envs,)``.
The RewardManager multiplies each term by its configured weight and by the env step dt,
so weights in the env config are expressed per-second.

Sign convention: terms meant as penalties return positive magnitudes and are given
negative weights in the config. This keeps every function readable in isolation and
makes ablations trivial (set the weight to 0.0 instead of editing code).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Task rewards (positive)
# ---------------------------------------------------------------------------


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential kernel on planar (xy) linear velocity tracking error, in the base frame.

    ``exp(-||v_cmd - v||^2 / std^2)`` saturates at 1.0 when tracking is perfect and decays
    smoothly with error. An exponential kernel is preferred over ``-||err||^2`` because it
    is bounded (no reward-scale explosion early in training when errors are large) and its
    gradient is steepest near the target, which sharpens tracking once the gait is stable.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential kernel on yaw-rate tracking error, in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward long swing phases, paid out once per touchdown.

    On the step a foot makes first contact, the reward is ``(air_time - threshold)``.
    Air time below the threshold is therefore *penalized*, which discourages rapid
    paddling gaits and pushes the policy toward a trot with a proper swing phase.
    The reward is gated to zero when the commanded speed is near zero so the robot
    is not encouraged to step in place while standing.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant survival reward.

    A small positive constant per step gives the policy a reason to avoid early
    termination even before it learns to track velocity. Kept small so it never
    dominates task rewards (a policy that just stands still should not be optimal).
    """
    return (~env.termination_manager.terminated).float()


# ---------------------------------------------------------------------------
# Regularization penalties (positive magnitude, use negative weights)
# ---------------------------------------------------------------------------


def flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-upright base orientation.

    Uses the xy components of gravity projected into the base frame: zero when the
    base z-axis is aligned with gravity, growing with tilt. This is preferred over
    Euler angles because it is singularity-free and cheap.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize deviation of base height from a nominal standing height.

    On rough terrain the world-frame z of the base is meaningless, so when a height
    scanner is provided the ground reference is taken as the mean of the ray hits
    underneath the robot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        # ray hits can be +/-inf when a ray misses the mesh; clamp before averaging
        ground_z = torch.nan_to_num(
            sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0
        ).mean(dim=1)
        height = asset.data.root_pos_w[:, 2] - ground_z
    else:
        height = asset.data.root_pos_w[:, 2]
    return torch.square(height - target_height)


def energy_consumption(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize mechanical power ``sum_j |tau_j * qdot_j|``.

    This is the physically meaningful energy proxy (unlike a pure torque penalty it
    does not punish isometric loading, e.g. holding a stance on a slope). Ablating it
    yields high-torque, high-frequency gaits with poor cost of transport.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize squared joint torques (actuator wear / heat proxy, complements energy)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize squared joint accelerations (smooths motion, protects gearboxes)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize first-order action difference ``||a_t - a_{t-1}||^2``.

    Without this the policy exploits the PD controller with bang-bang position targets,
    producing twitchy motion that destroys real actuators and transfers poorly.
    """
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


class action_smoothness_2nd_order(ManagerTermBase):
    """Penalize second-order action difference ``||a_t - 2 a_{t-1} + a_{t-2}||^2``.

    The first-order penalty allows constant-rate oscillation; the second-order term
    penalizes changes in the rate itself (jerk in target space). Stateful because the
    action manager only stores one previous action, so we buffer ``a_{t-2}`` ourselves.
    The buffer is zeroed on reset, so the first two steps of an episode are slightly
    over-penalized -- negligible over a 1000-step episode.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        curr = env.action_manager.action
        prev = env.action_manager.prev_action
        penalty = torch.sum(
            torch.square(curr - 2.0 * prev + self._prev_prev_action), dim=1
        )
        self._prev_prev_action[:] = prev
        return penalty


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that exceed the soft limits (magnitude of violation)."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos - asset.data.soft_joint_pos_limits[..., 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos - asset.data.soft_joint_pos_limits[..., 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize vertical base velocity (bouncing gaits)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize base roll/pitch rates (body rocking)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


# ---------------------------------------------------------------------------
# Contact-based penalties
# ---------------------------------------------------------------------------


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar foot velocity while the foot is in contact ("skating").

    Contact is detected from the force history (max over history) so brief contact
    flicker between physics substeps is not missed. Ablating this term produces a
    policy that drags its feet -- it looks fine in simulation with perfect friction
    but fails immediately on a real, lower-friction floor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.sum(foot_vel_xy * in_contact, dim=1)


def foot_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize swing-foot height deviation from a target clearance.

    The squared height error is weighted by ``tanh(tanh_mult * ||v_foot_xy||)`` so the
    penalty only applies to feet that are actually swinging (fast planar motion) and
    fades out for stance feet. Encourages deliberate foot lifting, which matters on
    rough terrain and stairs where toe-dragging causes trips.

    Note: uses world-frame foot z, which assumes locally flat terrain under the foot.
    Good enough as a shaping term; the height scanner / camera carries the real
    terrain information.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    swing_weight = torch.tanh(
        tanh_mult * asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    )
    return torch.sum(foot_z_error * swing_weight, dim=1)


def undesired_contacts(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Count bodies (e.g. thighs/calves) whose contact force exceeds a threshold.

    Feet are the only bodies that should touch the ground; contact on the upper legs
    means the robot is scraping obstacles or collapsing.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > threshold
    )
    return torch.sum(is_contact, dim=1).float()


def stand_still_joint_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from the default pose when the command is ~zero.

    Without this, a policy asked to stand still keeps shuffling because stepping is
    never explicitly discouraged at zero command.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    )
    is_standing = (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        < command_threshold
    )
    return deviation * is_standing


def termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-time penalty on non-timeout termination (falling / illegal contact)."""
    return env.termination_manager.terminated.float()
