"""Custom observation terms: terrain height scan, depth camera, and delayed observations.

Vector terms return ``(num_envs, obs_dim)``; the depth camera term returns a
``(num_envs, 1, H, W)`` image. Isaac Lab's ObservationManager concatenates terms within a
group in the order they are declared in the config, which fixes the observation layout
the deployment code relies on -- see ``source/deployment/``. Image observations live in
their own single-term group so nothing is concatenated onto them.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any, Callable

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    clip_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Terrain height map relative to the robot base, from a grid ray-caster.

    Returns ``base_z - hit_z - offset`` per ray: ~0 on flat ground at nominal standing
    height, positive over holes/drops, negative over obstacles. The offset centers the
    signal at the nominal base height so the network sees a zero-mean input. Values are
    clipped so rare deep pits do not produce out-of-distribution spikes.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    heights = torch.nan_to_num(
        heights, nan=clip_range[1], posinf=clip_range[1], neginf=clip_range[0]
    )
    return heights.clip(*clip_range)


def depth_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_depth: float = 5.0,
) -> torch.Tensor:
    """Normalized inverse depth image from a tiled (batched) camera, as ``(N, 1, H, W)``.

    Raw ``distance_to_image_plane`` is clamped to ``[0, max_depth]`` (sky / misses come
    back as inf) and mapped to ``1 - d / max_depth`` so *near = 1, far = 0*. Inverse
    depth concentrates resolution on nearby terrain -- the part that determines the next
    footstep -- and bounds the input to [0, 1] for the CNN.

    The channel-first 4-D layout is deliberate: rsl_rl dispatches observation groups by
    tensor rank, so a rank-4 group is routed to a CNN encoder while rank-2 groups go
    straight to the MLP. Flattening here would silently turn the depth image into 4096
    unstructured MLP inputs.
    """
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    img = sensor.data.output["distance_to_image_plane"]
    # camera returns (N, H, W, 1) -> (N, H, W)
    if img.dim() == 4:
        img = img.squeeze(-1)
    img = torch.nan_to_num(img, nan=max_depth, posinf=max_depth, neginf=0.0)
    img = img.clamp(0.0, max_depth)
    img = 1.0 - img / max_depth
    # (N, H, W) -> (N, 1, H, W)
    return img.unsqueeze(1)


def feet_contact_states(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Binary contact state per foot. Used only in the privileged critic observation --
    a real robot's foot contact estimate is unreliable, so the actor never sees it."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    return (forces > threshold).float()


class DelayedObservation(ManagerTermBase):
    """Wraps another observation function with a per-env random delay of 0..max_delay
    control steps, simulating sensor-pipeline latency for sim-to-real transfer.

    Each env draws its own delay at reset, so a single policy must be robust to the
    whole latency range rather than adapting to one fixed lag. Internally keeps a ring
    buffer of the last ``max_delay + 1`` values of the wrapped term; on reset the
    buffer rows for the affected envs are refilled with the current value so stale
    pre-reset data never leaks into a new episode.

    Config usage::

        joint_pos = ObsTerm(
            func=local_obs.DelayedObservation,
            params={"func": mdp.joint_pos_rel, "max_delay": 2, "func_kwargs": {}},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._max_delay: int = cfg.params["max_delay"]
        self._delays = torch.randint(
            0, self._max_delay + 1, (env.num_envs,), device=env.device
        )
        self._buffer: torch.Tensor | None = None
        self._needs_refill = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._env_ids = torch.arange(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._env_ids
        self._delays[env_ids] = torch.randint(
            0, self._max_delay + 1, (len(env_ids),), device=self._env.device
        )
        self._needs_refill[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        func: Callable,
        max_delay: int,
        func_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        value = func(env, **(func_kwargs or {}))
        if self._buffer is None:
            self._buffer = value.unsqueeze(1).repeat(1, self._max_delay + 1, 1)
        # newest value at index 0
        self._buffer = torch.roll(self._buffer, shifts=1, dims=1)
        self._buffer[:, 0] = value
        # freshly reset envs: fill the whole history with the current value
        if self._needs_refill.any():
            refill_ids = self._needs_refill.nonzero(as_tuple=False).squeeze(-1)
            self._buffer[refill_ids] = value[refill_ids].unsqueeze(1)
            self._needs_refill[refill_ids] = False
        return self._buffer[self._env_ids, self._delays]
