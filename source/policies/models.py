"""Custom network modules for rsl_rl: depth encoder + vision actor-critic.

Integration with rsl_rl: the ObservationManager hands the wrapper one flat vector per
env (proprio dims first, flattened depth image last -- the declaration order in
``VisionObservationsCfg``). ``ActorCriticVision`` splits that vector, runs the image
through a small CNN, and feeds ``[proprio, embedding]`` to the standard MLP actor.
Keeping the flat-vector interface means *zero changes to rsl_rl's PPO/rollout code* --
only the policy class differs, selected via ``class_name`` in the agent YAML.

The critic path is untouched: it receives the privileged critic observation group
(clean proprio + foot contacts + height scan), i.e. asymmetric actor-critic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic


class DepthImageEncoder(nn.Module):
    """Small strided CNN: (1, 64, 64) normalized inverse-depth -> embed_dim vector.

    Four stride-2 convs take 64 -> 4 spatial resolution; ~0.4M params. ELU matches the
    MLP trunk; the final LayerNorm keeps the embedding at a stable scale next to the
    proprioceptive features (we cannot use rsl_rl's running obs normalization on raw
    pixels -- 4096 running-stat channels on images is wasteful and slow to converge).
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (1, 64, 64),
        embed_dim: int = 128,
        channels: tuple[int, ...] = (16, 32, 64, 64),
    ):
        super().__init__()
        self.image_shape = tuple(image_shape)
        in_ch = image_shape[0]
        conv_layers: list[nn.Module] = []
        for out_ch in channels:
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        # infer the flattened conv output size once, with a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            conv_out = self.conv(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ELU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(image).flatten(1))


class ActorCriticVision(ActorCritic):
    """rsl_rl-compatible actor-critic whose actor consumes proprio + depth image.

    Constructor signature matches what OnPolicyRunner passes (obs dims + the ``policy``
    section of the agent YAML as kwargs). We tell the parent the actor input is
    ``prop_dim + embed_dim`` (post-encoding), then intercept every actor call to encode
    first. The critic is built by the parent on the raw privileged obs.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        prop_dim: int = 48,
        image_shape: tuple[int, int, int] = (1, 64, 64),
        embed_dim: int = 128,
        encoder_channels: tuple[int, ...] = (16, 32, 64, 64),
        **kwargs,
    ):
        image_numel = int(math.prod(image_shape))
        if num_actor_obs != prop_dim + image_numel:
            raise ValueError(
                f"Actor obs dim mismatch: env provides {num_actor_obs} dims but "
                f"prop_dim ({prop_dim}) + image ({image_numel}) = {prop_dim + image_numel}. "
                "Check the observation layout in VisionObservationsCfg vs ppo_vision.yaml."
            )
        super().__init__(
            num_actor_obs=prop_dim + embed_dim,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            **kwargs,
        )
        self.prop_dim = prop_dim
        self.image_shape = tuple(image_shape)
        self.encoder = DepthImageEncoder(self.image_shape, embed_dim, encoder_channels)

    def _split_and_encode(self, observations: torch.Tensor) -> torch.Tensor:
        """[..., prop+H*W] -> [..., prop+embed]. Handles both rollout batches (N, obs)
        and PPO minibatches (flattened over envs*steps)."""
        prop = observations[..., : self.prop_dim]
        image_flat = observations[..., self.prop_dim :]
        batch_shape = image_flat.shape[:-1]
        image = image_flat.reshape(-1, *self.image_shape)
        embedding = self.encoder(image).reshape(*batch_shape, -1)
        return torch.cat([prop, embedding], dim=-1)

    # -- intercept the actor paths; critic path (evaluate) is inherited unchanged
    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().act(self._split_and_encode(observations), **kwargs)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return super().act_inference(self._split_and_encode(observations))


def register_custom_modules() -> None:
    """Make ``class_name: ActorCriticVision`` resolvable by rsl_rl.

    OnPolicyRunner resolves the policy class by evaluating ``class_name`` in its own
    module namespace, so we inject our class into both ``rsl_rl.modules`` and the
    runner module. Call this before constructing the runner.
    """
    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as rsl_runner_module

    rsl_modules.ActorCriticVision = ActorCriticVision
    rsl_runner_module.ActorCriticVision = ActorCriticVision
