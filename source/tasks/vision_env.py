"""Vision-based locomotion: the robot infers terrain from a forward depth camera.

The actor's height scan is replaced by a 64x64 depth image from a head-mounted camera;
the critic keeps the (privileged) height scan. So the *value function* still knows the
true local terrain while the *policy* must learn to extract it from pixels -- a standard
asymmetric setup that stabilizes training massively compared to giving the critic
pixels too.

Why depth instead of RGB: depth is a direct geometric measurement, so there is no
appearance/texture/lighting domain gap to randomize away, and a small CNN suffices.
RGB would demand heavy visual domain randomization and a much larger encoder for
strictly less task-relevant information about where to step.

Why 64x64: foot placement needs coarse geometry (step edges, gap boundaries), not
texture detail. At 1024 envs a 64x64 tiled depth render keeps the vision pipeline from
dominating the simulation step budget.

Requires running with ``--enable_cameras``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from . import observations as local_obs
from .locomotion_env import (
    Go2RoughDREnvCfg,
    LocomotionSceneCfg,
    ObservationsCfg,
)

# image geometry shared with the policy config (configs/ppo_vision.yaml) and the
# vision actor-critic; change in one place only
DEPTH_IMAGE_SHAPE = (1, 64, 64)
DEPTH_MAX_RANGE_M = 5.0


@configclass
class VisionSceneCfg(LocomotionSceneCfg):
    """Adds a head-mounted, forward-facing depth camera (Go2 has one in the same spot).

    TiledCamera renders all envs' cameras into one big texture in a single pass --
    the only way camera-in-the-loop RL stays fast at ~1k envs.
    """

    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.02,  # 50 Hz, matching the control rate
        height=DEPTH_IMAGE_SHAPE[1],
        width=DEPTH_IMAGE_SHAPE[2],
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        # nose of the trunk, pitched down 20 deg (quaternion about +Y, w-x-y-z) so the
        # ground 0.5-3 m ahead -- where the next footsteps land -- fills the frame
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.32, 0.0, 0.04), rot=(0.9848, 0.0, 0.1736, 0.0), convention="world"
        ),
    )


@configclass
class VisionObservationsCfg(ObservationsCfg):
    """Actor: proprioception (48) + flattened depth (4096). Critic: unchanged
    (privileged height scan + contacts)."""

    @configclass
    class VisionPolicyCfg(ObservationsCfg.PolicyCfg):
        # remove the privileged height scan from the actor...
        height_scan = None
        # ...and append the depth image (declared last => concatenated last, so the
        # network can split proprio/image by index)
        depth_image = ObsTerm(
            func=local_obs.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "max_depth": DEPTH_MAX_RANGE_M,
            },
            noise=Gnoise(mean=0.0, std=0.02),  # depth sensor speckle
        )

    policy: VisionPolicyCfg = VisionPolicyCfg()


@configclass
class Go2VisionEnvCfg(Go2RoughDREnvCfg):
    """Extreme version: depth-vision locomotion with the full DR stack underneath."""

    scene: VisionSceneCfg = VisionSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # rendering must run every control step for the camera (not just for the GUI)
        self.sim.render_interval = self.decimation
        # camera-in-the-loop is memory-heavy; 1024 envs fits a 24 GB GPU comfortably
        self.scene.num_envs = 1024


@configclass
class Go2VisionPlayEnvCfg(Go2VisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator = (
            self.scene.terrain.terrain_generator.replace(num_rows=5, num_cols=8)
        )
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_vel = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
