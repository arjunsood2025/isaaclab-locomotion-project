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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
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

# image geometry shared with the CNN encoder config in configs/ppo_vision.yaml;
# change in one place only
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
    """Actor: "policy" (48 proprio dims) + "depth" (1x64x64 image). Critic: unchanged
    (privileged clean state + foot contacts + true height scan).

    The image is a *separate group*, not extra columns on the policy vector. rsl_rl
    dispatches observation groups by tensor rank, so the rank-4 "depth" group is routed
    through a CNN encoder while "policy" goes straight to the MLP; the two latents are
    concatenated inside the model. configs/ppo_vision.yaml wires this up with
    ``obs_groups: {actor: [policy, depth], critic: [critic]}``.
    """

    @configclass
    class VisionPolicyCfg(ObservationsCfg.PolicyCfg):
        # the actor loses the privileged height scan -- it must infer terrain from pixels
        height_scan = None

    @configclass
    class DepthCfg(ObsGroup):
        """Single-term image group: (N, 1, 64, 64) normalized inverse depth."""

        depth_image = ObsTerm(
            func=local_obs.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "max_depth": DEPTH_MAX_RANGE_M,
            },
            noise=Gnoise(mean=0.0, std=0.02),  # depth sensor speckle
        )

        def __post_init__(self):
            self.enable_corruption = True
            # single term: concatenation is a no-op, but it must not flatten the image
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObservationsCfg.PolicyCfg):
        """Privileged input for the distillation teacher.

        Inherits the base policy group *including* height_scan, so its layout is
        byte-identical to what the blind DR policy was trained on
        (48 proprio + 187 height scan = 235). That identity is not optional: the
        teacher checkpoint's first Linear layer is 235-wide, and rsl_rl loads it with
        ``strict=True``.
        """

    policy: VisionPolicyCfg = VisionPolicyCfg()
    depth: DepthCfg = DepthCfg()
    teacher: TeacherCfg = TeacherCfg()


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
        self.observations.depth.enable_corruption = False  # no sensor speckle at eval
        self.events.push_robot = None


@configclass
class Go2VisionDistillEnvCfg(Go2VisionEnvCfg):
    """Depth-vision locomotion trained by distillation from the blind DR policy.

    Kept separate from ``Go2VisionEnvCfg`` on purpose: the end-to-end RL task remains
    reproducible, because "we tried end-to-end and it collapsed into a stand-and-turn
    local optimum" is a result worth being able to re-run, not something to overwrite.

    Why distillation at all: learning to walk *and* to read terrain from pixels at the
    same time, under full domain randomization, is a hard exploration problem -- the
    end-to-end run maximized the survival bonus by standing still and never travelled
    far enough for the terrain curriculum to promote it, so the camera only ever saw
    flat ground and the encoder collapsed. Distillation removes the exploration problem
    entirely: the teacher supplies the correct action at every state, and the student
    only has to learn the perception mapping. This is the approach used by the
    egocentric-vision locomotion literature (Lee et al. 2020, Agarwal et al. 2022).
    """

    def __post_init__(self):
        super().__post_init__()
        # The teacher must see exactly what it was trained on, including the same
        # sensor-latency treatment. Go2RoughDREnvCfg applied delayed-observation
        # wrappers to the policy group; mirror them onto the teacher group. Each term
        # gets its own DelayedObservation instance (and so its own per-env delays),
        # which keeps the teacher inside its training distribution.
        import copy as _copy

        for term_name in ("joint_pos", "joint_vel", "base_ang_vel"):
            setattr(
                self.observations.teacher,
                term_name,
                _copy.deepcopy(getattr(self.observations.policy, term_name)),
            )
        # the teacher reads the privileged terrain map, so it needs the scan back
        self.observations.teacher.height_scan = _copy.deepcopy(
            ObservationsCfg.PolicyCfg().height_scan
        )


@configclass
class Go2VisionDistillPlayEnvCfg(Go2VisionDistillEnvCfg):
    """Evaluation variant of the distillation task.

    The teacher observation group is retained even though only the student is being
    scored: the distillation checkpoint stores both models, and rsl_rl rebuilds both
    when loading it, so the group has to exist for the load to succeed.
    """

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
        self.observations.depth.enable_corruption = False
        self.events.push_robot = None
