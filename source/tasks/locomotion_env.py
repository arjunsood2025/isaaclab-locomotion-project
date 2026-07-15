"""Manager-based locomotion environments for the Unitree Go2.

Variants (registered as gym tasks in ``source/tasks/__init__.py``):
    Go2FlatEnvCfg      -- flat plane, blind (proprioception only). Phase-1 baseline.
    Go2RoughEnvCfg     -- procedural rough terrain + height-scan observation + terrain
                          curriculum. Moderate randomization.
    Go2RoughDREnvCfg   -- rough terrain + full sim-to-real domain randomization stack
                          (mass, friction, PD gains, motor strength, joint params,
                          pushes, observation delay, actuator delay).
    *PlayEnvCfg        -- small evaluation variants: few envs, corruption/curriculum off.

Timing: physics at 200 Hz (dt=0.005), control at 50 Hz (decimation=4). 50 Hz matches
what a real onboard policy loop comfortably sustains and what Unitree's low-level PD
interface expects for target updates.

Action space: 12 joint position *offsets* around the default standing pose, scaled by
0.25 rad, tracked by a PD controller inside the actuator model. Position targets + PD
is far more stable to train than raw torques (the PD loop provides local feedback at
full physics rate between policy steps) and matches real quadruped control stacks.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import events as local_events
from . import observations as local_obs
from . import rewards as local_rewards
from . import terrain_curriculum as local_terrain


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class LocomotionSceneCfg(InteractiveSceneCfg):
    """Terrain + robot + sensors."""

    # procedural terrain grid; rows = difficulty, columns = terrain type
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=local_terrain.LOCOMOTION_TERRAINS_CFG,
        max_init_terrain_level=2,  # start the curriculum on easy rows
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Unitree Go2: 12 DoF, DC-motor actuator model with datasheet limits
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 1.6m x 1.0m grid of downward rays around the base, 0.1m resolution -> 187 values.
    # Rays originate 20m above the base and only follow base yaw (attach_yaw_only), so
    # the scan stays gravity-aligned when the body rolls/pitches -- matching how an
    # elevation map built from real sensors behaves.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # contact forces on every body; history_length=3 physics steps so brief contacts
    # between control steps are not missed; air time tracked for the gait reward
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.9, 0.9)),
    )


# ---------------------------------------------------------------------------
# MDP: commands, actions, observations
# ---------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Velocity command: (vx, vy, yaw-rate) resampled every 10 s.

    heading_command=True generates the yaw-rate from a target heading via a P-law,
    which produces smoother, more realistic turn commands than white-noise yaw rates.
    2% of envs get a ~zero command so standing is always in the training distribution.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Joint position offsets around the default pose: q_target = q_default + 0.25 * a.

    The 0.25 scale keeps a tanh-free Gaussian policy's typical output (roughly +/-2 std)
    within a safe +/-0.5 rad envelope around standing, so early random exploration
    cannot command self-destructive poses.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Actor (noisy, realistic) and critic (clean, privileged) observation groups.

    Asymmetric actor-critic: the critic is thrown away after training, so it may see
    noise-free states and privileged signals (foot contacts) that a real robot cannot
    measure. This tightens value estimates and speeds up learning without compromising
    the deployability of the actor.

    Actor layout (order matters -- deployment code reproduces it exactly):
        base_lin_vel(3) base_ang_vel(3) projected_gravity(3) velocity_commands(3)
        joint_pos(12) joint_vel(12) actions(12)                     = 48 "proprio" dims
        + height_scan(187) on rough terrain (removed in the vision variant)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # noise ranges approximate real IMU / encoder error magnitudes
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=local_obs.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged, noise-free copy of the state plus foot contacts."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        feet_contacts = ObsTerm(
            func=local_obs.feet_contact_states,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        height_scan = ObsTerm(
            func=local_obs.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Events: baseline (mild) randomization
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Baseline events: initial-state randomization + mild friction variety + pushes.

    The full sim-to-real randomization stack lives in ``DREventCfg`` so that
    baseline-vs-DR is a clean ablation.
    """

    # -- startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    # -- reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )

    # -- interval: shove the robot every 10-15 s to train push recovery
    push_robot = EventTerm(
        func=local_events.push_robot,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class DREventCfg(EventCfg):
    """Full domain-randomization stack for sim-to-real robustness (Policy D in the
    robustness study). Ranges follow common quadruped sim-to-real practice."""

    def __post_init__(self):
        # widen friction well below/above nominal: rubber feet on wet tile ~0.4,
        # on rough concrete ~1.2
        self.physics_material.params["static_friction_range"] = (0.3, 1.25)
        self.physics_material.params["dynamic_friction_range"] = (0.25, 1.0)
        # stronger pushes, more often
        self.push_robot.interval_range_s = (8.0, 12.0)
        self.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}

    # +/-20% payload/battery variance on the trunk
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    # PD gain error: real gains never match the URDF numbers exactly
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # joint friction / armature: unmodeled drivetrain losses and reflected inertia
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.05),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # motor strength: battery sag / motor variance (custom term)
    motor_strength = EventTerm(
        func=local_events.randomize_motor_strength,
        mode="reset",
        params={"strength_range": (0.85, 1.15)},
    )
    # constant unmodeled wrench on the trunk (e.g. a poorly balanced payload)
    base_external_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-1.0, 1.0),
        },
    )


# ---------------------------------------------------------------------------
# Rewards / terminations / curriculum
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Shaped locomotion reward. Weights are per-second (RewardManager multiplies by dt).

    Rough magnitude budget at convergence, per second: tracking ~ +3.0 dominates;
    each regularizer contributes ~ -0.05..-0.3 so it shapes *how* the task is done
    without competing with *whether* it is done.
    """

    # -- task
    track_lin_vel_xy = RewTerm(
        func=local_rewards.track_lin_vel_xy_exp,
        weight=2.0,
        params={"std": math.sqrt(0.25), "command_name": "base_velocity"},
    )
    track_ang_vel_z = RewTerm(
        func=local_rewards.track_ang_vel_z_exp,
        weight=1.0,
        params={"std": math.sqrt(0.25), "command_name": "base_velocity"},
    )
    alive = RewTerm(func=local_rewards.alive_bonus, weight=0.25)
    feet_air_time = RewTerm(
        func=local_rewards.feet_air_time,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    # -- base motion shaping
    lin_vel_z = RewTerm(func=local_rewards.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=local_rewards.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewTerm(func=local_rewards.flat_orientation_l2, weight=-1.0)

    # -- effort / smoothness
    joint_torques = RewTerm(func=local_rewards.joint_torques_l2, weight=-2.0e-4)
    energy = RewTerm(func=local_rewards.energy_consumption, weight=-1.0e-3)
    joint_acc = RewTerm(func=local_rewards.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=local_rewards.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(
        func=local_rewards.action_smoothness_2nd_order, weight=-0.005
    )

    # -- safety / contact
    joint_limits = RewTerm(func=local_rewards.joint_pos_limits, weight=-1.0)
    feet_slide = RewTerm(
        func=local_rewards.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    foot_clearance = RewTerm(
        func=local_rewards.foot_clearance,
        weight=-0.5,
        params={
            "target_height": 0.08,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    undesired_contacts = RewTerm(
        func=local_rewards.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_thigh", ".*_calf"]
            ),
        },
    )
    stand_still = RewTerm(
        func=local_rewards.stand_still_joint_deviation,
        weight=-0.2,
        params={"command_name": "base_velocity"},
    )
    termination = RewTerm(func=local_rewards.termination_penalty, weight=-100.0)


@configclass
class TerminationsCfg:
    """Fall detection: trunk contact = fell. time_out flagged so PPO bootstraps the
    value on truncation instead of treating it as failure."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=local_terrain.terrain_levels_vel)
    command_vel = CurrTerm(
        func=local_terrain.command_vel_curriculum,
        params={"max_speed": 1.5, "step_size": 0.1, "promotion_threshold": 0.8},
    )


# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------


@configclass
class Go2RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Rough-terrain locomotion with height-scan perception and terrain curriculum."""

    scene: LocomotionSceneCfg = LocomotionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 200 Hz physics, 50 Hz control
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # terrain and robot share the ground material settings
        self.sim.physics_material = self.scene.terrain.physics_material
        # sensors only need to refresh when they are consumed
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go2RoughDREnvCfg(Go2RoughEnvCfg):
    """Rough terrain + full domain-randomization stack + latency modeling."""

    events: DREventCfg = DREventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Actuation latency: replace the ideal DC-motor model with a delayed PD
        # actuator. 0-4 physics steps = 0-20 ms command-to-torque delay, resampled
        # per env, covering policy inference + bus transport on the real robot.
        self.scene.robot.actuators["base_legs"] = DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            min_delay=0,
            max_delay=4,
        )
        # Observation latency: proprioceptive channels arrive 0-2 control steps
        # (0-40 ms) late, per env. Wraps the same underlying obs functions.
        self.observations.policy.joint_pos = ObsTerm(
            func=local_obs.DelayedObservation,
            params={"func": mdp.joint_pos_rel, "max_delay": 2},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=local_obs.DelayedObservation,
            params={"func": mdp.joint_vel_rel, "max_delay": 2},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        self.observations.policy.base_ang_vel = ObsTerm(
            func=local_obs.DelayedObservation,
            params={"func": mdp.base_ang_vel, "max_delay": 2},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )


@configclass
class Go2FlatEnvCfg(Go2RoughEnvCfg):
    """Flat-ground blind baseline (Phase 1): no terrain, no exteroception."""

    def __post_init__(self):
        super().__post_init__()
        # flat plane instead of the generator
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no exteroception needed
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain rows to climb
        self.curriculum.terrain_levels = None
        # flat ground: keep the body at nominal height
        self.rewards.foot_clearance.weight = -0.25


# ---------------------------------------------------------------------------
# Play / evaluation variants: deterministic-ish, small, no curriculum
# ---------------------------------------------------------------------------


@configclass
class Go2RoughPlayEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        # smaller terrain grid spanning all difficulties, robots spread over all rows
        self.scene.terrain.terrain_generator = (
            local_terrain.LOCOMOTION_TERRAINS_CFG.replace(num_rows=5, num_cols=8)
        )
        self.scene.terrain.max_init_terrain_level = None
        # evaluation must not move the goalposts
        self.curriculum.terrain_levels = None
        self.curriculum.command_vel = None
        self.observations.policy.enable_corruption = False
        # pushes are injected explicitly by the evaluation harness, not randomly
        self.events.push_robot = None


@configclass
class Go2FlatPlayEnvCfg(Go2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.curriculum.command_vel = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
