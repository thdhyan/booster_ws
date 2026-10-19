"""Stage-0 lower-body base: one leg policy tracking (vx, vy, wz, H*) under random upper body.

PLAN_HRL_CONTROLLER.md WP2 (HOMIE / FALCON pattern). Built from the squat teacher env
(``K1VelocitySquatEnvCfg``: rough terrain, H* in the command, teacher obs group) plus:

1. Arms AND head (10 joints) are not in the action space. Each reset puts them at a
   curriculum-scaled random pose, and every 2-5 s the hold target moves by a random delta
   (partial env's events, widened to the head). Both obs groups observe their pos/vel.
2. Random stub-tip wrenches on both hand links, ramped 0 -> +-20 N (box-push reaction).
   The teacher observes the applied wrench; the policy group does not.
3. H* is sampled for half the envs from the start and its lower bound ramps 0.50 -> 0.40
   by TIME (lower_body_mdp.squat_height_ramp), replacing the height-gated curriculum that
   left the squat teacher standing.
4. base_height weight 2 -> 5 so ignoring H* costs more than the tracking terms gain.

Task ids: ``Isaac-LowerBody-K1-v0`` (train), ``Isaac-LowerBody-K1-Play-v0`` (video).
Symmetry augmentation is not wired yet (needs the WP0 mirror map).
"""
from __future__ import annotations

import os

import isaaclab.terrains as terrain_gen
from isaaclab.envs.mdp import apply_external_force_torque
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from k1_velocity.shove_mdp import applied_shove_wrench
from k1_velocity.tasks.partial import mdp as partial_mdp

from . import lower_body_mdp, squat_mdp
from . import agile_rewards as agile
from . import gait_rewards as gait
from . import reward_weight_ramp
from .squat_env_cfg import K1VelocitySquatEnvCfg, K1VelocitySquatPlayEnvCfg, SquatRewardsCfg, SquatTerminationsCfg
from .velocity_env_distill import STUDENT_HISTORY_LEN
from .velocity_env_cfg import K1RoughSceneCfg, K1_ARM_HEAD_JOINTS, K1_LEG_JOINTS, CurriculumCfg as VelocityCurriculumCfg, EventCfg as VelocityEventCfg, ObservationsCfg as VelocityObsCfg

try:  # Isaac Lab 3.0-EA layout (dl); isaac-lab image renamed this package
    import isaaclab_tasks.core.velocity.mdp as mdp
except (ImportError, ModuleNotFoundError):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

K1_STUB_BODIES = ["left_hand_link", "right_hand_link"]
_UPPER = SceneEntityCfg("robot", joint_names=K1_ARM_HEAD_JOINTS)


@configclass
class LowerBodyObservationsCfg(VelocityObsCfg):
    @configclass
    class PolicyCfg(VelocityObsCfg.PolicyCfg):
        upper_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _UPPER},
                                  noise=Unoise(n_min=-0.01, n_max=0.01))
        upper_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _UPPER},
                                  noise=Unoise(n_min=-1.5, n_max=1.5))

    @configclass
    class TeacherCfg(VelocityObsCfg.TeacherCfg):
        upper_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _UPPER})
        upper_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _UPPER})
        stub_wrench = ObsTerm(func=applied_shove_wrench,
                              params={"asset_cfg": SceneEntityCfg("robot", body_names=K1_STUB_BODIES)})

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class LowerBodyRewardsCfg(SquatRewardsCfg):
    # arms/head are randomized on purpose; a default-pose penalty would fight that
    joint_deviation_arms = None
    base_height = RewTerm(
        func=squat_mdp.base_height_cmd_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.05, "sensor_name": "height_scanner"},
    )
    # Teacher v3 (2026-10-12): L1 cadence regulariser - the Huber term is flat within
    # +-0.5 steps/s of 2.0 Hz so near the goal cadence it gives no gradient, and the
    # it-5750 gait gate still measured 5.19 steps/s. L1 prices every step/s of error
    # the same. Ramped alongside the other gait terms (see LowerBodyCurriculumCfg).
    gait_cadence_l1 = RewTerm(
        func=gait.gait_cadence_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS),
            "target_hz": 2.0,
            "dt": 0.02,
        },
    )
    # Teacher v3: yaw tracking 5 -> 8. error_vel_yaw sat at ~0.84 rad/s from
    # iteration 300 to 6100 of the v2 run (gate was < 0.3 by it 1500), while the
    # xy tracker reached 0.10. Weight here only - velocity/carry/kick untouched.
    track_ang_vel_z_exp = RewTerm(
        func=agile.track_ang_vel_z_world_exp_weighted,
        weight=8.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )


@configclass
class LowerBodyEventCfg(VelocityEventCfg):
    # declared after the generic joint resets so the sampled pose survives them
    upper_pose_random = EventTerm(
        func=partial_mdp.randomize_arm_pose,
        mode="reset",
        params={"asset_cfg": _UPPER, "offset_range": (-1.0, 1.0), "curriculum_scale": 0.15},
    )
    upper_pose_delta = EventTerm(
        func=partial_mdp.randomize_arm_pose_delta,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={"asset_cfg": _UPPER, "offset_range": (-1.0, 1.0), "curriculum_scale": 0.0},
    )
    stub_force = EventTerm(
        func=apply_external_force_torque,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=K1_STUB_BODIES),
            "force_range": (0.0, 0.0),   # set by the stub_force curriculum term
            "torque_range": (0.0, 0.0),
        },
    )


_GAIT_RAMPS = (
    "action_rate_regularization", "action_jerk_regularization", "gait_cadence_regularization",
    "phase_swing_regularization", "feet_clearance_regularization", "feet_alternation_regularization",
    "stride_length_regularization",
)


@configclass
class LowerBodyCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel, params={"asset_cfg": SceneEntityCfg("robot")})
    squat_height = CurrTerm(func=lower_body_mdp.squat_height_ramp)
    upper_pose = CurrTerm(func=partial_mdp.arm_pose_curriculum, params={
        "event_term": "upper_pose_random", "start_scale": 0.15, "end_scale": 1.0,
        "ramp_start_iter": 300, "ramp_end_iter": 2000, "steps_per_iter": 24})
    upper_delta = CurrTerm(func=partial_mdp.arm_pose_curriculum, params={
        "event_term": "upper_pose_delta", "start_scale": 0.0, "end_scale": 1.0,
        "ramp_start_iter": 600, "ramp_end_iter": 2500, "steps_per_iter": 24})
    stub_force = CurrTerm(func=lower_body_mdp.force_range_ramp)
    # Teacher v3 (2026-10-12): L1 cadence term, same upright-gated ramp as the rest.
    gait_cadence_l1_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={
            "reward_name": "gait_cadence_l1",
            "start_weight": -0.1,
            "terminal_weight": -0.6,
            "start_step": 20_000,
            "num_steps": 100_000,
            "upright_threshold": 0.10,
        },
    )


@configclass
class K1LowerBodyEnvCfg(K1VelocitySquatEnvCfg):
    observations: LowerBodyObservationsCfg = LowerBodyObservationsCfg()
    rewards: LowerBodyRewardsCfg = LowerBodyRewardsCfg()
    events: LowerBodyEventCfg = LowerBodyEventCfg()
    curriculum: LowerBodyCurriculumCfg = LowerBodyCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The velocity env's upright-gated weight ramps (smoothness + 5 gait terms) take the
        # stepping penalties to full strength (cadence -0.2 -> -1, clearance -1.6 -> -8, ...).
        # Replacing the curriculum class dropped them, leaving the gait terms at 1/5 strength
        # for the whole run (shuffle risk). Copy them back.
        base = VelocityCurriculumCfg()
        for name in _GAIT_RAMPS:
            setattr(self.curriculum, name, getattr(base, name))
        # Teacher v3 (2026-10-12): clearance is the priority gait term per user. The
        # measured it-5750 gait gate read +4 mm of lift against the 60 mm target; the
        # linear hinge only charges 0.056/step at the -8.0 terminal. -12 makes it the
        # dominant gait term (1.5x the 10-01 audit value, still under action_rate).
        self.curriculum.feet_clearance_regularization.params["terminal_weight"] = -12.0


@configclass
class K1LowerBodyPlayEnvCfg(K1VelocitySquatPlayEnvCfg):
    """Squat demo plus walking, with full-range random arm/head poses and stub forces."""

    observations: LowerBodyObservationsCfg = LowerBodyObservationsCfg()
    rewards: LowerBodyRewardsCfg = LowerBodyRewardsCfg()
    events: LowerBodyEventCfg = LowerBodyEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.5)
        self.commands.base_velocity.squat_height_range = (0.40, 0.50)
        self.events.upper_pose_random.params["curriculum_scale"] = 1.0
        self.events.upper_pose_delta.params["curriculum_scale"] = 1.0
        self.events.stub_force.params["force_range"] = (-20.0, 20.0)


@configclass
class K1LowerBodyDistillEnvCfg(K1LowerBodyEnvCfg):
    """Lower-body teacher -> deployable student: policy group stacked 10 steps (71 x 10 = 710).

    Teacher group (privileged: scan, clean proprio, upper joints, stub wrench) is unchanged,
    so the frozen teacher sees exactly the obs it was trained on.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = STUDENT_HISTORY_LEN


@configclass
class K1LowerBodyDistillPlayEnvCfg(K1LowerBodyPlayEnvCfg):
    """Render the lower-body STUDENT (same 710-dim stacked input, no corruption)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = STUDENT_HISTORY_LEN
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Teacher v4 (PLAN_HRL_CONTROLLER.md 6a): fresh start, survival-gated ramps, stairs/slopes,
# one PBT member per K1_TEACHER_PRESET (0-3).
# ---------------------------------------------------------------------------
# clearance / cadence_l1 / stride = terminal ramp weights, yaw = (weight, std), seed
TEACHER_PRESETS = {
    0: {"clearance": -12.0, "yaw": (8.0, 0.25), "cadence_l1": -0.6, "stride": -6.0, "seed": 42},
    1: {"clearance": -16.0, "yaw": (10.0, 0.5), "cadence_l1": -1.0, "stride": -6.0, "seed": 43},
    2: {"clearance": -8.0, "yaw": (6.5, 0.5), "cadence_l1": -0.6, "stride": -4.0, "seed": 44},
    3: {"clearance": -12.0, "yaw": (8.0, 0.25), "cadence_l1": -0.6, "stride": -6.0, "seed": 45},
}


def teacher_preset() -> tuple[int, dict]:
    n = int(os.environ.get("K1_TEACHER_PRESET", "0"))
    return n, TEACHER_PRESETS[n]


# rows = difficulty (step height / slope / amplitude interpolate low -> high over the rows)
V4_SUB_TERRAINS = {
    "flat": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.20, noise_range=(0.0, 0.01), noise_step=0.005, border_width=0.25),
    "rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.20, noise_range=(-0.04, 0.04), noise_step=0.01, border_width=0.25),
    "stairs_down": terrain_gen.HfPyramidStairsTerrainCfg(
        proportion=0.20, step_height_range=(0.04, 0.12), step_width=0.3, platform_width=2.0, border_width=0.25),
    "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.20, slope_range=(0.0, 0.36), platform_width=2.0, border_width=0.25),  # 0-20 deg
    "stairs_up": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
        proportion=0.10, step_height_range=(0.04, 0.10), step_width=0.3, platform_width=2.0, border_width=0.25),
    "wave": terrain_gen.HfWaveTerrainCfg(
        proportion=0.10, amplitude_range=(0.01, 0.04), num_waves=4, border_width=0.25),
}

SURVIVAL_GATE = 0.5  # ramps advance only while >= 50 % time out (0.6: 2026-10-16 walk 8k never passed it)


@configclass
class LowerBodyV4CurriculumCfg:
    # survival first: the gated ramps below read the EMA it updates (terms run in field order)
    survival = CurrTerm(func=lower_body_mdp.survival_gate)
    gated_iter = CurrTerm(func=lower_body_mdp.gated_iteration, params={"gate": SURVIVAL_GATE})
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel, params={"asset_cfg": SceneEntityCfg("robot")})
    squat_height = CurrTerm(func=lower_body_mdp.squat_height_ramp, params={"gate": SURVIVAL_GATE})
    upper_pose = CurrTerm(func=lower_body_mdp.upper_pose_ramp, params={
        "event_term": "upper_pose_random", "start_scale": 0.15, "end_scale": 1.0,
        "ramp_start_iter": 300, "ramp_end_iter": 2000, "gate": SURVIVAL_GATE})
    upper_delta = CurrTerm(func=lower_body_mdp.upper_pose_ramp, params={
        "event_term": "upper_pose_delta", "start_scale": 0.0, "end_scale": 1.0,
        "ramp_start_iter": 600, "ramp_end_iter": 2500, "gate": SURVIVAL_GATE})
    stub_force = CurrTerm(func=lower_body_mdp.force_range_ramp, params={"gate": SURVIVAL_GATE})
    gait_cadence_l1_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={"reward_name": "gait_cadence_l1", "start_weight": -0.1, "terminal_weight": -0.6,
                "start_step": 20_000, "num_steps": 100_000, "upright_threshold": 0.10},
    )


@configclass
class LowerBodyV4TerminationsCfg(SquatTerminationsCfg):
    # world z is meaningless over pits and stair pyramids; measure from the ground under the trunk
    root_height = DoneTerm(func=lower_body_mdp.root_height_below_terrain,
                           params={"minimum_height": 0.33, "sensor_name": "height_scanner"})


def _apply_v4(cfg, preset: dict) -> None:
    cfg.scene.terrain.terrain_generator.sub_terrains = V4_SUB_TERRAINS
    cfg.scene.terrain.terrain_generator.num_rows = 10
    w, std = preset["yaw"]
    cfg.rewards.track_ang_vel_z_exp.weight = w
    cfg.rewards.track_ang_vel_z_exp.params["std"] = std


@configclass
class K1LowerBodyV4EnvCfg(K1LowerBodyEnvCfg):
    curriculum: LowerBodyV4CurriculumCfg = LowerBodyV4CurriculumCfg()
    terminations: LowerBodyV4TerminationsCfg = LowerBodyV4TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()  # copies the 7 upright-gated gait ramps, clearance -12
        _, preset = teacher_preset()
        _apply_v4(self, preset)
        c = self.curriculum
        c.feet_clearance_regularization.params["terminal_weight"] = preset["clearance"]
        c.stride_length_regularization.params["terminal_weight"] = preset["stride"]
        c.gait_cadence_l1_regularization.params["terminal_weight"] = preset["cadence_l1"]


@configclass
class K1LowerBodyV4PlayEnvCfg(K1LowerBodyPlayEnvCfg):
    """v4 teacher on its own terrain mix (hardest rows), full arm/head poses and stub forces."""

    terminations: LowerBodyV4TerminationsCfg = LowerBodyV4TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = K1RoughSceneCfg().terrain  # play parent swapped in a plane
        self.sim.physics_material = self.scene.terrain.physics_material
        _apply_v4(self, teacher_preset()[1])
        self.scene.terrain.terrain_generator.curriculum = False  # random difficulty per tile
