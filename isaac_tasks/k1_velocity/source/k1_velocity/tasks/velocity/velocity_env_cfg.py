# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Modifications for Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""
K1 velocity locomotion — rough terrain training environment.
Ported from:
  isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py

Key differences from G1:
  - Robot: Booster K1 (22 DoF, 12 leg joints for policy)
  - Asset: BOOSTER_K1_CFG from booster_train (UrdfFileCfg, real actuator models)
  - K1 init height: 0.57m (from booster_train)
  - Real PD gains per joint from booster_train actuator specs

Requires:
  pip install -e isaac_tasks/booster_train_ref/source/booster_train
  pip install -e isaac_tasks/k1_velocity
"""

from __future__ import annotations
import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import agile_rewards as agile
from . import gait_clock
from . import gait_rewards as gait
from . import reward_weight_ramp
from . import velocity_curriculum

try:  # Isaac Lab 3.0-EA layout (dl); isaac-lab image renamed this package
    import isaaclab_tasks.core.velocity.mdp as mdp
except (ImportError, ModuleNotFoundError):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Use the real K1 articulation config from booster_train (correct actuators, URDF path, PD gains)
from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from k1_velocity.sim_backend import apply_physics_backend

# ---------------------------------------------------------------------------
# K1 joint name constants
# ---------------------------------------------------------------------------
# 12 leg joints controlled by locomotion policy
K1_LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]

# 10 arm+head joints — penalise deviation from default, regulated separately.
# NOTE: exact URDF names — Isaac Sim 6 importer preserves them verbatim:
# head yaw is 'AAHead_yaw', shoulder pitch joints carry an 'A' prefix.
K1_ARM_HEAD_JOINTS = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
]

# Nominal standing trunk height (matches BOOSTER_K1_CFG init_state z). Used as the
# upright target for the base_height reward and as the fall-termination reference.
K1_TRUNK_HEIGHT = 0.57

# K1 articulation — imported from booster_train (real actuators, K1_22dof.urdf, init_pos z=0.57)
#
# Leg-gain override for locomotion training.  booster_train derives
# stiffness = armature * (2*pi*f)^2 with natural_freq = 4 Hz, which yields hip
# gains of only 17.8-30.2 and knee 60.4.  Measured consequence (reward probe,
# zero action): the trunk sinks monotonically from 0.589 m to 0.076 m in 2.4 s --
# the robot cannot hold itself up, so no reward can produce a walk.  AGILE's
# Booster T1, a biped validated for standing and velocity tracking with sim2real,
# runs stiffness 100 on hips and knee.
#
# These gains are SIM-ONLY and deliberately do not live in booster.py: the
# shared hardware model keeps its real-robot values, and Track B is unaffected.
K1_P2_LEG_STIFFNESS = {
    ".*_Hip_Pitch": 100.0,
    ".*_Hip_Roll": 100.0,
    ".*_Hip_Yaw": 60.0,
    ".*_Knee_Pitch": 100.0,
}
K1_P2_LEG_DAMPING = {
    ".*_Hip_Pitch": 5.0,
    ".*_Hip_Roll": 5.0,
    ".*_Hip_Yaw": 3.0,
    ".*_Knee_Pitch": 5.0,
}
# Ankle gains (gait lever 1 of 3) -- same argument as the leg override above,
# applied one group further down. The stock ``feet`` group is two
# ``BoosterK1AnkleParaWrapperCfg`` around the E4310 (the hip-yaw motor), solved
# at natural_freq 4 Hz / damping_ratio 1.5 and then divided by the 4-bar
# armature ratio squared, which lands on 35.7 Nm/rad -- a third of the 100
# Nm/rad the hip pitch and knee run at. Measured consequence (GAIT_GATE on
# p2_gaitshuffle_2999: stride 0.03 m, cadence 8.25 steps/s, 8/8 fail): the foot
# twists under the stance moment instead of holding a rigid lever, the sole
# slides, and the policy answers with shuffling.
#
#   .*_Ankle_Pitch -> 100.0: the sagittal push-off axis, raised to the same
#       authority as hip pitch / knee. damping 5.0 matches the leg override
#       (~1.05x critical for the wrapper's 0.0565 kg.m^2 armature).
#   .*_Ankle_Roll  -> held at the wrapper-derived 35.69 (damping 4.26). It is a
#       small-range lateral axis under a 38.3 Nm limit, roll deviation is ~0 on
#       flat ground anyway, and stiffening it only risks contact chatter. Keep
#       the change to the sagittal axis so a gait-gate pass or fail means one
#       thing. Both keys must be listed: an actuator pattern that matches no
#       joint errors out, a joint matched by no pattern silently gets 0.0 gain.
K1_P2_FOOT_STIFFNESS = {
    ".*_Ankle_Pitch": 100.0,
    ".*_Ankle_Roll": 35.69,
}
K1_P2_FOOT_DAMPING = {
    ".*_Ankle_Pitch": 5.0,
    ".*_Ankle_Roll": 4.26,
}
K1_ARTICULATION_CFG = copy.deepcopy(BOOSTER_K1_CFG)
K1_ARTICULATION_CFG.actuators["legs"].stiffness = dict(K1_P2_LEG_STIFFNESS)
K1_ARTICULATION_CFG.actuators["legs"].damping = dict(K1_P2_LEG_DAMPING)
K1_ARTICULATION_CFG.actuators["feet"].stiffness = dict(K1_P2_FOOT_STIFFNESS)
K1_ARTICULATION_CFG.actuators["feet"].damping = dict(K1_P2_FOOT_DAMPING)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class K1RoughSceneCfg(InteractiveSceneCfg):
    """Scene: K1 on rough terrain."""

    # Rough terrain: low-angle slopes + random bumps only (no stairs).
    # Blind policy — no raycaster, proprioception only.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=6,
            num_cols=12,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                # 40% flat — for curriculum start
                "flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    noise_range=(0.0, 0.01),
                    noise_step=0.005,
                    border_width=0.25,
                ),
                # 40% mild rough — bumps up to ±4 cm
                "rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    noise_range=(-0.04, 0.04),
                    noise_step=0.01,
                    border_width=0.25,
                ),
                # 20% low-slope — max 12° incline
                "slope": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.2,
                    num_obstacles=5,
                    obstacle_height_mode="fixed",
                    obstacle_height_range=(0.0, 0.05),
                    obstacle_width_range=(0.5, 1.5),
                    platform_width=2.0,
                    border_width=0.25,
                ),
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = K1_ARTICULATION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Height scanner for the PRIVILEGED (teacher) observation group. The blind
    # student policy never sees it; the teacher uses it during PPO training and
    # the student recovers the behavior via distillation.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 17x11 = 187 pts
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Contact sensor for foot forces (feet_slide, undesired_contacts, terminations).
    # Fixed via scripts/flatten_k1_usd.py which applies PhysxContactReportAPI to all rigid bodies.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=["/World/ground"],
    )


# ---------------------------------------------------------------------------
# MDP — Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the blind locomotion policy (50-dim per step).

        48 dims of proprioception + command (see terms below) and the 2-dim
        phase clock. No height scan: the student is blind on purpose, it has to
        recover terrain from what the teacher felt.
        """
        # Base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Commands
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # Joint state (12 leg joints only)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # No height scan — blind policy, proprioception only
        # Last action
        actions = ObsTerm(func=mdp.last_action)
        # Gait lever 3: time base for stepping. LAST term, so it lands at
        # obs[48:50] and the existing 48 dims keep their indices. noise=None --
        # a signal the policy times itself from, not a sensor reading, so
        # corrupting it would desynchronise the clock from the body.
        phase_clock = ObsTerm(func=gait_clock.phase_clock, noise=None)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged observations for the TEACHER policy (PPO training only).

        Noise-free proprioception + terrain height scan. Foot contact forces are now
        available via scripts/flatten_k1_usd.py which applies PhysxContactReportAPI to
        all rigid bodies, enabling contact sensors to see nested feet.

        237 dims: the same 48 + 2 as the policy group, with a 187-point height
        scan in between. The phase clock lives here too -- PPO trains actor and
        critic on this group, and a teacher that never saw the clock could not
        produce a gait for the student to distil.
        """
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        # LAST term -> teacher obs[235:237]. Same clock, same phase, same step
        # as the policy group (see gait_clock.phase_clock on how the two reads
        # stay in sync), so PPO's actor and the student see identical timing.
        phase_clock = ObsTerm(func=gait_clock.phase_clock, noise=None)

        def __post_init__(self):
            self.enable_corruption = False  # privileged: no sensor noise
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


# ---------------------------------------------------------------------------
# MDP — Actions
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """Joint position targets for 12 leg joints (offsets from default)."""
    # AGILE uses scale=1.0 with an explicit +/-1.0 rad clip. Our old scale=0.25
    # capped every leg joint to a quarter-radian of authority, which is not
    # enough to recover a tilt -- the policy could not physically save itself.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_LEG_JOINTS,
        scale=1.0,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )


# ---------------------------------------------------------------------------
# MDP — Commands
# ---------------------------------------------------------------------------
@configclass
class CommandsCfg:
    """Direct Cartesian velocity commands for walking practice.

    P2 previously generated heading-mode commands for every environment.  That
    made lateral commands hard to learn and hid the requested ``vx/vy/wz``
    ranges.  Keep a small standing fraction, but sample the full Cartesian
    command space directly; terrain curriculum handles difficulty separately.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        # AGILE: 25% of envs get a true stand command, and the ranges are
        # deliberately small (0.5 m/s). Asking a biped that cannot yet stand for
        # 1.5 m/s guarantees collapse. Widen via curriculum, not day one.
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            # Kept in the schema but inactive while heading_command=False.
            heading=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# MDP — Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Velocity tracking plus AGILE/T1-style bipedal gait shaping.

    Weights and structure follow NVIDIA's AGILE velocity task for the Booster T1
    (``agile/rl_env/tasks/locomotion/t1/velocity_env_cfg.py``).  Three
    substantive differences from our previous gait-v2 set:

    1. **No air-time reward.**  ``feet_air_time`` pays for lifting a foot, and is
       maximised by hopping, skipping (never loading one foot) or bouncing.  It
       also fought our own ``lin_vel_z_l2``.  AGILE omits air time entirely and
       shapes the feet directly instead -- slip, roll, yaw-vs-base, stance width.
    2. **A trunk-height term.**  Previously the only upright signals were the fall
       *termination* (a cliff, not a gradient) and a weak ``flat_orientation_l2``,
       so nothing rewarded staying up before falling.  P3 fell face-first for
       exactly this reason.  ``base_height`` supplies the dense gradient.
    3. **Stronger tracking and orientation.**  Tracking is the task and gets the
       dominant weight; tilt is penalised hard because a biped leaning past ~30
       deg cannot recover.
    """

    # --- Task: velocity tracking (AGILE T1 weight 5.0, std 0.2) ---
    # AGILE-parity switch (2026-10-01): the *weighted* tracker, 1.0x at rest to
    # 2.0x at the top of the commanded range. The plain exponential this
    # replaces returned ~the same credit for 0.09 m/s as for 0.5 m/s, which is
    # the measured failure: the last retrain walked 0.09-0.31 m/s against a
    # 0.5 m/s command and was paid for the shortfall. Weight 10.0 (not AGILE's
    # 5.0) is kept because it was raised from 5.0 during the earlier audit when
    # the unweighted term proved too flat to earn a gradient; the 2.0x speed
    # multiplier now supplies the extra pull that raise was substituting for.
    # std stays at the audited 0.15, not AGILE's 0.2 -- that narrowing was the
    # other half of the same fix.
    track_lin_vel_xy_exp = RewTerm(
        func=agile.track_lin_vel_xy_exp_weighted,
        weight=10.0,
        params={"command_name": "base_velocity", "std": 0.15},
    )
    track_ang_vel_z_exp = RewTerm(
        func=agile.track_ang_vel_z_world_exp_weighted,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # --- Upright / posture (AGILE) ---
    # sensor_cfg shifts the target by the terrain height under the trunk, so this
    # stays correct on rough ground instead of pinning an absolute world height.
    # Positive, not a penalty.  As a -8.0 L2 bill it taxed every metre walked
    # out of the same budget as the task reward, and the first teacher learned
    # to stand perfectly (0 % falls) while walking 0.17 m in 36 s.  AGILE pays
    # for uprightness instead and penalises tilt separately.
    base_height = RewTerm(
        func=gait.base_height_exp,
        weight=2.0,
        params={
            "target_height": K1_TRUNK_HEIGHT,
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"])},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # --- Gait: foot shaping in place of air time (AGILE) ---
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
        },
    )
    feet_roll = RewTerm(
        func=gait.feet_roll_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])},
    )
    feet_yaw_diff = RewTerm(
        func=gait.feet_yaw_diff_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])},
    )
    feet_yaw_mean = RewTerm(
        func=gait.feet_yaw_mean_vs_base,
        weight=-4.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "base_body_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
        },
    )
    feet_distance = RewTerm(
        func=gait.feet_distance_from_ref,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "ref_distance": 0.2,
        },
    )

    # AGILE-parity additions (2026-10-01). Weights are AGILE T1's verbatim --
    # agile/rl_env/tasks/locomotion/t1/velocity_env_cfg.py. These three were the
    # only H2/T1 terms this config was missing; see docs/agile_parity_zzbw.md.
    # `ankle_torques` is NOT a duplicate of `ankle_roll_torques` below: that one
    # is 20x heavier and roll-only, so it covers a different failure (a weak
    # lateral axis letting the foot twist) rather than repeating it.
    ankle_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(?i).*_Ankle_.*"])},
    )
    # All joints, matching AGILE T1's `SceneEntityCfg("robot")` -- this is a
    # whole-body joint-speed penalty, not a leg-only one.
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Both feet off the ground at once. Threshold 1.0 N matches AGILE T1's
    # `feet_slip` contact threshold; the K1 weighs ~24 kg, so 1 N is a firm
    # criterion for "no longer supporting load" rather than a noisy near-zero.
    jumping = RewTerm(
        func=agile.jumping,
        weight=-0.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_foot_link", "right_foot_link"]
            ),
        },
    )

    # --- Regularization (AGILE) ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    # Smoothness -- STATIC value is AGILE's *start* weight; the ramp in
    # CurriculumCfg raises both smoothness terms and all five gait extras toward
    # their previous values as the policy proves it can stand.
    #
    # These two weights were raised (-0.5 -> -2.0, and -0.02 -> -0.5 in the 10-01
    # audit) because at the *end* of training the terms were inert against a
    # +1.18 tracking term. That reasoning is correct for a converged policy and
    # wrong for the first 1 000 iterations, where the policy falls in 99.3% of
    # episodes and is being charged for smoothness it cannot yet express. AGILE
    # makes the same call explicitly: update_reward_weight_step runs action_rate
    # at -0.5 until step 50 000 and ramps to -2.0 (see docs/agile_weight_comparison.md).
    # The static number is now the light one; the ramp owns the rest.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    action_jerk_l2 = RewTerm(
        func=gait.action_jerk_l2,
        # Audit 2026-10-01: at -0.02 this term contributed **-0.0028 per step** in
        # the 3000-iteration run -- inert -- while jerk was the metric the gait
        # gate actually failed (0.107 against a <0.06 bound). Two orders of
        # magnitude of headroom against a +1.18 tracking term, so it never
        # competed.
        #
        # Now the AGILE *start* value (-0.05), with a ramp in CurriculumCfg up to
        # -0.5 as the policy proves it can stand. Same reasoning as
        # action_rate_l2 above: -0.5 is right for a converged policy and wrong
        # while 99.3% of episodes end in a fall. AGILE ramps its equivalent term
        # from -0.05 to -1.0 over 100k steps for exactly this reason.
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
    )
    ankle_roll_torques = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Roll"])},
    )
    # ALL leg joints, not just the ankles.  Previously hips and knees could be
    # driven to their limits essentially for free -- a hyperextension collapse.
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS), "soft_ratio": 0.9},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS)},
    )
    body_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2, weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Keep the non-locomotion arms/head near the K1 default pose.
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_ARM_HEAD_JOINTS)},
    )
    # NOTE: the former ``stand_still`` term was removed.  With 25 % of envs
    # commanded to zero velocity it paid the policy to hold still, reinforcing
    # exactly the stand-in-place optimum that stopped the first teacher
    # walking.  Standing is now handled by the positive base_height reward.
    # Do not use hip/trunk ground contact to crawl or kneel forward.
    # ---- gait structure ----------------------------------------------------
    # The 3000-iteration run was stable and moved (2.66 m net, 0 falls) but did
    # not walk: mean cadence 5.7 steps/s against 1.8-2.2 for human walking, with
    # 5 of 8 envs chattering at 7.9-8.9 steps/s on 3-5 cm strides. Nothing in
    # the reward distinguished a walk from a jitter, so these terms make the
    # difference observable to the policy. Weights are deliberately modest --
    # this is shaping, and over-weighting it fights the tracking objective.
    #
    # The retrain with those terms still shuffled 8/8 (cadence 6.68, jerk
    # 0.099), for two measured reasons: gait_cadence read joint_pos, whose
    # postural drift hid the 4-6 Hz chatter and paid only -0.394/step, and the
    # phase clock sat unused at |corr(action, clock)| = 0.098. The estimator
    # now reads joint_vel, and phase_swing is the term that pays for using the
    # clock. Update both together if the gait moves again -- they are the two
    # numbers gait_gate.py actually measures (cadence from joint actions,
    # stride from speed/cadence).
    #
    # WEIGHTS BELOW ARE START VALUES, NOT FINAL. These five terms sum to -19.1,
    # which is 53% of our entire per-step penalty budget against AGILE T1's 13.4
    # total -- all of it non-AGILE, all of it applied from iteration 0 to a policy
    # that at iteration 12 falls in 99.3% of episodes. Each starts at a fifth of
    # its final weight and is ramped by CurriculumCfg once the robot stands; the
    # ramp's terminal weights are the values the 10-01 audit settled on.
    gait_cadence = RewTerm(
        func=gait.gait_cadence_penalty,
        # -1.0 final -> -0.2 start.
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINTS),
            "target_hz": 2.0,
            "dt": 0.02,
        },
    )
    # Term #24, and the one the 5000-iteration run was missing: a phase-locked
    # STRIDE, not a cadence rate and not a contact schedule.
    #
    # `gait_cadence` is bistable -- 0.4 steps/s and 11 steps/s are both "far from
    # 2.0" and the policy takes whichever is cheaper, which is how a 5000-iter
    # run ended at cadence 4.85 / jerk 0.114 / 8-of-8 gate failures.
    # `phase_swing` locked only CONTACT, a binary predicate a 4 cm twitch
    # satisfies as readily as a stride, so it could not fix it either.
    # Locking fore-aft POSITION against the clock makes slow and fast both
    # expensive, leaving one basin.
    #
    # Weight -4.0: squared metres of placement error, so ~0.1 m off-reference
    # costs -0.04/step against the +1.28 tracking term. Sized to compete, not
    # to dominate -- an over-weighted position prior makes the robot walk the
    # reference instead of the command, which is the failure the command-coupled
    # clock exists to prevent.
    phase_locked_stride = RewTerm(
        func=gait.phase_locked_stride,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
            ),
            "base_body_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
            "target_stride": 0.22,
        },
    )
    phase_swing = RewTerm(
        func=gait.phase_synced_swing,
        # -2.0 final -> -0.4 start.
        weight=-0.4,
        params={
            # contact for the swing/stance test; the phase comes from
            # gait_clock.get_phase(env), the same state the observation sees.
            # preserve_order=True pins body_ids to [left, right]: it defaults
            # to False, which silently follows the sensor's body order instead
            # of this list, and this term is the only one here that reads a
            # specific foot at a specific index.
            "contact_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )
    feet_clearance = RewTerm(
        func=gait.feet_clearance,
        # -8.0 final -> -1.6 start. Heaviest single term in the budget and it
        # reads -0.0000/step today (a hinge, zero until a foot actually lifts).
        weight=-1.6,
        params={
            # contact for the swing/stance test, robot for the foot heights.
            # These are different prims: a ContactSensor has no body_pos.
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "target_height": 0.06,
        },
    )
    feet_alternation = RewTerm(
        func=gait.feet_alternation_penalty,
        # -2.0 final -> -0.4 start.
        weight=-0.4,
        params={
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
        },
    )
    stride_length = RewTerm(
        func=gait.stride_length_penalty,
        # Audit 2026-10-01: at target 0.35 m this contributed only -0.051/step,
        # because 0.35 m fore-aft separation is roughly twice any stride the
        # policy actually takes, so the clamp zeroed out most of the gradient
        # and what remained rewarded a permanent split stance instead of steps.
        # 0.22 m sits inside the band the gate calls a stride (>=0.15) and inside
        # what a 2 m/s gait at 2 steps/s actually needs.
        # -6.0 final -> -1.2 start.
        weight=-1.2,
        params={
            # Foot separation comes from the articulation, not the sensor.
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "target_stride": 0.22,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["Trunk", ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw"]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Contact-free fall detection (height-based, not contact-based).
    # K1 trunk stands at ~0.57 m; 0.35 m means the robot has collapsed.
    # Contact-based termination is now possible via contact_forces sensor.
    root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35},
    )
    # AGILE uses 30 deg. Our previous 0.8 rad (~46 deg) let the robot get well
    # past the point of no return before the episode ended.
    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(30.0)},
    )
    # Trunk touching the ground is unrecoverable, and it is the failure mode we
    # actually observed (P3 fell face first with no termination to catch it).
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk"]),
            "threshold": 20.0,
        },
    )


# ---------------------------------------------------------------------------
# MDP — Curriculum
# ---------------------------------------------------------------------------
@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel,
                               params={"asset_cfg": SceneEntityCfg("robot")})
    # Widens the commanded velocity range as tracking improves, because a fixed
    # +/-0.5 m/s range makes 0 -> 4 m/s unreachable by construction: the policy is
    # never asked to go faster, so it never learns to.
    #
    # **Target raised to 4.0 m/s on user request (2026-10-01).** Keep the caveat
    # attached to the number: the K1 A2 factory walker peaked near 1.3 m/s over a
    # 12-minute, 254 m walk, and the curriculum *class* still defaults to 1.5 m/s
    # for that reason. 4 m/s is a sim stretch goal -- a sim number there is not a
    # deployable claim, and the gate to pass is GAIT_GATE at each speed, not
    # "it moved".
    #
    # step 0.5 (was 0.25) so the ramp 0.5 -> 4.0 takes 7 expansions rather than
    # 14; each expansion still needs the upright ratio held for `patience`
    # consecutive checks, so this is not a free speed-up.
    #
    # max_lin_vel_y caps lateral separately: the term applies one magnitude to
    # both axes, and +/-4 m/s sideways is not a gait, it is a fall.
    velocity_range = CurrTerm(
        func=velocity_curriculum.VelocityRangeCurriculumTerm,
        params={
            "init_lin_vel": 0.5,
            "init_ang_vel": 1.0,
            "target_max_lin_vel": 4.0,
            "target_max_ang_vel": 2.0,
            "max_lin_vel_y": 1.0,
            "step_lin_vel": 0.5,
            "step_ang_vel": 0.25,
            "success_threshold": 0.80,
            "patience": 5,
            "interval_steps": 50,
        },
    )

    # ---- regularization ramps (AGILE's idea, upright-gated) ------------------
    # Our per-step penalty budget was 35.9 vs AGILE T1's 13.4, with 53% of it in
    # five non-AGILE gait terms applied from iteration 0 -- while 99.3% of
    # episodes ended in a fall. AGILE ships update_reward_weight_step to run
    # action_rate at -0.5 until step 50k and ramp it to -2.0 over 100k; G1 simply
    # carries a static -0.01. We borrow the mechanism and change one thing: credit
    # accrues only while the robot is UPRIGHT, so a policy still falling at step
    # 3 000 is not charged for not being smooth yet.
    #
    # num_steps=100_000 is AGILE's value and is scaled for the 42 000-iteration run
    # (1.0M control steps): the ramp starts ~4% in and completes ~12% in, matching
    # AGILE's schedule as a fraction of training.
    #
    # Seven terms: the two smoothness terms plus the five gait extras that make
    # up the other half of the budget. start_weight must equal the static weight
    # in RewardsCfg -- the ramp reads it from the reward manager, and a mismatch
    # makes the static value dead. Terminal values are the 10-01 audit numbers.
    #
    # upright_threshold is 0.10, NOT AGILE's-everything-is-fine 0.80. This was
    # my error at 2ddec51: 0.80 is unreachable while the measured upright ratio
    # is ~0.02-0.15 (mean episode length 20 steps of 1000), so the ramp would
    # never fire and the gait penalties would sit at 1/5 strength for the whole
    # run -- entrenching exactly the shuffle we are trying to remove. 0.10 is
    # reachable from the first iteration and still means "standing far more than
    # it currently does".
    action_rate_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={
            "reward_name": "action_rate_l2",
            "start_weight": -0.5,
            "terminal_weight": -2.0,
            "start_step": 20_000,
            "num_steps": 100_000,
            "upright_threshold": 0.10,
        },
    )
    action_jerk_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={
            "reward_name": "action_jerk_l2",
            "start_weight": -0.05,
            "terminal_weight": -0.5,
            "start_step": 20_000,
            "num_steps": 100_000,
            "upright_threshold": 0.10,
        },
    )
    gait_cadence_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={
            "reward_name": "gait_cadence",
            "start_weight": -0.2,
            "terminal_weight": -1.0,
            "start_step": 20_000,
            "num_steps": 100_000,
            "upright_threshold": 0.10,
        },
    )
    phase_swing_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={
            "reward_name": "phase_swing",
            "start_weight": -0.4,
            "terminal_weight": -2.0,
            "start_step": 20_000,
            "num_steps": 100_000,
            "upright_threshold": 0.10,
        },
    )
    feet_clearance_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={
            "reward_name": "feet_clearance",
            "start_weight": -1.6,
            "terminal_weight": -8.0,
            "start_step": 20_000,
            "num_steps": 100_000,
            "upright_threshold": 0.10,
        },
    )
    feet_alternation_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={
            "reward_name": "feet_alternation",
            "start_weight": -0.4,
            "terminal_weight": -2.0,
            "start_step": 20_000,
            "num_steps": 100_000,
            "upright_threshold": 0.10,
        },
    )
    stride_length_regularization = CurrTerm(
        func=reward_weight_ramp.RewardWeightRampTerm,
        params={
            "reward_name": "stride_length",
            "start_weight": -1.2,
            "terminal_weight": -6.0,
            "start_step": 20_000,
            "num_steps": 100_000,
            "upright_threshold": 0.10,
        },
    )


# ---------------------------------------------------------------------------
# Events (randomization)
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                "mass_distribution_params": (-2.0, 2.0),
                "operation": "add"},
    )
    # Gait lever 2 of 3: friction domain randomisation. The scene pins the
    # terrain material at mu=1.0 (see K1RoughSceneCfg.terrain), so a policy
    # trained only there learns one sole/ground pair and its feet slide the
    # moment the floor is anything else -- one of the two halves of the
    # measured shuffle (stride 0.03 m). Randomising the robot's own material
    # over [0.4, 1.2] makes slip something the policy has seen and can push
    # against. Pre-allocated at startup, not per episode: PhysX buckets
    # materials, so re-rolling every reset would re-upload them 4096 * 64
    # times a run for no extra robustness.
    #
    # NOT "prestartup": the material impl needs asset.root_view, which only
    # exists after sim play; "startup" fires right after play. Same wiring as
    # push's K1PushEventCfg.randomize_friction.
    #
    # Fires in play runs too, so a gait-gate recording inherits whatever mu was
    # rolled. Pin it with env.events.randomize_friction.mode=none when the gate
    # needs the nominal mu=1.0 floor rather than a random point in [0.4, 1.2].
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


# ---------------------------------------------------------------------------
# Main env config
# ---------------------------------------------------------------------------
@configclass
class K1VelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """K1 velocity locomotion — rough terrain training."""

    scene: K1RoughSceneCfg = K1RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # IL 3.0: custom BoosterDelayedPDActuator cannot run through Newton-native
        # actuator authoring (see SimulationCfg.use_newton_actuators) → use the
        # Isaac Lab execution path that supports custom actuator configs.
        self.sim.use_newton_actuators = False
        # Physics backend: Newton/warp by default (K1_PHYSICS=newton|physx).
        apply_physics_backend(self)
        self.sim.dt = 0.005          # 200 Hz physics
        self.decimation = 4          # 50 Hz control
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Learn the gait on the flat/curriculum-start tiles first; the existing
        # terrain-level curriculum then advances the same policy to rougher tiles.
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
