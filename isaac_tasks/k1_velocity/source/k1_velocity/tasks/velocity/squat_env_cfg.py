"""Phase-1 squat-capable base teacher: K1 velocity env + commanded trunk height H*.

``K1VelocitySquatEnvCfg`` subclasses the validated P2 rough-velocity env and
changes exactly FOUR things — scene, observations, actions and events stay
byte-identical to the P2 teacher so the Phase-1 velocity-parity gate
(tracking within +/-10% of the P2 teacher) is a meaningful comparison:

1. ``commands.base_velocity`` -> ``SquatVelocityHeightCommandCfg``: the policy
   now sees (vx, vy, wz, H*), velocity ranges at VR teleop BASE_LIMITS
   (vx +/-0.5, vy +/-0.3, wz +/-0.8), H* sampled per resample
   (squat_command.py).
2. ``rewards.base_height`` -> ``squat_mdp.base_height_cmd_exp``: per-env H*
   target, Laplacian kernel sigma = 0.05 (audit default), weight unchanged
   at +2.0.
3. ``terminations.root_height`` -> ``squat_mdp.root_height_below_commanded_floor``:
   terrain-relative H*-relative floor for squat envs; the validated
   world-frame 0.35 m floor for stand envs (velocity-parity baseline is
   bit-identical as long as no squats are commanded).
4. ``curriculum`` -> ``SquatCurriculumCfg``: terrain levels + the squat depth
   curriculum. The P2 ``velocity_range`` widening term is deliberately NOT
   inherited: (a) Phase 1 must train exactly the VR command envelope —
   widening to 1.5 m/s would both invalidate parity and exceed what the
   teleop can send; (b) the widening term lives only in the sibling P2-gait
   campaign's branch (``velocity_curriculum.py``, not part of this branch),
   and squat's ranges are fixed by design — there is no widening stage to
   coordinate with the height stages.

Task ids (registered in velocity/__init__.py):
  * ``Isaac-Velocity-Squat-K1-v0``       — training (teacher obs group, 4096 envs)
  * ``Isaac-Velocity-Squat-K1-Play-v0``  — flat stationary squat->rise demo (video)
"""

from __future__ import annotations

from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils.configclass import configclass

from . import squat_mdp
from .squat_command import SquatVelocityHeightCommandCfg
from .velocity_env_cfg import K1_TRUNK_HEIGHT, CommandsCfg, RewardsCfg, TerminationsCfg, K1VelocityRoughEnvCfg
from .velocity_play_cfg import K1VelocityRoughPlayEnvCfg

try:  # Isaac Lab 3.0-EA layout (dl); isaac-lab image renamed this package
    import isaaclab_tasks.core.velocity.mdp as mdp
except (ImportError, ModuleNotFoundError):
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


# ---------------------------------------------------------------------------
# MDP — Commands
# ---------------------------------------------------------------------------
@configclass
class SquatCommandsCfg(CommandsCfg):
    """(vx, vy, wz, H*) at VR teleop BASE_LIMITS; everything else as in P2."""

    base_velocity = SquatVelocityHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        # Same standing fraction as P2 (the AGILE comment in velocity_env_cfg
        # says 25% but the shipped value has always been 0.02 — keep 0.02 for
        # parity). Standing zeroes the VELOCITY only; H* is independent, so a
        # standing-velocity env can hold a squat (free static squat-hold data).
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # VR teleop BASE_LIMITS (k1_teleop/teleop_mapping.py): vx +/-0.5,
        # vy +/-0.3, wz +/-0.8 — narrower than P2's +/-0.5/0.5/1.0 on purpose
        # (train the deployable envelope, see squat_command docstring).
        ranges=SquatVelocityHeightCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
            # Kept in the schema but inactive while heading_command=False.
            heading=(0.0, 0.0),
        ),
        stand_height=K1_TRUNK_HEIGHT,
        rel_squat_envs=0.0,           # raised by squat_depth_curriculum from iter ~500
        squat_height_range=(0.50, 0.55),
    )


# ---------------------------------------------------------------------------
# MDP — Rewards
# ---------------------------------------------------------------------------
@configclass
class SquatRewardsCfg(RewardsCfg):
    """Only ``base_height`` changes: per-env H* Laplacian kernel (audit default).

    Both tracking terms are inherited unchanged — they slice command columns
    0..2, so the H* column is invisible to them and velocity learning is not
    re-weighted in any way.
    """

    base_height = RewTerm(
        func=squat_mdp.base_height_cmd_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.05, "sensor_name": "height_scanner"},
    )


# ---------------------------------------------------------------------------
# MDP — Terminations
# ---------------------------------------------------------------------------
@configclass
class SquatTerminationsCfg(TerminationsCfg):
    """Only ``root_height`` changes (see squat_mdp for the split-basis rationale)."""

    root_height = DoneTerm(
        func=squat_mdp.root_height_below_commanded_floor,
        params={
            "command_name": "base_velocity",
            "margin": 0.05,
            "minimum_height": 0.35,
            "sensor_name": "height_scanner",
        },
    )


# ---------------------------------------------------------------------------
# MDP — Curriculum
# ---------------------------------------------------------------------------
@configclass
class SquatCurriculumCfg:
    """Terrain levels (unchanged) + squat depth. No velocity widening (VR parity).

    Stage plan and gates are documented on :func:`squat_mdp.squat_depth_curriculum`;
    watch ``Curriculum/squat_depth/*`` in wandb to see the gates open.
    """

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    squat_depth = CurrTerm(
        func=squat_mdp.squat_depth_curriculum,
        params={
            "command_name": "base_velocity",
            "steps_per_iter": 24,
            # transition k: allowed from stage_iters[k], applies stage_rel_squat[k]
            # + stage_height_range[k], only if stage_gates[k] passes.
            #   0 @500   vel-gated:   10% squat, H*~U(0.50, 0.55)  shallow intro
            #   1 @1100  height-gated: 25% squat, same shallow depth (fraction up)
            #   2 @1900  height-gated: H*~U(0.46, 0.53)
            #   3 @2700  height-gated: H*~U(0.42, 0.50)
            #   4 @3500  height-gated: H*~U(0.40, 0.48)  -> 0.40 floor, 17 cm squat
            "stage_iters": (500, 1100, 1900, 2700, 3500),
            "stage_rel_squat": (0.10, 0.25, 0.25, 0.25, 0.25),
            "stage_height_range": ((0.50, 0.55), (0.50, 0.55), (0.46, 0.53), (0.42, 0.50), (0.40, 0.48)),
            "stage_gates": ("vel", "height", "height", "height", "height"),
            "height_gate": 0.025,  # 2.5 cm EMA over squat envs (eval gate is 2 cm)
            "vel_gate": 0.30,      # m/s mean xy tracking error EMA (loose)
        },
    )


# ---------------------------------------------------------------------------
# Envs
# ---------------------------------------------------------------------------
@configclass
class K1VelocitySquatEnvCfg(K1VelocityRoughEnvCfg):
    """Phase-1 squat base teacher — the four deviations listed in the module docstring."""

    commands: SquatCommandsCfg = SquatCommandsCfg()
    rewards: SquatRewardsCfg = SquatRewardsCfg()
    terminations: SquatTerminationsCfg = SquatTerminationsCfg()
    curriculum: SquatCurriculumCfg = SquatCurriculumCfg()


@configclass
class K1VelocitySquatPlayEnvCfg(K1VelocityRoughPlayEnvCfg):
    """Flat, stationary squat->rise demo for the Phase-1 debug video.

    Inherits the play conventions (flat plane, no curriculum, no pushes, no
    obs corruption) and pins vx = vy = wz = 0 so the only thing moving is H*:
    with rel_squat 0.5 and a 4-6 s resample, about half the fleet is always
    mid squat/rise, giving several clean transitions per 30 s clip. The squat
    band (0.42-0.50 vs stand 0.57) is deeper than the early training stages
    on purpose — this env runs a TRAINED policy, and deeper reads better on
    camera.
    """

    commands: SquatCommandsCfg = SquatCommandsCfg()
    rewards: SquatRewardsCfg = SquatRewardsCfg()
    terminations: SquatTerminationsCfg = SquatTerminationsCfg()

    def __post_init__(self):
        # Order matters: the P2 play post_init pins a walking command
        # (vx = 0.8), so the stationary squat pinning must come after it.
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.rel_squat_envs = 0.5
        self.commands.base_velocity.squat_height_range = (0.42, 0.50)
