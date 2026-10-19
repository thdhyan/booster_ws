"""GEAR-style decoupled lower body: separate WALK and STAND policies, direct RL (no distillation).

Mirrors GR00T-WBC's decoupled lower-body controller (g1_gear_wbc_policy.py): two policies on
the 12 leg joints, picked at deploy time by the command norm

    ||(vx, vy, wz)|| < 0.05  -> stand policy       else -> walk policy

* ``Isaac-LowerBody-Walk-K1-v0``: command (vx, vy, wz, H*, f). f is the gait frequency in
  gait cycles/s (1 cycle = 2 steps), sampled per resample like GEAR's ``freq_cmd``. The phase
  clock and every phase/cadence reward use the commanded f (gait_clock.commanded_frequency).
  ||cmd|| is lifted to >= 0.1, so the walk policy never trains on standing.
* ``Isaac-LowerBody-Stand-K1-v0``: command (0, 0, 0, H*); no clock, no gait rewards, feet
  planted. H* still squats (GEAR's height_cmd).

Both keep the v4 world: upper-body random poses + stub forces on survival-gated ramps,
terrain-relative fall check. Terrain is flat + rough (+-4 cm) + mild slope + waves; stairs
are left to the steps residual.

Direct RL: actor = deployable ``policy`` group (no base_lin_vel, 10-step history; GEAR uses 6),
critic = privileged ``teacher`` group (height scan, stub wrench). See the runners in
agents/squat_ppo_cfg.py.
"""
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import gait_rewards as gait
from . import squat_mdp
from .gait_clock import commanded_frequency
from .lower_body_env_cfg import STUDENT_HISTORY_LEN, K1LowerBodyV4EnvCfg
from .squat_command import SquatVelocityHeightCommand, SquatVelocityHeightCommandCfg
from .velocity_env_cfg import mdp

FEET = ["left_foot_link", "right_foot_link"]

GEAR_SUB_TERRAINS = {
    "flat": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.30, noise_range=(0.0, 0.01), noise_step=0.005, border_width=0.25),
    "rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.30, noise_range=(-0.04, 0.04), noise_step=0.01, border_width=0.25),
    "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.20, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25),  # 0-14 deg
    "wave": terrain_gen.HfWaveTerrainCfg(
        proportion=0.20, amplitude_range=(0.01, 0.04), num_waves=4, border_width=0.25),
}


# ---------------------------------------------------------------------------
# Command: (vx, vy, wz, H*, f)
# ---------------------------------------------------------------------------
class GaitVelocityHeightCommand(SquatVelocityHeightCommand):
    """Squat command + per-env gait frequency ``gait_freq_command`` (Hz) at index 4."""

    cfg: "GaitVelocityHeightCommandCfg"

    def __init__(self, cfg: "GaitVelocityHeightCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        lo, hi = cfg.gait_freq_range
        self.gait_freq_command = torch.full((self.num_envs,), 0.5 * (lo + hi), device=self.device)
        if "gait_frequency" not in list(self.cfg.element_names or []):
            self.cfg.element_names = list(self.cfg.element_names or []) + ["gait_frequency"]

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([super().command, self.gait_freq_command.unsqueeze(-1)], dim=-1)

    def _resample_command(self, env_ids) -> None:
        super()._resample_command(env_ids)
        # the walk policy never stands: below min_command_norm the stand policy takes over
        v = self.vel_command_b[env_ids]
        v[v.norm(dim=-1) < 1e-6, 0] = self.cfg.min_command_norm
        norm = v.norm(dim=-1, keepdim=True)
        self.vel_command_b[env_ids] = torch.where(
            norm < self.cfg.min_command_norm, v * self.cfg.min_command_norm / norm, v)
        lo, hi = self.cfg.gait_freq_range
        self.gait_freq_command[env_ids] = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo


@configclass
class GaitVelocityHeightCommandCfg(SquatVelocityHeightCommandCfg):
    class_type = GaitVelocityHeightCommand
    # gait cycles/s; 1.0 = 2 steps/s (the old fixed target). GEAR G1: 1.0-2.0 (keyboard clamp).
    gait_freq_range: tuple[float, float] = (0.8, 1.5)
    min_command_norm: float = 0.1  # ||(vx, vy, wz)||, GEAR switches at 0.05


def _as_gait_command(old: SquatVelocityHeightCommandCfg) -> GaitVelocityHeightCommandCfg:
    new = GaitVelocityHeightCommandCfg()
    for k in old.__dataclass_fields__:
        if k != "class_type":
            setattr(new, k, getattr(old, k))
    return new


# ---------------------------------------------------------------------------
# Cadence rewards against the COMMANDED frequency (target steps/s = 2 f)
# ---------------------------------------------------------------------------
def gait_cadence_cmd_l1(env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    """(n,) |steps/s - 2 f| (gait_rewards.gait_cadence_l1 with a per-env target)."""
    hz = gait._cadence_steps_per_s(env, asset_cfg)
    return (hz - 2.0 * commanded_frequency(env, command_name)).abs()


def gait_cadence_cmd_penalty(env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    """(n,) Huber-ish cadence error (gait_rewards.gait_cadence_penalty with a per-env target)."""
    err = (gait._cadence_steps_per_s(env, asset_cfg) - 2.0 * commanded_frequency(env, command_name)).abs()
    return torch.where(err <= 0.5, 0.5 * err.pow(2), 0.125 + 0.5 * (err - 0.5))


# ---------------------------------------------------------------------------
# Envs
# ---------------------------------------------------------------------------
_STAND_DROPPED_REWARDS = ("gait_cadence", "gait_cadence_l1", "phase_locked_stride", "phase_swing",
                          "feet_clearance", "feet_alternation", "stride_length")
_STAND_DROPPED_RAMPS = ("gait_cadence_regularization", "gait_cadence_l1_regularization",
                        "phase_swing_regularization", "feet_clearance_regularization",
                        "feet_alternation_regularization", "stride_length_regularization")


def _apply_gear(cfg) -> None:
    """Shared walk/stand changes on top of v4: GEAR terrain, deployable actor obs with history."""
    gen = cfg.scene.terrain.terrain_generator
    gen.sub_terrains = GEAR_SUB_TERRAINS
    gen.num_rows = 10
    obs = cfg.observations.policy
    obs.base_lin_vel = None  # not measured on the robot (GEAR has none either)
    obs.history_length = STUDENT_HISTORY_LEN


@configclass
class K1LowerBodyWalkEnvCfg(K1LowerBodyV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_gear(self)
        self.commands.base_velocity = _as_gait_command(self.commands.base_velocity)
        self.commands.base_velocity.rel_standing_envs = 0.0
        r = self.rewards
        for name, func in (("gait_cadence", gait_cadence_cmd_penalty), ("gait_cadence_l1", gait_cadence_cmd_l1)):
            term = getattr(r, name)
            term.func = func
            term.params = {"asset_cfg": term.params["asset_cfg"]}


@configclass
class K1LowerBodyStandEnvCfg(K1LowerBodyV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_gear(self)
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = c.ranges.lin_vel_y = c.ranges.ang_vel_z = (0.0, 0.0)
        c.rel_standing_envs = 1.0
        self.observations.policy.phase_clock = None
        self.observations.teacher.phase_clock = None
        # AGILE's speed-weighted tracking needs a non-degenerate range; plain exp kernel on 0 command
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp
        for name in _STAND_DROPPED_REWARDS:
            setattr(self.rewards, name, None)
        for name in _STAND_DROPPED_RAMPS:
            setattr(self.curriculum, name, None)
        self.rewards.feet_planted = RewTerm(
            func=squat_mdp.feet_unloaded_when_still, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
        # standing never walks off a tile, so terrain_levels_vel would only demote: random rows instead
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None


def _apply_play(cfg) -> None:
    cfg.scene.num_envs = 32
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols, gen.curriculum = 5, 4, False
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.events.push_robot = None
    cfg.observations.policy.enable_corruption = False
    # end-of-curriculum disturbances, fixed
    for name in ("terrain_levels", "squat_height", "upper_pose", "upper_delta", "stub_force"):
        setattr(cfg.curriculum, name, None)
    cfg.commands.base_velocity.rel_squat_envs = 0.25
    cfg.commands.base_velocity.squat_height_range = (0.40, 0.50)
    cfg.events.upper_pose_random.params["curriculum_scale"] = 1.0
    cfg.events.upper_pose_delta.params["curriculum_scale"] = 1.0
    cfg.events.stub_force.params["force_range"] = (-20.0, 20.0)


@configclass
class K1LowerBodyWalkPlayEnvCfg(K1LowerBodyWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


@configclass
class K1LowerBodyStandPlayEnvCfg(K1LowerBodyStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)
