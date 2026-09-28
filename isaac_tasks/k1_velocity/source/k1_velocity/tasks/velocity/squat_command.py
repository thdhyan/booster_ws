"""Phase-1 squat base command: (vx, vy, wz) + commanded trunk height H*.

The height rides INSIDE the existing velocity command term instead of being a
second term, so every current consumer keeps working without config edits:

* obs ``velocity_commands = mdp.generated_commands`` grows 3 -> 4 dims by
  itself (it returns this term's ``command`` property verbatim);
* the tracking rewards slice explicitly — IL's
  ``track_lin_vel_xy_yaw_frame_exp`` reads ``command[:, :2]`` and
  ``track_ang_vel_z_world_exp`` reads ``command[:, 2]`` — column 3 is
  invisible to them;
* the debug velocity arrows use ``command[:, :2]`` — unaffected.

Phase 1b/2 (the reach task's ``FrozenBaseVelocityAction``) is where the extra
column gets consumed explicitly; that slice moves 3 -> 4 there.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

from .squat_mdp import trunk_height_rel


class SquatVelocityHeightCommand(UniformVelocityCommand):
    """UniformVelocityCommand + per-env commanded trunk height ``height_command``.

    ``command`` -> (vx, vy, wz, H*), shape (num_envs, 4), H* at index 3.

    H* semantics (sampled together with the velocity axes, i.e. every 8-12 s):

    * env is drawn "squat" with p = ``cfg.rel_squat_envs``
      -> H* ~ U(*cfg.squat_height_range);
    * otherwise H* = ``cfg.stand_height`` (full stand = the P2 baseline target).

    ``rel_standing_envs`` keeps its parent meaning (zero VELOCITY only): a
    standing-velocity env may still be told to squat, which trains the static
    squat hold for free.

    The squat-depth curriculum reads ``vel_err_ema`` / ``height_err_ema``
    (updated every control step in :meth:`_update_metrics`) and owns its stage
    counter here (``squat_stage``), so ``squat_mdp.squat_depth_curriculum``
    can stay a plain function (Isaac Lab build constraint, see squat_mdp).
    """

    cfg: "SquatVelocityHeightCommandCfg"

    def __init__(self, cfg: "SquatVelocityHeightCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        dev = self.device
        self.height_command = torch.full((n,), float(cfg.stand_height), device=dev)
        self.is_squat_env = torch.zeros(n, dtype=torch.bool, device=dev)
        # per-episode height error, finalized into metrics["height_err"] at reset
        self._h_err_sum = torch.zeros(n, device=dev)
        self._h_step_count = torch.zeros(n, device=dev)
        # step EMAs for the squat-depth curriculum gates. Both start at 1.0 —
        # deliberately pessimistic so no gate can pass before real data has
        # been observed at the current depth.
        self.vel_err_ema = 1.0
        self.height_err_ema = 1.0
        self._ema_beta = 0.99
        # applied-transition count; 0 = velocity baseline (no squats yet)
        self.squat_stage = 0
        self.metrics["height_err"] = torch.zeros(n, device=dev)
        # leapp export annotation follows the widened command (parent just set
        # the three velocity element names in its __init__)
        if "base_height" not in list(self.cfg.element_names or []):
            self.cfg.element_names = list(self.cfg.element_names or []) + ["base_height"]

    @property
    def command(self) -> torch.Tensor:
        """The desired base command (vx, vy, wz, H*); shape (num_envs, 4)."""
        return torch.cat([self.vel_command_b, self.height_command.unsqueeze(-1)], dim=-1)

    def _resample_command(self, env_ids) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        is_squat = torch.rand(n, device=self.device) <= self.cfg.rel_squat_envs
        u = torch.rand(n, device=self.device)
        lo, hi = (float(v) for v in self.cfg.squat_height_range)
        h_new = u * (hi - lo) + lo
        self.height_command[env_ids] = torch.where(
            is_squat, h_new, torch.full_like(h_new, float(self.cfg.stand_height))
        )
        self.is_squat_env[env_ids] = is_squat
        # NOTE: parent's _update_command zeroes vel_command_b for standing envs
        # only — the height command is intentionally NOT touched there, so a
        # standing-velocity env can still be told to hold a squat.

    def _update_metrics(self) -> None:
        super()._update_metrics()
        # per-env episode accumulator (finalized in reset) + curriculum EMAs
        h_err = torch.abs(trunk_height_rel(self._env) - self.height_command)
        self._h_err_sum += h_err
        self._h_step_count += 1.0
        if bool(self.is_squat_env.any()):
            m = float(h_err[self.is_squat_env].mean())
            self.height_err_ema = self._ema_beta * self.height_err_ema + (1.0 - self._ema_beta) * m
        vel_err = torch.linalg.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1
        )
        self.vel_err_ema = self._ema_beta * self.vel_err_ema + (1.0 - self._ema_beta) * float(vel_err.mean())

    def reset(self, env_ids=None) -> dict[str, float]:
        # Finalize the height metric BEFORE super().reset() logs and zeros the
        # metrics dict (same ordering trick as the parent's velocity metrics);
        # zero our own running sums afterwards.
        if env_ids is None:
            idx = slice(None)
        else:
            idx = env_ids
        denom = self._h_step_count[idx].clamp_min(1.0)
        self.metrics["height_err"][idx] = self._h_err_sum[idx] / denom
        extras = super().reset(env_ids)
        self._h_err_sum[idx] = 0.0
        self._h_step_count[idx] = 0.0
        return extras


@configclass
class SquatVelocityHeightCommandCfg(UniformVelocityCommandCfg):
    """Velocity ranges at VR teleop BASE_LIMITS + the height-channel config.

    The velocity ranges (vx +/-0.5, vy +/-0.3, wz +/-0.8) are deliberately
    NARROWER than the P2 teacher's (+/-0.5 / +/-0.5 / +/-1.0): they match what
    the VR teleop can actually send (``k1_teleop BASE_LIMITS``), so Phase 1
    trains exactly the deployable envelope — hence no velocity-range widening
    curriculum here either (see SquatCurriculumCfg).
    """

    class_type = SquatVelocityHeightCommand
    # full stand = velocity_env_cfg.K1_TRUNK_HEIGHT; the env cfg passes it
    # explicitly, this default is a belt-and-braces fallback.
    stand_height: float = 0.57
    rel_squat_envs: float = 0.0  # raised by squat_depth_curriculum (baseline first)
    squat_height_range: tuple[float, float] = (0.50, 0.55)  # H* ~ U(lo, hi) for squat envs
