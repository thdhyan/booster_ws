"""Carry task goal command: a ground goal the robot walks to while holding the box.

The term is registered as ``base_velocity`` and subclasses UniformVelocityCommand
so every velocity-env consumer keeps working unchanged:

* ``vel_command_b`` is not sampled but *derived* from the goal every step (walk
  toward it, turn to face it, stop inside the success radius). The gait clock
  and the AGILE tracking rewards read it exactly as in the velocity task, so the
  lower body keeps its walking prior.
* ``cfg.ranges`` stays, because agile_rewards reads it for the weight ramp; it
  also bounds the derived command.

The policy additionally observes the goal itself (``goal_pos_b``, ``goal_dist``,
see carry_mdp.goal_command_obs), so it can learn to deviate from the derived
velocity when carrying heavy boxes.

Success: trunk within ``success_radius`` of the goal for ``dwell_time`` seconds.
A ``stand_prob`` share of goals are stand goals: radius 0 (the robot's own spot),
held for ``stand_time``; they pay no goal_reached bonus (carry_tiers.reach_scale).
The max goal radius is ``max_radius``, owned by carry_mdp.goal_radius_curriculum.
Then ``just_reached`` is raised for one step (the reward at the next env step
reads it, because rewards are computed before the command manager) and a new
goal is drawn around the robot's current position.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

GOAL_RED, GOAL_GREEN = 0, 1


class GoalCarryCommand(UniformVelocityCommand):
    cfg: "GoalCarryCommandCfg"

    def __init__(self, cfg: "GoalCarryCommandCfg", env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.goal_pos_w = torch.zeros(n, 3, device=dev)
        self.goal_pos_b = torch.zeros(n, 2, device=dev)
        self.goal_dist = torch.zeros(n, device=dev)
        self.dwell = torch.zeros(n, device=dev)
        self.dwell_needed = torch.full((n,), cfg.dwell_time, device=dev)
        self.goal_r = torch.zeros(n, device=dev)
        self.reached_r = torch.zeros(n, device=dev)   # radius of the goal reached this step
        self.max_radius = float(cfg.goal_radius_range[1])
        self.just_reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.metrics["goal_dist"] = torch.zeros(n, device=dev)
        self.metrics["successes"] = torch.zeros(n, device=dev)

    def __str__(self) -> str:
        return (
            "GoalCarryCommand:\n"
            f"\tGoal radius range: {self.cfg.goal_radius_range}\n"
            f"\tSuccess: within {self.cfg.success_radius} m for {self.cfg.dwell_time} s"
        )

    # -- goal bookkeeping ---------------------------------------------------

    def _sample_goal(self, env_ids: torch.Tensor):
        k = len(env_ids)
        lo = min(self.cfg.goal_radius_range[0], self.max_radius)
        r = torch.empty(k, device=self.device).uniform_(lo, self.max_radius)
        stand = torch.rand(k, device=self.device) < self.cfg.stand_prob
        r = torch.where(stand, torch.zeros_like(r), r)
        self.goal_r[env_ids] = r
        self.dwell_needed[env_ids] = torch.where(
            stand, torch.full_like(r, self.cfg.stand_time), torch.full_like(r, self.cfg.dwell_time)
        )
        a = torch.empty(k, device=self.device).uniform_(-math.pi, math.pi)
        root = self.robot.data.root_pos_w.torch[env_ids]
        self.goal_pos_w[env_ids, 0] = root[:, 0] + r * torch.cos(a)
        self.goal_pos_w[env_ids, 1] = root[:, 1] + r * torch.sin(a)
        self.goal_pos_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2]
        self.dwell[env_ids] = 0.0

    def _refresh_goal_b(self):
        root_pos = self.robot.data.root_pos_w.torch
        delta = self.goal_pos_w - root_pos
        delta[:, 2] = 0.0
        yaw_q = math_utils.yaw_quat(self.robot.data.root_quat_w.torch)
        self.goal_pos_b[:] = math_utils.quat_apply_inverse(yaw_q, delta)[:, :2]
        self.goal_dist[:] = torch.linalg.norm(self.goal_pos_b, dim=1)

    # -- CommandTerm hooks --------------------------------------------------

    def _resample_command(self, env_ids):
        # Called at reset and when the (huge) resampling timer runs out.
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._sample_goal(env_ids)
        self.just_reached[env_ids] = False
        self.is_standing_env[env_ids] = False

    def _update_command(self):
        self.just_reached[:] = False
        self._refresh_goal_b()
        inside = self.goal_dist < self.cfg.success_radius
        self.dwell = torch.where(inside, self.dwell + self._env.step_dt, torch.zeros_like(self.dwell))
        done = (self.dwell >= self.dwell_needed).nonzero().flatten()
        if len(done) > 0:
            self.just_reached[done] = True
            self.reached_r[done] = self.goal_r[done]
            self.metrics["successes"][done] += 1.0
            self._sample_goal(done)
            self._refresh_goal_b()

        # derived velocity: approach speed shrinks near the goal, zero inside the stop radius
        rng = self.cfg.ranges
        v_max = min(abs(rng.lin_vel_x[1]), self.cfg.max_speed)
        speed = torch.clamp(self.cfg.speed_gain * (self.goal_dist - self.cfg.stop_radius), 0.0, v_max)
        direction = self.goal_pos_b / self.goal_dist.clamp_min(1e-6).unsqueeze(1)
        self.vel_command_b[:, 0] = (speed * direction[:, 0]).clamp(*rng.lin_vel_x)
        self.vel_command_b[:, 1] = (speed * direction[:, 1]).clamp(*rng.lin_vel_y)
        heading_err = torch.atan2(self.goal_pos_b[:, 1], self.goal_pos_b[:, 0])
        wz = (self.cfg.heading_gain * heading_err).clamp(*rng.ang_vel_z)
        self.vel_command_b[:, 2] = torch.where(self.goal_dist > self.cfg.turn_radius, wz, torch.zeros_like(wz))

    def reset(self, env_ids=None) -> dict[str, float]:
        idx = slice(None) if env_ids is None else env_ids
        self.metrics["goal_dist"][idx] = self.goal_dist[idx]
        extras = super().reset(env_ids)
        self.dwell[idx] = 0.0
        self.just_reached[idx] = False
        self._refresh_goal_b()
        return extras

    # -- debug vis: faint red cube, faint green while the trunk is inside --

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not self.robot.is_initialized:
            return
        pos = self.goal_pos_w.clone()
        pos[:, 2] += 0.5 * self.cfg.goal_cube_size
        idx = torch.where(self.goal_dist < self.cfg.success_radius, GOAL_GREEN, GOAL_RED)
        self.goal_visualizer.visualize(translations=pos, marker_indices=idx)


def _goal_cube(rgb) -> sim_utils.CuboidCfg:
    s = 0.5
    return sim_utils.CuboidCfg(
        size=(s, s, s),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, opacity=0.3, roughness=0.6),
    )


@configclass
class GoalCarryCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GoalCarryCommand
    asset_name: str = "robot"
    # goals are only redrawn on success or reset
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    rel_standing_envs: float = 0.0
    heading_command: bool = False
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(0.0, 0.0)
    )

    # (min, max); max is the curriculum's starting point if goal_radius_curriculum runs
    goal_radius_range: tuple[float, float] = (0.5, 4.0)
    stand_prob: float = 0.2
    stand_time: float = 4.0
    success_radius: float = 0.25
    dwell_time: float = 1.0
    stop_radius: float = 0.10
    turn_radius: float = 0.50
    speed_gain: float = 1.0
    heading_gain: float = 1.5
    max_speed: float = 0.5

    goal_cube_size: float = 0.5
    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/carry_goal",
        markers={"red": _goal_cube((0.9, 0.1, 0.1)), "green": _goal_cube((0.1, 0.9, 0.2))},
    )
