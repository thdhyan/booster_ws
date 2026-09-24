"""K1 partial-control play (eval) config — inherits the training env, disables curriculum/noise."""
from isaaclab.utils.configclass import configclass

from .partial_env_cfg import K1PartialCtrlEnvCfg


@configclass
class K1PartialCtrlPlayEnvCfg(K1PartialCtrlEnvCfg):
    """Play config: 50 envs, flat terrain, no curriculum, full-range random arms."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable terrain curriculum (and the arm ramp — see below)
        self.curriculum = None
        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Disable noise
        self.observations.policy.enable_corruption = False
        # Disable external pushes
        self.events.push_robot = None
        self.events.add_base_mass = None
        # Full-range arm randomization for eval: with the curriculum term gone,
        # nothing would update the scale, so pin it at the trained end value.
        self.events.arm_pose_random.params["curriculum_scale"] = 1.0
        # Mid-episode arm deltas: OFF (0.0) — Run-10 checkpoints were trained
        # without the arm_delta_change event. When playing checkpoints trained
        # with the arm_delta curriculum (run 11+), pin this to 1.0 instead.
        self.events.arm_delta_change.params["curriculum_scale"] = 0.0
        # Fixed command for play
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
