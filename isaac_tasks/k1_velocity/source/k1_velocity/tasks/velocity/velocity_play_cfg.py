"""K1 velocity play (eval) config — inherits rough env, disables curriculum/noise."""
from isaaclab.utils.configclass import configclass
from .velocity_env_cfg import K1VelocityRoughEnvCfg

@configclass
class K1VelocityRoughPlayEnvCfg(K1VelocityRoughEnvCfg):
    """Play config: 50 envs, flat terrain, no curriculum, no noise."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable terrain curriculum
        self.curriculum = None
        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Disable noise
        self.observations.policy.enable_corruption = False
        # Disable external pushes
        self.events.push_robot = None
        self.events.add_base_mass = None
        # Fixed command for play
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
