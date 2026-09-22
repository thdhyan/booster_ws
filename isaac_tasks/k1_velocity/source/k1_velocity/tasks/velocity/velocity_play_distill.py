# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Modifications for Booster K1 by thdhyan.
# SPDX-License-Identifier: Apache-2.0

"""K1 velocity distillation play (eval) config."""

from isaaclab.utils import configclass
from .velocity_env_distill import K1VelocityDistillEnvCfg

@configclass
class K1VelocityDistillPlayEnvCfg(K1VelocityDistillEnvCfg):
    """Play config: 1 env, flat terrain, no curriculum, no noise, no contact rewards."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # Disable terrain curriculum
        self.curriculum = None
        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Disable noise
        self.observations.policy.enable_corruption = False
        self.observations.teacher.enable_corruption = False
        # Disable external pushes
        self.events.push_robot = None
        self.events.add_base_mass = None
        # Disable contact-based rewards (require contact sensor setup)
        self.rewards.feet_air_time = None
        self.rewards.feet_slide = None
        # Fixed command for play
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
