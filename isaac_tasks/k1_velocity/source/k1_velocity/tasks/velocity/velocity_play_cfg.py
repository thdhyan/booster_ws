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
        # Fixed, directly expressed walking command for recordings/evaluation.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class K1VelocityRoughTeacherPlayEnvCfg(K1VelocityRoughPlayEnvCfg):
    """Play config for a TEACHER checkpoint (the 237-dim privileged policy).

    Added 2026-10-02. ``K1VelocityRoughPlayEnvCfg`` keeps the 50-dim student
    observation group, so loading a teacher checkpoint into it fails at
    ``runner.load`` with a shape mismatch::

        size mismatch for mlp.0.weight: copying a param with shape
        torch.Size([512, 237]) from checkpoint, the shape in current model is
        torch.Size([512, 50])

    The teacher consumes the privileged group (proprioception + a 187-point
    height scan), so the network width differs and the two are not
    interchangeable. This config promotes ``teacher`` to the ``policy`` group so
    the actor width matches a teacher checkpoint, and disables the training-only
    noise the same way the student play config does.

    The height scan is privileged, so this is an EVALUATION harness only -- it is
    not deployable, because no real robot carries a 187-ray terrain scan. A
    deployable policy still has to come from the distilled 50-dim student.
    """

    def __post_init__(self):
        super().__post_init__()
        # Promote the privileged group into the observation the actor reads.
        self.observations.policy = self.observations.teacher
        self.observations.teacher.enable_corruption = False
        # Rewards and curriculum are training-only; a play cfg should not carry
        # them, and the reward ramps would otherwise write weights on every step.
        self.rewards = None
        self.curriculum = None
