"""K1 velocity play (eval) config — inherits rough env, disables curriculum/noise."""
from isaaclab.utils.configclass import configclass
from .velocity_env_cfg import K1VelocityRoughEnvCfg
from .velocity_env_cfg import ObservationsCfg as ObsCfg

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
        # ...which pins the unused axes to zero, so the weighted tracking ramps
        # have no span to climb. Without this, any --cmd X 0 0 recording dies with
        # "weighted tracking needs hi > lo" on the first reward step, because the
        # yaw range upper bound is 0. See _weight_from_magnitude.
        for _term in (self.rewards.track_lin_vel_xy_exp,
                      self.rewards.track_ang_vel_z_exp):
            if _term is not None:
                _term.params["require_span"] = False


@configclass
class K1VelocityTeacherOnlyObsCfg:
    """Expose the privileged teacher group under BOTH names, as independent objects.

    Two consumers need the same 237 terms under different names:

    * ``play_record.py`` / ``play_keyboard_fixed.py`` read ``obs["policy"]``;
    * the teacher runner cfg declares ``obs_groups = {"actor": ["teacher", ...]}``
      and raises ``Observation 'teacher' ... not found`` if the env does not
      publish that name.

    They must be **separate instances**, not one object aliased into two slots.
    Aliasing makes Isaac Lab resolve the same ``SceneEntityCfg`` twice, and the
    second pass sees ``joint_ids`` already written by the first::

        ValueError: Both 'joint_names' and 'joint_ids' are specified, and are
        not consistent.

    Two independent ``TeacherCfg()`` instances each own their own
    ``SceneEntityCfg``, so each resolves cleanly. It costs one extra forward pass
    over the privileged terms, which is irrelevant for a play run.
    """

    @configclass
    class PolicyCfg(ObsCfg.TeacherCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsCfg.TeacherCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


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
    interchangeable.

    EVALUATION ONLY. The height scan is privileged, so this policy is not
    deployable: no real robot carries a 187-ray terrain scan. A deployable policy
    has to come from the distilled 50-dim student.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations = K1VelocityTeacherOnlyObsCfg()
        # Rewards and curriculum are training-only, and the reward ramps would
        # otherwise rewrite weights on every step of a play run.
        self.rewards = None
        self.curriculum = None
