"""K1 box carry: walk to a ground goal holding a box welded between both forearm stubs.

Built on the velocity (P2) env so the lower body keeps the full gait reward
stack; the carry pieces are layered on top:

* scene: flat plane (no terrain curriculum), ``replicate_physics=False`` for the
  per-env box width/mass DR, and ``carry_box`` -- a visual-only cuboid held by two
  FixedJoints to the forearm stubs (carry_mdp.spawn_carry_box);
* actions: the existing 12-leg ``joint_pos`` plus an 8-arm ``arm_pos`` around the
  per-env hold pose; the decoupled actor (decoupled_policy) gives each its own MLP;
* command: ``base_velocity`` is the goal command (carry_command), which derives
  the walking velocity the gait rewards track and draws the faint red/green goal cube;
* rewards: velocity stack + goal progress/success, box level, weld integrity,
  in tiers (carry_tiers): stand + hold always pays; the walking/goal terms are
  multiplied by hold quality; the goal radius grows only while falls are rare.

Arm gains: the K1 R14 arm actuator derives ~4 Nm/rad from its 10 Hz natural
frequency, which cannot hold even a 1 kg box against gravity at the hold pose
(the arms sag until the weld loop drags the trunk forward). 40 Nm/rad keeps the
static sag of an 8 kg box to a few degrees while staying inside the 14 Nm limit.
"""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from k1_velocity.tasks.velocity.velocity_env_cfg import (
    K1_ARTICULATION_CFG,
    CurriculumCfg,
    EventCfg,
    K1RoughSceneCfg,
    K1VelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

from . import carry_mdp
from .carry_command import GoalCarryCommandCfg

ARM_STIFFNESS = 40.0
ARM_DAMPING = 2.0
CARRY_WIDTH_RANGE = (0.25, 0.40)
CARRY_MASS_RANGE = (1.0, 8.0)


def _carry_robot_cfg():
    cfg = copy.deepcopy(K1_ARTICULATION_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actuators["arms"].stiffness = ARM_STIFFNESS
    cfg.actuators["arms"].damping = ARM_DAMPING
    # start in the mid-width hold pose; reset_carry then writes each env's own
    q_l, q_r = carry_mdp.hold_pose(0.325)
    joint_pos = dict(cfg.init_state.joint_pos)
    joint_pos.update({n: float(v) for n, v in zip(carry_mdp.ARM_JOINTS, list(q_l) + list(q_r))})
    cfg.init_state.joint_pos = joint_pos
    return cfg


@configclass
class K1CarrySceneCfg(K1RoughSceneCfg):
    replicate_physics = False

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot = _carry_robot_cfg()
    # 1 m prototype; spawn_carry_box scales its geometry per env to 0.18 x width x 0.18.
    # No collider: the box only has to hang off the welds, and a collider between the
    # forearms would fight the hold pose.
    carry_box = RigidObjectCfg(
        prim_path=carry_mdp.BOX_PRIM,
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.55, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.75)),
    )


_ARM_CFG = SceneEntityCfg("robot", joint_names=carry_mdp.ARM_JOINTS, preserve_order=True)
_HOLD = {"pos_std": 0.08, "tilt_std": 0.3}
# tier 2: these pay only in proportion to hold quality (wrapped in __post_init__)
GATED_REWARDS = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "goal_coarse", "goal_fine", "goal_reached")


@configclass
class K1CarryObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        goal = ObsTerm(func=carry_mdp.goal_command_obs)
        arm_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _ARM_CFG},
                                noise=Unoise(n_min=-0.01, n_max=0.01))
        arm_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _ARM_CFG},
                                noise=Unoise(n_min=-1.5, n_max=1.5))

    @configclass
    class TeacherCfg(ObservationsCfg.TeacherCfg):
        goal = ObsTerm(func=carry_mdp.goal_command_obs)
        arm_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _ARM_CFG})
        arm_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _ARM_CFG})
        box_params = ObsTerm(func=carry_mdp.box_params_obs)

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class K1CarryActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
            "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
            "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
            "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
        ],
        scale=1.0,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )
    # offset = per-env hold pose (reset_carry overwrites _offset once)
    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=carry_mdp.ARM_JOINTS,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )


@configclass
class K1CarryCommandsCfg:
    base_velocity = GoalCarryCommandCfg(debug_vis=False)


@configclass
class K1CarryRewardsCfg(RewardsCfg):
    goal_coarse = RewTerm(func=carry_mdp.goal_distance_tanh, weight=2.0, params={"std": 1.5})
    goal_fine = RewTerm(func=carry_mdp.goal_distance_tanh, weight=2.0, params={"std": 0.3})
    goal_reached = RewTerm(func=carry_mdp.goal_reached, weight=20.0, params={"full_radius": 1.0})
    # tier 1: always paid -- upright with the box in front of the trunk, more so while still
    stand_hold = RewTerm(func=carry_mdp.stand_hold, weight=1.0, params=dict(_HOLD))
    stand_still = RewTerm(func=carry_mdp.stand_still, weight=2.0,
                          params={"cmd_eps": 0.05, "sigma": 0.2, **_HOLD})
    box_tilt = RewTerm(func=carry_mdp.box_tilt_l2, weight=-2.0)
    weld_error = RewTerm(func=carry_mdp.weld_error_l1, weight=-10.0)
    # arms relative to the hold pose (default_joint_pos is rewritten per env)
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5, params={"asset_cfg": _ARM_CFG})
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )


@configclass
class K1CarryTerminationsCfg(TerminationsCfg):
    weld_broken = DoneTerm(func=carry_mdp.weld_broken, params={"threshold": 0.10})


@configclass
class K1CarryEventCfg(EventCfg):
    # prestartup: USD writes (box scale, mass, welds) before PhysX parses the stage
    spawn_carry_box = EventTerm(
        func=carry_mdp.spawn_carry_box,
        mode="prestartup",
        params={"width_range": CARRY_WIDTH_RANGE, "mass_range": CARRY_MASS_RANGE},
    )
    # after reset_scene / reset_robot_joints (field order = application order)
    reset_carry = EventTerm(func=carry_mdp.reset_carry, mode="reset")


@configclass
class K1CarryCurriculumCfg(CurriculumCfg):
    terrain_levels = None   # flat plane
    velocity_range = None   # the goal command derives velocity; ranges stay fixed
    # tier 3: the goal radius opens 1 -> 4 m in 0.25 m steps while < 8 % of resets are falls
    goal_radius = CurrTerm(
        func=carry_mdp.goal_radius_curriculum,
        params={"start": 1.0, "step": 0.25, "target": 4.0, "fall_threshold": 0.08, "window": 0},
    )


@configclass
class K1CarryEnvCfg(K1VelocityRoughEnvCfg):
    scene: K1CarrySceneCfg = K1CarrySceneCfg(num_envs=4096, env_spacing=10.0)
    observations: K1CarryObservationsCfg = K1CarryObservationsCfg()
    actions: K1CarryActionsCfg = K1CarryActionsCfg()
    commands: K1CarryCommandsCfg = K1CarryCommandsCfg()
    rewards: K1CarryRewardsCfg = K1CarryRewardsCfg()
    terminations: K1CarryTerminationsCfg = K1CarryTerminationsCfg()
    events: K1CarryEventCfg = K1CarryEventCfg()
    curriculum: K1CarryCurriculumCfg = K1CarryCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # flat and blind: no height scan anywhere (set after super, which configures it)
        self.scene.height_scanner = None
        self.observations.teacher.height_scan = None
        self.rewards.base_height.params["sensor_cfg"] = None   # plane: ground is z=0
        # goals reach 4 m from the start, so cells must be wider than 2 x 4 m
        self.scene.env_spacing = 10.0
        for name in GATED_REWARDS:
            term = getattr(self.rewards, name)
            term.params = {
                "inner": f"{term.func.__module__}:{term.func.__name__}",
                "inner_params": dict(term.params),
                **_HOLD,
            }
            term.func = carry_mdp.gated


@configclass
class K1CarryEnvCfg_PLAY(K1CarryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.base_velocity.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
