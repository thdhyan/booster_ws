"""K1 box carry: walk to a ground goal holding a box welded between both forearm stubs.

Built on the velocity (P2) env (scene, obs, leg action); the carry pieces are
layered on top:

* scene: flat plane (no terrain curriculum), ``replicate_physics=False`` for the
  per-env box width/mass DR, and ``carry_box`` -- a visual-only cuboid held by two
  FixedJoints to the forearm stubs (carry_mdp.spawn_carry_box);
* actions: the existing 12-leg ``joint_pos`` plus an 8-arm ``arm_pos`` around the
  per-env hold pose; the decoupled actor (decoupled_policy) gives each its own MLP;
* command: ``base_velocity`` is the goal command (carry_command), which derives
  the walking velocity the gait rewards track and draws the faint red/green goal cube;
* rewards (simplified 2026-10-11): goal progress/success, hold quality, box level,
  weld integrity, fall penalty and 4 gait terms; stand + hold always pays, the goal
  terms are multiplied by hold quality; the goal radius grows only while falls are rare.

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
GATED_REWARDS = ("goal_progress", "goal_reached")
_VEL_REWARDS = RewardsCfg()


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
class K1CarryRewardsCfg:
    """Simplified set (2026-10-11, REWARDS_SIMPLE_PROPOSAL.md option A): goal progress,
    hold, do not fall, light regularisation, plus 4 gait terms because the legs train
    from scratch (not a residual over the velocity student)."""

    # 1. carry the box to the goal (gated by hold quality in __post_init__)
    goal_progress = RewTerm(func=carry_mdp.goal_progress_rate, weight=5.0)       # 5 per metre closed
    goal_reached = RewTerm(func=carry_mdp.goal_reached, weight=500.0, params={"full_radius": 1.0})  # x dt = +10
    # 2. box at the hold spot, trunk upright
    stand_hold = RewTerm(func=carry_mdp.stand_hold, weight=1.0, params=dict(_HOLD))
    # 3. do not fall / do not drop the box (x dt = -10); carry has no success termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-500.0)
    weld_error = RewTerm(func=carry_mdp.weld_error_l1, weight=-10.0)
    box_tilt = RewTerm(func=carry_mdp.box_tilt_l2, weight=-2.0)
    # light regularisation
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"])})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    # arms relative to the hold pose (default_joint_pos is rewritten per env)
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5, params={"asset_cfg": _ARM_CFG})
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )
    # minimum gait for legs trained from scratch (same terms/weights as the velocity env)
    feet_slide = _VEL_REWARDS.feet_slide
    feet_clearance = _VEL_REWARDS.feet_clearance
    feet_alternation = _VEL_REWARDS.feet_alternation
    undesired_contacts = _VEL_REWARDS.undesired_contacts


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
    # box mass U(1, hi) per reset, hi 2 -> 8 kg with the goal-radius curriculum progress
    carry_mass = EventTerm(func=carry_mdp.carry_mass_curriculum, mode="reset",
                           params={"mass_lo": CARRY_MASS_RANGE[0], "hi_start": 2.0,
                                   "hi_end": CARRY_MASS_RANGE[1]})


@configclass
class K1CarryCurriculumCfg(CurriculumCfg):
    terrain_levels = None   # flat plane
    velocity_range = None   # the goal command derives velocity; ranges stay fixed
    # tier 3: the goal radius opens 1 -> 10 m in 0.5 m steps while < 8 % of resets are falls
    # (the box mass ceiling follows it, see K1CarryEventCfg.carry_mass)
    goal_radius = CurrTerm(
        func=carry_mdp.goal_radius_curriculum,
        params={"start": 1.0, "step": 0.5, "target": 10.0, "fall_threshold": 0.08, "window": 0},
    )


@configclass
class K1CarryEnvCfg(K1VelocityRoughEnvCfg):
    scene: K1CarrySceneCfg = K1CarrySceneCfg(num_envs=4096, env_spacing=22.0)
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
        # goals reach 10 m from the start, so cells must be wider than 2 x 10 m
        self.scene.env_spacing = 22.0
        # 10 m at the 0.5 m/s command cap is 20 s of walking; leave time to turn and dwell
        self.episode_length_s = 30.0
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
        # play runs a trained policy: start at the trained max radius (and max mass), not the 1 m
        # curriculum start (a fresh play env re-runs the schedule from `start`)
        self.curriculum.goal_radius.params["start"] = self.curriculum.goal_radius.params["target"]
