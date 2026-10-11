"""Multi-robot push (M1, PLAN_MAPPO_TEAM_SOCCER.md section 6): R K1s, ONE shared box.

Every robot is a full copy of the single-robot push stack: its own frozen squat base
(``base_velocity_k``), wrist IK terms (``wrist_left_k`` / ``wrist_right_k``), height
scanner (``height_scanner_k``), wrist-target command (``wrist_target_k``) and a 109-dim
``teacher_k`` obs group laid out exactly like the single-robot ``teacher`` group. So
the trained single-robot push policy drives each robot unchanged: stack the R groups
into (R*N, 109), run the actor once, split the actions back to (N, R*10).

Layout per cell: the robots stand side by side at x=0 (y = +-ROBOT_Y), facing +x; the
box spawns ahead (r 1.1-1.3 m, small bearing and yaw jitter) and the goal is sampled
in a forward cone, so the robots push it together. Each robot's wrist targets follow
its own lateral position on the near face (``follow_robot``), so they take separate
slots instead of both reaching for the face centre. The box is scaled 0.85-0.95 m
(single-robot: 0.67-0.76 m) so two robots fit behind one face.

Robots are hard-wired to R=2 because each scene entity, action term and obs group is
a config field.
"""
from __future__ import annotations

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import push_env_cfg as base
from . import push_mdp as mdp

NUM_ROBOTS = 2
ROBOT_Y = 0.5          # lateral offset of each robot from the cell centre line (m)
ROBOTS = [f"robot_{k}" for k in range(NUM_ROBOTS)]
ARM_JOINTS = mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS


def _robot_cfg(k: int):
    y = ROBOT_Y * (1.0 - 2.0 * k / max(NUM_ROBOTS - 1, 1))   # +0.5, -0.5
    cfg = base.BOOSTER_K1_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/Robot_{k}")
    cfg.init_state = cfg.init_state.replace(pos=(0.0, y, cfg.init_state.pos[2]))
    return cfg


def _scanner_cfg(k: int):
    return base.K1PushSceneCfg().height_scanner.replace(
        prim_path=f"{{ENV_REGEX_NS}}/Robot_{k}/Geometry/Trunk"
    )


@configclass
class K1PushMultiSceneCfg(base.K1PushSceneCfg):
    robot = None
    height_scanner = None
    robot_0 = _robot_cfg(0)
    height_scanner_0 = _scanner_cfg(0)
    robot_1 = _robot_cfg(1)
    height_scanner_1 = _scanner_cfg(1)


def _teacher_group(k: int):
    name = f"robot_{k}"
    a = SceneEntityCfg(name)

    @configclass
    class TeacherK(ObsGroup):
        """Single-robot TeacherCfg, term for term (noise included), bound to robot_k (109 dims)."""

        box_state = ObsTerm(func=mdp.push_box_teacher, params={"asset_name": name})
        base_lin_vel = ObsTerm(func=mdp.vmdp.base_lin_vel, params={"asset_cfg": a},
                               noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.vmdp.base_ang_vel, params={"asset_cfg": a},
                               noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.vmdp.projected_gravity, params={"asset_cfg": a})
        wrist_targets = ObsTerm(func=mdp.vmdp.generated_commands,
                                params={"command_name": f"wrist_target_{k}"})
        arm_joint_pos = ObsTerm(func=mdp.vmdp.joint_pos_rel,
                                params={"asset_cfg": SceneEntityCfg(name, joint_names=ARM_JOINTS)},
                                noise=Unoise(n_min=-0.01, n_max=0.01))
        arm_joint_vel = ObsTerm(func=mdp.vmdp.joint_vel_rel,
                                params={"asset_cfg": SceneEntityCfg(name, joint_names=ARM_JOINTS)},
                                noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action_of, params={
            "action_names": [f"base_velocity_{k}", f"wrist_left_{k}", f"wrist_right_{k}"]})

        def __post_init__(self):
            self.enable_corruption = True      # training; the Play cfg turns it off
            self.concatenate_terms = True

    return TeacherK()


@configclass
class K1PushMultiObservationsCfg:
    teacher_0 = _teacher_group(0)
    teacher_1 = _teacher_group(1)


def _base_action(k: int):
    return mdp.FrozenBaseVelocityActionCfg(
        asset_name=f"robot_{k}",
        base_policy_path=base._PUSH_BASE_POLICY,
        height_scanner_name=f"height_scanner_{k}",
    )


def _wrist_action(k: int, side: str):
    single = getattr(base.K1PushActionsCfg(), f"wrist_{side}")
    return single.replace(asset_name=f"robot_{k}")


@configclass
class K1PushMultiActionsCfg:
    """Per robot, in the single-robot order: base 4 | left wrist 3 | right wrist 3."""

    base_velocity_0 = _base_action(0)
    wrist_left_0 = _wrist_action(0, "left")
    wrist_right_0 = _wrist_action(0, "right")
    base_velocity_1 = _base_action(1)
    wrist_left_1 = _wrist_action(1, "left")
    wrist_right_1 = _wrist_action(1, "right")


@configclass
class K1PushMultiCommandsCfg:
    wrist_target_0 = mdp.WristTargetCommandCfg()
    wrist_target_1 = mdp.WristTargetCommandCfg()


@configclass
class K1PushMultiRewardsCfg:
    """Team terms (box -> goal, frame-invariant, evaluated once) + per-robot terms.

    The team terms take robot_0's frame only because the corner helpers need a frame;
    distances are the same in any frame. A per-agent split for MAPPO is M2's job.
    """

    corner_goal_tracking = RewTerm(func=mdp.corner_goal_tracking, weight=1.0, params={"asset_name": "robot_0"})
    centroid_goal_tracking = RewTerm(func=mdp.centroid_goal_tracking, weight=0.5, params={"asset_name": "robot_0"})
    box_goal_progress = RewTerm(func=mdp.box_goal_progress, weight=2.0, params={"asset_name": "robot_0"})
    box_vel_toward_goal = RewTerm(func=mdp.box_vel_toward_goal, weight=0.5)
    success_bonus = RewTerm(func=mdp.success_bonus, weight=50.0)
    wrist_box_proximity_0 = RewTerm(func=mdp.wrist_box_proximity, weight=0.25, params={"asset_name": "robot_0"})
    wrist_box_proximity_1 = RewTerm(func=mdp.wrist_box_proximity, weight=0.25, params={"asset_name": "robot_1"})
    base_height_command_0 = RewTerm(func=mdp.base_height_command, weight=0.5, params={"asset_name": "robot_0"})
    base_height_command_1 = RewTerm(func=mdp.base_height_command, weight=0.5, params={"asset_name": "robot_1"})
    termination_penalty = RewTerm(func=mdp.failure_terminated, weight=-200.0)


@configclass
class K1PushMultiTerminationsCfg:
    time_out = DoneTerm(func=mdp.vmdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"err_thresh": 0.08, "hold_s": 1.0})
    box_oob = DoneTerm(func=mdp.box_out_of_bounds, params={"radius": 3.0})
    box_tip = DoneTerm(func=mdp.box_tipped, params={"limit_deg": 45.0})
    height_below_command_0 = DoneTerm(func=mdp.height_below_command, params={"margin": 0.05, "asset_name": "robot_0"})
    height_below_command_1 = DoneTerm(func=mdp.height_below_command, params={"margin": 0.05, "asset_name": "robot_1"})
    bad_orientation_0 = DoneTerm(func=mdp.vmdp.bad_orientation,
                                 params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot_0")})
    bad_orientation_1 = DoneTerm(func=mdp.vmdp.bad_orientation,
                                 params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot_1")})


@configclass
class K1PushMultiEventCfg(base.K1PushEventCfg):
    # wider box so two robots fit behind one face; single-robot mass range
    randomize_box_geometry = EventTerm(
        func=mdp.randomize_box_geometry, mode="prestartup",
        params={"scale_range": (0.85, 0.95), "mass_range": (3.0, 10.0)},
    )
    # order matters: scene -> box + goal -> each robot's wrist targets
    reset_box = EventTerm(func=mdp.reset_box, mode="reset", params={
        "r_range": (1.1, 1.3),
        "phi_range": (-0.1, 0.1),
        "yaw_range": (-0.15, 0.15),
        "goal_psi_range": (-0.6, 0.6),        # forward cone: push away from the robots
        "workspace_r": 2.6,
    })
    reset_wrist_targets = None
    reset_wrist_targets_0 = EventTerm(func=mdp.reset_wrist_targets, mode="reset", params={
        "asset_name": "robot_0", "command_name": "wrist_target_0", "follow_robot": True})
    reset_wrist_targets_1 = EventTerm(func=mdp.reset_wrist_targets, mode="reset", params={
        "asset_name": "robot_1", "command_name": "wrist_target_1", "follow_robot": True})
    push_robot = None


@configclass
class K1PushMultiEnvCfg(base.K1PushEnvCfg):
    scene: K1PushMultiSceneCfg = K1PushMultiSceneCfg(num_envs=256, env_spacing=8.0)
    observations: K1PushMultiObservationsCfg = K1PushMultiObservationsCfg()
    actions: K1PushMultiActionsCfg = K1PushMultiActionsCfg()
    commands: K1PushMultiCommandsCfg = K1PushMultiCommandsCfg()
    rewards: K1PushMultiRewardsCfg = K1PushMultiRewardsCfg()
    terminations: K1PushMultiTerminationsCfg = K1PushMultiTerminationsCfg()
    events: K1PushMultiEventCfg = K1PushMultiEventCfg()

    def __post_init__(self):
        super().__post_init__()
        for k in range(NUM_ROBOTS):
            getattr(self.scene, f"height_scanner_{k}").update_period = self.decimation * self.sim.dt
        # play the trained goal range from the start (no curriculum here)
        self.curriculum = None


    # train.py wraps the env in AgentsInBatchWrapper(R) when this is > 0
    agents_in_batch: int = NUM_ROBOTS


@configclass
class K1PushMultiTrainEnvCfg(K1PushMultiEnvCfg):
    """M2: train the shared push policy with both robots in the loop (warm-start p6).

    The box is heavier than the single-robot range (6-16 kg vs 3-10) so the team has
    a reason to push together; corruption stays on as in single-robot training.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.randomize_box_geometry.params["mass_range"] = (6.0, 16.0)


@configclass
class K1PushMultiPlayEnvCfg(K1PushMultiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        for k in range(NUM_ROBOTS):
            getattr(self.observations, f"teacher_{k}").enable_corruption = False


@configclass
class K1PushMultiTrainPlayEnvCfg(K1PushMultiTrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        for k in range(NUM_ROBOTS):
            getattr(self.observations, f"teacher_{k}").enable_corruption = False
