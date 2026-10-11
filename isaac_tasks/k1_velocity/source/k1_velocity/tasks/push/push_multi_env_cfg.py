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
    CurriculumTermCfg as CurrTerm,
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
    """Simplified set (same three ideas as push_simple_env_cfg), team + per-robot.

    Team terms (no suffix) go to both agents; ``_k`` terms go to robot_k only
    (agents_in_batch.RewardSplit), so a robot that falls is charged alone.
    The corner helpers need a frame; distances are frame-invariant, robot_0's is used.
    """

    # box closer to goal (team)
    box_progress = RewTerm(func=mdp.box_goal_progress_rate, weight=5.0, params={"asset_name": "robot_0"})
    box_near_goal = RewTerm(func=mdp.box_goal_near, weight=2.0, params={"std": 0.5})
    success_bonus = RewTerm(func=mdp.success_bonus, weight=500.0)
    box_failed = RewTerm(func=mdp.term_fired, weight=-500.0, params={"term_names": ["box_oob", "box_tip"]})
    # hands to the box (per robot)
    hands_to_box_0 = RewTerm(func=mdp.wrist_box_proximity, weight=0.5, params={"asset_name": "robot_0"})
    hands_to_box_1 = RewTerm(func=mdp.wrist_box_proximity, weight=0.5, params={"asset_name": "robot_1"})
    # do not fall (per robot)
    fallen_0 = RewTerm(func=mdp.term_fired, weight=-500.0, params={"term_names": ["fallen_0"]})
    fallen_1 = RewTerm(func=mdp.term_fired, weight=-500.0, params={"term_names": ["fallen_1"]})
    flat_orientation_0 = RewTerm(func=mdp.vmdp.flat_orientation_l2, weight=-1.0,
                                 params={"asset_cfg": SceneEntityCfg("robot_0")})
    flat_orientation_1 = RewTerm(func=mdp.vmdp.flat_orientation_l2, weight=-1.0,
                                 params={"asset_cfg": SceneEntityCfg("robot_1")})


@configclass
class K1PushMultiTerminationsCfg:
    time_out = DoneTerm(func=mdp.vmdp.time_out, time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached_centroid, params={"err_thresh": 0.15, "hold_s": 1.0})
    box_oob = DoneTerm(func=mdp.box_out_of_bounds, params={"radius": 3.6})
    box_tip = DoneTerm(func=mdp.box_tipped, params={"limit_deg": 45.0})
    # Real falls only (z < 0.30 m or tilt > 0.8 rad). height_below_command (z < H* - 5 cm)
    # ended 98.7 % of MAPPO episodes after ~8 steps (robot_1, 2026-10-11 run).
    fallen_0 = DoneTerm(func=mdp.robot_fallen, params={"asset_name": "robot_0"})
    fallen_1 = DoneTerm(func=mdp.robot_fallen, params={"asset_name": "robot_1"})


@configclass
class K1PushMultiCurriculumCfg:
    box_dist = CurrTerm(func=mdp.box_dist_curriculum, params={
        "start_iter": 200, "end_iter": 1800, "r0": (0.85, 0.95), "r1": (1.1, 1.4)})
    goal_dist = CurrTerm(func=mdp.goal_dist_curriculum, params={
        "start_iter": 200, "end_iter": 1800, "d_max": 1.8})


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
        "workspace_r": 3.2,
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
    curriculum: K1PushMultiCurriculumCfg = K1PushMultiCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        for k in range(NUM_ROBOTS):
            getattr(self.scene, f"height_scanner_{k}").update_period = self.decimation * self.sim.dt


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


def _play(cfg) -> None:
    """Small scene, no obs noise, curricula pinned at their final values."""
    cfg.scene.num_envs = 4
    for k in range(NUM_ROBOTS):
        getattr(cfg.observations, f"teacher_{k}").enable_corruption = False
    for term in (cfg.curriculum.box_dist, cfg.curriculum.goal_dist):
        term.params["start_iter"], term.params["end_iter"] = -2, -1


@configclass
class K1PushMultiPlayEnvCfg(K1PushMultiEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class K1PushMultiTrainPlayEnvCfg(K1PushMultiTrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
