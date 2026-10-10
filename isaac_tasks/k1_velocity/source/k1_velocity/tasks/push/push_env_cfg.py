"""P6 box-push envs (v3): frozen squat base + wrist IK, fixed-goal rewards.

Two ids share this cfg:
  Isaac-Push-Reach-K1-v0 : walk up + extend wrists to contact (heavy box, no
                           push rewards) -> warm-start for the push task.
  Isaac-Push-K1-v0       : full corner-goal pushing (box DR 0.7-1.5x size,
                           3-25 kg, friction 0.3-1.2; FIXED randomized goal:
                           distance 0.3 -> 1.5 m + yaw +-30 -> +-180 deg on
                           phased curricula).

Action (10) = [vx, vy, wz, H*] | left wrist EE delta (3) | right (3).
The 4-dim command slice feeds the FROZEN squat base teacher (TorchScript,
236-dim obs assembled in FrozenBaseVelocityAction - blocker 2a); the wrist
slices feed DifferentialIK position terms whose EE is the hand link. Obs
group is TeacherCfg = fully observable by design (mass, size, current
corners, FIXED goal pose + corners, goal-box delta).

v3 per the Phase-0 degenerate audit (TRAINING.md "P6 Phase-0 degenerate
audit"): fixed randomized goal (A1-A3), world-frame box velocity toward goal
(A4), two-scale proximity (A7), track_cmd cut to 0.25/0.10 (A8), sparse
success + failure accounting (A9-A11/A14), command-relative height terms
(A12), box tip/OOB terminations (A11/A13), spin penalty retired (A3).
"""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as env_mdp
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import push_mdp as mdp

#: Thickness of the per-env ground patch. Top face at z=0 (the trained standing
#: height), so the world plane sits one thickness lower as a backstop.
GROUND_PATCH_THICKNESS = 0.05
from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from k1_velocity.sim_backend import (
    apply_physics_backend,
    ensure_physx_gpu_capacity,
)
from k1_velocity.tasks.partial.mdp import randomize_arm_pose

# Frozen-base policy path. PUSH_BASE_POLICY overrides it (docker/dl runs; env
# vars are stripped by singularity --containall, so the zz-bw chain points the
# default models/k1_push_base.pt symlink at the wanted export instead).
_PUSH_BASE_POLICY = os.environ.get("PUSH_BASE_POLICY", "models/k1_push_base.pt")


@configclass
class K1PushSceneCfg(InteractiveSceneCfg):
    """K1 + one box; replicate_physics=False for per-env USD DR."""

    replicate_physics = False

    # Ground material MUST match the velocity/partial terrain the frozen base
    # trained on (static/dynamic 1.0, combine multiply). GroundPlaneCfg()
    # defaults to 0.5/0.5 default-combine — quantified env-side delta in the
    # A/B hunt (home@cmd=0 stands, P6@cmd=0 falls in ~17 steps; chain2 gate
    # DIAG_FAILED_BASE_STILL_FALLS done_rate=0.0425).
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -GROUND_PATCH_THICKNESS)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            )
        ),
    )
    # The WORLD ground is only a backstop now. Per-env friction lives on
    # `ground_patch`, whose TOP surface sits at z=0 -- the height the frozen base was
    # trained standing at -- with the world plane dropped 5 cm below so the patch is
    # unambiguously the contact surface rather than z-fighting with it.
    #
    # It must be dropped, not left coincident: two colliders sharing a plane make the
    # resolved contact normal depend on the solver, and the base is a frozen policy that
    # cannot compensate for a contact that jitters between episodes.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    # Per-env ground, so each cell can carry its own friction. Sized to the env cell
    # (see the env_spacing note in __post_init__) and static, but kept as a RigidObject
    # so it has a rigid view the standard material randomizer can write through.
    ground_patch = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ground_patch",
        spawn=sim_utils.CuboidCfg(
            size=(7.9, 7.9, GROUND_PATCH_THICKNESS),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -GROUND_PATCH_THICKNESS * 0.5)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(fixed=True),
    )
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),  # prototype; per-env scale 0.7-1.5x
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.0, 0.5)),
    )
    # v3: the frozen SQUAT base observes the privileged height scan - exact
    # copy of the velocity env's sensor (same robot = same
    # {ENV_REGEX_NS}/Robot/Geometry/Trunk prim, same 17x11 grid = 187 rays).
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 17x11 = 187 pts
        debug_vis=False,
        # The patch is listed FIRST and is the real contact surface. If it were missing
        # here the frozen base's 187-ray privileged obs would measure the backstop 5 cm
        # below the feet -- no crash, just a systematically wrong terrain height that the
        # base cannot compensate for, since it is frozen.
        mesh_prim_paths=["{ENV_REGEX_NS}/ground_patch", "/World/ground"],
    )


@configclass
class ObservationsCfg:
    @configclass
    class TeacherCfg(ObsGroup):
        """Fully observable (privileged by design): box mass/size/corners,
        FIXED goal pose & corners, goal-box delta, wrist targets, proprio."""

        box_state = ObsTerm(func=mdp.push_box_teacher)          # 68
        base_lin_vel = ObsTerm(func=mdp.vmdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.vmdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.vmdp.projected_gravity)
        wrist_targets = ObsTerm(
            func=mdp.vmdp.generated_commands, params={"command_name": "wrist_target"}
        )                                                        # 6
        arm_joint_pos = ObsTerm(
            func=mdp.vmdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )                                                        # 8
        arm_joint_vel = ObsTerm(
            func=mdp.vmdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )                                                        # 8
        actions = ObsTerm(func=mdp.vmdp.last_action)             # 10 (v3)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


@configclass
class K1PushActionsCfg:
    """10-dim (v3): [vx, vy, wz, H*] frozen-base command + two wrist IK terms.
    Legacy partial mode resolves the command slice to 3 (blocker 2a: the cmd
    slice grew 3 -> 4 when the squat base brought H*)."""

    base_velocity = mdp.FrozenBaseVelocityActionCfg(
        asset_name="robot",
        base_policy_path=_PUSH_BASE_POLICY,
    )
    wrist_left = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=mdp.K1_LEFT_ARM_JOINTS,
        body_name=mdp.LEFT_EE,
        scale=0.05,  # max 5 cm wrist delta per step
        controller=mdp.DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=True, ik_method="dls",
        ),
    )
    wrist_right = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=mdp.K1_RIGHT_ARM_JOINTS,
        body_name=mdp.RIGHT_EE,
        scale=0.05,
        controller=mdp.DifferentialIKControllerCfg(
            command_type="position", use_relative_mode=True, ik_method="dls",
        ),
    )


@configclass
class K1PushCommandsCfg:
    wrist_target = mdp.WristTargetCommandCfg()


@configclass
class K1PushRewardsCfg:
    """Corner-goal pushing rewards + reach shaping + regularization.

    SIGN CONVENTION (2026-09-28): the *_tracking / *_penalty funcs return the
    SIGNED quantity itself (corner/centroid/wrist return NEGATIVE error),
    so their weights must be POSITIVE to make the product a penalty. The
    original negative weights double-negated and the policies were literally
    PAID for keeping wrists/box far (0 % contact in v1 runs) - see HANDOFF
    "Reward sign inversion". v3 weights below carry the Phase-0 audit fixes
    (A8/A10/A12) on top of that sign correction.
    """

    # primary: corners -> FIXED goal corners (normalized), centroid, progress
    corner_goal_tracking = RewTerm(func=mdp.corner_goal_tracking, weight=1.0)
    centroid_goal_tracking = RewTerm(func=mdp.centroid_goal_tracking, weight=0.5)
    box_goal_progress = RewTerm(func=mdp.box_goal_progress, weight=2.0)
    box_vel_toward_goal = RewTerm(func=mdp.box_vel_toward_goal, weight=0.5)   # world-frame (A4)
    # A3: the goal owns its yaw now - rotation is part of the task, so the
    # spin penalty is retired (corners track orientation; func kept for A/B)
    box_spin_penalty = None
    # reach/contact shaping
    wrist_target_tracking = RewTerm(func=mdp.wrist_target_tracking, weight=0.3)
    wrist_box_proximity = RewTerm(func=mdp.wrist_box_proximity, weight=0.5)   # two-scale (A7)
    # commanded-velocity tracking (command = action's command slice)
    # A8: cut 0.5/0.25 -> 0.25/0.10 - v2's stand-at-zero was worth ~0.5/step
    # and beat every positive shaping term before the box ever moved
    track_cmd_lin_vel = RewTerm(func=mdp.track_cmd_lin_vel_exp, weight=0.25, params={"std": 0.5})
    track_cmd_ang_vel = RewTerm(func=mdp.track_cmd_ang_vel_exp, weight=0.1, params={"std": 0.5})
    # A12/E1: trunk height tracks the CURRENT command H*, not a fixed 0.57
    base_height_command = RewTerm(func=mdp.base_height_command, weight=1.0, params={"std": 0.05})
    # A10/A14: sparse success, paid once on the goal_reached firing step
    success_bonus = RewTerm(func=mdp.success_bonus, weight=50.0)
    # regularization
    flat_orientation_l2 = RewTerm(func=mdp.vmdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.vmdp.action_rate_l2, weight=-0.005)
    dof_torques_l2 = RewTerm(
        func=mdp.vmdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS)},
    )
    joint_pos_limits = RewTerm(
        func=mdp.vmdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=mdp.K1_LEFT_ARM_JOINTS + mdp.K1_RIGHT_ARM_JOINTS)},
    )
    # A9: failures only - excludes time_out AND success (vmdp.is_terminated
    # would have charged -200 to a successful episode)
    termination_penalty = RewTerm(func=mdp.failure_terminated, weight=-200.0)


@configclass
class K1PushReachRewardsCfg(K1PushRewardsCfg):
    """Reach stage: no push rewards; box is heavy so bumps do not slide it.
    Height/success/failure terms stay active: reaching the contact height is
    exactly the squat-to-box-height skill (wrist/height interplay, blocker 2a),
    and a heavy box can never cover the 0.3+ m goal distance, so the success
    machinery is inert here."""

    corner_goal_tracking = None
    centroid_goal_tracking = None
    box_goal_progress = None
    box_vel_toward_goal = None
    # func returns NEGATIVE distance -> positive weight = penalty for error
    wrist_target_tracking = RewTerm(func=mdp.wrist_target_tracking, weight=1.0)
    wrist_box_proximity = RewTerm(func=mdp.wrist_box_proximity, weight=1.0)
    track_cmd_lin_vel = RewTerm(func=mdp.track_cmd_lin_vel_exp, weight=0.25, params={"std": 0.5})
    track_cmd_ang_vel = RewTerm(func=mdp.track_cmd_ang_vel_exp, weight=0.1, params={"std": 0.5})


@configclass
class K1PushTerminationsCfg:
    time_out = DoneTerm(func=mdp.vmdp.time_out, time_out=True)
    # A12: command-relative floor H* - 0.05 replaces the fixed 0.35 root_height
    height_below_command = DoneTerm(func=mdp.height_below_command, params={"margin": 0.05})
    bad_orientation = DoneTerm(func=mdp.vmdp.bad_orientation, params={"limit_angle": 0.8})
    # A10/A14: success (updates st.success_hold, drives success_bonus)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"err_thresh": 0.08, "hold_s": 1.0})
    # A11/A13: failures counted as terminations (Episode_Termination/*)
    box_oob = DoneTerm(func=mdp.box_out_of_bounds, params={"radius": 1.9})
    box_tip = DoneTerm(func=mdp.box_tipped, params={"limit_deg": 45.0})


@configclass
class K1PushEventCfg:
    # pre-startup USD-level DR (this build has no "usd" event mode; prestartup
    # fires before sim start / PhysX parse)
    randomize_box_geometry = EventTerm(
        func=mdp.randomize_box_geometry,
        mode="prestartup",
        params={"scale_range": (0.7, 1.5), "mass_range": (3.0, 25.0)},
    )
    randomize_friction = EventTerm(
        func=mdp.vmdp.randomize_rigid_body_material,
        # NOT prestartup: the material impls need asset.root_view, which only
        # exists after sim play; startup fires right after play.
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
        },
    )
    # Per-env GROUND friction, so each cell has its own surface. Separate from the
    # robot-material randomizer above, which varies the robot's own friction and
    # therefore the pair -- two different knobs for two different halves of
    # (mu_robot, mu_ground).
    #
    # LOWER BOUND 0.7, NOT THE REFERENCE'S 0.3. push_env_cfg's ground comment records a
    # measured failure at the GroundPlaneCfg default of 0.5: the frozen base stood at
    # home@cmd=0 but fell in ~17 steps in this scene (chain2 DIAG_FAILED_BASE_STILL_FALLS,
    # done_rate=0.0425). The base is FROZEN, so it cannot adapt to a slicker floor -- it
    # just falls, and the push policy inherits that as an unrecoverable early
    # termination. Widening the range downward would buy robustness we cannot pay for
    # until the base is trainable again, so the floor is kept where the base is known to
    # stand. Exposed as cfg.ground_friction_range to widen deliberately.
    randomize_ground_friction = EventTerm(
        func=mdp.vmdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ground_patch"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.7, 1.2),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 16,
        },
    )
    green_alpha = EventTerm(func=mdp.apply_box_green_alpha, mode="startup")
    # per-episode (order matters: pose -> goal -> wrist targets)
    reset_scene = EventTerm(func=mdp.vmdp.reset_scene_to_default, mode="reset")
    reset_box = EventTerm(func=mdp.reset_box, mode="reset")      # spawn + FIXED goal
    reset_wrist_targets = EventTerm(func=mdp.reset_wrist_targets, mode="reset")
    # v3: advance_goal (interval integrator, A1/A2) deleted - the goal is
    # sampled once in reset_box and never moves.
    push_robot = EventTerm(
        func=mdp.vmdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class K1PushCurriculumCfg:
    goal_dist = CurrTerm(
        func=mdp.goal_dist_curriculum,
        params={"start_iter": 200, "end_iter": 1800, "d_max": 1.5},
    )
    # A3: goal yaw delta phased AFTER the distance walk (1800 -> 3000)
    goal_yaw = CurrTerm(
        func=mdp.goal_yaw_curriculum,
        params={"start_iter": 1800, "end_iter": 3000, "yaw0_deg": 30.0, "yaw_max_deg": 180.0},
    )


@configclass
class K1PushEnvCfg(ManagerBasedRLEnvCfg):
    """Box-push env (push stage)."""

    scene: K1PushSceneCfg = K1PushSceneCfg(num_envs=256, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: K1PushActionsCfg = K1PushActionsCfg()
    commands: K1PushCommandsCfg = K1PushCommandsCfg()
    rewards: K1PushRewardsCfg = K1PushRewardsCfg()
    terminations: K1PushTerminationsCfg = K1PushTerminationsCfg()
    events: K1PushEventCfg = K1PushEventCfg()
    curriculum: K1PushCurriculumCfg = K1PushCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.use_newton_actuators = False
        apply_physics_backend(self)
        # GPU broadphase pair buffer must scale with env count or PhysX
        # silently drops contacts (see ensure_physx_gpu_capacity).
        ensure_physx_gpu_capacity(self)
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        # ENV PITCH 2.5 -> 8.0 m. The old value let neighbouring boxes interpenetrate.
        #
        # Arithmetic: reset_box spawns the box in an annulus r in [0.9, 1.4] m with FULL
        # direction, and randomize_box_geometry scales the 1.0 m prototype 0.7-1.5x, so a
        # box reaches 1.4 m from its env origin with a half-extent up to 0.75 m. With env
        # origins 2.5 m apart, two boxes on the line between adjacent envs land
        # 2.5 - 2.8 = -0.3 m apart at their CENTRES: overlapping by 0.3 m plus 1.5 m of
        # combined extent. That corrupts the task twice -- the box being pushed is merged
        # with a neighbour's, and every corner/goal reward is measured against geometry
        # another env is also pushing.
        #
        # 8.0 m clears it with margin (8 - 2.8 = 5.2 m between the furthest centres) and
        # matches the Push-Things reference, which uses the same 8 m for the same reason.
        # The cost is stage memory, not physics: the gap between patches is empty air.
        self.scene.env_spacing = 8.0
        # See randomize_ground_friction: the lower bound is held at 0.7 because a slicker
        # floor was measured to drop the FROZEN base. Widen deliberately, not by accident.
        self.ground_friction_range = (0.7, 1.2)
        self.scene.ground_patch.spawn.size = (
            self.scene.env_spacing - 0.1,
            self.scene.env_spacing - 0.1,
            GROUND_PATCH_THICKNESS,
        )
        # frozen-base obs parity: the height scanner must tick once per
        # control step, exactly like the velocity env the teacher trained in
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # Diag A/B (arm-hold + reset-semantics hunt), PUSH_DIAG_STIFF_ARMS=1:
        # replicate the partial (home) env's reset/hold semantics, which the
        # frozen base trained against (its step-1 action is identical to
        # home's — the first cmd snaps hip_yaw ~0.9 rad in BOTH envs, yet
        # only P6 tips):
        #   * drop the two wrist IK terms — stock relative-mode IK rebases
        #     ee_pos_des = current EE pose + cmd EVERY control step
        #     (DifferentialIKController.set_command), so zero wrist actions
        #     leave the arms with no hold stiffness at all;
        #   * pin arm pose AND persistent PD targets via randomize_arm_pose
        #     (the exact partial-env function). Dropping the IK terms alone
        #     is NOT enough: nothing writes the target buffer, it stays at
        #     stale/zero and the arms ran away to joint-zero at 6+ rad/s
        #     (done_rate only 0.0434 -> 0.0294);
        #   * scale leg joints U(0.5,1.5) like velocity reset_robot_joints —
        #     home never resets to the exact default pose, P6 did.
        # Baseline pass-B done_rate 0.0434; friction parity already refuted.
        # Diag-only: default runs unchanged.
        if os.environ.get("PUSH_DIAG_STIFF_ARMS", "") == "1":
            self.actions.wrist_left = None
            self.actions.wrist_right = None
            # Appended AFTER the existing reset terms (config field order =
            # event application order) so the sampled arm pose survives
            # reset_scene/reset_box/reset_robot_joints.
            self.events.reset_robot_joints = EventTerm(
                func=env_mdp.reset_joints_by_scale,
                mode="reset",
                params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
            )
            self.events.arm_pose_random = EventTerm(
                func=randomize_arm_pose,
                mode="reset",
                params={
                    "asset_cfg":
                        SceneEntityCfg("robot",
                                       joint_names=list(mdp.K1_LEFT_ARM_JOINTS)
                                       + list(mdp.K1_RIGHT_ARM_JOINTS)),
                    "offset_range": (-1.0, 1.0),
                    "curriculum_scale": 1.0,
                },
            )


@configclass
class K1PushReachEnvCfg(K1PushEnvCfg):
    """Reach stage: same world, heavy box (12-25 kg), reach-only rewards."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = K1PushReachRewardsCfg()
        self.events.randomize_box_geometry.params["mass_range"] = (12.0, 25.0)
