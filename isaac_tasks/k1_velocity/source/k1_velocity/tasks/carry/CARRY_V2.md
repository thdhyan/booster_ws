# Carry v2: pick the box up, carry it, put it down (design)

Status: design only (2026-10-11). Nothing in this file is implemented yet.

Reference: ULTRA, arXiv 2603.03279 (Unitree G1, OMOMO box lift/carry). ULTRA's pipeline:

1. Neural retargeting of human-object MoCap (OMOMO, 4 box-like objects) to the robot.
2. A universal tracking teacher that tracks the robot motion and the object pose. Rewards are a product of
   exponential kernels: joints k=10, rotation k=5, object position k=5, object rotation k=0.5,
   palm-to-surface k=20, contact k=5; termination −50.
3. A multimodal student distilled with DAgger. The goal is sparse (object and root target transforms) and
   object sensing is MoCap or an egocentric 64-point cloud. It has a 64-d Gaussian latent.
4. RL finetuning on goal reaching with dense goal, clipped progress and success-bonus terms.

Object domain randomisation: mass 0.15–1.5 kg, friction 0.2–1.2. Real G1 success rate on sparse goals: 80–90 %
with MoCap object pose, 50–60 % with egocentric depth.

## What carries over to K1, and what does not

| ULTRA piece | K1 v2 | Why |
|---|---|---|
| OMOMO MoCap + retarget + tracking teacher | **Not in v2.0**; optional v2.1 | K1 has no hands (forearm stubs) and 4-DoF arms. OMOMO retargeting needs a separate retarget pipeline. The SMPL-X neutral model is on disk, but this is a multi-week detour |
| Sparse goal = object target transform | **Yes**: the goal is the box target pose on the ground (xy, yaw) plus a phase flag | Same interface as v1, so v1 goal/curriculum code is reused |
| Physical object, contact grasp | **Yes**: the box becomes a real rigid body with collisions. No weld at deployment | This is the whole point of v2. v1 welds a visual-only box to the forearms |
| Egocentric point cloud | Later. v2.0 uses box pose in the robot frame (teacher), with noise for the student | The K1 head camera (depth panel now in the recordings) can feed a 64-point cloud in v2.2 |
| Object DR | mass, friction, size, all on curricula | v1 mass now has a 2→8 kg curriculum. ULTRA tops out at 1.5 kg; K1 arm torque limit is 14 Nm |
| Product-of-kernels tracking reward | Not used (no reference motion). Phase-structured task reward instead | See below |

## Task and phases

The box spawns on the ground, 0.6–1.5 m in front of the robot (curriculum). A second goal sits 1–10 m away for
the put-down. One policy handles all phases. A phase id (one-hot, 4) goes into the observation, and the env
advances it from state, never from time:

| Phase | Ends when | Main rewards |
|---|---|---|
| 0 approach | pelvis within 0.35 m of the box's pre-grasp stance, facing it (yaw err < 0.3) | stance-pose progress rate, heading kernel |
| 1 grasp | both forearm stubs in contact with opposite box side faces, squeeze force > f_min for 0.3 s | stub-to-face distance kernel (like palm-to-surface, k=20), contact flag both sides, squat depth allowed (squat skill) |
| 2 lift + carry | box bottom > 0.15 m above ground (lifted), then box within 0.3 m of the goal | box height kernel, box tilt penalty, goal progress rate (v1 terms), box-drop termination |
| 3 place | box resting on the ground at the goal (z < 0.02 m above rest, tilt < 10°), contacts released, robot upright, for 1 s | success bonus 500 (×dt = +10), release reward |

Terminations: robot fallen (z < 0.30 m or tilt > 0.8), box dropped (phase ≥ 2 and box z at rest away from the goal),
box tipped > 45°, box out of bounds, time out of 40 s.

## Grasping without hands: the assist curriculum

K1 can only squeeze the box between its two forearms. Learning that from scratch with PPO is a known hard
exploration problem, so the plan is a **grasp-assist curriculum** (this is not in ULTRA):

- During phases 2 and 3, a virtual spring-damper pulls the box toward the hold pose midway between the stubs, but
  only while both stubs touch the box. Force is `alpha * (k * err + d * err_dot)`, clipped to the box weight.
- `alpha` decays 1 → 0 over the curriculum and is gated on the success rate, like v1's radius schedule (fall rate
  low → advance). Friction then has to do the work.
- Squeeze force is observed for the teacher only (privileged). The student gets joint torques/efforts, which a
  real robot also measures.

Arm control: 8 arm joints, actions around the v1 hold pose (`hold_pose(width)`). In phases 0 and 1 the targets
start from a "ready" pose with arms forward at box height.

## Reuse of existing skills

- **Legs**: residual over the frozen velocity student (the same `FrozenStudentResidualActionCfg` used now for squat
  and track/kick). Phase 1 needs a deep squat to reach a box on the ground, so start from the squat residual
  policy once it trains (`Isaac-Velocity-Squat-Residual-K1-v0`).
- **Carry phase**: warm-start the arm head from v1 (p7_carry) where the observation layout allows.
- **Goal + curricula**: `carry_command` (ghost-box goal marker), `goal_radius_curriculum`, `carry_mass_curriculum`.

## Curricula

| Knob | Start | End | Gate |
|---|---|---|---|
| box spawn distance | 0.6–0.8 m | 0.6–1.5 m | success > 50 % |
| box start height | on a 0.4 m pedestal (easy reach) | on the ground | success > 50 % |
| grasp assist alpha | 1.0 | 0.0 | success > 60 %, then step down 0.1 |
| box mass | 1–2 kg | 1–6 kg | follows the goal radius |
| put-down goal distance | 1 m | 10 m | as in v1 |
| box width | 0.30 m | U(0.25, 0.40) m | fixed per env at startup, so `replicate_physics=False` |

Starting on a pedestal is the cheapest way to separate "learn to grasp" from "learn to squat to the floor".

## Observations

Teacher, in addition to the v1 policy obs:

- box pose and velocity in the pelvis frame
- box size and mass
- phase one-hot
- goal pose in the pelvis frame
- stub-to-face vectors (2 × 3)
- contact forces on the stubs (2)

Student: the same minus forces and mass. Box pose gets noise (σ 0.02 m, 0.05 rad, as in ULTRA Table K) and 10 %
dropout. A later version swaps box pose for a head-camera point cloud.

## Implementation steps

1. `carry_v2_env_cfg.py`:
   - make the box a dynamic rigid body
   - add stub contact sensors
   - add phase state in `carry_mdp._state`
   - add phase-advance events
   - add the assist force applied via `set_external_force_and_torque`
2. Rewards and terminations as tabled. Smoke test with the zero policy and scripted arms to check that the assist
   holds an 8 kg box at alpha = 1 (panel video with stub contacts drawn as markers).
3. Teacher PPO, 4096 envs. Legs: residual over the velocity student. Arms: own MLP (`decoupled_policy`).
4. Distil the student (as for P2/track), then optionally RL-finetune it on the student obs (ULTRA stage 3).
5. v2.1 (optional): OMOMO box-lift clips. Retarget SMPL-X to K1 (the licensed neutral model is on disk; never
   commit or upload it or derived data), then use the clips as an AMP/tracking prior for the grasp phase.

Open questions:

- Is 14 Nm per arm joint enough to squeeze-hold more than 3 kg by friction alone at μ ≈ 0.8? If not, cap mass at
  the max that a static check (step 2) can hold.
- Should the stubs get a rubber material (friction 1.0+)? On the real robot that would need sleeves on the
  forearms.
