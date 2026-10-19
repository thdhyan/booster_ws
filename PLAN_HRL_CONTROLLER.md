# PLAN — K1 Hierarchical RL Controller: from proposal to action

> **Date:** 2026-10-12 · **Branch:** `main` @ `0c61944`
> **Source proposal:** Claude Doc "Booster K1 Hierarchical RL Controller" (Oct 12, 2026) —
> team policy → task policy → {head, upper body, lower body} → joint PD; tokens only at the command interfaces.
> **Companion docs:** `HANDOFF-PUSH.md` (live runs, §15–17), `PLAN_PHASE6_SOCCER_HRL.md` (older P1–P5 plan, superseded
> where this file disagrees), `PLAN_MAPPO_TEAM_SOCCER.md`.

---

## 1. Where the repo stands against the proposal

| Proposal stage | What exists today | Gap to close |
|---|---|---|
| Setup (booster_train, booster_deploy, MuJoCo) | `isaac_tasks/booster_train_ref` (K1 asset + actuators, gains fixed 10-02), IsaacLab v3-EA on dl/zz-bw, MuJoCo fleet scripts in `src/k1_sim_gazebo/scripts/mujoco_fleet_*.py` | `booster_deploy` not cloned/wired; no per-module MuJoCo sim-to-sim exit test |
| Interface contract as code | Nothing. Each task defines its own command layout (`residual_action.py` CMD_RANGES, push 10-dim action, carry cmd) | **Missing — blocks everything above Stage 0.** Need command dataclasses, clamp ranges, skill-token enum, L↔R mirror map |
| Stage 0 lower body (vx, vy, ωz, h) | `k1_velocity_student_gaitfix.pt` (walk, 3 cmds); `Isaac-Velocity-PartialCtrl-K1-v0` (arm random-pose curriculum ±0.15→±1.0 rad); squat **residual** on the walk student (`squat_residual_env_cfg.py`, zz-bw, stuck at curriculum stage 0 until the 10-12 gate change) | h is a residual, not part of the base command; no stub-tip force curriculum; no symmetry augmentation; gait gate history (shuffle); no benchmark vs built-in WALK; no MuJoCo pass |
| Stage 1 upper body (8 arm joints, stub-tip targets) | Push/reach/carry drive wrists by **differential IK** (5 cm/step) inside task-specific envs; reach v2 running on dl/0 | No task-agnostic arm controller. Reach/push are fused task+arm policies, not a frozen-able module with a `(tipL, tipR, mode)` interface |
| Stage 2 head (ball in view, filter-fed) | `Isaac-HeadTrack-K1-v0` (detection obs, ball roll curriculum); ball-track residual `p8` usable (reward 15.6, 6 % falls) | Ball-track residual also writes legs (violates "own joints only"); no Kalman filter in loop, no covariance/time-since-seen obs, no dropout/latency model |
| Stage 3 kick residual (L, R by mirror) | `Isaac-Kick-Residual-K1-v0` on walk student; v1 failed (goal 6 %), v2 running dl/1 | No skill phase φ, no gate α(φ), no kick params (dir/strength), last layer not zero-init, no abort path, no mirror → kick-right |
| Stage 4 jump residual | Nothing | Whole stage |
| Stage 5 task policies (striker, pusher; hybrid token action) | Push MAPPO (skrl fork, goal_reached ~45 %) and carry are end-to-end over the frozen base; old P5 HRL manager plan | No hybrid categorical+Gaussian PPO head, no token masking, no 5–10 Hz task loop, no increment-integrated locomotion command |
| Multi-agent | skrl MAPPO fork + `train_mappo.py --resume` working on push | Not on top of task-layer tokens; no fictitious self-play / opponent pool for soccer |
| Tokenization (FAST/BEAST) | Nothing | Deferred until Stage 5 logs exist (as the proposal says) |
| Sim-to-real | ROS deployment plan `docs/k1_ros_deployment_plan.md`, CUSTOM-mode SDK probes in `sdk/` | 50 Hz single inference loop for 3 body controllers + async task loop; safety clamps |

**Summary:** we have most of the *ingredients* (frozen walk base, arm-pose curriculum, residual action plumbing,
head task, kick residual, MAPPO), but they are wired as separate end-to-end tasks with private interfaces.
The proposal's core value — frozen modules behind a fixed contract — is not in place yet.

## 2. Decisions (user, 2026-10-12)

1. **Stage 0 = (a):** retrain ONE lower-body base with `h` in the command, on **zz-bw GPU0** (long run).
   Reason: walk student + squat residual "keeps standing, not squatting" in the videos.
   Root cause found: `squat_depth_curriculum` is gated on height_err_ema <= 0.025 (0.03 in v2), which never
   passed, so H* stayed in 0.50-0.55 (a 2-7 cm dip). The new env ramps H* by TIME instead.
2. **Stage 1 = (a):** IK arms, with two fixes the user asked for: no arm flailing, push with the arms (videos
   show the robot going face first and pushing with its trunk). Root causes found:
   - relative diff IK at 5 cm/step: policy noise = random wrist jumps, zero action does not hold the arm;
   - `left/right_hand_link` origin is the ELBOW (forearm mesh runs 0 -> 0.228 m along link +-y), so
     `hands_to_box` paid elbows (and the chest behind them) for touching the box; box progress was
     paid no matter what touched the box.
3. Live 10-12 runs: no answer; default is to let them finish.

### 2a. What was built (2026-10-12)

| piece | files | task id | status |
|---|---|---|---|
| Stage-0 lower-body base: (vx, vy, wz, H*), arms+head (10 j) random pose + interval deltas (curriculum), stub-tip forces 0 -> +-20 N (time ramp), H* low bound 0.50 -> 0.40 over it 200-3000 (time ramp), base_height weight 5, teacher obs + 20 upper-joint obs + 6 stub wrench (264), policy 71 | `tasks/velocity/lower_body_env_cfg.py`, `lower_body_mdp.py`, `agents/squat_ppo_cfg.py:K1LowerBodyPPORunnerCfg` | `Isaac-LowerBody-K1-v0` (+Play) | train smoke OK; 20k-it run on zz-bw/0, tmux `k1_lowerbody` |
| Push v4 arms: action 8 = base 4 + (u, v) per arm on the near box face (left arm = robot-left half), target 2 cm inside the face, clamped to 0.40 m reach sphere, EMA 0.8, ABSOLUTE DLS IK on the stub tip (link + 0.20 m). Rewards: stub-tip proximity, box progress only with both tips < 6 cm from the box, trunk/head within 0.18 m of box -5, flat orientation -2, arm joint vel -2e-3, action rate -0.01 | `tasks/push/push_arms_mdp.py`, `push_arms_env_cfg.py`, `agents/rsl_rl_ppo_cfg.py:K1PushArmsPPORunnerCfg` | `Isaac-Push-Arms-K1-v0` (+Play) | smoke on zz-bw/0 |

Not done yet: symmetry augmentation (needs WP0 mirror map); warm start from gaitfix (obs layout differs:
teacher 264 vs student 50, so the base trains from scratch); a `FrozenBaseVelocityAction` mode for the new
base (obs adds H* + 20 upper-joint obs) before push v4 can use it.

### 2b. Reward audit for the hierarchy (2026-10-12 ~18:30, Opus)

The hierarchy needs: Stage 0 learns ONLY "track (vx, vy, wz, H*) and survive whatever the upper body does";
each upper layer is paid ONLY for its own task and treats the frozen layer below as an actuator.

Stage 0 lower-body teacher (zz-bw, it 670/20000, ~5 it/min):

| finding | evidence | proposal |
|---|---|---|
| OK: H* is independent of velocity (crouch-walk is trained), arms/head default-pose penalty removed, tracking terms blind to H* | `squat_env_cfg.py` cmd, `LowerBodyRewardsCfg` | keep |
| Yaw tracking weak | `error_vel_yaw` 0.93 rad/s with wz range +-0.8; track_ang_vel_z 0.90 vs lin 7.3 | wait for it ~1500; if still > 0.3 raise ang weight 5 -> 8 (Haiku) |
| Falls high | `base_orientation` term. 35 % | expected while arm poses ramp; gate: < 10 % by it 3000 |
| `action_rate_l2` is the largest penalty (-1.94) | may flatten squats | leave; re-check when H* ramp ends (it 3000) |
| Run length | 20k it at 5 it/min = ~66 h; all curricula end by it 4000 | stop at ~6000 or on plateau; distill student from the it>=4000 teacher |
| Not wired | symmetry aug (WP0 mirror map) | WP0 |
| **Gait ramps missing (shuffle risk)** | `LowerBodyCurriculumCfg` replaced the velocity curriculum and dropped the 7 upright-gated `RewardWeightRampTerm`s, so cadence / phase_swing / clearance / alternation / stride / action_rate / jerk stayed at 1/5 of their terminal weight | FIXED 2026-10-12 ~19:00: `K1LowerBodyEnvCfg.__post_init__` copies them back. Teacher restarted as v2 (warm start from model_750, 10000 it); ramps finish at ~it 5000 upright. Student must be re-distilled from a v2 teacher |

Push v4 arms (dl, v2, it 437/3000):

| finding | evidence | proposal |
|---|---|---|
| Base is the OLD walk student (no H*, never saw stiff arms in a hold pose) | `fallen` 41 % of episodes | the push layer must sit on the new Stage-0 student (`FrozenBaseVelocityAction` 710-d mode) — biggest fix |
| `box_progress` never pays | 0.0000 at it 437; gate = BOTH tips < 6 cm AND zero body contact; `hands_to_box` 0.16/s (tips ~0.5 m away) | pay progress x (tips in contact)/2 so one hand already earns half; keep the body-contact gate |
| `box_near_goal` is free reward | 0.73/s while progress is 0: box spawns 0.63 m from goal (curriculum), exp(-d/0.5) pays without pushing | drop it, or make it a potential difference (pay only for the change) |
| Arm penalty larger than arm reward | `arm_joint_vel_l2` -0.51 vs `hands_to_box` +0.16 | 2e-3 -> 5e-4 |
| Body-contact penalty works | `body_box_contact` -0.002, `body_box_intrusion` -0.006 | keep (-10 / -5) |

User decision 2026-10-12 ~19:15: push v4 keeps running to it 3000 on the OLD walk base; push does not wait
for the new student. Whether push needs H* at all is judged from the v4 result.

Push v4 final (2026-10-12 ~20:15): 3000/3000 rc=0, box_progress 0.0 all run, hands_to_box 0.075 —
never moved the box once. Push v5 launched 2026-10-12 ~20:24 on dl GPU0 (tmux `k1_push`,
`Isaac-Push-ArmsV5-K1-v0`, 3000 it, 1:31): progress x (tips in contact)/2, box_near_goal
dropped, arm_joint_vel_l2 -5e-4. Smoke 3 it passed.

Push v5 final (2026-10-12 ~22:05): 3000/3000 rc=0, box_progress ~0 again, hands_to_box 0.084,
fallen 37%, video `push_v5_final.mp4` (Drive `18ryspb2ZQmHlA3HkuHzBw4SUUNEk9DRI`). The reward
fixes changed nothing — degenerate-reward analysis 2026-10-12 ~22:30 (opencode):
- `box_goal_progress_rate` is correct (telescopes); the box genuinely never moves.
- Learned behavior is an avoidance optimum: falls 0→42→29%, time_out 1%→71%, body contact/
  intrusion →0, hands_to_box decays 0.13→0.08. Policy stands still, stays upright, avoids the box.
- Missing: (a) dense robot-to-box approach reward (tips-only proximity gives ~zero gradient at
  0.9–1.8 m spawn: exp(-1.2/0.3)≈0.02); (b) contact BONUS to counterbalance the -10/-5 contact
  fear + -500 fall risk; (c) stand-still penalty while far from the box. Frames confirm: robot
  near box at step 50, standing at a distance by step 200+.
- v6 proposals: torso-to-box exp approach (~1 m scale) + per-tip contact bonus + still-penalty
  when far; optional box-spawn curriculum (start 0.5 m so contact happens by accident). Still on
  the old walk base for fast (~1.5 h) iteration; new student only if v6 pushes.

Push v6 BUILD + LAUNCH (2026-10-12 ~23:37, opencode, dl GPU0 tmux `k1_push`, ETA ~1 h 45):
`Isaac-Push-ArmsV6-K1-v0` = v5 set + user additions:
- `stub_contact` +2.0: per-tip contact bonus, big enough to outpay the -10 body-contact /
  -5 intrusion fear; tips are exempt from those two terms (filtered sensor).
- `torso_to_box` +1.0 std 0.6: dense trunk-to-box pull with gradient from any spawn
  (exp(-1.4/0.6)=0.10 vs tip proximity exp(-1.4/0.3)=0.01).
- `stub_pointing` +1.0: relu(cos(shoulder->tip, tip->box)) mean over arms - arms aimed at the
  box pay even at range.
- `stand_still_far` -0.5: standing still with the box > 0.7 m away now costs.
- box_dist curriculum r0 (0.75,0.9) -> (0.5,0.65): spawn point-blank early, reach grows to
  1.8 m by it 1800 as before.
Smoke 3 it PASSED (all 4 terms logged, box_dist reads 0.65). Student test distill
(teach-750 pipeline test) finished RUN_RC=0; dl GPU1 free.

Teacher v3 STAGED (opencode, NOT restarted - awaiting kill decision):
- `gait_rewards.gait_cadence_l1`: L1 |steps/s - 2.0| on the shared cadence estimator
  (new `_cadence_steps_per_s` cache so Huber + L1 share one filter advance per step).
- `LowerBodyRewardsCfg.gait_cadence_l1` -0.1 with ramp -0.1 -> -0.6, same upright gate.
- Clearance priority: `feet_clearance_regularization` terminal weight -8.0 -> -12.0, now the
  dominant gait term.
- Still open: yaw track_ang_vel_z 5 -> 8 (proposed at it 1500, never approved); bundle with
  the v3 restart if wanted. Files synced to zz-bw clone.

Teacher v3 LIVE (2026-10-12 ~23:54 user OK, own slot: **dl GPU1**, tmux `k1_lbv3`, log
`~/k1_logs/lowerbody_v3_dl.log`, 10000 it, warm start lb_v2 model_7000.pt; v2 KEEPS RUNNING
on zz-bw GPU0):
- `track_ang_vel_z_exp` 5.0 -> 8.0 via a LowerBodyRewardsCfg override (velocity/carry/kick
  keep 5.0).
- `gait_cadence_l1` -0.1 -> -0.6 ramp (L1 on |steps/s - 2 Hz|, no flat zone; cadence
  estimator refactored into `_cadence_steps_per_s` with a per-step cache so both cadence
  terms share one filter advance).
- `feet_clearance_regularization` terminal -8.0 -> -12.0 (clearance is the priority gait
  term; it-5750 gate read +4 mm vs 60 mm target).
- Smoke 3 it: all terms + ramp registered (weights table shows 8.0 / -0.1), warm start
  loaded; v3 needs to be re-gated with gait_gate.py around it 8000 wall (ramps finish
  earlier in upright-seconds than nominal because of the fall rate).

Teacher v3 FINAL (2026-10-13 ~13:45): **DIVERGED at ~it 850** (warm start from v2@7000 +
curricula restart). Mean reward hit -1e28, later oscillated ~-2e11; action_jerk grew to
-1.1e10. Gait gate on the final ckpt FAIL 8/8 (jerk 0.43 rad/step2, clearance +9 mm vs 60 mm,
speed 0.10-0.28 m/s). **Discard v3.** Lesson: warm-starting must pin the upright-gated ramps
at their TERMINAL values (restart relaxes the gait penalties and the trained policy slides),
or train the v3 reward set from scratch. v2 (zz-bw, 8245/10000, yaw 0.77) stays the best
teacher candidate; student distill should come from v2's final, not v3.

Push v6 FINAL (2026-10-13 ~01:20, 3000 it rc=0): approach/pointing/contact all active
(pointing 0.49, tips on box 9 % of steps, torso_to_box 0.09) - but box_progress ~0 again and
falls ROSE to 55 % (v5: 37 %). Reward-side reaching is fixed; the frozen OLD walk base is the
limiter. Push on it is closed until the new Stage-0 student exists.

Slides 130-135 (opencode 2026-10-12/13): status 23:00 + push v5 (Drive 18ryspb2ZQmHlA3...),
teacher v2 gait (1tB3GpZMoym-...), status 13:05 + push v6 video (1eystts-Je2jS5...), teacher
v3 final video (1bJiF4aRG03-e28VzKRwj9go5Q1eroozi).

Gait gate on teacher v2 @ it 5750 (2026-10-12 ~23:00, `lb_v2_it5000.npz`): **FAIL 4/4 envs** —
mean cadence 5.19 steps/s (want 1.2–4.0), mean jerk 0.328 rad/step² (want < 0.06), swing
clearance +4 mm vs the 60 mm target, speed 0.06–0.27 m/s. Interpretation: the restored ramps
are upright-gated and ~100 % of episodes terminate at it ~5300, so effective upright time is a
fraction of wall-clock iterations → ramps still mid-growth, not a hard gate failure yet. If it
still fails at ~it 8000, extend past 10000 (original concession) or restart the teacher with a
softer fall spiral (yaw weight 8 + shallower squat end). Videos: push v5 final Drive
`18ryspb2ZQmHlA3HkuHzBw4SUUNEk9DRI`, teacher gait Drive `1tB3GpZMoym-Zf-Eom0P304p5T33W6t2_`;
Slides 130–132 (`s_status_1012_2300`, `s_p5v_1012`, `s_lbg_1012`).

Order (H* student path, separate from push): (1) Stage-0 student distill (`Isaac-LowerBody-K1-Distill-v0`, 710-d); (2) push v5 = proposals above + new
student as base; (3) the Haiku agent tunes weights one change per restart.

## 3. Work packages (in order; each ends in an exit test + video + Slides note)

| WP | Content | Exit test | Who |
|---|---|---|---|
| WP0 | `k1_hrl/contract.py`: `LocoCmd(vx,vy,wz,h)`, `ArmCmd(tipL,tipR,mode)`, `GazeInput`, `SkillToken` enum {WALK,KICK_L,KICK_R,JUMP,REACH,PUSH,IDLE}, `ResidualTrigger(skill,phi,params)`, clamp ranges from trained envelopes, L↔R joint/obs mirror map (verified by unit test: mirror∘mirror = id, mirrored default pose = default pose) | `pytest tests/test_contract.py` | Opus: draft + review |
| WP1 | Clone/wire `booster_deploy`; MuJoCo sim-to-sim harness that loads any exported module behind the contract | gaitfix student walks in MuJoCo, tracking error logged | Opus: draft code |
| WP2 | Stage 0 v2: PartialCtrl env + `h` command + stub-tip force curriculum + symmetry aug + full DR; warm from gaitfix | flat tracking error under random arm poses + pushes; gait gate PASS; MuJoCo pass; beats built-in WALK | Opus: env code; **Haiku: reward tuning** |
| WP3 | Stage 2 head v2: Kalman ball filter in env loop (frustum projection, dropout, depth noise, latency); head-only actions; obs = estimate + cov + time-since-seen | ball in view ≥ X %, reacquire < Y s, legs unaffected (Stage 0 test re-run) | Opus: filter + env; Haiku: rewards |
| WP4 | Stage 3 kick residual v3: phase clock φ, gate α(φ), params (ball rel. foot, dir, strength), zero-init last layer, abort on tilt, kick-right via mirror | kick success + direction error, upright after handoff, abort survives | Opus: residual action; Haiku: rewards |
| WP5 | Stage 1 arm: diff-IK v0 behind `ArmCmd`; later RL 8-joint reach/push on frozen Stage 0 with random walk underneath | tip error low while walking; box pushed straight | Opus / Haiku |
| WP6 | Stage 4 jump residual with widening output bound | height reached, clean landing, walk resumes | Opus / Haiku |
| WP7 | Stage 5 task layer: hybrid PPO head (categorical token + per-token Gaussian, masking), 5–10 Hz loop, increment commands; striker + pusher | task success in Isaac + MuJoCo | Opus |
| WP8 | Multi-agent: MAPPO on task layer (push 2→3 robots), soccer 1v1 → 2v2 with FSP + scripted opponent | goal/push success vs baselines | Opus / Haiku |
| WP9 | Hardware on harness, one module at a time; then full stack | — | user + Opus |

After every WP: re-run all earlier exit tests with the new module live (proposal rule).

## 4. Agent roles (user rule, 2026-10-12)

- **Opus agents:** planning, code review, drafting code (env cfgs, contract, residual action, filter, hybrid PPO).
- **Haiku agents:** reward fine-tuning (weight/std sweeps, reading TensorBoard terms, proposing one-change restarts),
  and all uploads (Drive files, Slides notes, report videos).

## 5. Next actions (this week)

1. Watch Stage-0 run (3 h reports, Haiku uploads); push v4 smoke video -> training on dl when a GPU frees.
2. WP0 contract + mirror map + tests (Opus draft, Opus review).
3. WP1 booster_deploy/MuJoCo harness, verified on the gaitfix student.
4. WP2 Stage 0 v2 smoke (16 envs, debug video) on dl GPU0 once reach v2 / carry v2 finish.
5. Keep the 3 h report cadence on the running 10-12 restarts (Haiku uploads).

GPU budget unchanged: training on dl GPU0/1 + zz-bw GPU0 only; render bursts 15–20 min.

## 6. Teacher v4 PBT + stair-climber residual — plan (2026-10-13, user request)

Status inputs: teacher v2 (zz-bw GPU0) still running past it 8300 with yaw ~0.8 and no gait-gate
pass; teacher v3 diverged after a warm start that restarted all curricula and ramps (do not
repeat); push v4/v5/v6 proved the OLD walk base cannot push. v2 finishes ~17:30 on zz-bw.

### 6a. Teacher v4 = population-based sampling (PBT-lite), fresh start, rough+steps+slopes

**Why population, not one more single run:** v1-v3 history has 3 confounds: warm-start kills the
policy (v3), ramps stall when the policy falls too much (v2), and there is no single "right"
weight triplet (yaw 5 vs 8, clearance -8 vs -12). Sample a small population, let them run, keep
the survivors.

- **Population:** M = 4 members, ALL from scratch (no warm start — the v3 trap), same task
  `Isaac-LowerBody-PBT-K1-v0`, same base reward set as v3 (cadence L1, yaw 8) with per-member
  jitter around the v3 values:
  | member | clearance terminal | yaw weight | cadence_l1 terminal | stride terminal | seed |
  |---|---|---|---|---|---|
  | P0 | -12 | 8.0 | -0.6 | -6 | 42 |
  | P1 | -16 | 10.0 | -1.0 | -6 | 43 |
  | P2 | -8 | 6.5 | -0.6 | -4 | 44 |
  | P3 | -12 | 8.0 | -0.6 | -6 | 45 |
  Preset chosen by env var `K1_TEACHER_PRESET` read in `K1LowerBodyEnvCfg.__post_init__` and
  `K1LowerBodyPPORunnerCfg.__post_init__` (experiment_name suffix `pbt_p<N>`) — one env var,
  no new registrations, auditable diff per member in wandb run name.
- **Terrain (the "rough + steps + slopes" ask):** extend `K1RoughSceneCfg` sub-terrains in a
  `K1LowerBodyV4SceneCfg` (lower-body only; velocity task untouched):
  | sub-terrain | proportion | cfg |
  |---|---|---|
  | flat | 0.20 | HfRandomUniformTerrainCfg noise ±1 cm |
  | rough | 0.20 | HfRandomUniformTerrainCfg ±4 cm |
  | pyramid stairs | 0.20 | HfPyramidStairsTerrainCfg step 4-12 cm, width 0.3 m |
  | pyramid slope | 0.20 | HfPyramidSlopedTerrainCfg 0-20 deg |
  | inverted stairs (descend) | 0.10 | HfInvertedPyramidStairsTerrainCfg 4-10 cm |
  | wave (uneven) | 0.10 | HfWaveTerrainCfg amp 1-4 cm, 1.0-2.5 Hz |
  All verified to exist in the installed IsaacLab 17.0.2 (`terrains/trimesh`/`height_field`).
  Curriculum stays `terrain_levels_vel` + rows; keep `max_init_terrain_level=5` so early
  resets randomize over easy levels. NOTE: v2's terrain_levels metric sat at ~0 the whole run
  (promotion needs walking) — expect stairs to only be reached once the member walks.
- **Selection (segment-free, because warm start kills policies):** all 4 run to it 6000 (~4-5 h).
  At it 2500 and it 5000, run `rec_burst.sh`-style 200-step plays + `gait_gate.py` per member.
  Selection at 6000: best 2 members by (gate pass count, then falls, then yaw error) continue
  to it 10000 on their GPUs; the other 2 are killed. No weight transfer between members.
- **Exit test:** gait gate PASS on the winning member's final trace + a stair/slope play video;
  then distill the Stage-0 student from it.

### 6b. Stair-climbing residual skill (new task `Isaac-Steps-Residual-K1-v0`)

Pattern from `squat_residual_env_cfg.py` (proven): frozen walk student
(`k1_velocity_student_gaitfix.pt`, or the v4 student later) does the legs; the policy adds a
12-dim residual (`FrozenStudentResidualActionCfg`, scale 0.5) and sees the teacher obs incl.
the height scanner. Curriculum exactly as the user wrote it:

| stage | terrain | command |
|---|---|---|
| A | ONE pyramid step, 6-8 cm, directly in path | velocity on flat around it |
| B | 2-3 steps, 6-10 cm | full velocity ranges |
| C | pyramid stairs rows + mild slopes | full ranges |

Mechanism: stairs-only terrain generator rows = step count (1, 2, 3, 5, 8) with
`terrain_levels_vel` promoting by distance walked, plus a custom `steps_mdp` curriculum gating
on step crosses (height-gain over track distance). Rewards: velocity tracking (same), feet
clearance hinge (already L1), step-contact (do not kick the riser), termination on fall.
Files: `tasks/steps/steps_env_cfg.py`, `steps_mdp.py`, registration loop mirroring the
squat-residual task, runner `K1StepsResidualPPORunnerCfg`.

### 6c. Machines and schedule

GPU budget rule applies (training only on dl GPU0/1 + zz-bw GPU0; zz-bw GPU1 and dl GPU2/3 for
other users; zz-bw GPU0 hosts v2 until ~17:30 and CAN host a second job in parallel — a 4096-env
lower-body run uses ~7 GB / 33 % util).

| time | dl GPU0 | dl GPU1 | zz-bw GPU0 |
|---|---|---|---|
| now → ~17:30 | PBT P0 | PBT P1 | v2 (finishing) |
| ~17:30 → +5 h | PBT P0 | PBT P1 | PBT P2 + P3 (shared) |
| after 6000 it | PBT winner conts. | PBT 2nd conts. (or steps residual after PBT P1 dies) | PBT P2/P3 winners |
| overnight | steps residual (3000 it, ~3-4 h) if slots free | | |

Total to a v4 winner + gate: ~9-10 h from v2's finish. Steps residual can start ~03:00 on dl GPU1.

### 6d. Next agent checklist (handoff)

1. `K1_TEACHER_PRESET` wiring + 2 new scene/cfg classes + task registration (this doc's §6a/6b).
2. Sync to dl + zz-bw, smoke each member 3 it, smoke steps residual 3 it.
3. Launch P0/P1 on dl, P2/P3 on zz-bw after v2 ends.
4. it-2500 gate videos per member (`rec_burst.sh`; on dl the `play_record.py` pattern), gait gate
   each trace, log to this doc; 3-h report cadence only if another agent picks the training watch.

### 6e. Status + audit (2026-10-16, Opus)

**Nothing is training** (dl GPU0-3 and zz-bw GPU0/1 idle). The §6c schedule never ran; §6a/6b
had no code until today.

| item | latest artifact | result | verdict |
|---|---|---|---|
| Teacher v2 | zz-bw `.../k1_lower_body/2026-10-12_17-15-59_k1_lower_body/model_9999.pt` | falls (base_orientation) 66 %, time_out 0 %, ep len 144, yaw err 0.77, xy err 0.10, terrain lvl 0 | not usable |
| Teacher v3 | dl `.../k1_lower_body/2026-10-12_23-54-18_k1_lower_body/model_9999.pt` | reward -2e11 (action_jerk_l2 -1.1e10) | discard |
| Stage-0 student | dl `k1_lower_body_student/2026-10-12_18-09-10/model_4999.pt` | distilled from teacher it 750 (pipeline test) | not usable |
| Kick v2 | dl `p9_kick_residual/2026-10-12_13-41-23/model_3999.pt` | goal 59 %, falls 6 %; on OLD gaitfix base; no phase/gate/mirror/abort (WP4) | partial |
| Ball track | dl `p8_balltrack_residual/2026-10-11_23-12-56/model_2999.pt` | time_out 0.905, falls 6 %; writes legs, no Kalman (WP3) | partial |
| Climb steps | — | — | code built today (below), untrained |
| Decoupled hierarchy | — | needs a valid frozen Stage-0 base; WP0/WP1 not done | NOT reached |

Audit of v2 (every 500 it): falls rose 23 % → 66 % and ep length fell 800 → 140 exactly while
the TIME-based ramps finished (upper_pose 1.0 at it 2000, H* 0.40 at 3000, stub force 20 N at
4000); they never recovered. Cause: ramps ran on the clock, not on competence. Yaw error sat
at 0.8-1.0 from it 0 in v1 and v2 (plain velocity teacher: 0.3). A zero-action probe
(`scripts/probe_lowerbody_yaw.py`, all variants) gives wz_mean 0.10-0.15, so there is no
physics spin; it is a learning problem (cause open; P1/P2 vary yaw weight + std to test it).
v3 cause: `action_jerk_l2` is a 3rd difference of the RAW (unclipped) actions; once the mean
drifted it grew cubically.

Built 2026-10-16 (smoked 3 it each on dl, 256 envs, clean exit):
- `Isaac-LowerBody-V4-K1-v0` (+Play): `lower_body_env_cfg.K1LowerBodyV4EnvCfg`. Presets P0-P3
  via `K1_TEACHER_PRESET` (table §6a; P1/P2 yaw std 0.5); rough+stairs up/down+slope+wave
  terrain, 10 rows; ALL disturbance ramps (squat H*, upper pose, upper delta, stub force)
  gated on survival (EMA of time-out share of resets ≥ 0.6, `lower_body_mdp.gated_iteration`,
  logged as `Curriculum/survival`, `Curriculum/gated_iter`); terrain-relative fall check
  (0.33 m above ray-mean ground). Runner `K1LowerBodyV4PPORunnerCfg`: logs
  `k1_lower_body_v4/<time>_pbt_p<N>`, seed 42+N, `clip_actions = 2.0` (v3 fix).
- `Isaac-Steps-Residual-K1-v0` (+Play): `skills/steps_env_cfg.py`. Residual 0.3 on the frozen
  gaitfix student; actor/critic on the teacher group (height scan); stairs-up 50 % / down 40 %
  / flat 10 %, step 4-14 cm by row, `terrain_levels_vel`; rewards tracking, termination,
  flat orientation, feet stumble, residual_l2, action_rate; commands vx -0.3..0.5, vy ±0.2,
  wz ±0.5. Simpler than the A/B/C stage plan in §6b: row curriculum only. Runner
  `p10_steps_residual`, 5000 it.
- `scripts/k1_launch_hrl.sh` (dl tmux launcher) and `scripts/k1_train_status.sh` (GPU, tmux,
  last iteration + key metrics per log on dl and zz-bw).

Start (dl, from laptop):
```
scripts/k1_launch_hrl.sh teacher 0 0     # P0 on dl GPU0
scripts/k1_launch_hrl.sh teacher 1 1     # P1 on dl GPU1
scripts/k1_launch_hrl.sh steps 1         # or steps instead of P1
scripts/k1_train_status.sh lbv4\|steps   # watch
```
zz-bw (P2/P3) runs from a git clone inside the SIF: commit + push first, then
`K1_TEACHER_PRESET=2 K1_GPU=0 ~/k1_train_tmux.sh lbv4_p2 /export/scratch/thakk100/k1/logs/lbv4_p2.log --task Isaac-LowerBody-V4-K1-v0 --num_envs 4096 --viz none`
(check that `run_k1_train.sh` passes `K1_TEACHER_PRESET` through `--containall`; today it does not).

### 6f. Runs started 2026-10-16 (user request)

- **Teacher v4 P0** (base velocity controller: vx, vy, wz + base height H*) on zz-bw GPU0,
  tmux `lbv4_p0`, log `/export/scratch/thakk100/k1/logs/lbv4_p0.log`. Files rsynced straight
  into the zz-bw clone (`.../k1/tmp/booster_ws`), no git push; no env var → preset 0. The first
  launch segfaulted in Kit startup (`libomni.platforminfo`, RC 139, before task code;
  log `lbv4_p0_crash1.log`) and was relaunched with `~/k1_train_tmux.sh`.
- **Steps residual, height scan** (`Isaac-Steps-Residual-K1-v0`) on dl GPU0, tmux `steps_res`.
- **Steps residual, depth camera** (`Isaac-Steps-Depth-Residual-K1-v0`,
  `skills/steps_depth_env_cfg.py`): 24x32 depth from a camera on `Head_2` pitched 45 deg down,
  clipped to 2 m. Actor = deployable proprio (no base_lin_vel, no scan) + depth (768);
  critic = privileged teacher group with the height scan. 1024 envs, needs `--enable_cameras`
  (`scripts/k1_launch_hrl.sh steps_depth <gpu>`). Target: dl GPU1.
- Depth run launched on dl GPU1 (tmux `steps_depth`): ~9.8k steps/s, 2.5 s/it → 5000 it ≈ 3.5 h.
  IL3 has no `--enable_cameras` flag; `train.py` now enables cameras when `K1_ENABLE_CAMERAS=1`.
  Depth probe (zero action, 16 envs): values 0.07-1.0, per-image std 0.17, 33 % at the 2 m clip.

### 6g. GEAR-style decoupled lower body: walk + stand, direct RL (2026-10-16)

User asked for a gait-frequency command and a separate stand/walk switch, trained like
GR00T-WBC's decoupled lower body (GEAR: two ONNX policies, stand when ||cmd|| < 0.05, else walk;
walk obs carries `freq_cmd`). GEAR's training code is not released, so the recipe is ours:

- `tasks/velocity/lower_body_gear_env_cfg.py`, built on the v4 env (upper-body random poses +
  stub forces on survival-gated ramps, terrain-relative fall check).
  - Terrain: flat 30 % / rough +-4 cm 30 % / slope 0-14 deg 20 % / wave 1-4 cm 20 %. No stairs
    (steps residual).
  - **Walk** `Isaac-LowerBody-Walk-K1-v0`: command (vx, vy, wz, H*, f). f ~ U(0.8, 1.5) gait
    cycles/s (1 cycle = 2 steps), resampled with the velocity. ||(vx, vy, wz)|| lifted to >= 0.1
    (never trains standing). `gait_clock.get_frequency` returns the commanded f when the command
    term has `gait_freq_command`, so the phase clock, phase_swing and phase_locked_stride follow it;
    both cadence terms target 2 f steps/s.
  - **Stand** `Isaac-LowerBody-Stand-K1-v0`: command (0, 0, 0, H*). No phase clock, no gait
    rewards/ramps, `feet_planted` (-1, `squat_mdp.feet_unloaded_when_still`). Terrain rows random
    (terrain_levels_vel only demotes a robot that does not walk).
  - Play variants: `-Play-v0` (32 envs, end-of-curriculum poses/forces, squats 0.40-0.50).
- Direct RL runners (`agents/squat_ppo_cfg.py`): actor = deployable `policy` group (no
  base_lin_vel, 10-step history), critic = privileged `teacher` group; obs normalization on;
  `clip_actions = 2.0`. Logs: `k1_lower_body_walk/*_walk`, `k1_lower_body_stand/*_stand`.
- Deploy switch (WP1 harness, not built yet): ||(vx, vy, wz)|| < 0.05 -> stand policy, else walk;
  both get the same H*; walk also gets f (default 1.0).
- Runs: v4 P0 on zz-bw and the depth steps run on dl GPU1 stopped (user choice). Walk on zz-bw
  GPU0 (tmux `lb_walk`), stand on dl GPU1 (`scripts/k1_launch_hrl.sh stand 1`, tmux `lb_stand`).
  Height-scan steps residual keeps dl GPU0. Status: `scripts/k1_train_status.sh 'walk|stand|steps'`.

### 6h. Walk/stand restart: std fix, 8192 envs, 2-day run (2026-10-16 ~20:10)

Audit of the first walk/stand runs (Opus): walk (zz-bw, killed at it 6633) peaked at reward 42 @ it 2000
then slid to 16; falls ~50 %, survival 0.22 < 0.6 gate so `gated_iter` stayed 0 (no curriculum ever ran),
yaw err 0.70. Stand (dl, 5000 it done): reward 143, falls 33 %, yaw drift 0.38. Common cause: action std
1 -> 103 (walk) / 34 (stand). `clip_actions = 2.0` makes noise past the clip free and the inherited
`entropy_coef = 0.01` kept pushing it up, so rollouts became near-random.

Changes (`agents/squat_ppo_cfg.py`, walk + stand runners only):
- `entropy_coef` 0.01 -> 0.0; `init_std` 1.0 -> 0.5; std capped at 1.0 by `CappedGaussianDistribution`
  (own class: the zz-bw SIF rsl_rl has no `std_range` kwarg).
- PhysX: `train.py` re-runs `ensure_physx_gpu_capacity` after `--num_envs`; the helper now also sizes
  `gpu_found_lost_pairs_capacity` (512/env) and `gpu_found_lost_aggregate_pairs_capacity` (16384/env).
  At 8192 envs the defaults overflowed (2.9M / 86.9M asked) and PhysX drops contacts when that happens.

Runs (8192 envs, max 40000 it, stop after ~48 h = ~2026-10-18 20:00 from the latest checkpoint):
- Walk: zz-bw GPU0, tmux `lb_walk`, log `/export/scratch/thakk100/k1/logs/lb_walk_8k.log`.
- Stand: dl GPU1, tmux `lb_stand`, log `~/k1_logs/lb_stand_8k.log`.
Early iterations: walk ~25 s/it, stand ~6 s/it (early its are slow in every run; the old walk went
9.5 -> 1.7 s). Checks at it ~500: std <= 1 and falling, survival rising toward 0.6, `gated_iter` > 0.
Not changed yet: per-joint action scale (Booster `K1_ACTION_SCALE`), SURVIVAL_GATE 0.6, gait freq range.

**6h result (2026-10-18 23:20, Opus):** walk stopped at it 38213 (51 h; another user's 3 jobs joined
zz-bw GPU0 at ~22:25 and pushed it to 80 s/it). Std no longer explodes, it COLLAPSES: 0.5 -> 0.10 by
it 1000 -> 0.02 (entropy 0 = no floor). Best window it 1000-4000: reward 191-212, falls 36 %, ep len
700-740, xy err 0.12-0.15, yaw err 0.22-0.27 (was 0.70-0.96 in the first walk run). After that reward
slid 211 -> -3, falls 36 -> 48 %, ep len 740 -> 190, survival 0.53 -> 0.11; survival never reached the
0.6 gate, so `gated_iter` = 0 for the whole run (no arm pose / squat / stub-force curriculum ever ran).
Likely cause of the slide: upright-gated penalty ramps kept growing while std ~0.02 left the policy no
exploration to adapt. Keep `model_2000.pt` / `model_4000.pt` (zz-bw `.../k1_lower_body_walk/2026-10-16_20-10-19_walk/`).
dl was unreachable over SSH at 23:17 (timeout), so the stand run status is unknown.
Next proposals: entropy 0.001-0.003 with the 1.0 cap plus a std floor ~0.15; SURVIVAL_GATE 0.6 -> 0.5.

**6h stand + videos (2026-10-19 00:40, Opus):** stand run on dl died at it 20853 (2026-10-17 14:41): dl
`/` 100% full, tensorboard write raised. Last: reward 266, ep len 736, std 0.05, gate reached 1659 then
stalled (survival 0.52-0.56 < 0.6); early ends 47% (illegal_contact 37%, orientation 9.5%). Checkpoint
copied to zz-bw `logs/rsl_rl/k1_lower_body_stand/dl_2026-10-16_stand/model_20750.pt`.
Play clips on zz-bw GPU0 (`logs/videos/burst/{walk8k_it4000,walk8k_it38000,stand8k_it20750}.mp4`, 250 steps,
4 envs). Defaults (1920x1080, spp 8, 4 subframes) ran ~10 s/step on the shared GPU and hit the 1200 s
timeout; use `--panel_spp 4 --panel_subframes 1 --panel_size 1280 720` (~13 min/clip). Resets in 5 s over
4 envs: walk 4000 = 13, walk 38000 = 9 (crouched, pelvis z 0.55 vs 0.72), stand = 11. Play falls are far
above the train rate, so check whether the Play cfgs use max curriculum ranges instead of the trained ones.
Plots + clips: Drive "K1 HRL 4-layer controller" / `2026-10-19_lowerbody_8k_results`.
