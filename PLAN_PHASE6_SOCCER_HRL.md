# PLAN — Phase 6: Soccer HRL (single robot, ball → goal)

> **Date:** 2026-09-22 · **Branch:** `dev/phase-0` → (merge to `main`) → **`dev/phase-6-soccer-hrl`**
> **Stack:** Isaac Sim **6.1.0** + IsaacLab **v3.0.0-EA** (https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA), Python 3.12, RSL-RL 5.x, PyTorch 2.11
> **Rewards:** ✅ approved by user in chat (2026-09-22) — tables below are LOCKED
> **Companion docs:** `STATE.md`, `TASKS.md` (T6.*), `HANDOFF.md`, `PLAN_MULTILAYER.md` (architecture background)

---

## 0. Goal

Train a **hierarchical policy stack** for a single K1 to see a ball, keep it in camera frame (YOLO-verified), chase it, and kick it into a goal:

```
P5  HRL manager (5 Hz)  ── selects option + (vx,vy,wz) params
 │
 ├── P1 BASIC   stand/balance         (50 Hz, legs)      } teacher–student
 ├── P2 MOVE    velocity walk         (50 Hz, legs)      } teacher–student
 ├── P3 TRACK   head-only ball track  (50 Hz, head, YOLO rewards)
 └── P4 KICK    chase & kick ball→goal(50 Hz, legs, head driven by P3)
```

Train **4 base policies first**, then the **P5 top-level** in sim, single robot. All runs headless, wandb-logged, periodic verification videos, on the local 8 GB box without crashing it.

### Deployability invariant (user requirement — overrides earlier drafts)

**The final stack must run on the real robot with ZERO ground-truth state.** Therefore:

- **Deployable observation groups never read GT ball/goal state.** GT (ball pose/vel, goal pose, geometric in-frame flags) is allowed **only** in (a) *reward functions* (training-time signals — standard practice) and (b) *teacher* observation groups (distillation-time only, weights discarded).
- **Ball position comes from vision**: frozen **P3 head policy keeps the ball centered in the Booster depth camera** → YOLO bbox @10 Hz + depth lookup → relative ball position (§3.4). P4 and P5-student consume this *estimate*.
- **Goal position**: a **rectangular goal post is guaranteed** → goal estimator *tracks + remembers* relative goal position (visible → noisy detection snap; occluded → odometry dead-reckon; never seen → memorized field-layout prior) (§3.4).
- Every distilled student trains **with the estimator in the loop** (noise, latency, dropouts randomized), so train obs == deploy obs. A static test asserts deployable obs groups reference only estimator outputs.

---

## 1. Phase-0 closure (git)

1. Commit leftover untracked/modified work onto `dev/phase-0` (head/kick drafts, `k1_soccer_compose.py`, `PLAN_MULTILAYER.md`, `ROBOTS.md`, `isaac_fleet_vis.py`, `docker_isaac_fleet.sh`).
2. `git checkout main && git merge dev/phase-0 && git push origin main`
3. Tag `v0.7-phase0-complete` (no tags exist yet), push tags.
4. `git checkout -b dev/phase-6-soccer-hrl` — all Phase-6 work here.

---

## 2. Environment rebuild (BLOCKING — nothing trains until this is done)

Old `.venv-isaac` (Isaac Sim 6.0.1 + IL 3.0.0b2) was deleted. Fresh empty `IsaacLab/isaac6/.venv` (uv, py3.12) exists.

1. Install **Isaac Sim 6.1.0 pip** into `isaac6/.venv` (`pip/uv install isaacsim[all,extscache]==6.1.*` ≈ 15–25 GB; disk has 54 GB free → monitor).
2. Clone/checkout IsaacLab **`v3.0.0-EA`** as a **separate worktree** `~/Projects/IsaacLab-ea` (do NOT move the shared `~/Projects/IsaacLab` checkout off `perf-2026-07-06` — other projects use it). Install editable into the venv. (Note: `release/3.0.0` branch has 49 bug-fix commits past the EA tag — evaluate switching after smoke passes.)
3. Install `rsl_rl`, `ultralytics` (yolov8n), `imageio[ffmpeg]` into the venv.
4. `scripts/start_training.sh` — rewrite (stale venv path + wrong wandb entity).

### 2.x Migration checklist (IL 2.x/b2 → 3.0-EA — release notes)

| Change | Impact on us |
|---|---|
| Quaternions **WXYZ → XYZW** | audit every hard-coded quat: kick mdp, spawn poses, `flatten_k1_usd.py`, `soccer_sim.py`, head-cam optical rotation |
| `.data.*` returns **`ProxyArray`** → need `.torch` / `.warp` | every env cfg/mdp touch of `robot.data.*`, `ball.data.*` |
| `write_*_to_sim(data, env_ids)` **removed** → `write_*_to_sim_index` / `_mask` | kick mdp OmniReset ball/robot reset (known code path) |
| URDF importer rewritten, **nested rigid bodies by design** | re-validate: IL 3.0 may fix contact sensors natively (PR #6378) → if so, flatten may be droppable; else re-run `flatten_k1_usd.py` on 6.1 output |
| `--headless` removed → `--viz none`; `gym.wrappers.RecordVideo` removed → **`VideoRecorderCfg`** | all train scripts + video plan (§3.5) |
| `isaaclab train/play` unified CLI replaces per-library `train.py` | port our `scripts/train*.py` wrappers to the shared task-composition helpers |
| `ActuatorCollection`, `actuator_effort_limit` renames | `BOOSTER_K1_CFG` (booster_train fork — patch our fork if needed) |
| Contact-force contracts split (normal/aggregate/filtered) | velocity + kick contact reward params |
| `enable_extension()` needed for direct `isaacsim.*` imports | any direct Isaac import in our code |

Gate: migrate `k1_velocity` first (known-good reference), smoke `Isaac-Velocity-Flat-K1-v0 --num_envs 16`, then migrate kick/head/kick drafts.

---

## 3. Shared MDP infrastructure (used by multiple policies)

### 3.1 Domain randomization (user requirement)
| Axis | Spec |
|---|---|
| **Ball position** | OmniReset families extended: `at_ball_shoot` 0.3–0.6 m · `stand_ready` 0.6–1.2 m · `walk_up` 2–3.5 m; ±y cone widened to ±1.5 m; **rolling resets**: initial ball vel 0–1.5 m/s, random direction |
| **Ball color** | per-reset diffuse RGB: white · orange · high-vis yellow · black/white-panel · red |
| **Ball physics** | mass 0.35–0.5 kg · restitution 0.3–0.8 · friction 0.4–1.0 · radius 0.10–0.12 m |
| **Lighting** | dome intensity 400–1200 + color-tint jitter (YOLO robustness) |
| **Camera** | brightness/exposure jitter + small image noise pre-YOLO |
| **Ground** | friction 0.5–1.0, restitution 0–0.1 (existing) + rough terrain (P1/P2 teachers, §4) |
| **Robot spawn** | position/yaw jitter per reset (existing) |

### 3.2 Auto-stability push randomization (P1 + P2 required; P4/P5 enabled too)
Custom `EventTerm` `random_body_push` (mode=`interval`), complementing the existing velocity push:
- Sample body from **{Trunk/waist, pelvis, chest, upper-leg links}** (final list from `probe_k1_bodies.py` on the 6.1-imported asset).
- Horizontal random direction (±180°), magnitude **20–80 N**, duration **0.05–0.15 s**, interval **3–8 s**, active in ~60 % of envs.
- Applied via `robot.set_external_force_and_torque()` at the sampled body index — a localized waist shove, exactly the real-robot test: policy must self-balance (P1: stiffen/stance) and recovery-step (P2: keep walking).
- Present in **teacher training, student distillation, and evaluation** (it is an env property, not an obs-group property).

### 3.3 Camera + YOLO (user requirement)
- Head/scene camera 320×240 on a **camera subset** of envs (P3: all camera-enabled; P4/P5: 1-in-8 envs) → `yolov8n` (COCO `sports ball`) → **YOLO vector = (visible flag, du, dv)** normalized bbox-center offset.
- All other envs: cheap **geometric FOV check** as dense proxy for the same rewards.
- YOLO rewards (P3 R2/R3, P4 R5, P5 R3) computed from YOLO where cameras exist, geometric elsewhere; wandb metric `ball_detect_rate` logged always.
- Fine-tune `yolov8n` on auto-labeled synthetic frames (sim gives perfect ball bbox; palette = §3.1 colors) once first camera runs produce frames.

### 3.4 Vision → state estimator (deployable obs — user requirement)

Runs inside the env as an obs provider for the **deployable (student) groups** of P3/P4/P5. Two interchangeable backends producing identical vectors:

**Ball estimate (3+3+flag):**
- **Real pipeline (camera envs):** head-cam RGB @10 Hz → YOLO bbox → depth image lookup (median over bbox pixels) → bearing from intrinsics + known head-joint angles → camera→base transform → `ball_pos_est (3)`; `ball_vel_est (3)` = finite difference over 0.2–0.5 s; `ball_visible` flag. Occlusion → hold + constant-velocity propagate from history, flag 0.
- **Estimator model (all non-camera envs):** GT ball pose + range-scaled Gaussian noise (σ ≈ 0.05–0.15 m) · random dropouts 0–400 ms · latency 50–150 ms · visibility = geometric FOV test through current head pose. Cheap, differentiable-free, runs on all envs.
- Detector/model parameters (noise σ, dropout rate, latency) randomized per episode = part of §3.1 DR.

**Goal estimate (2+flag)** — rectangular goal guaranteed:
- Visible → noisy goal detection (σ 0.05–0.2 m) = "track".
- Occluded → last-seen pose propagated by odometry with drift noise = "remember".
- Never seen → memorized prior: goal fixed in field layout, transformed by (noisy) odometry = "remember without ever seeing".
- Deploy path: fine-tune detector on auto-labeled synthetic goal-post frames; same estimator code in sim and real.

**Composition inside P4/P5 rollouts:** frozen P3 runs every control step (keeps ball in frame) → estimator feeds P4-student/P5-student obs → legs from P4/P5. The head policy is literally "the other policy that gets the relative ball location".

### 3.5 Videos (user requirement)
- **Every 200 training iterations → 30 s clip**, saved `logs/videos/<run>/iter_XXXX.mp4` **and** logged to wandb (`wandb.Video`).
- 4 dedicated low-res video envs per run (persistent camera — P1/P2 have no YOLO camera, only these).
- Mechanism: IL 3.0 `VideoRecorderCfg` where it fits; fallback = custom runner callback (`env.render()` → imageio-ffmpeg) — decide at smoke time.
- ⚠️ "200 episodes" is ambiguous (with 256–512 envs, 200 episodes fire every few seconds) → **interpreted as every 200 iterations**; configurable `--video_interval`.

### 3.6 WandB
- Entity `thakk100-dhyan-home`, **new project `booster_k1_soccer_hrl`**.
- One run per training: `p1_basic_teacher`, `p1_basic_student`, `p2_move_teacher`, `p2_move_student`, `p3_head_track`, `p4_chase_kick`, `p5_hrl_teacher`, `p5_hrl_student`.

### 3.7 PC guardrails (user requirement — do not crash this box)
- Sequential runs only; `num_envs` ≤ 512 (256 where cameras heavy); **4096 forbidden locally**.
- Every run: `systemd-run --user --scope -p MemoryMax=9G -p MemorySwapMax=4G nice -n 10 …`
- `scripts/train_guard.sh` watchdog: poll 5 s → kill run if GPU > 7.0 GB, free RAM < 1 GiB, or disk < 5 GB.
- Headless (`--viz none`) always; one RTX camera pipeline max at a time.

---

## 4. Policy stack — locked I/O and rewards

Joint sets: **legs (12)** hip/knee/ankle ×2 · **head (2)** `AAHead_yaw`,`Head_pitch` · **arms (8)**.
All policies: `JointPositionActionCfg`, legs scale 0.25, head scale 0.5, `use_default_offset=True`. Physics 200 Hz / control 50 Hz (decimation 4).

### Teacher–student formulation (P1, P2, P4, P5) — user requirement

Same environment for teacher and student (identical terrain, pushes, DR, **and the same §3.4 vision estimator for P4/P5**); only the **observation group** differs (asymmetric observation). Two-stage: (1) PPO with privileged obs → teacher ckpt; (2) distillation with rsl_rl DistillationRunner — **student acts**, teacher provides action targets (replicate the `output_std` class-property guard from `train_student.py`). P3 is single-stage: its deployable obs (YOLO vector) needs no privileged variant — GT is used only in its reward.

**Architecture decision — students are MLPs with a 10-step history stack; no LSTM/transformer/VLA:**
- RSL-RL 5.x has no recurrent-policy support → LSTM/GRU would force an RL-library swap (skrl/SB3) and lose the proven distill pipeline.
- Frame-stack MLP = proven in this workspace (velocity student 40.89 > teacher 25.08), native to rsl_rl, trivially 50 Hz-deployable.
- Frame-stack also covers **partial observability introduced by the §3.4 estimator** (occlusion → held/propagated estimates): the student reads motion from its own history — this is exactly the RMA-style adaptation loop, no recurrence needed.
- Transformer / **Pi0.5 / smolVLA / GR00T** are language-conditioned VLA-class models: 100 M–1 B+ params, pretrain-heavy, impossible to train on an 8 GB GPU and pointless for a ≤54-dim proprio/estimator input at 50 Hz. Revisit only if a future semantic high-level ("go to the far ball") needs language grounding — not this phase.
- Net (all actors/critics): **MLP (512, 256, 128), tanh**; students input = flatten(K=10 stack). History at native rate: P1/P2/P4 @50 Hz (0.2 s), P5 @5 Hz (2.0 s).

### P1 BASIC — stand / balance (teacher–student)

| | Teacher (privileged) | Student (deployable) |
|---|---|---|
| **Input** | base 42 = gravity 3 + ang_vel 3 + joint_pos 12 + joint_vel 12 + last_action 12 · **+ height scan (~171, rough terrain)** · **+ foot contact/slip (4)** ≈ **217** | **42 × K10 = 420** (history only, blind) |
| **Output** | 12 leg joint targets | identical |

| # | Reward (LOCKED) | W | Signal |
|---|---|---|---|
| R1 | `flat_orientation_l2` | −1.0 | base quat |
| R2 | `still_lin_vel` (‖v_xy‖²) | −1.0 | base lin vel |
| R3 | `still_ang_vel` (‖ω‖²) | −0.5 | base ang vel |
| R4 | `joint_deviation_default` (L1 legs) | −0.5 | joint_pos vs default |
| R5 | `action_rate_l2` | −0.005 | action history |
| R6 | `dof_torques_l2` | −1.5e−7 | torques |
| R7 | `joint_pos_limits` | −1.0 | limits |
| R8 | `feet_slide` | −0.1 | foot contact |
| R9 | `termination` fall (h<0.35 / tilt>0.8) | −200 | base state |

Env: **rough terrain** (teacher curriculum; student blind on same terrain) + `random_body_push` (§3.2) + push events.

### P2 MOVE — velocity walk (teacher–student)

| | Teacher (privileged) | Student (deployable) |
|---|---|---|
| **Input** | base 48 = lin_vel 3 + ang_vel 3 + gravity 3 + joint_pos 12 + joint_vel 12 + **vel_commands 3** + last_action 12 · **+ height scan (~171)** · **+ foot contact/slip (4)** ≈ **223** | **48 × K10 = 480** |
| **Output** | 12 leg joint targets | identical |

| # | Reward (LOCKED) | W | Signal |
|---|---|---|---|
| R1 | `track_lin_vel_xy_exp` | 1.0 | lin vel vs cmd |
| R2 | `track_ang_vel_z_exp` | 2.0 | yaw rate vs cmd |
| R3 | `feet_air_time` | 0.25 | foot contact |
| R4 | `feet_slide` | −0.1 | foot contact |
| R5 | `flat_orientation_l2` | −1.0 | base quat |
| R6 | `action_rate_l2` | −0.005 | action history |
| R7 | `dof_acc_l2` | −1.25e−7 | joint acc |
| R8 | `dof_torques_l2` | −1.5e−7 | torques |
| R9 | `joint_deviation_arms_head` | −0.05 | arms/head vs default |
| R10 | `joint_pos_limits` | −1.0 | limits |
| R11 | `termination` fall | −200 | base state |

Env: **rough terrain curriculum** (uneven ground — user requirement) + `random_body_push` (§3.2) + existing velocity cmds. Existing flat policy is **retrained** as this teacher (6.0.1 checkpoints are not reused).

> **Updated 2026-09-24 (implemented, `Isaac-HeadTrack-K1-v0`).** Deployable obs is
> the **detection vector** (12-dim), not GT angles: head-cam RGB (320x240, on
> `Head_2`, 10 Hz) -> YOLOv8n `sports ball` bbox centre -> `(visible, du, dv)`,
> held between detections (geometric FOV proxy = fallback when no detector).
> **Ball speed is a curriculum** (0 -> 0.8 m/s, iters 200->1200): centre a
> static ball first, then track a rolling one. **CCW search**: no detection for
> 0.5 s -> locomotion gets an in-place counter-clockwise command
> `(0, 0, +0.6 rad/s)` (`head_mdp.ccw_search_command`, compose `update_search`).

### P3 HEAD TRACK — ball in frame, head motors only (single-stage)

| | |
|---|---|
| **Input (12)** — deployable, **no GT** | **YOLO vector 3** (visible, du, dv — the ONLY ball signal; updated @10 Hz, held between detections) · head joint_pos 2 · head joint_vel 2 · base ang_vel 3 · last_action 2 |
| **Output (2)** | `AAHead_yaw`, `Head_pitch` |
| **Legs** | frozen (P1 runs); `time_out` only |
| **Note** | R1's geometric ball angle is a **reward-time** signal only (GT allowed in rewards). A YOLO→angle mapping (intrinsics) gives the same steering information as input. |

| # | Reward (LOCKED) | W | Signal |
|---|---|---|---|
| R1 | `track_ball_angle_exp` | +3.0 | geometric yaw/pitch error |
| R2 | `ball_in_frame_yolo` | +0.5 | YOLO detects ball (geometric FOV proxy on non-cam envs) |
| R3 | `ball_centered_yolo` | +0.5 | exp kernel on (du,dv) |
| R4 | `action_rate_l2` | −0.1 | head action history |
| R5 | `joint_pos_limits` (head) | −1.0 | head limits |
| R6 | `time_penalty` | −0.01/step | step count |

DR: ball position (fast movers included), **ball color**, lighting, camera noise.

### P4 CHASE & KICK — ball → goal (**teacher–student**; no GT at deploy — user requirement)

Head joints are driven by **frozen P3 every control step** (ball kept in the depth camera); the leg policy consumes the **vision estimator** output (§3.4), never GT.

| | Teacher (privileged, discarded after distill) | Student (deployable) |
|---|---|---|
| **Input** | proprio 45 = lin_vel 3 + ang_vel 3 + gravity 3 + joint_pos 12 + joint_vel 12 + last_action 12 · **+ true ball_pos_rf 3 + true ball_vel_rf 3 + true goal_pos_rf 2 = 53** | proprio 45 · **+ ball_pos_est 3 + ball_vel_est 3 + goal_pos_est 2 + ball_visible/​goal_visible flags… = 54** (estimator in the loop: noise, 10 Hz latency, dropouts) → **× K10 = 540** |
| **Output (12)** | leg joint targets | identical |
| **Scene** | `Isaac-Kick-Ball-K1-v0` migrated: dynamic ball, rectangular goal +4 m, OmniReset (§3.1 extended), camera subset runs real YOLO+depth | same env, estimator-model on other envs |

| # | Reward (LOCKED) | W | Signal |
|---|---|---|---|
| R1 | `goal_scored` (terminal) | +100 | ball in goal volume |
| R2 | `ball_to_goal_progress` | +1.0 | ball→goal dist decrease |
| R3 | `approach_ball` (if robot-ball > 0.5 m) | +1.0 | robot→ball dist decrease |
| R4 | `kick_toward_goal` | +2.0 | ball vel along ball→goal axis (exp) |
| R5 | `ball_in_frame` hybrid (while dist > 0.75 m) | +0.5 | YOLO/FOV |
| R6 | `align_stance` | +0.5 | robot→ball vs ball→goal angle |
| R7 | `flat_orientation_l2` | −1.0 | base quat |
| R8 | `action_rate_l2` | −0.005 | action history |
| R9 | `dof_torques_l2` | −1.5e−7 | torques |
| R10 | `joint_pos_limits` | −1.0 | limits |
| R11 | `joint_deviation_arms` | −0.05 | arms vs default (head excluded) |
| R12 | `termination` fall | −200 | base state |

Terminals: `time_out` · h<0.35 · tilt>0.8 · `goal_scored`.
Rewards use GT (allowed — training-time only). All R-terms stay exactly as approved.

### P5 HRL MANAGER — top-level (teacher–student), 5 Hz

Selects one of 4 frozen options (BASIC/MOVE/TRACK/CHASE_KICK) + continuous params (vx, vy, wz); option executes 10 control steps at 50 Hz.

| | Teacher (privileged) | Student (deployable, **no GT**) |
|---|---|---|
| **Input** | **true** ball_pos_rf 3 + ball_vel_rf 3 + goal_pos_rf 2 · lin_vel 3 + ang_vel 3 · **ground-truth visibility flags 2** · last_skill one-hot 4 · last_params 3 · progress 1 = **23** | **YOLO vector 3 + ball_pos_est 3** · **goal_pos_est 2 + ball/goal visible flags 2** · lin_vel 3 + ang_vel 3 · last_skill 4 · last_params 3 · progress 1 = **20 × K10 = 200** (2 s @5 Hz) |
| **Output (7)** | 4 skill logits + 3 params (vx, vy, wz) | identical |

Student never sees true ball/goal state → it must **keep the ball detectable by YOLO** (P3 keeps it framed) to know where it is; occluded → estimator holds/propagates → history recovers motion. Goal estimate comes from §3.4 (track + remember). This couples the user's YOLO + ball-in-frame + no-GT requirements directly into HRL learning.

| # | Reward (LOCKED) | W | Signal |
|---|---|---|---|
| R1 | `goal_scored` (terminal) | +100 | ball in goal |
| R2 | `ball_to_goal_progress` | +0.5 | ball→goal dist decrease |
| R3 | `ball_in_frame_yolo` | +0.5 | YOLO at manager step |
| R4 | `ball_proximity_exp` (while CHASE_KICK) | +0.3 | robot-ball dist |
| R5 | `time_penalty` | −0.02/step | step count |
| R6 | `skill_switch_penalty` | −0.05/switch | option change events |
| R7 | `termination` fall | −100 | base state |

Options P1–P4 frozen during P5 training. `random_body_push` enabled.

---

## 5. Training campaign (sequential, local, guarded)

| # | Run | Task id (new) | envs (cam) | iters | wandb run |
|---|---|---|---|---|---|
| 1 | P1 teacher | `Isaac-Basic-Teacher-K1-v0` | 256 (4 vid) | 2000 | `p1_basic_teacher` |
| 2 | P1 student | `Isaac-Basic-Student-K1-v0` `--distill` | 256 | 1500 | `p1_basic_student` |
| 3 | P2 teacher | `Isaac-Move-Teacher-K1-v0` | 512 (4 vid) | 3000 | `p2_move_teacher` |
| 4 | P2 student | `Isaac-Move-Student-K1-v0` `--distill` | 512 | 3000 | `p2_move_student` |
| 5 | P3 | `Isaac-HeadTrack-K1-v0` | 512 (64 yolo+vid) | 2000 | `p3_head_track` |
| 6 | P4 teacher | `Isaac-Kick-Teacher-K1-v0` | 512 (4 vid) | 3000 | `p4_kick_teacher` |
| 7 | P4 student | `Isaac-Kick-Student-K1-v0` `--distill` | 512 (64 yolo+vid) | 3000 | `p4_kick_student` |
| 8 | P5 teacher | `Isaac-HRL-Teacher-K1-v0` | 256 (32 yolo+vid) | 2000 | `p5_hrl_teacher` |
| 9 | P5 student | `Isaac-HRL-Student-K1-v0` `--distill` | 256 | 1500 | `p5_hrl_student` |

- Each run's gate: headless smoke (16 envs, 2 it, video writer + YOLO + wandb each exercised once) → 200-iter benchmark to measure it/h → ETA logged in `STATE.md`.
- Iter counts are ceilings: stop early on plateau; reward curves + videos are the review signals (user verifies videos every 200 iters).
- Estimated total: several days sequential on the 4060 — acceptable; nothing parallel.
- Export: TorchScript `models/k1_{basic,move,head_track,chase_kick,hrl}_policy[_student].pt` (LFS) + `k1_soccer_compose.py` extended to load P3/P4 and run the full stack in the MuJoCo/Isaac fleet for integration verification.

---

## 6. Verification gates

1. **G0** env rebuild: `Isaac-Velocity-Flat-K1-v0` smoke 16 envs under IL 3.0-EA.
2. **G1** migration: contact rewards fire (`feet_air_time` > 0 in logs) on flattened/re-validated USD.
3. **G2** each base run: reward increases by iter 100; video at iter 200 looks sane; `ball_detect_rate` > 0.9 for P3/P4/P5 camera envs.
4. **G2.5** estimator sanity: `ball_pos_est` error < 15 cm when visible (camera envs); goal estimate snaps to GT (within σ) after sightings and drifts bounded while occluded.
5. **G3** students: distill reward ≥ 80 % of teacher on same env — **evaluated with GT physically disconnected from the deployable obs group** (static test: policy obs terms reference estimator outputs only).
6. **G4** auto-stability: eval script applies scripted 60 N waist shove → P1 recovers (no fall in ≥ 90 % trials), P2 resumes walking — captured as video.
7. **G5** HRL: **P5 student** (estimator-only inputs, no GT) scores goals in ≥ X % eval episodes (set X after P4 baseline); videos show sensible skill switching.
8. **G6** integration: P5 student export drives one robot ball→goal in fleet sim with YOLO fed by the real rendered camera stream end-to-end.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| 8 GB VRAM OOM (cameras + terrain) | env caps, camera subsets, guard watchdog kills at 7 GB |
| IL 3.0-EA is Early Access (APIs still shifting) | pin tag, separate worktree, fall back to `release/3.0.0` branch (49 fixes) |
| `booster_train` fork incompatible (actuator/quat changes) | patch our fork `thakk100/booster_train`, not the wheel |
| YOLO `sports ball` fails on stylized colors | auto-labeled fine-tune from sim frames (free labels) |
| Estimator domain gap (sim occlusion/noise ≠ real Booster depth cam) | estimator-model trained-in noise range + randomized latency/dropout; G2.5/G3/G5 all evaluate estimator-only; real-camera eval is G6 |
| YOLO @50 Hz unaffordable on 4060 | detector runs @10 Hz (hold-between), estimator-model covers non-camera envs |
| Contact sensors still broken after importer rewrite | flatten script re-run; worst case drop R3/R4 (P2) and keep teacher contact-free |
| Disk < 10 GB from videos/checkpoints | retention script keeps last 10 videos + latest/best ckpts |
| zz-bw stays unreachable | local-only plan already sized for it; remote is an optimization, not a dependency |

---

## 8. Execution order (summary)

```
T6.0 git closure (merge phase-0 → main, tag, branch phase-6)
T6.1 env rebuild (Isaac Sim 6.1.0 + IL v3.0.0-EA + smoke G0)
T6.2 migrate existing tasks to IL 3.0 API (velocity first, G1)
T6.3 shared infra: DR (ball pos/color/physics/light), random_body_push,
      camera+YOLO wrapper, video recorder, wandb, train_guard.sh
T6.4 base policies: P1 T/S → P2 T/S → P3 → P4 T/S   (each: smoke → train → gate)
T6.5 P5 HRL: teacher → student → eval (G5)
T6.6 integration + export + handoff update (G4, G6)
```

**No training starts until the user gives the final go-ahead on this plan.**
