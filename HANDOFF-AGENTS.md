# Booster K1 — Two-Agent Handoff (contact-sensor unblock + kick task)

> **Repo:** `/home/thakk100/Projects/booster_ws` (GitHub: `thdhyan/booster_ws`)
> **Branch/HEAD:** `dev/phase-0` @ `430190f` "fix(train_student): guard MLPModel.output_std class property for deterministic distillation"
> **Status:** two parallel agents mid-build, no code landed yet. This doc lets fresh agents (Claude Code / another OpenCode session / a teammate) pick up.

---

## Machine / environment (validated — do not change)

- **Venv:** `/home/thakk100/Projects/IsaacLab/.venv-isaac/bin/python` (Isaac Sim 6.0.1, Isaac Lab 3.0.0b2, rsl-rl-lib 5.0.1, torch 2.11.0+cu128, numpy ≥2).
- **Training target:** `cs-zhang-net-01` (alias `zz-bw`), SingularityCE 4.1.1 (no sudo, proot). Proven flow:
  `~/run_k1_train.sh` + SIF at `/export/scratch/thakk100/k1-train.sif`, logs/env at `/export/scratch/thakk100/k1/`.
  Teacher→student already proven here (velocity task: teacher 5000 it / reward 25.08; student 3000 it / 40.89; final ckpts at
  `/export/scratch/thakk100/k1/logs/rsl_rl/{k1_velocity_teacher/2026-08-25_22-56-12,k1_velocity_student/2026-08-26_01-23-58}/`).
- **HARD RULE: no local training.** Only tiny headless smoke (`--num_envs 16 --max_iterations 2`) locally. Real training = zz-bw. No commits by agents.

---

## BLOCKING issue being solved (root cause, confirmed)

Isaac Sim 6.0 URDF importer NESTS child link prims under parents
(`/Robot/Trunk/Arm_1/foot_link`). This breaks Isaac Lab 3.0.0b2 contact sensors:
1. `activate_contact_sensors` (schemas.py) uses `apply_nested`, stops at first rigid body → only root `Trunk` gets `PhysxContactReportAPI`.
2. `ContactSensor` builds a single parent-level alternation pattern (assumes FLAT hierarchy, like Isaac Sim 5 / G1 assets) → can't address nested bodies.

Upstream: issue IsaacLab#5918; fixed by merged PR #6378 (+ companion #6384) — but on `develop`, NOT in pinned 3.0.0b2. **We do NOT patch the wheel; we FLATTEN the USD instead.**

Decision made: **Flatten the K1 USD → then re-add contact rewards.**

---

## AGENT A — USD flatten + re-add contact rewards

**Session id (for resume):** `ses_fb4b7fd1fffe6p2HJo7Z2lsm12`

### Deliverables
1. **NEW `scripts/flatten_k1_usd.py`** — import K1 (config `BOOSTER_K1_CFG` from `booster_train.assets.robots.booster`), find the converted USD (URDF `UrdfFileCfg` + `force_usd_conversion=True` cache — read `booster.py`/`actuator.py` for the path), `stage.Traverse()` to find all `UsdPhysics.RigidBodyAPI` prims, REPARENT them to robot root (siblings), re-express local transforms (preserve world pose), re-apply `UsdPhysics.PhysxContactReportAPI.Apply()` on every body, write to NEW file `src/k1_description/assets/robots/K1/K1_flat.usd`. Idempotent; never touch source URDF/USD.
2. **Verify** via extended `scripts/probe_k1_bodies.py` (or new `..._flat.py`): spawn flattened USD, assert `ContactSensor` addresses foot bodies; articulation still sims; `robot.body_names` unchanged.
3. **Edit `velocity_env_cfg.py`**: add `ContactSensorCfg` (prim_path `{ENV_REGEX_NS}/Robot/.*`, history_length=3, track_air_time=True, filter `["/World/ground"]`); re-add `feet_air_time` + `feet_slide` rewards (exact G1 rough_env_cfg weights — READ `isaaclab_tasks/.../config/g1/rough_env_cfg.py` in site-packages for signatures); add `foot_contact` obs to TEACHER group only (student stays contact-free). Update the now-stale "contact sensors unavailable" comments.

### Key reference files (agent must read first)
- `scripts/probe_k1_bodies.py` (existing probe)
- `isaac_tasks/booster_train_ref/source/booster_train/booster_train/assets/robots/{booster,actuator}.py`
- `isaac_tasks/k1_velocity/source/k1_velocity/tasks/velocity/velocity_env_cfg.py` (has the BLOCKED-comment block to replace)
- site-packages `isaaclab_tasks/manager_based/locomotion/velocity/{mdp/rewards.py,config/g1/rough_env_cfg.py}`

### Scope limits
- Don't modify installed wheels. Don't touch `tasks/kick/` or `scripts/train*.py` (Agent B's tree). No commit, no long training.

---

## AGENT B — OmniReset-style kick task (state-based, NO vision yet)

**Session id (for resume):** `ses_fb4bcc03cffe2tBna6N33cpVuA`

### Deliverables (all under `isaac_tasks/k1_velocity/`)
1. `source/k1_velocity/tasks/kick/kick_env_cfg.py` — `K1KickSceneCfg`, `ObservationsCfg`, `ActionsCfg`, `RewardsCfg`, `TerminationsCfg`, `EventCfg`, `K1KickEnvCfg`. Task id **`Isaac-Kick-Ball-K1-v0`**.
2. `source/k1_velocity/tasks/kick/mdp.py` — ball pose/vel in robot frame, goal-scored detect, ball-to-goal progress, reset-family samplers.
3. `source/k1_velocity/tasks/kick/agents/rsl_rl_ppo_cfg.py` + `rsl_rl_distill_cfg.py`.
4. `source/k1_velocity/tasks/kick/__init__.py` (gym.register) + update `.../tasks/__init__.py`.
5. `scripts/train_kick.py` + `scripts/train_kick_student.py` (replicate the `output_std` guard from `train_student.py` — see note below).

### MDP design (locked)
- **Scene:** K1 (`BOOSTER_K1_CFG`) + ball as **moving `RigidObject`** (radius 0.11 m, mass 0.43 kg, init (0,0,0.111); needs `root_pos_w`/`root_lin_vel_w`) + kinematic goal posts/crossbar at +x + flat ground + 3 cameras (left/right/disparity, defined but NOT wired yet).
- **Actions:** `JointPositionActionCfg` on `K1_LEG_JOINTS` (12), scale 0.25.
- **Obs:** `policy` = 48-dim proprio (base lin/ang vel, gravity, joint pos/vel ×12, last action); `teacher` = policy + ball pose (3+4) + ball vel (3) in robot frame, `enable_corruption=False`.
- **Reset families (OmniReset core):** `at_ball_shoot` (ball 0.3–0.6 m at one foot), `stand_ready` (0.6–1.2 m behind ball), `walk_up` (2–3.5 m, ball in wide cone). Population-sampled, ratios circulizable + annealed (0.5/0.3/0.2 → uniform).
- **Rewards (minimal, no shaping):** sparse `goal_scored` (dominant) + light `ball_to_goal_progress` + velocity-task regularization (`flat_orientation_l2`, `action_rate_l2`, `dof_torques_l2`, `joint_pos_limits`, `joint_deviation_arms`). **No contact reward (blocked — see Agent A).**
- **Terminations:** time_out, root_height<0.35, bad_orientation>0.8, + goal_scored.

### Known gotchas (must honor)
- rsl_rl 5.x model schema: `actor=`/`critic=` `RslRlMLPModelCfg`, NOT old `policy=`.
- Distillation crash: `OnPolicyRunner.learn()` logs `policy.output_std` → raises for deterministic student (distribution=None). `train_student.py` already has a **class-property guard** — replicate in `train_kick_student.py`.
- Joint A-prefix names (`AAHead_yaw`, `ALeft/ARight_Shoulder_Pitch`).
- Port scene from `src/k1_sim_isaac/scripts/soccer_sim.py` (has field/ball/goal constants).

### Key reference files
- The 5 velocity files in `isaac_tasks/k1_velocity/.../velocity/` (env_cfg, agents ×3, scripts).
- `src/k1_sim_isaac/scripts/soccer_sim.py`.

---

## Verified training commands (for after agents land)

```bash
# LOCAL smoke only (venv-isaac python):
python isaac_tasks/k1_velocity/scripts/train.py --task Isaac-Kick-Ball-K1-v0 --num_envs 16 --max_iterations 2 --headless

# zz-bw real training (kick teacher→student; SIF already built):
ssh zz-bw 'tmux new -d -s k1kick "K1_GPU=1 ~/run_k1_train.sh --task Isaac-Kick-Ball-K1-v0 --num_envs 4096 --headless --max_iterations N"'

# CONTACT-REWARD velocity variant (Agent A): MUST commit + push dev/phase-0 first
# (container clones it at runtime) AND clear the clone cache on zz-bw, else the
# flattened USD + env changes won't be picked up:
ssh zz-bw 'rm -rf /export/scratch/thakk100/k1/tmp/booster_ws'
# then run teacher→student as before.
```

---

## What to tell a fresh agent

"The K1 workspace has two half-done jobs: (A) flatten the K1 USD so contact sensors work, then re-add feet_air_time/feet_slide/foot_contact to the velocity task; (B) build a state-based OmniReset-style kick task `Isaac-Kick-Ball-K1-v0`. Pick one agent and continue from its session id above (follow its Deliverables + Scope limits). No local training — only headless smoke, then train on zz-bw via `~/run_k1_train.sh` + the existing SIF. Read the reference files listed before writing. Respect the gotchas. Don't commit."