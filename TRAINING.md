# TRAINING.md — Booster K1 soccer HRL campaign

**Created:** 2026-09-22. Living table of every planned training run: task ids, sizes,
estimates, checkpoint flow, status. Source of truth for run order = `PLAN_PHASE6_SOCCER_HRL.md` §5.

## Operating rules (user directives, 2026-09-22)

1. **tmux for all training** — launch via `scripts/tmux_train.sh` so runs survive quitting
   OpenCode / closing terminals.
2. **Backend = PhysX for ALL local runs** (user decision 2026-09-22, after measuring
   Newton ~2.5–3× slower @256 and PhysX→Newton checkpoint transfer exploding to NaN).
   Env var `K1_PHYSICS` (default **`physx`**; `newton` = opt-in for probes/other boxes)
   in `k1_velocity/sim_backend.py`, applied in each env cfg `__post_init__`.
   Headless via `HEADLESS=1` (auto-exported by the tmux launcher).
3. **No GT state** for deployable policies (T6.3.5 static test enforces); estimator in loop.
4. **Locked rewards** (PLAN §4) — no silent edits.

## Remote execution on dl (decided 2026-09-22, user directive)

**Why:** the laptop's 8 GB GPU + ~2.6–3.2 GB desktop baseline tripped the guard's
7000 MB cap mid-run-1 (7741 MB peak). All training moves to a server box, headless
(always), tmux, one run at a time. Target **dl** (4× RTX 6000 Ada 48 GB, 503 GB RAM,
64 cores, x86_64) once VPN routes exist — campus subnets (128.101/10.131) unreachable
from home without the tunnel; **zz-bw** (2× RTX PRO 6000 Blackwell 96 GB, no SLURM,
3 TB home) = verified fallback but both GPUs currently compute-saturated by another
user (`abrar008`). **Sparks (GB10, aarch64)**: Isaac Sim unsupported *natively* on arm64,
BUT the `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1` container runs the full training
stack (S0 smoke → S1/P1 + S2/P2 student smokes PASSED 2026-09-23; image's
`activate_contact_sensors` stops at the first rigid body vs the nested 6.0 importer —
worked around via the `_spawn_k1_urdf` wrapper in `booster.py`, idempotent on dl/EA) →
**P1/P2 full students run on spark04** (`scripts/spark_full_host.sh p1|p2`, tmux
`k1_spark_p1`/`k1_spark_p2`); outputs pulled to dl `logs/spark04/` by tmux
`k1_spark_sync` → existing `k1_gdrive_sync` → GDrive (wandb = live dashboard).

**dl recipe (same paths as laptop):**
1. `git clone` booster_ws → `$HOME/Projects/booster_ws` + `git lfs pull`.
2. Clone isaac-sim/IsaacLab, `git fetch origin tag v3.0.0-EA --no-tags`,
   `git worktree add $HOME/Projects/IsaacLab-ea v3.0.0-EA` (tag = `ae37b028e`,
   what the laptop worktree runs; no local IsaacLab patches exist — all ours live
   in booster_ws).
3. `uv sync --extra isaacsim --extra wandb --extra video --extra importers
   --extra rsl-rl` + `uv pip install ultralytics "imageio[ffmpeg]"`
   (= `scripts/install_phase6_env.sh`, ≈ 30 GB incl. Isaac Sim 6.1.0 pip wheels).
4. rsync resume checkpoint `model_1500.pt` from laptop.
5. Launch via `scripts/tmux_train.sh` (exports `HEADLESS=1`; tmux server auto-started
   in its own systemd scope) with server guard env:

   | var | laptop default | dl value |
   |---|---|---|
   | `CUDA_VISIBLE_DEVICES` | (all) | chosen idle GPU (e.g. `2`) |
   | `GPU_IDX` (guard sample) | `0` | same index as `CUDA_VISIBLE_DEVICES` |
   | `GPU_MAX_MB` | `7000` | e.g. `46000` (48 GB card − headroom) |
   | `MEM_MAX_GB` / `SWAP_MAX_GB` | `9` / `4` | e.g. `64` / `64` |
   | `DISK_PATH` | `/` | `$HOME` (NVMe) |

   `GPU_IDX` MUST match `CUDA_VISIBLE_DEVICES` or the guard watches the wrong GPU.
6. `WANDB_API_KEY` exported at launch (same entity/project).

**dl active since 2026-09-22 11:55 (laptop clock)** — `scripts/setup_remote.sh` brought
the box up in 3 idempotent iterations (https rewrite for public submodules on a keyless
host; checkout `dev/phase-6-soccer-hrl` since `main` lacks phase-6 scripts; reuse the
pre-existing clean `~/Projects/IsaacLab` @ beta2 → fetch `v3.0.0-EA` tag → worktree
`ae37b028e`). uv venv + import sanity OK (`isaaclab`/`rsl_rl`/`ultralytics`).
`tmux_train.sh` now snapshots caller env into `logs/tmux_env_<name>` (600) — tmux
sessions inherit the SERVER's env, not the ssh client's. dl joins the **pre-existing**
tmux server (user's `g1`/`isaac-jupyter-*`); the laptop's cgroup/restart hazard (gotcha 4)
is OpenCode-local and does not apply to remote servers.

**Parallel scheme (user directive 2026-09-22):** independent runs on SEPARATE dl GPUs,
all in tmux, **no client-side polling** — wandb is the dashboard. Single-run guard lock
became per-run (`k1_train_guard_<name>.lock`) to allow this. `chain_train.sh` waits for
a session server-side and launches the next run (newest `model_*.pt` picked after the
wait). GDrive: rclone remote `gdrive-dhyan` mirrors `logs/` (guard logs, checkpoints,
videos) to `gdrive-dhyan:Booster/logs` every 10 min from tmux session `k1_gdrive_sync`
(wandb key `logs/.wandb_key` excluded, chmod 600).

| GPU | run | session | envs × iters |
|---|---|---|---|
| GPU1 | P1 teacher finish → **extension** (chained) | `k1_p1_teacher_dl` → `k1_p1_chain` → `k1_p1_teacher_dl_ext` | 256×2000 (running) → 512×3000 from newest ckpt |
| GPU2 | P2 velocity teacher (fresh) | `k1_p2_teacher_dl` | 512×3000 |
| GPU3 | ⚠️ not viable for our runs (20 GiB resident by neighbors vs 30 GiB guard cap) | — | — |
| GPU0 | kick smoke ✅ PASSED (16×2) → **P4 teacher** queued behind P3 | `k1_p4_kick_smoke` → planned | 16×2 → 512×3000 |
| GPU2 (frees when P2 ends) / GPU0 | **P3 head-track** — launches right after T6.3.x authoring | planned | 512×2000 |

Video interval per size: 256 envs → 4800; 512 envs → 2 457 600 env-steps (= every
200 iters). `max_iterations` counts FRESH iterations per launch (resume restarts the
counter: the 09:57-derived dl run does a full 0→2000 block from `model_1500`).

**Wall-time estimates below are laptop-measured PhysX**; dl measured ≈ **1.45 s/iter @256
with video, 4353 steps/s, GPU util 40 %, 6.5 GB VRAM** — update table rows as runs land.

**Run-1 handoff to dl:** resume point = `model_1500.pt` (the 09:57 window saved only
the start copy before tripping the guard at iter ~1574 — its ~74 thrashing iterations
are discarded; rewards were sane: mean −16, all 9 locked rewards logged). Remaining
≈ 430–500 iters × 256 envs.

## Launch / monitor cheat-sheet

```bash
# launch (session k1_<name>, done marker logs/tmux_<name>.done)
# tmux_train.sh exports HEADLESS=1 automatically — no Kit window ever opens
# (pass HEADLESS=0 to opt out and watch the window).
scripts/tmux_train.sh --name p1_teacher_newton -- isaaclab train --rl_library rsl_rl \
  --task <gym-id> --external_callback k1_velocity.register_tasks.register_tasks \
  --num_envs N --max_iterations M --seed 42 \
  [--checkpoint <path/to/model_K.pt>] \
  --headless --enable_cameras --video --video_length 1500 --video_interval 4800

# monitor
tmux attach -t k1_<name>        # detach: Ctrl-b d
tmux capture-pane -t k1_<name> -p | tail -30
tail -f logs/guard_<name>.log
```

- Video: `--video_length 1500` = 30 s @50 Hz; `--video_interval 4800` = every 200 iterations
  (24 env-steps/iter). Clips: `logs/rsl_rl/<wandb-run>/<timestamp>/videos/train/clip_*.mp4`,
  auto-uploaded to wandb as `wandb.Video`.
- One run at a time (guard lock `/tmp/k1_train_guard.lock`). Guard: GPU total < 7000 MB
  (desktop baseline ≈ 1.5 GB → process budget ≈ 5.5 GB), MemAvailable > 1024 MB, disk > 5 GB,
  3 consecutive 5 s samples before kill.
- 4096 envs forbidden locally; ≤ 512 envs. Heavy RL box = remote (currently unreachable).

## Run 10 — hierarchical partial-control base policy (user request 2026-09-23)

**Task `Isaac-Velocity-PartialCtrl-K1-v0`** (new family `k1_velocity/tasks/partial/`):
base policy for hierarchical partial control — **14-dim action = 12 leg DoF + 2 head
DoF**; the **8 arm DoF are outside the action space**: each reset places them at a
curriculum-scaled random pose (default + scale·U(±1 rad), clamped to soft limits) and
holds it via their delayed-PD actuators (episode target written into the articulation
joint-position-target buffer, untouched by any action term). Policy observes arm
pose/vel + head state (68-dim), so it balances & tracks velocity commands under
**variable arm configurations** — precondition for carrying/pushing under an
upper-body controller later.

- **Curriculum `arm_pose`**: `start_scale=0.15` (small perturbations) held for iters
  0–300 → linear ramp to `1.0` (full random poses) by it 2000; writes
  `events.arm_pose_random.params["curriculum_scale"]` each reset (CurriculumManager
  runs before reset events in `_reset_idx`). Play cfg pins scale 1.0.
- **Rewards** = velocity task's locked set with `joint_deviation_arms` (vs fixed
  default) **disabled** (None) and replaced by `arm_pose_deviation` (vs episode hold
  target, −0.05). Scene/commands/terminations/PPO runner inherited unchanged.
- Static gate: `tests/test_deployability.py` passes (10 policy obs terms, proprio only).

| # | Run | PLAN task id | actual gym id (registered) | envs | iters | backend | est. wall time | ckpt in | status |
|---|-----|--------------|----------------------------|------|-------|---------|----------------|---------|--------|
| 10 | partial-ctrl base | — (user request, outside PLAN §5) | `Isaac-Velocity-PartialCtrl-K1-v0` | 512 | 3000 | physx (`isaac-lab:3.0.0-beta2-post1` container) | measured 3.89 s/iter → ETA ≈ 3 h 13 m | `logs/rsl_rl/k1_partialctrl_base/<ts>_k1_partialctrl_base/model_2999.pt` (spark02) | **RUNNING 2026-09-23 22:22 on idle `spark02`** (dl: all 4 GPUs busy ~38/49 GB by other users; sparks01/03/04 busy; spark02 load 0.35, 1.9 T free ≥500 G ✓) — smoke 16×3 in-container **PASSED** (gated: full run only on `HP_SMOKE_RC=0`), tmux `k1_spark_hp`, log `scripts/hp.full.log`, wandb run [`1zr6w8a2`](https://wandb.ai/thakk100-dhyan-home/booster_k1_soccer_hrl/runs/1zr6w8a2) (`k1_partialctrl_base`); startup verified: action shape **14** (2 terms), policy obs **68**, 11 reward terms (`arm_pose_deviation` active, −0.0003/ep early), `Curriculum/arm_pose=0.15`, `Curriculum/terrain_levels≈2.9`, 0 Tracebacks; **progress 2026-09-24 00:45 UTC: iter 1419/3000 (47 %), `Curriculum/arm_pose=0.7098`, `Episode_Reward/arm_pose_deviation≈−0.007/ep`, `model_1400.pt`, 0 Tracebacks, ETA ≈ 1 h 45 m** |

  Launch recipe (reproducible): `scripts/spark_partial_host.sh` → tmux `k1_spark_hp` →
  `scripts/spark_partial_container.sh` (pip -e ×3 → smoke 16×3 `train.py` → gate →
  full `--num_envs 512 --max_iterations 3000 --seed 42 --video`). Outputs stay on
  spark02 (`~/Projects/booster_ws/logs/...`); `k1_spark_sync`/gdrive currently mirrors
  **spark04 only** — pull spark02 `logs/` manually if a checkpoint is needed elsewhere.

## Campaign table (11 runs)

Wall-time estimates: **PhysX measured baseline** — 256 envs = 1.59–1.63 s/iter fresh
(≈ 2.1–2.5 s/iter with video capture + swap pressure); startup ≈ 2–3 min.
512-env tasks scale ≈ 1.6–2× per iteration + YOLO-env overhead where noted.
**Newton measured (2026-09-22):** 256 envs = **4.8–5.0 s/iter** steady (24 s first-iter
warp JIT), startup ≈ 2–4 min → 2000 it ≈ 2.8–3 h — **~2.5–3× SLOWER than PhysX on this
RTX 4060**, despite headless. 16 envs = 3.6 s/iter. Newton-at-256 = stable with a fresh
policy (probe `g3_newton_256_fresh` rc=0, 0 NaN) but **PhysX→Newton checkpoint transfer
explodes to NaN within 1 iteration** (run-1 resume attempt, 2026-09-22 09:37).

| # | Run | PLAN task id | actual gym id (registered) | envs | iters | backend | est. wall time | ckpt in | status |
|---|-----|--------------|----------------------------|------|-------|---------|----------------|---------|--------|
| 1 | P1 teacher | `Isaac-Basic-Teacher-K1-v0` | `Isaac-Basic-Teacher-K1-v0` | 256 | 2000 | physx | PhysX dl actual ≈ 46 min finish + ~68 min ext | `model_1500.pt` (physx runs 0→~1548) | **DONE 2026-09-22** — finish run GUARD_RC=0 (2000 it → model_3500) auto-chained ext GUARD_RC=0 (512 envs, 3000 it) → **best `model_6498.pt`** (`p1_basic_teacher/2026-09-22_13-41-33_p1_basic_teacher`); peak mean reward **−0.27** (single-iter spike), settles −16/−17; finish-run clips non-black (mean 89 / std 53) |
| 2 | P1 student | `Isaac-Basic-Student-K1-v0` | `Isaac-Basic-Student-K1-v0` via Path B `train_student.py --teacher_checkpoint` (EA `isaaclab train` distill strict-loads the PPO ckpt → `std_param` crash) | 256 | 1500 | physx | PhysX ≈ 45–60 min (dl) | #1 final `model_6498.pt` | **RUNNING spark04 2026-09-23** — S1 smoke PASSED on `isaac-lab:3.0.0-beta2-post1` (16×3, contact obs fix `_spawn_k1_urdf`, no Traceback) → full run `tmux k1_spark_p1` (`spark_full_host.sh p1`); **dl queue CANCELLED** (all 4 dl GPUs busy w/ tang0836, no ETA); outputs pulled to dl `logs/spark04/` by `tmux k1_spark_sync` → gdrive |
| 3 | P2 teacher | `Isaac-Move-Teacher-K1-v0` | `Isaac-Velocity-Rough-K1-Teacher-v0` | 512 | 3000 | physx | PhysX ≈ 2–3 h / dl actual ≈ 73 min | — | **DONE 2026-09-22 14:27 dl** — GUARD_RC=0, 3000 it → **`model_2999.pt`** (`k1_velocity_teacher/2026-09-22_13-14-06`), reward −2.94 → peak **16.22** → final 14.0; logs+ckpt synced to gdrive; wandb run renamed → `p2_move_teacher` (2026-09-23, API) |
| 4 | P2 student | `Isaac-Move-Student-K1-v0` | `Isaac-Velocity-Distill-K1-v0` via Path B `train_student.py --teacher_checkpoint` | 512 | 3000 | physx | PhysX ≈ 2–3 h | #3 final `model_2999.pt` | **RUNNING spark04 2026-09-23** — S2 smoke PASSED on isaac-lab image (feet_air_time > 0, ckpt+video+wandb) → full run `tmux k1_spark_p2` (`spark_full_host.sh p2`); **dl queue CANCELLED**; runs parallel with #2 |
| 5 | P3 head-track | `Isaac-HeadTrack-K1-v0` | `Isaac-HeadTrack-K1-v0` | 512 (camera + YOLO) | 2000 | physx (`isaac-lab:3.0.0-beta2-post1`) | current ≈ 21–22 s/iter; P3 ETA ≈ 10 h; Track A ETA ≈ 06:30–08:00 UTC 25 Sep | `logs/rsl_rl/p3_head_track/2026-09-24_14-58-47_p3_head_track/model_1999.pt` | **RUNNING on spark02** (tmux `k1_spark_p3`, full started 2026-09-24 14:58 UTC; iteration 333/2000, mean reward 30.53, `model_300.pt` saved at 17:20 UTC). Zero-step and real-YOLO 16×3 smoke **PASSED**; fixed `(N,3)`→`(N,6)` rigid-body velocity write, isolated Isaac cache, explicit final-iteration marker, 200-iteration video cadence, and FP16 320px full-batch YOLO. WandB [`oavkxiwf`](https://wandb.ai/thakk100-dhyan-home/booster_k1_soccer_hrl/runs/oavkxiwf), log `scripts/p3.full.train.log`. |
| 6 | P4 teacher | `Isaac-Kick-Teacher-K1-v0` | `Isaac-Kick-Ball-K1-Teacher-v0` | 512 | 3000 | physx (`isaac-lab:3.0.0-beta2-post1`) | ≈ 3–5 h | `logs/rsl_rl/p4_kick_teacher/<run>_p4_chase_kick/model_2999.pt` | **SMOKE PASSED; FULL QUEUED** on spark02. Teacher now has 55-dim privileged obs and 14-dim legs+head action. Per-environment goals, goal observation, and approach/kick/align shaping are fixed; 16×3 smoke produced `model_2.pt` + video, WandB smoke [`b4f5ag2d`](https://wandb.ai/thakk100-dhyan-home/booster_k1_soccer_hrl/runs/b4f5ag2d). Server chain `k1_soccer_chain` waits for `SOC_FULL_MARKER=OK` from P3, then runs `spark_soccer_host.sh p4t`. |
| 7 | P4 student | `Isaac-Kick-Student-K1-v0` | `Isaac-Kick-Ball-K1-Distill-v0` `--distill` | 512 (64 YOLO+vid) | 3000 | physx | PhysX ≈ 2.5–3.5 h | #6 final | queued (Gate G5; estimator/student wiring remains) |
| 8 | P5 teacher | `Isaac-HRL-Teacher-K1-v0` | *not yet authored* (T6.3.4) | 256 (32 YOLO+vid) | 2000 | physx | PhysX ≈ 1–1.5 h | #1–#7 | blocked: T6.3.4 |
| 9 | P5 student | `Isaac-HRL-Student-K1-v0` | *not yet authored* (T6.3.4) | 256 | 1500 | physx | PhysX ≈ 50–70 min | #8 final | queued (Gate G6) |
| 10 | P6 push reach | `Isaac-Push-Reach-K1-v0` | `Isaac-Push-Reach-K1-v0` | 256 | 1500 | physx (`isaac-lab:3.0.0-beta2-post1`) | ≈ 1–2 h | `models/k1_partialctrl_base.pt` (frozen) | **CRASHED spark04 2026-09-24** — CUDA `cuda-EvtHandlr` wedge at iter ~690 (container killed; logs `scripts/reach.push.crashed.log`). Not learning anyway: reward flat −3.85 → −3.60, ep_len 16–23, 99.7 % `root_height` terminations. Root cause (open): frozen Run-10 TorchScript base **falls every ~17 steps in P6 even with zero commands** (hold-mode diag stands perfectly; home partial env walks — see `scripts/diagfrozen.log`). Suspect: export-path semantics (`as_jit()`); pending offline parity test. |
| 11 | P6 push | `Isaac-Push-K1-v0` | `Isaac-Push-K1-v0` | 256 | 3000 | physx | ≈ 3–5 h | #12 re-export (`model_2999.pt`) | queued behind **#12** — `spark_push_host.sh push` warm-starts via `--checkpoint` (auto-picked latest `p6_push_reach` model); box DR edge 0.7–1.5 m / 3–25 kg / friction 0.3–1.2, goal curriculum 0.3→1.5 m, wandb `p6_push` |
| 12 | P6 base retrain (gait-v2) | `Isaac-Velocity-PartialCtrl-K1-v0` | `Isaac-Velocity-PartialCtrl-K1-v0` | 512 | 3000 | physx (`isaac-lab:3.0.0-beta2-post1`) | ≈ 3–6 h dedicated GB10 | → re-export `models/k1_partialctrl_base.pt` via `scripts/export_base_policy.sh` | **RUNNING spark04 2026-09-24** — tmux `k1_spark_push_base`, launcher `scripts/spark_push_base_{host,container}.sh` (smoke marker-gated → FULL), log `scripts/push_base.full.log`, wandb `k1_partialctrl_base`. Retrains the frozen base on the reviewed gait-v2 velocity cfg (`feet_air_time` +0.5, `feet_slide` −0.25, `stand_still`, `undesired_contacts`, direct Cartesian cmds ±1.5/±0.75/±1.5 — all verified firing via `scripts/reward_probe.sh` on spark04 2026-09-24). Fresh base + fresh export = Track B unblock path. |

  **Totals (PhysX for all runs — user decision 2026-09-22):** ≈ **15–21 h sequential**.
Newton measured slower on this box (4.8–5.0 vs 1.6–2.1 s/iter @256) and can't resume
PhysX checkpoints → retained only as `K1_PHYSICS=newton` opt-in. Each run preceded by a
16-env smoke (gate); iter counts are ceilings — stop early on plateau; user reviews
videos every 200 iters.

### Notes

- Run 1 checkpoint chain: `2026-09-22_08-19-32` (PhysX, 0→~1100, SIGKILL at 1147 — no
  cgroup/OOM trace, swap 9.2 GB) → `2026-09-22_08-59-01` (PhysX resume, 1100→~1548) →
  newton resume from `model_1500.pt` **CRASHED: NaN obs @iter 1 (PhysX→Newton policy
  transfer explodes; fresh-policy probe proved scale@256 is fine)** → **PhysX headless
  finish from `model_1500.pt`** (tmux `k1_p1_teacher_physx_finish`, 09:57) → target
  `max_iterations 2000` absolute (rsl_rl counts from the checkpoint's iteration, so
  `--max_iterations 2000` = finish the campaign).
- Newton feasibility verified statically (2026-09-22): `NewtonCfg` selection pxr-free for all
  3 cfg families; `isaaclab_newton` has contact_sensor + ray_caster factories; custom
  BoosterDelayedPDActuator forced through Isaac Lab path (`use_newton_actuators=False`).
  Runtime = G3 smoke (`k1_g3_newton_smoke`, 16×3) before committing run 1.
- **Newton bring-up gotchas (G3 smoke, 2026-09-22):**
  1. `ContactSensorCfg.filter_prim_paths_expr=["/World/ground"]` crashes Newton init
     (`No bodies matched the counterpart pattern(s)` — ground plane isn't a body label
     in the Newton model). Fix: `sim_backend._clear_contact_filters` clears sensor
     filters under Newton only. Semantically safe: all consumers read **net** forces +
     air/contact timers (unfiltered in PhysX too); `filtered_forces`/`force_matrix`/
     `contact_pos` are never read in k1_velocity (verified).
  2. Scene cfg field `class_type: ResolvableString` has a lazy `__getattr__` that
     resolves the path template **and imports 43 pxr modules** — a plain
     `getattr(cfg, 'filter_prim_paths_expr', None)` walk over the scene would
     re-introduce the G2 Kit-poisoning pxr import pre-SimulationApp. Fix: read sensor
     fields from the instance `__dict__` directly (bypasses `__getattr__`).
     Static check after fix: newton → `NewtonCfg` + filter `[]` + pxr 0;
     physx → `NoneType` + filter kept + pxr 0.
  3. **`--headless` CLI arg is NOT honored by the unified `train` backend** (the
     export path forces `args_cli.headless = True` programmatically; train does not
     parse/forward it). Every run — including the morning PhysX runs — was actually
     **windowed**: the "Isaac Lab 3.0.0" window on the desktop was Isaac Lab's env UI
     (`base_env_window.py:73 "Creating window for environment."`). Fix: `HEADLESS=1`
     env var (AppLauncher `app_launcher.py:890`), now exported by
     `scripts/tmux_train.sh`. Verified: headless smoke log = 0 window lines,
     no Kit window at exit, clips still non-black + motion.
   4. **A plain detached `tmux` server is NOT restart-proof here**: the server
      inherits OpenCode's cgroup (`vte-spawn-*.scope`), so an OpenCode server restart
      kills the server → sessions die → training dies mid-run (observed 2026-09-22
      09:51: killed at iter ~1502, stale lock left behind — clear with
      `rmdir /tmp/k1_train_guard.lock`). Fix in `tmux_train.sh`: if no server is
      running, start one via `systemd-run --user --scope tmux new-session -d -s
      k1_tmux_holder 'while :; do sleep 3600; done'` → server lives in its own
      `run-*.scope`, sessions in `tmux-spawn-*.scope`, both outside OpenCode's
      cgroup (verified via `/proc/<pid>/cgroup`; 09:57 relaunch survived).
- **G3 smoke PASSED (2026-09-22 — two runs, rc=0, windowed then headless):**
  banner `Physics newton_mjwarp`; `Registered backend 'newton'` for Articulation +
  **ContactSensor** + **RayCaster**; 9 locked rewards logged; 2 clips/run written
  (32 frames, 640×360, first-vs-last frame diff 2.7 = motion; cubric warning is
  precautionary — frames NOT black); iter time @16 envs ≈ 3.6 s steady
  (first iter 7.7 s = warp JIT); startup ≈ 46–59 s; no guard LIMIT HIT.
- Env counts: 512-env rows planned per PLAN; actual envs capped by GPU guard (256 = ~6.3 GB
  device peak under PhysX+video). Bump only if measured peak leaves headroom — record here:
  - P1 (256): device peak 6315 MiB (PhysX+RTX, video every 200 it) → at cap, stay 256.
- Registry: **17** K1 gym ids registered (basic T/S **+ P1f teacher/student F**, velocity rough/distill/play + teacher,
  **+ P2f teacher/student F**, kick ball base/teacher/distill, **partial-ctrl train/play**). PLAN ids renamed → actual id column above (aliases
  `Move`/`HeadTrack`/`HRL` may be added later for PLAN parity).

## Run 11 — partial control + mid-episode arm-delta curriculum (queued 2026-09-24)

User request 2026-09-23: *random delta changes to arm-joint positions during the
episode, on a curriculum that starts with **0 changes** and slowly increases as the
model improves.* Implementation lives in the env cfg only — **Run-10 is unaffected**
(it launched before this code existed and never saw deltas):

- **Event** `arm_delta_change` (`mode="interval"`, 2–5 s/env) —
  `partial/mdp.py:randomize_arm_pose_delta` adds `curriculum_scale · U(−1,1)` rad to
  **each arm joint's current PD hold target** (clamped to soft limits; hard no-op
  while scale == 0). The delayed-PD actuators glide to the new pose mid-walk;
  `arm_pose_deviation` (reward) tracks the moved target automatically — no
  joint-state teleport, so it's a smooth standing disturbance.
- **Curriculum** `arm_delta` (reuses the `arm_pose_curriculum` ramp helper):
  scale **0.0 → 1.0** linearly over iterations **600 → 2500** of 3000 — starts
  with zero changes, grows only after the policy has mastered static random poses
  (the reset-pose `arm_pose` ramp 0.15 → 1.0 over 300 → 2000 still runs first).
- **Play cfg** (`partial_play_cfg.py`) pins `arm_delta_change` scale to **0.0** —
  Run-10 checkpoints never saw deltas; raise to 1.0 to play Run-11 ckpts.
- **Launch:** after Run-10 finishes on spark02 — pull `model_2999.pt`, preserve
  `hp.full.log` → `hp.run10.log` (host script truncates the log), rsync spark02,
  then `scripts/spark_partial_host.sh` (512×3000, seed 42, tmux `k1_spark_hp`).

## Runs F1–F4 — force-variant P1f / P2f (2026-09-24, spark04)

Isaac Lab's **native** `envs.mdp.apply_external_force_torque` event drives a
sustained random wrench on the Trunk — *teacher observes the shove, deployable
student does not* (user request 2026-09-23: use Isaac Lab's built-in force/torque
events; researched + implemented as `shove_mdp.py` + per-family F cfgs).

| F-run | Variant | gym id | envs × iters | experiment | chain |
|---|---|---|---|---|---|
| F1 | P1f teacher | `Isaac-Basic-Teacher-K1-F-v0` | 256 × 2000 | `p1f_basic_teacher` | 16×3 smoke gate → full |
| F2 | P1f student | `Isaac-Basic-Student-K1-F-v0` | 256 × 1500 | `p1f_basic_student` | F1 final ckpt |
| F3 | P2f teacher | `Isaac-Velocity-Rough-K1-Teacher-F-v0` | 512 × 3000 | `p2f_move_teacher` | 16×3 smoke gate → full |
| F4 | P2f student | `Isaac-Velocity-Distill-K1-F-v0` | 512 × 3000 | `p2f_move_student` | F3 final ckpt |

Design notes:

- Event `shove_force_torque`: interval 4–8 s/env, force ±30 N, torque ±10 N·m on
  `Trunk`, written into the `permanent_wrench_composer` (persists across resets
  until the next resample).
- Teacher obs group gains `shove_wrench` 6-dim
  (`shove_mdp.applied_shove_wrench` reads the composer's
  `out_force_b`/`out_torque_b`): P1f teacher 233+6=**239**, P2f teacher
  235+6=**241**; student `policy` groups untouched (**42 / 48** blind).
- P1f **disables** P1's custom `random_body_push` — both drive the same wrench
  buffer and would clobber each other (one wrench authority per task).
- Launch: `scripts/spark_force_host.sh p1f|p2f` → tmux `k1_spark_p1f` /
  `k1_spark_p2f`, logs `scripts/p1f.f.log` / `scripts/p2f.f.log`,
  teacher→student chained in one container session (smoke-gated, exits 10/11/13
  on gate failure).
- `tests/test_deployability.py` green with these cfgs: **13 tasks, TEST_RC=0**;
  F-teacher groups 8/9 terms (privileged). Group classes must end in `Cfg`
  (`ForceTeacherCfg` — `_iter_groups` skips others).

## Play recordings (2026-09-24)

`scripts/record_policies_host.sh` → container → `play_record.py`: headless
750-step (15 s @ 50 fps, 1024×576, 4 envs) rollouts with a live HUD (velocity
command, every obs group as value bars, action bars, step + episode reward,
task/ckpt/step badge), a full-fidelity `videos/*_trace.npz` sidecar (git-ignored),
and TorchScript exports to `models/` (Git LFS):

| video | task | checkpoint | export |
|---|---|---|---|
| `videos/p1_teacher_stand.mp4` | `Isaac-Basic-Teacher-K1-v0` | `p1_basic_teacher/…13-41-33…/model_6498.pt` | `models/p1_basic_teacher.pt` |
| `videos/p1_student_stand.mp4` | `Isaac-Basic-Student-K1-v0` | `p1_basic_student/…16-20-36…/model_1499.pt` | `models/p1_basic_student.pt` |
| `videos/p2_teacher_rough.mp4` | `Isaac-Velocity-Rough-K1-Teacher-v0` | `k1_velocity_teacher/…13-14-06/model_2999.pt` | `models/p2_move_teacher.pt` |
| `videos/p2_student_walk.mp4` | `Isaac-Velocity-Distill-K1-Play-v0` | `p2_move_student/…13-46-54/model_2999.pt` | `models/p2_move_student.pt` |
| `videos/partial_walk.mp4` | `Isaac-Velocity-PartialCtrl-K1-Play-v0` | Run-10 final (after finish) | `models/k1_partialctrl.pt` |

- Walking tasks use `--cmd 0.6 0.0 0.5` (circle path keeps the robot in the
  fixed camera frame); PPO vs distill ckpts auto-detected from the agent cfg.
- **Gotcha fixed 2026-09-24:** `RslRlVecEnvWrapper.reset()` returns a
  **TensorDict** — `for k in obs` falls back to the sequence protocol
  (`obs[0]`, `obs[1]`… = batch slices) and never yields the group keys, so the
  first batch crashed with `KeyError: 'obs_policy'` (4× ~10 KB header-only mp4s).
  Fix: iterate `obs.keys()`; the runner now also **verifies artifacts**
  (mp4 ≥ 200 KB + trace + export) because `python.sh` reported rc=0 on a crashed
  run.
- **Results:** `REC_FAILS=0`, mp4s 0.96 / 1.82 / 1.67 / 2.16 MB, JIT parity
  max|Δa| = 2.7 / 2.7 / 3.7 / 2.2e-03 (TF32-level); parity ref now follows the
  live policy's device (the original check fed CPU obs into the CUDA policy and
  crashed *after* the export was saved).
- **Mirrors:** [Drive folder](https://drive.google.com/drive/folders/1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy)
  (uploaded via `GOOGLEDRIVE_UPLOAD_FROM_URL` from the public raw.githubusercontent
  URLs, anyone-with-link reader) · [Slide deck with every video embedded](https://docs.google.com/presentation/d/1KamnVS6DEQMrtXbk9z8Mp5qdG9_XTeqRJZTkRXMexyI/edit)
  (8 slides: overview, recordings table, one video slide per policy, new
  features, status — embeds via Slides `createVideo`, `source: DRIVE`).

## P3 head tracking + P4 teacher update (2026-09-24, branch `dev/soccer-p3p4`)

**P3 `Isaac-HeadTrack-K1-v0`** (new family `tasks/head/`) — head-only ball
tracking, **independent of locomotion/balance** (legs held at the standing
default; the robot never walks in this task) and **no ground truth in obs**:

- **Detector = the only ball signal.** Head cam (320×240 RGB on `Head_2`, 10 Hz,
  moves with the head joints) → YOLOv8n COCO `sports ball` bbox centre →
  `(visible, du, dv)`, held between detections. A geometric FOV proxy through
  the current head angles is the automatic fallback when the detector or its
  weights are unavailable (keeps smoke/deployability green offline).
- **Obs (12):** detection 3 · head pos 2 · head vel 2 · base ang vel 3 · last
  action 2.
- **Action (2):** `AAHead_yaw`, `Head_pitch` @ scale 0.5.
- **Rewards:** `ball_centered` **+2.0** = `exp(−(du²+dv²)/0.35²)` on the
  detection (the "keep the bounding box centred" reward) · `track_ball_angle`
  +1.0 (geometric, reward-time only) · `ball_in_frame` +0.5 · `action_rate_l2`
  −0.1 · head `joint_pos_limits` −1.0 · `time_penalty` −0.01.
- **Curriculum `ball_speed`:** **0 → 0.8 m/s** linear over iters **200 → 1200**
  — static ball first (centre the box), then a slowly rolling ball the head
  must keep tracking. Direction resampled per reset; a 0.5 s interval event
  re-asserts speed against ground friction.
- **CCW search (conditional):** no detection for **0.5 s** (25 steps) → the
  locomotion policy is issued an **in-place counter-clockwise** command
  `(0, 0, +0.6 rad/s)` (zero x/y so the robot rotates in place to re-acquire
  an out-of-FOV ball). Implemented as `head_mdp.ccw_search_command` (env-side,
  tensor) and `PolicyComposer.update_search` (runtime, `active_cmd`).
  Compose head-obs rebuilt to the new 12-dim detection format.

**P4 teacher update** (`K1KickTeacherEnvCfg`, same `…-Teacher-v0` id): teacher
now also **drives the head** (legs 12 + head 2 = 14-dim action) and gains
`ball_in_frame` (+0.5, while robot-ball > 0.75 m) + `head_ball_aim` (+0.5
exp kernel) shaping; arm deviation penalty excludes the now-active head. The
GT ball/goal `teacher` obs group is unchanged → the distilled student (which
will consume P3 detections + the vision estimator) is unaffected.

- Launchers: `scripts/spark_soccer_host.sh p3|p4t` → container
  (`spark_soccer_container.sh`): installs pkgs (+ultralytics + yolov8n for p3),
  **smoke-gated 16×3** (cameras + video), then full 512×2000 (p3) /
  512×3000 (p4t).
- Gate: `tests/test_deployability.py` **HOLDS — 14 tasks** incl.
  `Isaac-HeadTrack-K1-v0` (policy 5 terms) and the 3 kick ids (teacher 8 terms).
- Compose runtime: `src/k1_sim_isaac/scripts/k1_soccer_compose.py` (12-dim head
  obs + CCW search state machine).

### Interim Track A recordings (2026-09-24)

Recorded from the current server checkpoints with `play_record.py` (4 envs,
750 steps / 15 s, HUD + trace):

| video | checkpoint | Drive file |
|---|---|---|
| `track_a_p3_head_track_interim.mp4` | P3 `model_200.pt` | [open](https://drive.google.com/file/d/1HQCztYaw4s8aR37Vf2VvCJ6xP_n8ycYa/view) |
| `track_a_p4_teacher_smoke.mp4` | P4 teacher smoke `model_2.pt` | [open](https://drive.google.com/file/d/1BBivuueov76p1uOrmjpVNd2cvCHvVEmb/view) |

Both files are in the reader-shared [Track A Interim folder](https://drive.google.com/drive/folders/1BWLirPTpn_lPEaMHwCUkAEiCAD9uA6ge).
The [slide deck](https://docs.google.com/presentation/d/1KamnVS6DEQMrtXbk9z8Mp5qdG9_XTeqRJZTkRXMexyI/edit)
has labeled poster-preview slides and the shared-folder link. Direct Drive
`createVideo` embedding was rejected by the Slides connector; final recordings
should replace these interim previews when P3/P4 finish.

### P1/P2 recording-camera correction (2026-09-24 18:40 UTC)

The P1 clips were not missing the robot: the RGB recorder initialized its camera
at world origin while rough-terrain env 0 was offset (for example, `(-15,-55)`).
`play_record.py` now updates the recorder camera from the env-0 robot root on
reset and every frame; P1 teacher/student were re-recorded for 750 steps with
finite float32 traces and verified root traces. The corrected files are in the
same reader-shared Drive folder:

- [P1 teacher — corrected](https://drive.google.com/file/d/1E4ej6sOnXfPQOZQsVoTDNirDhL1goE14/view) · file ID `1E4ej6sOnXfPQOZQsVoTDNirDhL1goE14`
- [P1 student — corrected](https://drive.google.com/file/d/1r4IswjbF32fvjL5f2oNaYSlaIZjHQDS1/view) · file ID `1r4IswjbF32fvjL5f2oNaYSlaIZjHQDS1`

The P2 old checkpoint was also re-framed to show all four environments, but it
still does not walk; it is not presented as a gait success. The gait-v2 teacher
and student smoke gates passed on spark04. The full P2 campaign is queued on
spark02 in tmux `k1_spark_p2_gait` behind the P3/P4 Track A markers:
`scripts/spark_p2_gait_{host,container}.sh`. Final P2 clips must come from the
new `model_2999.pt` checkpoints and be recorded only after both final markers.

## Drive video sync from training servers (2026-09-24)

Training videos (`logs/rsl_rl/<exp>/<run>/videos/*.mp4` from the `--video`
recorder) now mirror to Google Drive from the **training spark itself**:
`rclone` (linux-arm64) + the existing `gdrive-dhyan` remote, tmux loop
`k1_gdrive_sync` copying every 10 min:

```bash
rclone copy ~/Projects/booster_ws/logs gdrive-dhyan:Booster/logs \
  --include '*.mp4' --min-age 30s
```

Policy-recording mirrors remain the Drive folder + slide deck linked above.
