# Booster K1 Workspace — Handoff

## 📌 SESSION UPDATE — 2026-10-01, later (gait REWARD repair landed; training host moved zz-bw → dl → **aim_spark02** — read this first)

### Round 2: the three levers were not enough — root cause was in the reward

Retrained teacher (`7ce0049`, 4096 envs × 3000 iters on zz-bw, 2.4 h) **still
failed `gait_gate.py` 8/8**: cadence **6.68** steps/s (want 1.2–4.0), stride
~0.07 m (want ≥0.15), jerk **0.099** (want <0.06), speed 0.23–0.53 m/s. Video
`isaac_tasks/k1_velocity/videos/p2_gaitfix_teacher.mp4` (207 MB, 750 frames,
8 envs) — robot upright and moving, feet shuffling.

Trace diagnosis (`p2_gaitfix_teacher_trace.npz`):

- **The phase clock works and is ignored.** Phase advances 0.02002/step =
  **1.001 Hz**, full sin/cos range, last term in both groups — but
  max |corr(action, clock)| = **0.098** over all 24 joint×channel pairs.
  Nothing in the objective paid for reading it.
- **`gait_cadence_penalty` was measuring the wrong signal.** It zero-crossed
  `joint_pos`, whose slow postural drift (0.07–1.0 Hz) swamps the knee's 4–6 Hz
  stepping oscillation, so it read ~1–3 steps/s on a 6.68 shuffle and paid only
  **−0.394/step**. `feet_clearance` contributed −0.0002 (dead) and
  `stride_length` measured *static* foot separation (0.21 m) against a 0.07 m
  gate stride — it rewarded a wide stance, not steps.
- Two failure modes in the fleet: 5/8 envs chattering at 4–6 Hz (physically
  real: `max |qvel| 17.8 rad/s`), 3/8 barely stepping at all (cadence 0.13–1.2).

**`10f7fc9` + `4e2ff82`** (both on `main`, pushed):

1. `gait_cadence_penalty` reads `data.joint_vel` (drift-free) instead of
   `joint_pos`.
2. New `phase_synced_swing` reward (`gait_rewards.py`, wired as term #23,
   weight −2.0): each foot owes contact over its clock half-cycle, so
   alternating single support locked to the 1 Hz reference (2.0 steps/s) is the
   cheap state. Contact-based, so it holds on slopes where world-z lies.
   Reads the phase through `gait_clock.get_phase(env)` — the same state the
   observation encodes, idempotent per control step.
3. `stride_length_penalty` projects the foot separation onto the **body forward
   axis** (lateral splay no longer counts as stride).
4. **`SceneEntityCfg.preserve_order=True`** on the `phase_swing` contact cfg:
   `preserve_order` defaults to **False**, which makes `body_ids` follow the
   *sensor's* body order, not the requested `[left, right]` list. Index 0 was
   only the left foot because the URDF declares `left_foot_link` (line 935)
   before `right_foot_link` (line 1277).

Tests: **108 passed** (+5 new guards: cadence must read `joint_vel`, clock
sharing, `preserve_order`, `phase_swing` wiring/negative weight), locomotion
`4 passed, 1 skipped`.

### Gates for round 2 (all green)

| gate | result |
|---|---|
| `probe_rewards.py` zz-bw | `REWARD_PROBE_MARKER=OK`, `STANDS` 0.553 m, `phase_swing` −1.25/step |
| `probe_rewards.py` spark02 | `REWARD_PROBE_MARKER=OK`, `STANDS` 0.559 m, `phase_swing` **−1.00000** (= penalty 0.5: standing, both feet planted, exactly one foot always owes swing) |
| 16-env smoke zz-bw | 0 tracebacks, `policy (50,)`/`teacher (237,)`, `phase_swing` #23 live, `model_0/1.pt` |
| 16-env smoke dl | same |
| 16-env smoke spark02 | same, `model_0/2.pt` |
| smoke debug video | `gaitfix_smoke2.mp4` (200 frames × 4 envs) frame-checked: HUD `step 100/200`, 4 labelled panels render on the 50-dim obs |
| `pytest tests/` | **108 passed** |

### Host migration — the 26× slowdown was GPU contention, NOT the code

Same code family, three hosts, measured `Collection time` per iteration:

| run | host / GPU | co-tenant on that GPU | collection |
|---|---|---|---|
| first retrain (worked, 2.4 h) | zz-bw GPU1 | **exclusive** | **1.139 s** |
| 2nd attempt | zz-bw GPU1 | `abrar008` `train_v8` moved onto GPU1 | 25.7 s |
| 3rd attempt | dl GPU0 | ollama serving 37.5 GB on GPU0 | 17.2 → 29.4 s |

`Learning time` never moved (0.075 → 0.13 s): the regression is **entirely
rollout collection**, i.e. our kernels time-sliced behind someone else's big
job. Ruled out: no cgroup quota (`cpu.max` n/a), no throttling
(`nr_throttled 0`), 64 real cores on both boxes, `Environment device: cuda:0`
identical, GPU memory fine. GPU `utilization.gpu` reads a misleading 4–19 %
because the *neighbour's* kernels are what saturate.

**A/B that exonerated the reward repair** (dl, same box, same GPU, same minute,
4096 envs × 3 iters):

| code | collection time |
|---|---|
| `7ce0049` three-lever (no `phase_swing`, cadence on `joint_pos`) | 13.193 s / 15.389 s |
| `4e2ff82` round-2 rewards (current) | 12.868 s / 15.007 s |

Identical → the reward changes cost nothing; the code stands as committed.

### Host setups that now exist

**dl** (`128.101.125.152`) — native, no container:

- `~/k1_run_ws` = clone of `main` at `4e2ff82` + submodules (its
  `~/Projects/booster_ws` is a **dirty `dev/soccer-p3p4` tree — left untouched**).
  Gotcha: `git clone <local path>` makes `origin` the *local* repo, so
  `origin/main` resolved to dl's stale local `main`; re-point with
  `git remote set-url origin https://github.com/thdhyan/booster_ws.git`.
- `~/run_k1_train_dl.sh` (sources `scripts/phase6_env.sh`, `CUDA_VISIBLE_DEVICES=$K1_GPU`,
  `nice -n 10`, `K1_TRAIN_SCRIPT` override).
- **dl's IsaacLab is v3: `--headless` is NOT a valid flag** — use `--viz none`
  (what the container on zz-bw, being v2-era, accepted).
- 4× RTX 6000 Ada but only ~11 GB free each (ollama resident on all four);
  64 cores, load ~35. Usable, just not fast while ollama serves.

**aim_spark02** (`10.131.37.135`, GB10, 20 cores, 121 GB unified) — **the
training host**:

- Idle GB10 (0 % util, load 0.52) and **`nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`
  already built** — the image `scripts/spark_p2_gaitfix_*.sh` was written for.
  IsaacLab **3.0.0** in-image, `SceneEntityCfg.preserve_order` present ✓.
- `~/k1_run_ws` = clone of `main` @ `4e2ff82`; submodules needed dl's
  `~/.gitconfig` (`url.https://github.com/.insteadOf git@github.com:`) because
  submodule URLs are SSH and the spark has no GitHub key. `logs/.wandb_key`
  copied from the old `~/Projects/booster_ws`.
- `~/k1_stages/{common,preflight,smoke,timeit,train_full,record_gate}.sh`, run
  one per container via the proven invocation (fresh cache dir each time —
  the container runs as `--user 0` so caches are root-owned and a stale
  `ov/_cache.lock` wedges the next boot):
  ```
  docker run --rm --gpus all --user 0 --entrypoint bash \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1 -e TERM=xterm \
    -e NVIDIA_DRIVER_CAPABILITIES=all -e K1_PHYSICS=physx \
    -v ~/k1_stages/common.sh:/stage_common.sh:ro -v ~/k1_stages/<stage>.sh:/stage.sh:ro \
    -v ~/k1_cache_<stamp>:/root/.cache -v ~/k1_run_ws:/workspace/booster_ws \
    nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 /stage.sh <args>
  ```
- **The old campaign scripts omit `PYTHONPATH`** (our packages are not
  pip-installed) — `common.sh` sets it explicitly; without it the run cannot
  import `k1_velocity`.
- Throughput on the idle GPU: **0.79 s/iter @ 16 envs, 1.19 s @ 512 envs**
  (vs 1.14 s @ 4096 on the exclusive Blackwell).

### Mid-run check: the shuffle is gone (model_1100, iteration 1100/3000)

Recorded with `~/k1_stages/record_gate.sh <run>/model_1100.pt p2_gaitfix_it800`
(fresh container; artifacts pulled to
`isaac_tasks/k1_velocity/videos/p2_gaitfix_it800.{mp4,trace.npz}`, frame-checked —
robot upright and stepping in all four panels).

`gait_gate.py`, 8 envs, 750 steps, cmd 0.5 m/s — **before → after**:

| metric | pre-repair (`p2_gaitfix_teacher`) | now (model_1100) | target |
|---|---|---|---|
| cadence | 6.68 steps/s | **1.02** | 1.2–4.0 |
| jerk | 0.099 | **0.0478** | < 0.06 ✅ |
| stride | ~0.07 m | 0.045–0.50 (env0 0.197) | ≥ 0.15 |
| speed | 0.23–0.53 | 0.09–0.31 | cmd 0.5 |
| verdict | 8/8 fail | 7/8 fail (env0 `walk-like`) | — |

So the failure **inverted**: the 4–6 Hz chatter is gone (cadence 6.68 → 1.02,
jerk now inside spec) and the policy steps, but at this checkpoint it
**under-steps** (cadence just below the 1.2 floor) and **under-tracks speed**
(0.09–0.31 m/s against 0.5 commanded). Training was still improving when
recorded — at iteration 1597 mean reward −2.21 (from −28.5), `track_lin_vel`
term 0.153 → **0.769**, `gait_cadence` −0.061 → −0.288 (cadence error ~0.85 steps/s
off the 2.0 target). Gate the **final** `model_2999.pt` before judging.

### Google Slides deck — decoded, and where it is blocked

Deck: **"K1 RL Policy Play Videos — booster_ws (2026-09-24)"**, id
`1KamnVS6DEQMrtXbk9z8Mp5qdG9_XTeqRJZTkRXMexyI`, 33 slides, 9144000×5143500 EMU
(10×5.625 in), revision `E2JzzunzZdGe-w`, all slides on layout `p12`. Slides 4–15
hold videos; each is **Drive-hosted**, e.g. slide 5:

```json
"video": { "id": "1dz4TZMtXCOfk4kQ6a6U14e4IJkj40UzW", "source": "DRIVE",
           "url": "https://docs.google.com/file/d/1dz4TZ…/preview" }
```

Reachable through Composio (toolkits `googledrive` + `googleslides` are connected
for thakk100@umn.edu): `GOOGLESLIDES_PRESENTATIONS_GET` (accepts a `fields`
mask), `GOOGLESLIDES_PRESENTATIONS_BATCH_UPDATE` for the insert. Note
`COMPOSIO_MULTI_EXECUTE_TOOL` needs `sync_response_to_workbench` at the **top
level**, not per item.

**Blocked on the upload, not the insert:** every upload tool
(`GOOGLEDRIVE_UPLOAD_FILE` 5 MB, `GOOGLEDRIVE_RESUMABLE_UPLOAD`,
`YOUTUBE_UPLOAD_VIDEO`) takes `file_to_upload.s3key` — a staged reference
returned by some *other* app's download action — not local bytes; the only
content-taking tool is `GOOGLEDRIVE_CREATE_FILE_FROM_TEXT`. `createImage` also
needs a publicly fetchable URL. So the mp4 has to reach Drive by hand (user
drag-drop into folder `1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy`) or via a temporary
public host; **then** the deck insert is one guarded batch
(`createSlide` on `p12` + `createVideo` with the Drive file id, guarded by a
fresh `revisionId`).

### Next steps (in order)

1. **Gate the final checkpoint** on spark02 when the run exits (~1597/3000 at the
   time of writing, ETA ~32 min): `~/k1_stages/record_gate.sh <run>/model_2999.pt
   p2_gaitfix_teacher`, then compare against the table above. If cadence is still
   under 1.2, the next lever is the speed/stride side (the gait terms are now
   doing their job — the policy trades steps for staying planted).
2. `~/k1_stages/record_gate.sh <run>/model_2999.pt p2_gaitfix_teacher` →
   panel video + trace + `gait_gate.py` inside the container; scp both to the
   laptop, extract a frame and **look at it**, then read MOVEMENT_GATE /
   GAIT_GATE. The pre-repair baseline fails 8/8, so this is the comparison.
3. **Distill** `Isaac-Velocity-Distill-K1-v0` with the fresh teacher
   (student input **500**, teacher 237) → smoke gate, then full → export.
4. Export `models/*.pt` (still legacy 48/480 on disk) so
   `test_export_matches_obs_dim` flips from skip to pass.
5. Do **not** re-run on zz-bw while `abrar008`'s jobs hold both GPUs; check
   `nvidia-smi --query-compute-apps=pid --format=csv` + owners first.

---

## 📌 SESSION HANDOFF — 2026-10-01 (gait fix: three levers landed + smoke PASSED)

**Goal:** kill the P2 foot-shuffle (`p2_gaitshuffle_2999.npz` fails 8/8 GAIT_GATE
metrics: cadence 8.25, stride 0.03 m, jerk 0.10) with three cheap levers, then
retrain → distill → export → `isaac_tasks/k1_velocity/scripts/gait_gate.py`.

### Landed on `main` (worktree merges done earlier this session)

- All 5 worktrees merged into `main`, pushed; 3 unique commits + 15 superseded
  duplicates dropped (backup `/tmp/opencode/dirty_backup/` — **wiped by the
  2026-10-01 reboot**, `/tmp` is gone).
- **`7ce0049` feat(p2): ankle-gain override, friction DR, phase clock** — the
  three levers, exactly 14 files.
- **`98f3be3` fix(p2): last obs-layout consumers move to 50-dim + phase clock** —
  `isaac_fleet_vis.py`, `mujoco_fleet_policy.py`, `k1_soccer_compose.py`,
  `locomotion.launch.py`, plus doc notes (`guide_real.md`, `train_student.py`,
  `push_mdp.py`, `push_export_parity.py`).
- `main` is **ahead of origin by 2** (`7ce0049`, `98f3be3`) — push when ready.

### The three levers (all in `isaac_tasks/k1_velocity/.../velocity/`)

1. **Ankle pitch gain override** (`velocity_env_cfg.py`): `.*_Ankle_Pitch`
   100.0 / 5.0 applied to `K1_ARTICULATION_CFG.actuators["feet"]` (a deepcopy of
   `BOOSTER_K1_CFG` — **stock verified untouched**: 35.69/4.26 both ankles).
   `.*_Ankle_Roll` explicitly re-listed at wrapper values 35.69/4.26 (an
   actuator pattern matching nothing errors; a joint matched by nothing gets
   0.0). Verified live: velocity `feet = {Ankle_Pitch: 100.0, Ankle_Roll: 35.69}`.
2. **Friction DR** (`EventCfg.randomize_friction`): `mdp.randomize_rigid_body_material`,
   `mode="startup"`, `body_names=".*"`, static/dynamic (0.4, 1.2),
   restitution (0.0, 0.05), `num_buckets=64`, `make_consistent=True`.
   Confirmed in the smoke run's Event Manager table.
3. **Phase clock** (`gait_clock.py`, new): `phase_clock(env, frequency_hz=1.0)` →
   `[sin, cos]`, state keyed `id(env)`, advanced on `episode_length_buf` change
   (two group reads in one step count once), `noise=None`, **last** ObsTerm in
   both `PolicyCfg` and `TeacherCfg`.

### Obs layout (post-change, everywhere)

| Group / file | Dim | Clock slot |
|---|---|---|
| `policy` | **50** (48 proprio + 2) | `obs[48:50]` |
| `teacher` | **237** (48 + 187 height scan + 2) | `obs[235:237]` |
| distill stack | **500** (50 × 10) | per-step last 2 |
| squat teacher (inherits) | **238** (237 + H*) | last |
| push / kick / P1 / partial-ctrl | unchanged (own `ObservationsCfg`) | — |

Consumers updated: `locomotion_node.py` (`OBS_DIM=50`, phase accumulator in
`_build_obs`, fail-fast on layout mismatch), `test_locomotion_node.py`,
`isaac_fleet_vis.py`, `mujoco_fleet_policy.py`, `k1_soccer_compose.py`
(`VEL_OBS_DIM=50`; `KICK_OBS_DIM` still 48 — different obs group),
`locomotion.launch.py`. **`colcon build --packages-select k1_locomotion` after
any node/launch edit** — `install/` holds a copy.

### Gates run (all green, after the reboot)

- `PYTHONPATH=/tmp/opencode/pytest_pkgs:$PYTHONPATH python -m pytest tests/ -q`
  → **103 passed**.
- `src/k1_locomotion/test/test_locomotion_node.py` → **4 passed, 1 skipped**
  (the skip is `test_export_matches_obs_dim`: on-disk export is still 48-dim).

  > **The 2026-10-01 reboot wiped `/tmp/opencode`** → `pytest_pkgs` had to be
  > reinstalled: `pip install --target /tmp/opencode/pytest_pkgs "pytest==8.3.5" lark`
  > (**must be pytest < 9** — pytest 9 breaks the ROS `launch_testing` plugin
  > hook) and `lark` for the `launch` import. The locomotion test also needs
  > `source /opt/ros/jazzy/setup.bash && source install/setup.bash`.

### 16-env smoke gate — PASSED (this is the gate that was blocked on the driver)

Driver mismatch (`580.173.02` module vs `580.178` NVML) blocked it yesterday;
**user rebooted, `nvidia-smi` 580.178.04 + torch CUDA now healthy.**

```
source scripts/phase6_env.sh
scripts/train_guard.sh --name gaitfix_smoke -- isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Rough-K1-Teacher-v0 \
  --external_callback k1_velocity.register_tasks.register_tasks \
  --num_envs 16 --max_iterations 2 --seed 42 --viz none
```

→ `rc=0`, log `logs/guard_gaitfix_smoke.log`, run
`logs/rsl_rl/p2_move_teacher/2026-10-01_00-33-26/`:

- Observation tables: `policy (50,)` terms 0–7 with `phase_clock` **last**;
  `teacher (237,)` terms 0–8 with `height_scan` then `phase_clock` last.
- Event Manager `startup`: `add_base_mass`, `randomize_friction` ✓.
- `model_0.pt` / `model_1.pt` saved; actor **and** critic first layer
  `(512, 237)` → **teacher input 237 verified** ✓ (wandb `thakk100-dhyan-home` /
  `booster_k1_soccer_hrl`).

### zz-bw (cs-zhang-net-01) — chosen target; teacher retrain RUNNING 2026-10-01

User picked zz-bw over local/Sparks. Box: 2× RTX PRO 6000 Blackwell (98 GB
each), `k1-train.sif` present, `~/run_k1_train.sh` runner, home 3 T free,
`/export/scratch` 198 G free.

- **Clone switched to `main`** (`/export/scratch/thakk100/k1/tmp/booster_ws`).
  It was on `feat/velocity-squat` with a fetch refspec pinned to `dev/phase-0`
  only (that is why `origin/main` was invisible) → refspec fixed to
  `+refs/heads/*:refs/remotes/origin/*`, fetched, 7 dirty rsync'd files
  **backed up to `/export/scratch/thakk100/k1/dirty_backup_20261001/`** then
  hard-reset — all 7 were verified *older* than `origin/main` (4 identical, 3
  only missing today's phase-clock notes), so nothing unique was lost.
  Submodule `booster_train_ref` clean at the recorded pointer (the historical
  reverted-files root cause), untracked `run_unbuffered.py` kept.
- **zz-bw 16-env smoke PASSED** (`k1_gaitfix_smoke`, GPU1, 16×2, seed 42):
  0 Tracebacks, `policy (50,)` / `teacher (237,)` with `phase_clock` last,
  `randomize_friction` in the startup Event table, `model_0/1.pt` saved to
  `/export/scratch/thakk100/k1/logs/rsl_rl/p2_move_teacher/2026-10-01_01-09-27/`.
- **Debug video recorded + frame checked**: `play_record.py --panel_video`,
  session `k1_gaitfix_smokevid` → `isaac_tasks/k1_velocity/videos/gaitfix_smoke.mp4`
  (200 frames × 4 envs) + `gaitfix_smoke_trace.npz`; frame pulled to
  `/tmp/opencode/gait_smoke/frame_02.png` — 4 labelled panels + HUD render fine
  on the new obs (robot collapses: it is a 2-iteration checkpoint).
- **FULL RETRAIN RUNNING**: tmux `k1_gaitfix_teacher`, GPU1,
  `Isaac-Velocity-Rough-K1-Teacher-v0` **4096 envs × 3000 iters, seed 42,
  `--headless`**, log `/export/scratch/thakk100/k1/gaitfix_teacher.log`,
  checkpoints → `/export/scratch/thakk100/k1/logs/rsl_rl/p2_move_teacher/<run>/`.
  GPU0 is another user's job (15 GB resident) — **never co-locate on GPU0**.

### Next steps (in order)

1. ✅ **Push `main`** — done (`7ce0049`, `98f3be3`, `91859dd`; origin in sync).
2. ⏳ **Teacher retrain on zz-bw** — running (`k1_gaitfix_teacher`). When it
   ends: gate the log for `Learning iteration 2999/3000`, **0 Tracebacks**,
   and a fresh `model_2999.pt`.
3. **Record + gate the teacher** (GPU1, never GPU0 — debug video is compulsory
   for play/eval runs, and `gait_gate.py` consumes the trace):
   ```
   tmux new -d -s k1_gaitfix_record \
    "K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/play_record.py K1_GPU=1 \
     $HOME/run_k1_train.sh --task Isaac-Velocity-Rough-K1-Teacher-v0 \
     --checkpoint /workspace/mounts/logs/rsl_rl/p2_move_teacher/<RUN>/model_2999.pt \
     --num_envs 8 --steps 750 --headless --panel_video --cmd 0.5 0 0 \
     --video_out isaac_tasks/k1_velocity/videos/p2_gaitfix_teacher.mp4 \
     --trace_out isaac_tasks/k1_velocity/videos/p2_gaitfix_teacher_trace.npz \
     >> /export/scratch/thakk100/k1/gaitfix_record.log 2>&1"
   scp zz-bw:.../videos/p2_gaitfix_teacher_trace.npz /tmp/opencode/
   python isaac_tasks/k1_velocity/scripts/gait_gate.py /tmp/opencode/p2_gaitfix_teacher_trace.npz
   ```
   Pull a frame of the mp4 and **look at it** before calling anything done.
4. **Distill smoke** with the fresh teacher (student input **500**, teacher
   237): `K1_TRAIN_SCRIPT=isaac_tasks/k1_velocity/scripts/train_student.py
   ~/run_k1_train.sh --task Isaac-Velocity-Distill-K1-v0 --teacher_checkpoint
   /workspace/mounts/logs/rsl_rl/p2_move_teacher/<RUN>/model_2999.pt
   --num_envs 16 --max_iterations 2 --headless` → gate `500`/`237` tables +
   0 Tracebacks, then the full distill run.
5. **Export** `models/*.pt` (`play_student.py` / `play_record.py --export`) —
   the shipped `k1_velocity_policy.pt` etc. are still the legacy **48/480**
   exports and `locomotion_node` refuses them until this — then re-run
   `src/k1_locomotion/test/test_locomotion_node.py` (the
   `test_export_matches_obs_dim` skip must turn into a pass).
6. **`isaac_tasks/k1_velocity/scripts/gait_gate.py`** on the exported student —
   MOVEMENT_GATE + GAIT_GATE must pass (baseline shuffle fails 8/8).
7. `scripts/spark_p2_gaitfix_host.sh` / `spark_p2_gaitfix_train_container.sh`
   **do not exist** (planned names only) — smoke/retrain go through
   `scripts/train_guard.sh` (local) or `~/run_k1_train.sh` (zz-bw).

### Still open / gotchas

- **G4 (action_scale)** — node default `0.25` vs training `scale=1.0`
  (landed `5d5af78`, after the deployed models): safety-relevant, out of
  three-lever scope, tracked in `docs/k1_ros_deployment_plan.md:200`.
- **Push frozen base is PRE-phase-clock** — `FrozenBaseVelocityAction` assembles
  the 236-dim legacy squat layout to match the shipped export; a retrained squat
  teacher is **238** (needs code change if re-exported). Documented in
  `push_mdp.py` + `scripts/push_export_parity.py`.
- `logs/rsl_rl/p2_move_teacher/2026-10-01_00-33-26/` is a **2-iteration smoke**
  run — never use it as a checkpoint.
- **zz-bw clone gotcha:** its `remote.origin.fetch` was pinned to a single
  branch, so `origin/main` never appeared (looked like "no main on origin").
  Fixed to `+refs/heads/*:refs/remotes/origin/*`. The zz-bw scratch disk was at
  **95 % / 198 G free** — checkpoints are tiny, but purge old SIFs before any
  model install.
- **`/tmp` does not survive a reboot** — `/tmp/opencode` (pytest_pkgs,
  local dirty backups) was wiped on 2026-10-01; anything durable must live in
  the project dir or on scratch.

---

## 📌 HARDWARE / CAMERA FINDINGS — 2026-09-29 (unit A2, `10.37.11.3`)

Full writeup: **[`docs/k1_hardware_camera_findings.md`](docs/k1_hardware_camera_findings.md)**

**A2 has no camera sensor fitted.** The ROS camera graph is fully intact
(135 topics, 13 nodes, every camera topic `Publisher count: 1`) but there is no
hardware behind it:

- `lsusb` — 9 devices, all accounted for (hubs, GbE, AX210 **Bluetooth**,
  audio). No RealSense, no USB camera.
- `lspci` — no multimedia/camera device at all.
- `/dev/video*` — none. **`/sys/class/video4linux/` exists but is EMPTY**, which
  proves V4L2 is compiled in but no device ever registered.
- No `/dev/nvargus*`, no sensor node in device-tree, no sensor lines in dmesg.

⚠️ **The pair did publish frames once, then died.** Immediately after a reboot
(robot up ~2 min) `rclpy` measured **7.8 Hz on both
`/boostercamera/head/rgb` and `/boostercamera/head/right/rgb`** (and depth; raw at
29.6 Hz) with `booster-video-stream` alive. About five minutes later: **0.0 Hz**
on all three, raw down to 0.375 Hz, and `booster-video-stream` had died with
nothing to restart it. No OOM (4.5 GiB free throughout). So: graph → brief live
frames → collapse, sensor never re-enumerating. This is the most diagnostic
finding and rules out "there has never been a camera here".

**QoS is required to see anything.** The publishers offer a non-standard
durability that silently rejects default subscribers, so a healthy-looking
`Publisher count: 1` can sit next to a subscriber receiving nothing. Match
`RELIABLE` + `VOLATILE` + `KEEP_LAST(10)`. Also note `booster-video-stream` is a
**RealSense** subscriber (logs `Publishing: '0'` forever), not the producer of
`/boostercamera/head/*`.

**Installed on A2:** cuVSLAM v17.0.0 native Orin build at `~/cuvslam`
(aarch64, 0 unresolved libs) — working. `boosteros` 1.2.0 at `/tmp/venv_bos`
(**will not survive reboot**). Watchdog service `k1-stereo-watchdog.service`
enabled; it restarts what it can and reports `UNRECOVERABLE` rather than
flapping when no sensor is present.

**Next:** physical check that a head camera is fitted / reseat the MIPI ribbon —
the watchdog will pick it up automatically. Then check A1/A3/B1–B3, which were
off-network. **A2 also rebooted for low battery and its perception stack is
currently degraded** — check charge and `booster-daemon-perception` before the walk.

---

## 📌 SESSION HANDOFF — 2026-09-24 (read this first; older history below)

> **Branches:** `dev/soccer-p3p4` (active — holds BOTH tracks; Track A gates landed) ·
> `dev/phase-6-soccer-hrl` @ `f3000d6` (pushed — policy recordings, docs, videos, exports).
> **GitHub:** https://github.com/thdhyan/booster_ws · **Plan:** `PLAN_PHASE6_SOCCER_HRL.md` · **Run log:** `TRAINING.md`
> **Drive:** [policy videos folder](https://drive.google.com/drive/folders/1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy) ·
> [Slide deck (4 videos embedded)](https://docs.google.com/presentation/d/1KamnVS6DEQMrtXbk9z8Mp5qdG9_XTeqRJZTkRXMexyI/edit)

---

# 🟠 TRACK A — SOCCER (P3 head-track, P4 kick teacher) — *agent A owns this*

**Goal:** make the K1 see and kick a ball without ground truth in its inputs
(P3 keeps the ball in frame from camera detections; P4 teacher knows the ball;
the student will consume P3 + the vision estimator).

**Code map (all in `isaac_tasks/k1_velocity/source/k1_velocity/`):**
- `tasks/head/` — P3 `Isaac-HeadTrack-K1-v0` (head-only; obs 12 = YOLO
  detection(3)+head(4)+ang-vel(3)+action(2); centredness reward; ball-speed
  curriculum 0→0.8 m/s; CCW search helper), `tasks/head/agents/rsl_rl_ppo_cfg.py`
  (wandb `p3_head_track`).
- `tasks/kick/` — P4. Teacher id now drives head+legs (14-dim) with
  `ball_in_frame` + `head_ball_aim` (`K1KickTeacherEnvCfg`). Student id unchanged.
- Launchers: `scripts/spark_soccer_{host,container}.sh p3|p4t` (smoke-gated,
  tmux `k1_spark_p3`/`k1_spark_p4t`, logs `scripts/p3.soccer.log`/`p4t.soccer.log`).
- Smokes/verifiers: `scripts/smoke_p3_head.sh`, `scripts/smoke_p4_teacher.sh`,
  `scripts/zero_step.sh Isaac-HeadTrack-K1-v0 cameras` (zero-agent, ~5 min).

**Status (2026-09-24 17:20 UTC):**
- Deployability gate is green for **16 tasks**. P3 zero-step and the real
  camera+YOLO 16×3 smoke both pass; the smoke loaded `yolov8n`, exercised
  periodic resets, produced `model_2.pt` and three non-black clips.
- Fixed the P3 runtime blocker: `write_root_velocity_to_sim_index` requires
  linear+angular velocity `(N,6)` in this build. Also gave P3/P4 separate Isaac
  caches (the shared cache could hang OmniHub), switched to `--viz none`, and
  added log-marker gates because `python.sh` can return 0 after a crash.
- **P3 full is TRAINING on spark02**: tmux `k1_spark_p3`, 512 envs × 2000,
  log `scripts/p3.full.train.log`, run
  `logs/rsl_rl/p3_head_track/2026-09-24_14-58-47_p3_head_track/`, W&B
  [`oavkxiwf`](https://wandb.ai/thakk100-dhyan-home/booster_k1_soccer_hrl/runs/oavkxiwf).
  At 17:20 UTC it is at iteration 333/2000, mean reward 30.53, with
  `model_300.pt` saved. Current rate is ≈21–22 s/iter; P3 ETA ≈03:30 UTC
  on 25 Sep, and Track A completion including P4 is ≈06:30–08:00 UTC.
  Inference is 10 Hz, full-batch, FP16, 320 px. Videos are every 6400 control
  steps = 200 iterations.
- **P4 teacher 16×3 smoke PASSED** (`P4T_SMOKE_RC=0`,
  `P4T_SMOKE_MARKER=OK`): teacher obs `(55,)`, legs+head action `(14,)`,
  `model_2.pt` + video, W&B smoke
  [`b4f5ag2d`](https://wandb.ai/thakk100-dhyan-home/booster_k1_soccer_hrl/runs/b4f5ag2d).
  Fixed per-environment goal replication/scoring, added teacher goal obs and
  the locked approach/kick/align shaping terms.
- Server-side tmux `k1_soccer_chain` waits for P3's explicit
  `SOC_FULL_MARKER=OK`, then runs `spark_soccer_host.sh p4t` (fresh smoke-gated
  512×3000 full). It aborts P4 if P3 lacks the final marker.
- Interim recordings are complete: P3 `model_200` (15 s, real YOLO) and P4
  teacher smoke `model_2` (15 s) are in the shared
  [Track A Interim Drive folder](https://drive.google.com/drive/folders/1BWLirPTpn_lPEaMHwCUkAEiCAD9uA6ge).
  The deck has clean poster-preview slides plus the folder link; the Slides
  connector rejected direct Drive `createVideo` embedding, so the MP4s remain
  the playable source in Drive.
- **P1 camera correction is shipped:** the missing-robot clips were a recorder
  camera-origin bug, not a policy failure. `play_record.py` now aims the RGB
  recorder at env 0's robot root on every frame; corrected 15 s P1 teacher and
  student MP4s are in the same Drive folder, and the P1 poster slides were
  replaced with verified previews.
- **P2 gait-v2 is smoke-gated and queued:** teacher and student 16×3 smokes
  passed on spark04. The full teacher→student campaign is waiting in
  `k1_spark_p2_gait` on spark02 until both P3 and P4 emit
  `SOC_FULL_MARKER=OK`; final P2 videos must use the new `model_2999.pt` files.
- Run-11 remains independently training on spark02 in tmux `k1_spark_hp`.

**Agent A — next steps (in order):**
1. Monitor without restarting healthy runs:
   ```bash
   ssh aim_spark02 'cd ~/Projects/booster_ws && grep -E "Learning iteration|Mean reward|SOC_FULL_MARKER|Traceback|Error" scripts/{p3,p4t}.full.train.log | tail -20'
   ```
2. Require final markers, not just process exit: P3 `SOC_FULL_MARKER=OK` plus
   `model_1999.pt`; P4T marker plus `model_2999.pt`. If either run dies, relaunch
   with the newest checkpoint via `scripts/train.py --checkpoint ...`.
3. Pull representative P3/P4 clips, play-record final checkpoints using the
   Track B recipe, mirror them to Drive, and add slides to the existing deck.
4. Replace these running/queued statuses with final W&B/reward/video evidence in
   `TRAINING.md` and this handoff.

**Watch-outs:** `WANDB_API_KEY` comes from `logs/.wandb_key`; ultralytics gets
a **list of HWC uint8 frames**; rigid-body root velocity is `(N,6)`; reward
terms return 1-D `(N,)`; P3 smoke/full share no Isaac cache with P4 or Run-11.

---

# 🟢 TRACK B — BOX PUSHING (P6: hierarchical velocity + wrist-IK controller) — *agent B owns this*

**Goal:** a policy that walks the K1 up to a box, places both wrist stubs on it
and pushes its corners to a moving goal set. Hierarchy: **velocity command
override → frozen Run-11 locomotion policy → legs**; **wrist-stub targets →
DifferentialIK → arms**.

**Code map:** `tasks/push/` — `push_env_cfg.py` (`K1PushEnvCfg`,
`K1PushReachEnvCfg`), `push_mdp.py` (frozen-base action term, wrist command,
corner/goal math, USD DR, green alpha), `agents/rsl_rl_ppo_cfg.py`
(wandb `p6_push_reach` → `p6_push`). Ids: `Isaac-Push-Reach-K1-v0`,
`Isaac-Push-K1-v0`. Frozen base: `models/k1_partialctrl_base.pt` (exported from
Run-10 `model_2999`, JIT parity 2.87e-03; present on local + spark02 + spark04;
re-export with `scripts/export_base_policy.sh` if Run-11 supersedes it).

**Design (locked):** action 9 = `[vel override 3 | left wrist EE delta 3 | right 3]`;
frozen base assembled with the exact 68-dim partial obs; IK `body_name=*_hand_link`,
`scale=0.05`, dls, position/relative; box = 1 m prototype with USD-time per-env DR
(edge 0.7–1.5 m, mass 3–25 kg / 12–25 for Reach, friction 0.3–1.2,
`replicate_physics=False`), green alpha 0.35; goals = box pose + cumulative
offset integrator (curriculum 0.3→1.5 m), rewards on 8 corners + corner
centroid + progress; teacher-group obs (fully observable by design).

**Status (2026-09-28 16:55 CDT) — ✅ PHASE 1 TRAINING LIVE ON zz-bw:**
- `feat/velocity-squat` = **`7e61244`** pushed (branched off
  `origin/feat/video-to-motion` tip ⇒ sibling's 8 unpushed commits NOT
  republished; mis-based `77494fe` reverted on `dev/soccer-p3p4` as `f37dfd5`).
- zz-bw smoke **PASSED** (`import OK`, `[k1_contact] 23 rigid bodies`,
  0 Traceback, iters 0/2–1/2, ckpts saved); debug video recorded + frame
  checked vs the accepted teacher baseline (zz-bw render path now proven).
- **Contact-sensor root cause:** zz-bw's `booster_train_ref` working tree had
  15 files reverted to pre-`bf3342e` (incl. `booster.py` without
  `_spawn_k1_urdf`) → stock spawn, root-only contact, `('Trunk',)`. Fixed with
  `git -C isaac_tasks/booster_train_ref checkout -- .`. Parent status hides
  submodule content dirt — check inside the submodule.
- **Full run:** tmux `k1_squat_full` (GPU1) 4096 envs × 5000 iters, log
  `/export/scratch/thakk100/k1/squat_full.log`, ETA ~18h45. Gates before
  Phase 1b: velocity parity ±10%, height MAE <2 cm, done_rate 0.0000, then
  Play-task squat→rise panel video + frame check.

**Status (2026-09-27 21:30 CDT) — ⚠️ spark04 DEAD, BASE RECOVERED + RESUME RUNNING ON dl:**
- **spark04 offline since 2026-09-24 ~00:32 UTC** (`No route to host`; dl's
  sync loop logged 2352 consecutive FAILs since then). Needs the usual
  power/network check by the user. The gait-v2 base retrain was at ~iter 840
  when the box died — it did NOT finish. Auto-chain never fired.
- **Rescue:** dl's `k1_spark_sync` had synced the run to
  `dl:Projects/booster_ws/logs/spark04/rsl_rl/k1_partialctrl_base/
  2026-09-24_18-44-39_k1_partialctrl_base/` through `model_800.pt`. Checkpoints
  `model_0..800` are now materialized under the standard dl root
  `logs/rsl_rl/k1_partialctrl_base/2026-09-24_18-44-39_k1_partialctrl_base/`.
- **dl recovery stack (committed `5875c50` + follow-ups):**
  `scripts/dl_push_base_resume_{host,container}.sh` (smoke marker-gated → FULL
  512-env resume with `--checkpoint model_800`), `scripts/dl_push_chain2_host.sh`
  (auto-chain, see gates below), `scripts/dl_partial800{,_walk}_record_host.sh`.
  Sessions: `k1_dl_push_base_resume` (GPU 1), `k1_dl_push_chain2`,
  `k1_dl_partial800_rec`/`k1_dl_partial800_walk` (GPU 2, done).
- **Two dl-specific startup fixes (both marker-gated catches):**
  1. `isaac_tasks/booster_train_ref` is a **submodule with a stale/uncommitted
     pointer** — a git materialization (dl) gets the OLD fork whose bulk import
     lists `resolve_joint_parameter` → `ImportError` on the image (rsync'd boxes
     were fine because they carry the fixed working tree with the try/except
     vendored fallback). Fixed by rsync'ing the fork source to dl. **Commit the
     submodule pointer bump when convenient.**
  2. `train.py --checkpoint` computes `load_run` as a relpath from
     `logs/rsl_rl/<experiment>` and `get_checkpoint_path` **regex-matches run
     dir NAMES**, so a `logs/spark04/...` path never matches → materialize the
     ckpt under the standard root instead (done).
- **Resume run:** smoke 16×3 `Learning iteration 2/3` OK → FULL launched
  21:14 CDT, display total **3800** (rsl_rl adds `--max_iterations` to the ckpt
  iter: 800+3000). chain2 gates: reaches `/3800` (iter ≥3700 line) AND newest
  `model_*.pt` ≥ 3700 → export (`BASE_EXPORT_RC=0` + fresh mtime) →
  `diag_frozen.sh` → pass-B `done_rate < 0.02` → `dl_push_host.sh reach`.
  v1 chain's `2999/3000` markers were WRONG for resume (never print) — do not
  resurrect them.
- **Iter-800 progress videos recorded + frame-checked** (GPU 2, no training
  concurrency): `videos/partial_gaitv2_iter800.mp4` (stand; 4/4 robots upright
  through t=5.7 s, episode reward +4.90) and `videos/partial_gaitv2_iter800_walk.mp4`
  (`cmd vx=+0.80`; all 4 upright with active stepping, **no forward translation
  yet** — matches `error_vel_xy ≈ 0.95` in the curves). HUD shows ckpt + task +
  cmd + step/episode reward + action/obs traces.
- **Base-run curves @ iter 800** (synced tfevents): ep_len 10 → peak 190 →
  158; `termination/base_orientation` 0.92 → 0.15 (good);
  `termination/root_height` 0.03 → **0.80** (kneeling falls now dominant —
  the remaining 2900 iters must fix this); `feet_air_time` ≈ 0 (gait not
  emerged yet); timeout term 0 → 0.12.

**Status (2026-09-24 19:05 UTC — historical): TRACK B: TRAINING BLOCKED ON FROZEN-BASE BUG; GAIT-V2 BASE RETRAIN RUNNING:**
- **Zero-step OK:** `ZERO_STEP_RESULT=OK` (3 steps, obs `teacher (n,108)`, rewards
  compute, no terminations; box rests at `z = half_extents`, DR prints
  `edge 0.71-1.47 m, mass 3.6-19.2 kg`). Gate log `scripts/pushzero.log`.
- **Smoke OK:** `scripts/smoke_push.sh push` 16×3 reached
  `Learning iteration 2/3`, video written, no Traceback. The last blocker was a
  **partial-reset broadcast bug**: `reset_wrist_targets` builds an
  `env_ids`-subset `(n,2,3)` target but `_to_base` subtracted the **full-batch**
  `root_pos_w` (4 envs never subset → zero-step hid it; 16-env smoke crashed at
  the first 2-env reset: `size of tensor a (2) must match (16)`). Fix:
  `_to_base(env, pts, env_ids=None)` indexes `root_pos_w`/`root_quat_w` by
  `env_ids`. Earlier rounds also fixed: `quat_apply` batching (`_rot_batch`),
  `goal_quat_bf` (`quat_mul(quat_inv(base),box)`), event modes (`prestartup`
  for USD geometry, `startup` for friction — material impls need `root_view`),
  USD `MassAPI` + `Gf.Vec3f` inertia DR, `stage.GetPrimAtPath`,
  `UsdShade.Material.Define` 2-arg, warp `joint_ids` int32,
  `_update_command()` no-args, `reset_box` z=half-extents, box-frame
  `wrist_box_proximity`, `find_joints` lists→tensor.
- **Deliverables shipped** (commit `6f7db13`, pushed `dev/soccer-p3p4`):
  README **P6 section** (obs/action/reward/DR tables) + `videos/push_smoke.mp4`
  + 2 PNG frames; TRAINING.md campaign rows 10–11; launchers
  `scripts/spark_push_{host,container}.sh`.
- **Drive:** `push_smoke.mp4` → folder
  [Booster videos](https://drive.google.com/drive/folders/1TDRzuMYN_mFZVrJqN8DiTtwwRT_D5EQy)
  (anyone/reader, file id `14lbLlTdANq6qNy1vGc5DcGoSJWjY4I-j`); deck slide
  added with `createVideo` source `DRIVE` (slide `slide_fe11a147`, Composio
  session `trip`).
- **Run-10 reach (256×1500) CRASHED/WEDGED on spark04** — CUDA
  `cuda-EvtHandlr` spin at iter ~690, log stalled 11:47 CDT; container killed,
  logs preserved as `scripts/reach.push.crashed.log` +
  `scripts/reach.push.full.crashed.log`. Training itself was NOT learning
  anyway: mean reward flat −3.85 → −3.60, ep_len 16–23 (0.3–0.46 s),
  99.7 % late terminations `root_height`, `wrist_box_proximity` ≈ 0.0001.
- **Root cause (RESOLVED 2026-09-28): joint-wiring mismatch in the frozen
  term.** `FrozenBaseVelocityAction` resolved joint ids with
  `find_joints(K1_*_JOINTS, preserve_order=True)` = left-then-right list
  order, but the partial (home) env the policy trained in resolves BOTH its
  action terms (`JointActionCfg.preserve_order=False`, confirmed in the home
  play log: `Resolved ... JointPositionAction: [Left_Hip_Pitch, Right_Hip_Pitch, ...] [[3,4,8,9,...]]`)
  and its obs terms (`SceneEntityCfg.preserve_order=False`) in
  **articulation order** (interleaved L/R per joint type). Result in P6:
  **11/12 leg dims drove the wrong joints** (hip-roll commands onto knees,
  ankle commands onto hip-yaws) + permuted leg/arm obs blocks → the base
  cartwheeled within ~17 steps. Everything else was exonerated along the
  way: export bit-exact (`push_export_parity.py`), step-0 obs canonical,
  home step-1 actions ≈ frozen `out0` (identical ±4 first output — only
  *where it landed* differed). **Fix:** `push_mdp.py` resolves
  `preserve_order=False` everywhere (arms: ONE find over all 8 — two
  per-side finds concatenated would still be L-block-then-R-block).
  **A/B ladder (dl GPU 2, same export, pass-B done_rate):** baseline
  0.0425/0.0434 → ground-friction parity (1.0/1.0 multiply) 0.0434
  **REFUTED** → stiff-arms (wrist terms dropped; nothing pinned targets →
  arms ran to joint-zero at 6 rad/s) 0.0294 → full home-reset mimic
  (legs ×U(0.5,1.5) + `randomize_arm_pose` with pinned PD targets,
  verifiably applied in the dump) 0.0431 **REFUTED** → **wiring fix:
  0.0000** (8 envs × 400 steps, zero falls, stand_frac 1.0; step-1
  ang_vel absmax 0.981→0.169, leg_vel 3.2→1.3). Logs
  `scripts/dl_ab_push_{frict,stiff,wiring}.log`; pre-fix gate evidence
  preserved in `scripts/dl_push_chain2.v1gate-fail.log`. Gate `< 0.02`
  PASSES → chain2 re-run 2026-09-28 launches reach.
- **ROOT CAUSE #2 — reward sign inversion in P6 (RESOLVED 2026-09-28):** the
  v1 reach+push runs trained "successfully" (reach `PUSH_FULL_MARKER=OK`,
  24.7/ep; push `model_4498.pt`, 52.1/ep; base stood, stand_frac 1.0) but the
  GPU-2 eval exposed **zero task performance**: reach `contact_both=0.0000`,
  `wrist_tgt_err=0.839 m`; push `box_disp=0.0009 m`, `contact_both=0`,
  `goal_err` 0→0.300 (= curriculum `d_max` cap at iter 0, not box motion).
  Per-term `Episode_Reward/*` in the tb events gave the mechanism:
  `wrist_target_tracking` logged **+0.931** (rate, reach) and
  `corner_goal_tracking +1.012` / `centroid_goal_tracking +0.506` (push) —
  i.e. the policy was PAID for keeping wrists/box FAR: those funcs return
  the SIGNED quantity (`−error`, `−|ω|`) and the cfg paired them with
  negative weights → **double-negative = reward for error** (reconciliation:
  per-step = rate×dt: reach 1.266×0.02 ≈ 0.025 ≈ logged mean 24.7/1000 ✓).
  Confirmed policy-indifferent vs controllable split: `corner/centroid/
  box_goal_progress` depend only on `goal_offset` (goal = current box pose
  + offset → d ≡ |offset|, env-driven by `advance_goal`), so only
  `wrist_target_tracking`, `wrist_box_proximity` (exp(−gap/0.08), dead until
  ~8 cm), `box_vel_toward_goal` and `track_cmd` shape behavior — and the
  strongest of those (wrist) was inverted → contact actively avoided.
  **Fix (`push_env_cfg.py`, sign-convention docstring added):** weights made
  positive where the func is signed — corner 1.0, centroid 0.5, spin 0.1,
  wrist push 0.3, wrist reach 1.0; all other penalties verified correct via
  their logged negative contributions (positive func × negative weight).
  Eval untouched (`tgt_err = −func` still right). v1 evidence preserved:
  `scripts/{reach,push}.v1inverted.push.log`,
  `scripts/dl_push_chain2.v2-wiringfix-reach.log`,
  `scripts/dl_push_chain3.v1.log`, v1 ckpt run dirs
  (`p6_push_reach/2026-09-28_06-25-35`, `p6_push/2026-09-28_07-51-35`).
  **Follow-up (design, not a bug):** `goal_err=||goal_offset||` cannot be
  reduced by pushing (goal glued to the box's current pose); the anchored
  reading `goal_err_anchor = spawn+offset−box` is the true box-position
  error and only `box_vel_toward_goal` (+0.5) currently rewards box motion —
  consider anchoring the goal to the box spawn pose if box-position-error
  should enter the reward. v2 retrain chain relaunched and **completed the
  same day** (reach wrist gap halved 0.50→0.25 m, campaign-first 5.7 %
  any-wrist contact, box still static → TRAINING.md § P6 quality report).
- **Velocity gait-v2 review (2026-09-24) — APPROVED + committed:** the
  velocity cfg was reworked (H1/G1-style gait shaping + direct Cartesian
  commands; see README P2 tables). Verified on the training image via the new
  `scripts/reward_probe.sh` (16 envs × 120 random steps, per-term raw reward
  means + foot-contact diagnostics): velocity rough + partialctrl both
  `REWARD_PROBE_RESULT=OK`, 15/15 terms resolve, `feet_air_time` > 0,
  `feet_slide` active, `stand_still`/`undesired_contacts` fire, foot contact
  sensor peaks 1222 N. Logs `scripts/reward_probe{,_partial}.log` (spark04).
  Runtime note: on the image the mdp import falls back to
  `manager_based.locomotion.velocity.mdp` (no `isaaclab_tasks.core` there) —
  same function set, signatures inspected.
- **Base retrain (GAIT-V2) RUNNING on spark04** — tmux `k1_spark_push_base`,
  `scripts/spark_push_base_{host,container}.sh`: smoke 16×3 (marker-gated) →
  FULL 512×3000 `Isaac-Velocity-PartialCtrl-K1-v0`, log
  `scripts/push_base.full.log`, wandb `k1_partialctrl_base`. Rationale: the
  shipped base predates gait-v2 AND its export is the open blocker; a fresh
  base + fresh validated export is the clean unblock path. (Parallel, other
  session: P2 gait campaign teacher→student on spark02, tmux
  `k1_spark_p2_gait`.)

**Remaining (updated 2026-09-28):**
0. **NEXT PIPELINE (user 2026-09-28): squat-base → reach → push-v3 — Phase 0
   audit DONE** (TRAINING.md § P6 Phase-0 degenerate audit: 15 findings —
   goal anchoring makes corner/centroid/progress pure-env (A1–A3),
   `track_cmd` stand-at-zero pull (A8), no success/OOB/tip termination
   (A10/A11/A13), squat vs fixed `root_height`/`base_height` (A12)). User
   decisions: **train on zz-bw** (SLURM, faster), **randomize box spawn AND
   goal position**, **corner-point tracking** for the box (reward/eval/success
   all via the 8 corners). Phase 1 next: `UniformHeightCommandCfg` squat
   teacher in `velocity/` (cmd = vx, vy, wz, H* with ranges matching VR
   `BASE_LIMITS`), depth curriculum gated on stand-back-up, then frozen-base
   reach (10-dim action incl. H*) → push v3 (fixed randomized goal pose +
   success/OOB/tip terms).
1. ~~v2 RETRAIN + eval + record + report~~ **DONE 2026-09-28 (dl)** —
   chain2 → reach 1500/1500 (`PUSH_FULL_MARKER=OK`; tb sanity passed:
   `wrist_target_tracking` **−0.681**, negative as required) → re-armed
   chain3 → push **3000/3000** (display 4498/4499, `PUSH_FULL_MARKER=OK`,
   `PUSH_FULL_RC=0`, 0 tracebacks, ckpt
   `p6_push/2026-09-28_11-34-52_p6_push/model_4498.pt`; goal curriculum
   0.3→1.5 m complete, `box_vel_toward_goal` **+0.0025 positive**, falls
   0.13 %). Evaluated on **GPU 2** (8×400 + 8×1000), both 4-panel deliverable
   videos recorded + frame-checked (`videos/push_{reach,stage2}_policy.mp4`),
   and the standing **mean-box-position-error / success-rate / reach-quality
   report** is in TRAINING.md § “TRACK B — P6 quality report (v2,
   2026-09-28)”: reach improved materially (wrist gap 0.50→0.25 m, tgt
   0.84→0.63 m, campaign-first contacts 5.7 % any-wrist) but **success 0 %**
   — dual contact 0, box displacement ≤1.5 mm, `goal_err_anchor` stuck at the
   initial ≈0.198 m (box never moves). v1 protocol artifacts kept for the
   record: warm-start fix in `scripts/spark_push_container.sh` (symlink child
   `p6_push/warm_from_reach`, smoke without `--checkpoint`, gate
   `END=ITER0+ITERS`), v1 logs `*.v1inverted.push.log`, v1 run dirs + wandb.
   **Open (user decision):** iterate training (longer reach / proximity
   dead-zone tune / anchor goal to box spawn — design note above) vs ship v2
   as-is. **Delivery chain DONE 2026-09-28:** batch committed via detached
   worktree `~/Projects/wt-trackb` on `dev/soccer-p3p4` (`fb7115b` fix(push)
   root causes, `833cf03` docs+videos) → pushed to GitHub; videos uploaded
   byte-exact to Drive folder `1TDRzuMYN_...` (reach `11nDdUDn...`, push
   `1p5yUBOth...`) → embedded as slides `slide_p6v2_reach` /
   `slide_p6v2_push` in deck "K1 RL Policy Play Videos — booster_ws"
   (`1KamnVS6...`). Main tree (`feat/video-to-motion`) keeps these same edits
   uncommitted on purpose — do not re-commit them there.
2. ~~In-env A/B hunt for the P6 fall~~ **DONE 2026-09-28** — root cause was
   the frozen term's joint-wiring (`preserve_order` mismatch vs the training
   env); full A/B ladder + fix in the Root cause bullet above. Gate passed
   (`done_rate=0.0000` < 0.02).
3. GPU discipline per box: dl uses pinned GPUs (training GPU 1, records GPU 2)
   so record/eval and training can coexist across GPUs; never run two heavy
   jobs on the SAME GPU. spark04 (when revived) keeps the old rule: no
   eval/record while training (it wedged Run-10's CUDA context).
4. spark04 revive checklist: power/network, then `tmux ls` (old sessions are
   dead), pull any newer ckpts if the base had progressed past model_800
   before it died (unlikely — dl's sync saw nothing after model_800), and
   re-sync its repo from git before any run (its tree also has the stale
   booster_train_ref submodule).

**Loop recipe (used):** patch → `rsync …/tasks/push/ aim_spark04:…/tasks/push/`
(with `--no-owner --no-group --no-perms --exclude='__pycache__'`) → relaunch
`k1_pushzero` → grep markers (never rc: `PUSH_SMOKE_RC=0` prints even on
Traceback). Full smoke `k1_pushsmoke` only after zero-step OK.

---

**Status (2026-09-28 ~19:00 CDT) — push v3 AUTHORED + PUSHED (`c0fe6fb`),
rsync'd to zz-bw, preflight PASS; Phase-1 squat teacher RUNNING on zz-bw:**
- **v3 = Phase-0 audit A1–A15** (fixed randomized goal + yaw curriculum,
  success/failure terminations with corner-err-0.08-held-1 s success, squat
  frozen base 4-dim cmd / 236-dim obs, height_scanner, dual-mode legacy
  partial kept for A/B). Full change table + validation log: this branch's
  TRAINING.md § "TRACK B — push v3".
- **Blockers:** 2a resolved (mode = basename of `models/k1_push_base.pt`
  symlink; squat export happens chain stage 3), 4 ships as chain stage 4
  (parity `--layout squat`, gate `PARITY_OK`), **5 + 6 resolved**
  (`scripts/push_preflight.sh` PASSED on the clone; `zzbw_push_{smoke,host,
  chain_host}.sh` marker-gated), **7 CLEARED-enough** (user approved `ltx2`
  119 G + old SIFs `cosmos3-gen`/`so101-train` → zz-bw scratch **88 G → 228 G
  free, 94 %**; hf-cache 298 G / sing-cache 49 G retained by choice; kept
  `k1-train.sif` in use, `pe-isaaclab.sif`, `isaaclab23_sim51.sif`).
- **Chain design:** `tmux new -s k1_push_chain 'scripts/zzbw_push_chain_host.sh'`
  — waits for Phase-1 `k1_squat_full`, gates its log, exports the squat base,
  parity, smokes, reach 1500 → push 3000 (warm). **Never switches the clone's
  branch** (stays `feat/velocity-squat`; v3 push files ride in via rsync;
  all gates are content markers — `python.sh` returns 0 on Traceback, so
  gates are grep-only).
- **dl validation smokes DONE, both PASS:** A = `reach` partial
  (`mode=partial cmd=3 out=14`), B = `push` squat-structural w/ fake 236→12
  policy (`mode=squat cmd=4 out=12`, `[frozenobs]` block sanity OK), each
  gated on `Learning iteration 2/3` + `PUSH_SMOKE_DONE` + 0 Tracebacks, PNG
  frames viewed (`v3_smoke_{a,b}_frame.png`).
- **Next:** Phase-1 gate (velocity parity ±10 %, height
  MAE < 2 cm, done_rate 0.0000) → Play-task squat→rise panel video + frame
  check → start the chain → Phase 4 (A15 eval harness, report, videos,
  Drive/Slides).

---

# 🔧 SHARED INFRASTRUCTURE (both tracks)

- **Registration chain (a new family must touch all 4):**
  `tasks/__init__.py`, `source/k1_velocity/register_tasks.py`,
  `scripts/train.py`, `scripts/play_record.py` (+ `tests/test_deployability.py`
  conventions: obs group classes end in `Cfg`; `TeacherCfg` is the privileged
  exemption). Gate: container on spark04/02 —
  `python tests/test_deployability.py` → currently **HOLDS, 14 tasks**.
- **Container:** `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`, in-container
  python `/isaac-sim/python.sh`, repo mounted at `/workspace/booster_ws`, wandb
  project `booster_k1_soccer_hrl` (entity `thakk100-dhyan-home`).
- **Drive video mirror:** rclone linux-arm64 + `gdrive-thakk100` remote, tmux
  `k1_gdrive_sync` on spark02 AND spark04 (copies `logs/**/*.mp4` every 10 min).
  Config lives in `~/.config/rclone/rclone.conf` on both.
- **Server restart lesson (2026-09-24 ~12:55 UTC):** both sparks rebooted and
  killed every container/tmux session. Only `k1_gdrive_sync` survived. Salvage
  came from per-100-iter checkpoints + the new `train.py --checkpoint` resume
  flag (`RESUME_CKPT` env in `spark_force_*`). After any restart: list
  `logs/rsl_rl/*/*/model_*.pt`, then relaunch with `--checkpoint`.
  Run-10 had already finished cleanly (`HP_FULL_RC=0`).
- **Gotchas (full list was in the previous revision — top ones):**
  - `python.sh` returns rc=0 on crashes → gate on log markers/artifacts
    (`ZERO_STEP_RESULT=OK`, `REC_VERIFY_OK_*`), never rc.
  - Reward terms: 1-D `(N,)`. `(N,1)` broadcasts to `(N,N)` and the reward
    manager traceback does NOT name the term — guard or bisect.
  - Event/curriculum term signatures: `(env, env_ids, …)`; all names must be
    passed in cfg params (defaults make the validation fail).
  - `joint_pos[:, ids][:, 0]`; `joint_pos[:, ids, 0]` silently gives `(N, 2)`.
  - `randomize_rigid_body_scale`/mass/material are USD-cooked: `mode="usd"`,
    `replicate_physics=False`, per-env fixed.
  - ultralytics: pass a list of HWC uint8 frames, not a tensor.
  - `DifferentialInverseKinematicsActionCfg(body_name=...)`, not `body`.
  - configclass `class_type` must be set at decoration time.
  - Sparks: container files are root-owned; rsync with
    `--no-owner --no-group --no-perms --exclude='__pycache__'`.
    Occasional `Could not resolve hostname aim_sparkNN` → retry.
  - Never `git add src/k1_description/assets`.
  - tmux launchers that CREATE their own session (`spark_*_host.sh`,
    `record_policies_host.sh`) must be invoked directly, not wrapped in another
    `tmux new-session` (duplicate-session error).

---

> **Older handoff (2026-09-22)** — kept below for history.
> **Date:** 2026-09-22 · **Branch:** `dev/phase-6-soccer-hrl` @ `16a6803` (`main` == `origin/main` @ `88f73fc`, tag `v0.7-phase0-complete` pushed)
> **GitHub:** https://github.com/thdhyan/booster_ws
> **Read first:** `STATE.md` (snapshot) → `PLAN_PHASE6_SOCCER_HRL.md` (the active plan) → `TASKS.md` T6.* (execution list)
> Supersedes the older two-agent handoff (`HANDOFF-AGENTS.md` — contact-sensor + kick-task jobs **landed**: `faefa3e`, `89b13a9`).

---

## Current State

### ✅ T6.2 — IsaacLab 3.0-EA task migration (2026-09-22 — **Gate G1 PASSED**, `16a6803`)
- **Canonical launch (supersedes `start_training.sh` + old `scripts/train.py` wrappers — 3.0 removed backend-local scripts):**
  ```bash
  source scripts/phase6_env.sh
  scripts/train_guard.sh --name <run> -- isaaclab train --rl_library rsl_rl \
      --task Isaac-Velocity-Rough-K1-v0 \
      --external_callback k1_velocity.register_tasks.register_tasks \
      --num_envs 256 --max_iterations 5000 --seed 42 --viz none
  ```
  Third-party tasks register via `--external_callback` (→ `k1_velocity/register_tasks.py`); packages ride `PYTHONPATH`. Distillation runs are native: same command, `--task Isaac-Velocity-Distill-K1-v0 --checkpoint <teacher model.pt>`.
- **G1 evidence:** 16 envs × 2 iters `rc=0` · wandb online `booster_k1_locomotion/x1ptwkae` (that cfg predated the project switch; runs now log to **`booster_k1_soccer_hrl`** per PLAN §3.6) · `logs/rsl_rl/k1_velocity_rough/<ts>/model_0,1.pt` · `feet_air_time`=3e-05 (>0) + `feet_slide`=-2.2e-04 → **contact fires directly from `K1_22dof.urdf` on the 6.1 importer — `flatten_k1_usd.py` is off the training path** (kept for reference).
- **Key 3.0 deltas applied:** `isaaclab_tasks.core.velocity` imports · `sim.use_newton_actuators=False` (custom `BoosterDelayedPDActuator` needs the Isaac Lab execution path) · fork **`booster_train` `2879b1a`** (`_parse_joint_parameter` → `resolve_joint_parameter`) · `SceneEntityCfg.body_names` = bare link names (`left_foot_link`, no `Robot/` prefix; prim_paths keep it) · wandb project `booster_k1_soccer_hrl` · guard fix (invalid `ManagedOOMPreference=nodest` killed every scope on systemd 255) · `phase6_env.sh` puts venv bin on PATH (`isaaclab` CLI).
- **Kick (T6.2.3, ~half done):** cfg imports+instantiates ✅; mdp restored to authentic `89b13a9` **244-line** version (the `88f73fc` LFS restore had silently landed a stale 210-line cached object missing `reset_ball_omnireset` — see STATE.md) + `write_root_pose/velocity_to_sim_index` + `ProxyArray.torch` in `_set_ball_pos`. **Remaining:** obs-fn ProxyArray/quat audit (`ball_pos_in_robot_frame` etc.) + kick smoke.

### ✅ Training environment rebuilt (2026-09-22, T6.1 — Gate G0 PASSED)
- **Venv:** `~/Projects/IsaacLab-ea/.venv` (uv-managed, py3.12) — Isaac Sim **6.1.0** + IsaacLab **v3.0.0-EA** (worktree `~/Projects/IsaacLab-ea`; shared `~/Projects/IsaacLab` untouched on `perf-2026-07-06`), torch 2.11.0+cu128 (CUDA OK), rsl_rl, **ultralytics 8.4.158**, wandb, imageio+ffmpeg.
- **G0**: headless Kit 110.3 launch on the RTX 4060 ✅. 9 K1 tasks register (velocity rough/distill/contact/play + kick teacher/distill).
- **Use:** `source scripts/phase6_env.sh` → `phase6-python <script>` (packages ride PYTHONPATH; not pip-installed: root-owned egg-info removed, and EA dropped the `omni.isaac.lab.tasks` entry-point group so metadata isn't consumed). **Reproduce/repair:** `scripts/install_phase6_env.sh` — any future `uv sync` strips the extras, rerun it after.
- ⚠️ `~/Projects/IsaacLab/isaac6/.venv` belongs to a **parallel G1_sim session** (recreated it twice mid-run) — do not use or touch.
- `zz-bw` HPC still unreachable → **local-only training**; disk now **16 GB free** (guard floor 5 GB).

### ✅ Landed (still valid)
- Velocity task trained (old env): student reward 40.89 (`models/k1_velocity_student.pt`), teacher 25.08 (`velocity_teacher_4999.pt`).
- Kick task `Isaac-Kick-Ball-K1-v0` (OmniReset, goal detection) — `89b13a9`.
- USD flatten + contact rewards re-enabled — `faefa3e` (`scripts/flatten_k1_usd.py`, `K1_flat.usd` 4.2 MB). *Contact end-to-end never verified; IL 3.0 importer may make flatten obsolete — re-check.*
- Fleet sim ×3 backends (Gazebo/MuJoCo/Isaac), stereo ZED 2i rig, RoboCup MuJoCo capture — see `PHASE0-DONE.md`, old details in git history.

### 🟡 Drafts landed in git (T6.0, `6ae1d94`) — still not runnable
`isaac_tasks/k1_head_tracking/` + `isaac_tasks/k1_kicking/` (lone env-cfg files, NOT runnable: no `__init__.py`/agents/scripts, obs wired via velocity-command hacks) · `k1_soccer_compose.py` (3-policy state machine, needs trained head/kick policies) · `PLAN_MULTILAYER.md`.

### ✅ Phase 6 plan + rewards LOCKED
Written this session, approved by user in chat:
- **`PLAN_PHASE6_SOCCER_HRL.md`** — full plan: git closure, env rebuild + IL 3.0-EA migration table, DR spec (incl. **ball position + ball color**), **auto-stability push randomization** (`random_body_push` on waist/pelvis/chest), camera+YOLO, **vision estimator (no-GT deployability)**, video cadence (200 iters → 30 s), wandb, PC guardrails, 9-run training matrix, gates G0–G6.
- **`TASKS.md` § PHASE 6** — T6.0–T6.6 checklist.
- **`STATE.md`** — repo/hardware/env snapshot.

---

## Phase 6 design (one-paragraph version)

Five policies, all MLP (512,256,128): **P1 stand** and **P2 walk** are *teacher–student* (teacher: rough terrain + height scan + foot contacts; student: blind 10-step history) with **random 20–80 N body-part pushes** so the robot self-balances when shoved at the waist. **P3 head-track** (2-dim head, YOLO-only obs) keeps the ball in the depth camera. **P4 chase&kick** is *teacher–student*: teacher sees true ball/goal, **student sees only the vision estimator** (YOLO bbox + depth → ball pos/vel; rectangular-goal tracker/rememberer → goal pos) — head driven by frozen P3 during rollouts. **P5 HRL manager** (5 Hz) picks among frozen P1–P4 + (vx,vy,wz) params; its student is estimator-only (200-dim history MLP). Rewards are user-approved and LOCKED in the plan — GT allowed only in rewards/teacher obs, never in deployable obs (static test T6.3.5 enforces). Trained sequentially, 256–512 envs, headless, wandb project `booster_k1_soccer_hrl` (entity `thakk100-dhyan-home`), video every 200 iterations.

---

## Machine rules (do not violate)

- **RTX 4060 8 GB / 16 GiB RAM / 16 GB disk free (post-install).** One training at a time; ≤512 envs; `train_guard.sh` watchdog + `systemd-run --scope MemoryMax=9G`; GPU >7 GB ⇒ kill. Never 4096 envs locally.
- Headless only (`--viz none` in IL 3.0).
- Videos: every 200 iterations, 30 s → `logs/videos/<run>/` + wandb.
- Never source system ROS2 and Isaac ROS2 together.
- wandb entity is **`thakk100-dhyan-home`** (never `thakk100`/`thdhyan`).

## Gotchas carried forward

- K1 joint A-prefix: `AAHead_yaw`, `ALeft_Shoulder_Pitch`, `ARight_Shoulder_Pitch`.
- URDF-imported USD nests link prims → contact sensors needed flat hierarchy (flatten script). IL 3.0 importer still nests by design; PR #6378 (merged on develop) may fix contact addressing — verify before re-flattening.
- rsl_rl ≥4: `actor=`/`critic=` model cfg, not `policy=`; deterministic distillation needs the `output_std` class-property guard (copy from `scripts/train_student.py`).
- IsaacLab 3.0 breaking changes: **quats XYZW**, `ProxyArray` (`.data.*.torch`), `write_*_to_sim_index/mask`, `isaaclab train` CLI, `VideoRecorderCfg`, `--viz none`, `enable_extension` for direct isaacsim imports — full table in plan §2.
- IsaacLab ships its own agent skills (`isaaclab-migrating-2x-to-3x` etc.) — use them during migration.
- LFS incident (fixed 2026-09-22): `002a647` (Sep 2) added `*.py filter=lfs` to `.gitattributes`; `dd3f6b2` (Sep 4) removed the rule but never de-LFS'd the 62 already-converted `.py` files, so a later checkout overwrote real sources with raw pointer blobs. Restored all 62 from local `.git/lfs/objects` (exact content, zero edit loss) + repaired a truncated `ball_to_goal_progress` RewTerm in `kick_env_cfg.py` (weight +1.0 per PHASE0-DONE.md). **If any file ever reads `version https://git-lfs.github.com/spec/v1`, it is corruption, not content** — recover from git history or `.git/lfs/objects/<oid[0:2]>/<oid[2:4]>/<oid>`, never edit around it.

## Verified commands (post-rebuild — placeholders until T6.1 done)

```bash
# smoke (local, any time):
<venv>/python isaac_tasks/.../train.py --task <ID> --num_envs 16 --max_iterations 2 --viz none

# real run (guarded):
./scripts/train_guard.sh "<venv>/python <train script> --task <ID> --num_envs 512 --max_iterations N"
# → wraps in systemd-run scope + watchdog; logs to logs/train_<id>.log
```

## Key files

| File | Purpose |
|---|---|
| `PLAN_PHASE6_SOCCER_HRL.md` / `TASKS.md` (T6.*) / `STATE.md` | plan / tasks / state — the contract for this phase |
| `isaac_tasks/k1_velocity/.../kick/` | kick task (migrate in T6.2.3) |
| `scripts/flatten_k1_usd.py`, `scripts/probe_k1_bodies.py` | USD flatten / body-name probe |
| `src/k1_sim_isaac/scripts/k1_soccer_compose.py` | runtime policy composition (extend in T6.6.2) |
| `src/k1_sim_isaac/scripts/{soccer_sim,stereo_cam_test}.py` | field/ball constants · stereo rig (depth source) |
| `models/*.pt` | exported policies (LFS) |

## Next steps

1. User **go-ahead on the plan** → T6.0 git closure → T6.1 env rebuild → T6.2 migration → T6.3 infra → T6.4 base policies → T6.5 HRL → T6.6 integration.
2. Every gate (G0–G6) blocks the next stage; user reviews wandb + videos every 200 iters.

---

## 📌 SESSION UPDATE — 2026-10-01, latest: AGILE parity + upright-gated weight ramps + 1M-step run on zz-bw

**Read this before `docs/agile_weight_comparison.md`.**

### zz-bw is NOT unreachable (STATE.md §3.3 is stale)

`ssh zz-bw` works. The `kex_exchange_identification: Connection reset` symptom is a
**stale local port-forward bind**, not a down host — `~/.ssh/config` has
`LocalForward 3001 localhost:3001` for zz-bw, and once 3001 is held, every
subsequent connection dies during kex:

```bash
ssh -o ClearAllForwardings=yes zz-bw     # <- works
```

Do **not** use `ssh -F ~/.ssh/config zz-bw`; the `-F` path fails with
`Can't open user config file` from the remote side. Use the form above. The box
also resets connections under load (load avg ~50, 17 users) — wrap remote calls in
a retry loop, that is normal and not a failure.

### AGILE parity (`bbf01ac`) — 23 terms, matching AGILE's Booster T1

Closed the gap against `nvidia-isaac/WBC-AGILE` (the reference NVIDIA ships for
T1 is the closest analogue to K1). Was 20/23, now 23/23 plus 8 non-AGILE extras.

| added | AGILE T1 weight | why |
|---|---|---|
| `ankle_torques` | -1e-4 | all 4 ankle joints, NOT roll-only — that is `ankle_roll_torques`' job at 20× |
| `dof_vel` | -2e-4 | whole robot; a `joint_names` filter would make it weaker than the reference |
| `jumping` | -0.5 | both feet off ground, clock-independent |
| **weighted trackers** | — | replaces the plain exponential; **this is the speed fix** |

The weighted tracker (`agile_rewards.py`) ramps 1.0×→2.0× on commanded speed.
The plain exponential returns nearly the same credit for creeping as for arriving,
which is why the retrain walked **0.09-0.31 m/s against a 0.5 m/s command** and
was paid for the shortfall. Weight stays 10.0 and std stays 0.15 (audited values,
not AGILE's 5.0/0.2) — reverting either undoes half the same fix.

Two port traps, both documented in the module: AGILE reads
`command_term.cfg.min_vel_norm`, which exists only on AGILE's command class, and
the weight ramp must read `cfg.ranges` **live** because
`VelocityRangeCurriculumTerm` mutates those ranges mid-run (a ramp pinned to the
initial range goes flat the moment the curriculum moves).

### Upright-gated weight ramps (`2ddec51`) — the main change

**Static weights are now light; ramps supply the rest.** Seven terms:

| term | start (static) | terminal (ramped) |
|---|---|---|
| `action_rate_l2` | -0.5 | -2.0 |
| `action_jerk_l2` | -0.05 | -0.5 |
| `gait_cadence` | -0.2 | -1.0 |
| `phase_swing` | -0.4 | -2.0 |
| `feet_clearance` | -1.6 | -8.0 |
| `feet_alternation` | -0.4 | -2.0 |
| `stride_length` | -1.2 | -6.0 |

New `reward_weight_ramp.py`. **The deviation from AGILE, and why it matters:**
AGILE's `update_reward_weight_step` ramps on `common_step_counter` alone. We
credit progress **only while upright** (≥0.80, same signal
`VelocityRangeCurriculumTerm` uses). A step-based schedule would hand a still-
falling policy the terminal smoothness penalty for not being smooth yet. A
missing upright measurement is never read as success.

`max_iterations` 5 000 → **42 000** (= 1.008M control steps vs AGILE's 1.2M).
AGILE's absolute 50k/150k schedule would have been **unstarted** for the whole of
our earlier 3 000-iteration (72k-step) runs — scaled to 20 000/100 000 instead.

### Do these terms make learning easier? No — measured

| term | weight | measured contribution |
|---|---|---|
| `feet_clearance` | -8.0 | **-0.0000/step** |
| `stride_length` | -6.0 | -0.0015 |
| `gait_cadence` | -1.0 | -0.0015 |
| `phase_swing` | -2.0 | -0.0043 |
| `feet_alternation` | -2.0 | -0.0004 |
| **tracking** | **10.0** | **+1.18** |

53% of the penalty budget, four orders of magnitude below the task signal. More
gating of this kind makes the objective **harder**, which is why they ramp.

### Running job

`k1_ramp_1m` on zz-bw GPU1, **42 000 iters × 4096 envs**, log
`/export/scratch/thakk100/k1/ramp_1m.log`, session `k1_ramp_1m`.
Verified live at iteration 10/42000: 0 tracebacks, all 7 ramps registered as
curriculum terms 2–8, `Mean episode length 20.07` (full episode = the ramp gate
is open early). Video every 2000 iters. Superseded control `agilepar_teacher`
killed at iteration 41. Iteration time ~16.2 s — expect **~7.5 days** at this
rate; that is the contended-GPU number, not the 1.14 s/iter exclusive one.

### Base height (squat/jump/lift) — proposal only, NOT in the run

See `docs/k1_base_height_parameter.md`. Three conflicts must be fixed before any
height command: `jumping` (-0.5) currently **forbids jumping**;
`base_height` (+2.0) pins 0.57 m and so fights squatting; AGILE's
`UniformVelocityBaseHeightCommandCfg` (`min_walk_height`, `squatting_threshold`)
already solves the second. Gate height before commanding it, and keep it out of
the 1M run so that run's result stays attributable.

### Test state

`169 passed`. The single failure, `test_deployability_invariant`, is **pre-existing**
(`k1_velocity` is not on PYTHONPATH on the laptop) and fails identically at
`ea7179b`. `test_action_rate_was_strengthened` was **rewritten, not deleted**: it
correctly failed on the new light static weight, so it now asserts the
static/ramp **pair** — a static -2.0 regression or a missing ramp both fail.

---

## 📌 SESSION UPDATE — 2026-10-01/02, latest: phase-locked STRIDE + command-coupled clock + video fix

Supersedes the `2ddec51` ramp entry below. **Read this section first.**

### Why: the 5000-iteration run failed 8/8 and it was structural

`p2_gaitfix_5k`: cadence **4.85** steps/s (gate wants 1.2-4.0), jerk **0.114**,
8/8 fail. The audited weights did **not** fix it, and the reason is not a weight:

1. **A cadence *rate* penalty is bistable.** 0.4 steps/s (standing) and 11 steps/s
   (chatter) are both "far from 2.0"; the policy takes whichever is cheaper given
   tracking pays +1.28/step against ~0.7 of gait penalty. No weight fixes an
   objective with no preferred basin.
2. **`phase_synced_swing` locked only CONTACT** — a binary predicate a 4 cm twitch
   satisfies as well as a real stride. That it also failed is the evidence the
   contact channel is too weak.
3. **The clock was FIXED at 1.001 Hz** regardless of command, so any position
   reference would step on our schedule rather than the command's. The measured
   two-mode split shows it: chattering envs moved ~0.44 m/s effective
   (11.5 steps/s × 4 cm) — the top of the command range — while under-steppers sat
   near zero. Chatter was the policy's solution to "go fast".

AGILE avoids (3) by exposing `gait_process` **and `gait_frequency`** from its
command term. We had the phase without the frequency coupling.

### Landed in `f941d15` / `d22cf3b`

- **`phase_locked_stride`** (term #24, weight −4.0): fore-aft **position** against a
  sinusoid in the clock's phase, feet antiphase, yaw-aligned forward axis,
  amplitude scaled by the clock's own frequency, `preserve_order=True` so index 0
  is the left foot.
- **`gait_clock` follows the command**: `2.0 Hz per m/s`, clamped to
  `[0.6, 2.0]` cycles/s so steps/s stays inside the gate band, EMA-smoothed
  (`COMMAND_SMOOTHING=0.15`) so an 8-12 s resample does not step the reference
  mid-stride, reset envs jump straight to target. `MIN_HZ` keeps a standing
  reference — at 0 Hz the phase freezes and the term goes uninformative exactly
  when the policy is least confident.
- **`upright_threshold` 0.80 → 0.10 on all 7 ramps.** This was **my error** at
  `2ddec51`: measured upright ratio is ~0.02-0.15 (mean episode length 20-24 steps
  of 1000), so the ramp would **never have fired** and all seven gait penalties
  would have sat at 1/5 strength for the whole run — entrenching the shuffle.
- **Video**: `--video_interval` 2000 → **1000**; recording now opt-in behind
  `--video_during_training`.

### VIDEO WAS THE 25× — and the launcher hides it

| | collection time |
|---|---|
| with `--video`, 4096 envs | **21 → 23 s/iter** (degrading over ~30 iters) |
| without | **0.85-1.14 s/iter** |

`render_mode="rgb_array"` makes the renderer capture **every step**, including the
~990 of every 1000 that no clip covers. `python.sh` returns **rc=0 on crash** —
the stride run's first attempt died with a shape `RuntimeError` at the first
reward step and still exited 0. **Gate on log markers, never rc.**

### Bug that cost a run (`d22cf3b`)

`phase_locked_stride` built a 2-component forward axis against 3-component
positions → `RuntimeError: The size of tensor a (3) must match the size of tensor
b (2)` on step 1. Fixed to `(x, y, 0)`; test added for the shape specifically.

### Running now

`k1_stride_1m` on zz-bw **GPU 1**, 42 000 iters × 4096 envs, log
`/export/scratch/thakk100/k1/stride_1m.log`. Verified at iteration 12/42000:
0 tracebacks, `phase_locked_stride` firing (−0.0046/step), ~12 s/iter, 6.5 GB, 4 %
util. ETA at 12 s/iter ≈ **14 days** — the box is loaded (avg 58, 23 users) and SSH
resets constantly; wrap remote calls in a retry loop, and use
`ssh -o ClearAllForwardings=yes zz-bw`.

**Do not scale further until `gait_gate.py` passes at 3 000 iterations.** The 4 m/s
curriculum still has never fired (its gate needs upright ≥0.80 sustained, which we
have never reached) — that speed goal remains untested.

### Base height (squat/jump/lift)

`docs/k1_base_height_parameter.md`. Blocked on: `jumping` (−0.5) currently
*forbids* jumping; `base_height` (+2.0) pins 0.57 m and fights squatting;
AGILE's `min_walk_height`/`squatting_threshold` already solve the second.

---

## 📌 SESSION UPDATE — 2026-10-02, ROOT CAUSE: the actuator gains were 4-5x wrong

**This supersedes every reward-tuning section above it. Four consecutive 8/8
gait-gate failures were caused by the ankle gains, not by the rewards.**

### The bug

Our fork of `booster_train` had dropped `armature_ratio=(1.4, 0.4)` from both K1
ankle joints, silently falling back to the wrapper's class default `(2.0, 2.0)`.
The K1 ankle is a 4-bar parallel linkage — pitch and roll axes have genuinely
different effective inertia, which is exactly what that per-axis ratio encodes.
Dropping it is silent: the config loads, runs, and is simply wrong.

On top of that, `velocity_env_cfg` hand-overrode every leg and ankle gain
(hips/knee → 100.0, ankle pitch → 100.0), reasoning that the published 4 Hz
natural frequency was "too soft to stand" and that AGILE's T1 runs 100.

| joint | broken fork | **upstream** | what we forced |
|---|---|---|---|
| ankle pitch | 35.69 | **24.98** | 100.00 → **4x the real robot** |
| ankle roll | 35.69 | **7.14** | 35.69 → **5x the real robot** |

Source: `BoosterRobotics/booster_train@main`,
`source/booster_train/booster_train/assets/robots/booster_k1.py`. Derived via
`stiffness = armature * (2*pi*f)^2`, E4310 armature `0.0282528`, f = 4 Hz.

**All overrides removed.** The articulation is now used exactly as published. A
policy cannot be taught to walk on gains 4x the robot's — which is why every
reward experiment kept failing.

### Verified on the robot's own gains (AGILE Stage 1 gate)

`sweep_standing_gains.py`, zero action, 300 steps:

```
  mult  stiff@start     z@0   z@min   z@end  verdict
   1.0         32.5   0.587   0.522   0.559  STANDS
```

**The K1 stands on the manufacturer's gains.** That gate had never been passed
before this session.

### Training effect, measured

| | before (broken gains) | after (upstream) |
|---|---|---|
| mean episode length | **6 steps** / 1000 | **34 steps** / 1000 (**5.7x**) |
| base_orientation terminations | 96% | 85% |
| upright ratio | 0.006 | 0.034 |

### Ramp gate — third and final fix

Two hand-picked gates (0.80, then 0.10) were both unreachable against a measured
0.006-0.024 ratio, silently pinning all seven gait penalties at 1/5 strength.
Now: baseline sampled **once**, gate = `baseline + 0.05` (additive).
Two multiplicative forms were tried first and **both were caught by tests, not by
reading the number** — instantaneous `baseline*1.25` (gate rises with the
policy, never overtaken) and fixed `baseline*1.25` (gate starts *above* where the
robot already is). The noise floor is capped by the baseline, since a floor above
the ratio is itself an unreachable gate.

### Running now

`k1_gains`, zz-bw **GPU 1**, **6322 iterations × 8192 envs**, log
`/export/scratch/thakk100/k1/gains_48h.log`. Smoke: 0 tracebacks, 26.1 s/iter.
Verified at iteration 13: 0 tracebacks, `phase_locked_stride` firing.

**Gate at ~1000 / ~3000 / ~6000 iterations with `gait_gate.py`.** Realistic
expectation: the gains fix is necessary but not sufficient — 34 steps of 1000 is
standing, not walking.

### Video cost, for the record

`--video` cost **~25x** (21-23 s/iter vs 0.85-1.14) because
`render_mode="rgb_array"` captures every step, not just the ~200 a clip covers.
Opt-in behind `--video_during_training` since `f941d15`. And `python.sh` returns
**rc=0 on crash** — a run that died at the first reward step still exited clean.
Gate on log markers, never rc.

---

## 📌 SESSION UPDATE — 2026-10-02, later: model_6321 is ungated and ungateable; student inherits the shuffle

### The gate that had never run, now run

`model_6321` came off zz-bw as the newest policy. Two things were wrong with the
assumption that it was deployable:

1. **Its gait gate had never actually executed.** `gate_6321.log` ends in an
   `AssertionError` (host path passed where the container path was needed) and
   `gait_gate.py: error: unrecognized arguments: --checkpoint` (the script takes
   a trace `.npz`, not a checkpoint). So no cadence/stride/jerk number had ever
   been computed for it. A trace existed at `logs/gate_6321_trace.npz`; scoring it
   locally took seconds.
2. **It is a *teacher* checkpoint**: `actor_state_dict`, first layer `(512, 237)`
   → 237-dim input with the terrain height scan. `locomotion_node.py` calls
   `torch.jit.load()` on a dict, and probes `torch.zeros(1, 50)`. It cannot load
   at all. `k1_teacher_6321_gains.pt` is the same 6,906,869 bytes — a copy, not a
   conversion.

### Scored it, then swept it — it shuffles at every speed (`GAIT_GATE=FAIL` 8/8)

| cmd | cadence | stride | jerk | speed tracking | verdict |
|---|---|---|---|---|---|
| 0.1 | **8.48/s** | 0.011 m | 0.064 | 0.087–0.108 | 8/8 FAIL |
| 0.2 | 7.83/s | 0.022 m | 0.074 | 0.116–0.191 | 8/8 FAIL |
| 0.3 | 7.57/s | 0.037 m | 0.086 | 0.268–0.285 | 8/8 FAIL |
| 0.5 | 5.43/s | 0.096 m | 0.112 | 0.485–0.502 | 8/8 FAIL |

**Cadence rises as the command falls, and stride collapses to 11 mm — smaller
than the foot.** Speed tracking is near-perfect at every speed, which is exactly
why it reads as walking: the base translates at the commanded rate while the feet
buzz. The user's hypothesis (a shorter robot needs more steps, so try slower) was
reasonable and the data says the opposite — slower is *worse*.

This is the same bistability `f941d15` diagnosed, now measured on a second policy
and across a speed sweep. It is the strongest argument yet for the command-coupled
clock: a fixed-rate reference cannot ask for 11 cm of travel at 0.1 m/s without
fighting velocity tracking.

### Student: distilled from model_6321, 1500 iters, 0 tracebacks — inherits the shuffle

zz-bw GPU1, `Isaac-Velocity-Distill-K1-v0`, `policy (500,)` / `teacher (237,)`,
0.96 s/iter at the end. Student `model_1499.pt`, mean reward ~270.

| cmd | teacher cad. | student cad. | teacher stride | student stride | teacher jerk | student jerk |
|---|---|---|---|---|---|---|
| 0.1 | 8.48 | 8.07 | 0.011 | 0.021 | 0.064 | **0.063** |
| 0.2 | 7.83 | 7.40 | 0.022 | 0.030 | 0.074 | **0.064** |
| 0.3 | 7.57 | 6.82 | 0.037 | 0.046 | 0.086 | **0.077** |
| 0.5 | 5.43 | 6.32 | 0.096 | 0.079 | 0.112 | **0.087** |

8/8 fail at all four speeds. The student is a slightly *smoother* copy of the
same shuffle (jerk ~20% better at 0.5 m/s) and it **overshoots at low speed**
(0.150–0.201 against a 0.1 command). Distillation cannot manufacture a gait the
teacher does not have — it is a lossy copy, and the loss is not in the copy.
Videos: `videos/student1499_v05_panels.mp4` + `_trace.npz` (frame checked: robot
upright and translating, HUD `vx=+0.50`, consistent with the ~0.49 m/s tracking).

**So there is still no certified policy, and the robot stays disconnected.**

### Bug found and fixed: a fixed command killed every play run (`da10d61`)

All four student recordings died on the first reward step:

```
ValueError: weighted tracking needs hi > lo, got lo=0.1, hi=0.0
```

`track_ang_vel_z_world_exp_weighted` sets `hi = abs(ranges.ang_vel_z[1])`, and both
play cfgs pin the axes they do not use to zero (`--cmd 0.5 0 0` → `ang_vel_z =
(0, 0)`), so `hi > lo` is false. This blocked straight-line gait recordings *and*
the keyboard forward-walk session. It hid because the teacher play cfg replaces
RewardsCfg wholesale and never evaluated a tracking term.

A zero-width range has two causes that are not the same bug, so the guard now
distinguishes them rather than being deleted: during training the velocity
curriculum rewrites `cfg.ranges` and a collapse means tracking weights stopped
ramping silently (still a hard error, still the default); in play the collapse is
the intent, and the floor weight is correct. `require_span` carries that.

The Rough play cfg needed it as much as the Distill one — it also keeps the
training rewards, and a spin-only recording collapses the *linear* term the same way.

### Trap: importing the task package breaks the test suite (`3e9e3e4`)

The first version of the new tests did `from k1_velocity.tasks.velocity import
agile_rewards`, which runs that package's `__init__.py` and **registers every gym
task**. `tests/test_deployability.py` then failed *later in the run* on
`ImportError: cannot import name 'AdditiveUniformNoiseCfg'` — a symbol this
IsaacLab no longer exports, i.e. a pre-existing break in the `booster_train_ref`
submodule that the error message pointed straight past. Tests here load the module
by file path and check the play cfgs with `ast`, so nothing global is touched.
204 passed.

### Duplicate-work reconciliation (reverted my own `79b76ff`)

I implemented a phase-locked stride term (`phase_synced_stride`) from the
`model_6321` evidence before noticing `f941d15` had already added
`phase_locked_stride` for the same reason. Mine was worse: a fixed 0.25 m demand
at every commanded speed is precisely what the command-coupled clock exists to
avoid. Reverted mine, kept theirs, and moved my sweep table into their docstring
as the supporting evidence rather than discarding it. **Check for an existing
implementation before adding a reward term.**

### spark02: transient Kit startup stall, and the watchdog for it

The first 512-env launch hung **60 minutes**: `python3` at 0.0% CPU, 2 threads,
blocked in `futex_do_wait` on `/dev/shm/carb-RStringInternals-58`, never printing
one Isaac line. A 16-env smoke in a fresh container minutes later was clean
(1741 lines, 0 tracebacks, rc=0), so it is a transient Kit stall, not the code.
Because a stall is indistinguishable from a slow boot by log content alone,
`teacher_supervised.sh` watches for the first `Learning iteration` and restarts
after 20 min, up to 3 tries. Containers are `--name`d so cleanup can never match
another tenant's by image ancestor — spark02 is shared and currently has 4 other
compute apps on the GPU.

Note `python.sh` returns **rc=0 on crash** (already noted below); the watchdog
greps log markers, never rc.

### Running now

`aim_spark02`, tmux `k1_phasestride`, `p2_move_teacher/2026-10-02_19-14-19`,
512 envs × 5000 iters at **1.14 s/iter**, 0 tracebacks. Gait terms confirmed
*live and biting* at iteration 1109 — `gait_cadence -0.0879`, `phase_swing
-0.0445`, **`phase_locked_stride -0.0409`** (the new position channel),
`action_jerk_l2 -0.0457`, `feet_alternation -0.0153`, `stride_length -0.0056`,
`feet_clearance -0.0000` (hinge; the foot still never lifts).

### Next

1. Gate `p2_move_teacher/2026-10-02_19-14-19/model_*.pt` at 0.1–0.5 m/s **and** at
   the top of the curriculum — the gate has only ever been run at 0.5, and a
   command-coupled clock needs checking at both ends of its range.
2. Distil **only** from a gate-passing checkpoint.
3. Still open: velocity-curriculum gating fired 0 times in 5000 iters, so the
   4 m/s goal may not actually be exercised.
4. Still blocked: Drive/Slides upload needs the user to drag the mp4s into the
   Drive folder and send the file ids.

---

## 📌 SESSION UPDATE — 2026-10-02, latest: the phase-locked run FAILED, and the reason was a dimensional error

### The run: 5000 iters, clean, and worse than the policy it was meant to replace

`p2_move_teacher/2026-10-02_19-14-19`, 512 envs, ~1.1 s/iter, **0 tracebacks**,
mean reward 3.85 -> 17.72, tracking `+2.87 -> +4.89`. Gated 8/8 FAIL at all four
speeds:

| cmd | cadence | stride | jerk |
|---|---|---|---|
| 0.1 | **10.65** | 0.012 m | 0.102 |
| 0.2 | 8.07 | — | 0.117 |
| 0.3 | 7.38 | — | 0.137 |
| 0.5 | 7.25 | — | **0.176** |

vs `model_6321` at 0.1 m/s: cadence 8.48, jerk 0.064. **Worse on both.** Jerk
also degrades monotonically with speed. Not distilled.

Frame check of `videos/teacher4999_v05_panels.mp4` (rule: look before believing):
at step 300 all 8 envs stand **upright, legs together, arms spread horizontally**;
by step 500 (t=10 s) several have fallen and reset. Standing micro-jitter, not
walking — which is what 8-16 mm strides at 7-10 steps/s means. The arms-out pose
also says `joint_deviation_arms` (-0.1) is too weak to be worth anything.

### ROOT CAUSE: `amp = target_stride / hz` is not a length, and it inverts the physics

```python
amp = target_stride / hz.clamp(min=0.1)   # m / (1/s) = m*s
```

1. **Dimensional error.** m divided by 1/s is m*s, not a distance. It looks
   correct at exactly 1 Hz (`0.22/1 = 0.22 m`) and is wrong at every other
   frequency. A defect that is invisible at the one value anyone would sanity-check.
2. **Inverted physics.** Dividing demands a *bigger* excursion when the clock is
   *slower*, while a real gait's stride grows with speed.

The clock clamps to `MIN_HZ = 0.6` for every command from 0.1 to 0.3 m/s, so the
old law demanded a `0.22/0.6 = 0.367 m` excursion across that whole band -- about
**0.88 m/s of foot speed while the base moved at 0.1 m/s.** The policy could only
respond by running faster, chasing a reference it could not reach: cadence 10.65
against a *demanded* 1.2. Above 0.5 m/s the clock leaves the clamp and the old law
is plausible, which is exactly why every earlier gate at 0.5 m/s missed it and the
defect only ever showed up at low speed.

Fixed to `amp = v / (2 * hz)` — the stride the command implies, since the clock's
`hz` is half the step rate. No tuning knob: 0.083 m at 0.1 m/s rising to 0.25 m at
0.5 m/s. `gait_clock._command_magnitude` is now public as `command_magnitude`;
`target_stride` is kept only as the fallback for envs with no readable command
term. `3b01a42`, 213 tests. The new test asserts the invariant *arithmetically*
(implied stride must not shrink as speed rises; must stay under the gate's 0.15 m
at 0.1 m/s), not by substring.

**Lesson worth keeping: a quantity that is only correct at one value of its own
input is almost always a units bug.** Check the dimensionality before tuning.

### My own bug: four Kit launches in one container

The first gate sweep ran all four speeds as one stage, i.e. four sequential Kit
launches in a single container. Speeds 0.1 and 0.2 recorded fine; 0.3 and 0.5 died
with `record_rc=124` and wrote no trace — and `gait_gate.py` then reported
`GAIT_GATE_RC=1`, which reads exactly like a failing verdict but is only a missing
file. **Never treat a gate line without a trace behind it as a result.** Now one
container per speed (`gate_one_speed.sh` + `gate_sweep.sh`), each with its own
cache dir, and the stage exits early with `GATE_SKIPPED_NO_TRACE` if the trace is
missing rather than scoring nothing.

### Also confirmed: episodes never time out

`episode_length_s = 20.0` -> 1000-step budget. `Mean episode length` went
190 -> 238 -> 339 -> **401** and plateaued at 35-45% of budget, so episodes end by
**termination** (trunk < 0.35 m, tilt > 30 deg, or Trunk contact), never by
timeout. `termination_penalty` corroborates it: `mdp.is_terminated` excludes
timeouts, and Isaac Lab normalises an episode's reward sum by `max_episode_length`,
so one fall per episode reads as -200/1000 = **-0.2** against a measured -0.167
(a running average over partly-complete episodes). Dividing by the *actual*
episode length instead gives ~1195 steps, which contradicts the 401 — the 1000-step
denominator is the one that reconciles. That is also why the velocity curriculum
finished the run at **0 firings**: its gate reads the same survival ratio (0.40
against a required 0.80).

### Running now

`aim_spark02`, tmux `k1_phasestride2`, **seed 43** (so it cannot be confused with
the failed seed-42 run), 512 envs x 5000 iters, same stall watchdog. Gated across
0.1-0.5 m/s before anything is distilled.
