# K1 Locomotion: Plan, Framework Choice, and Evidence

Status as of **2026-09-28 04:50 UTC**. P2 velocity teacher mid-training.
This document is the compressed record of *why* the current approach was chosen
and *what evidence* supports it. New file — nothing here modifies the shared
`HANDOFF.md` / `TRAINING.md`.

---

## 1. TL;DR

We never had a working K1 walking policy. Three separate defects were found and
fixed, in this order:

| # | Defect | Symptom | Fix |
|---|---|---|---|
| 1 | Reward peaked on a degenerate state | P3 "converged" to a policy that ignored the ball | Mask centring reward by detector visibility |
| 2 | `TerminationsCfg` lost `@configclass` | **Zero** terminations loaded, silently; falls were invisible | Restore decorator + add a test that catches it |
| 3 | Leg gains too soft to support the body | Trunk sank 0.589 m → 0.076 m in 2.4 s under zero action | Task-local gain override, AGILE/T1 values |

Only after all three was it worth training. The current P2 teacher is the first
run where the robot learns to stay upright: **episode length 8 → 350 steps,
reward −14 → +3.4**.

The walking policy itself is **not** finished. 350 of 1000 steps is "stays up",
not yet "walks".

---

## 2. What was actually wrong

### 2.1 A reward that paid for failing

P3 (head tracking) ran 2000 iterations and produced a confident-looking
checkpoint. It did nothing. W&B showed `ball_in_frame = 0.13` and
`track_ball_angle = 0.005` — the ground-truth pointing term was essentially
zero, i.e. the head pointed nowhere near the ball.

Cause: `ball_centered = exp(-(du² + dv²)/σ²)`. When the detector sees nothing,
`_detect_yolo` leaves `du = dv = 0`, so the kernel returns **its maximum, 1.0**.
Not seeing the ball paid the *full* centring reward. The policy correctly
exploited this: park the head at a fixed yaw (+0.99 rad) and collect the reward
forever. It also fought our own `lin_vel_z_l2` penalty.

This class of bug — a term maximised by a do-nothing state — is invisible in
aggregate metrics and only surfaces by reading the policy's behaviour. It now
has a dedicated test.

### 2.2 Every fall termination was inert

`K1VelocityRoughEnvCfg` assigned `terminations: TerminationsCfg =
TerminationsCfg()` but the class had lost its `@configclass` decorator.

`ManagerBase._prepare_terms()` iterates `self.cfg.__dict__.items()`. On a
`@configclass` the terms are dataclass *fields* and land in the instance dict.
On a plain class they remain *class attributes*, the instance dict is empty, and
the manager loads **zero** terms — no warning, no exception. The manager help
output confirmed it: `<TerminationManager> contains 0 active terms`.

Consequence: `time_out`, `root_height`, `base_orientation` and
`illegal_contact` never ran. The robot could sink to the floor mid-episode and
training never saw it. This was the same blind spot that let P3 lie face-down
for 750 steps with `dones` all zero.

### 2.3 The robot physically could not stand

A reward probe stepping the env with a **zero** action (default joint targets):

```
step   0  0.589 m     step  60  0.386 m
step  30  0.550 m     step 100  0.145 m
step  50  0.469 m     step 119  0.076 m
```

Monotonic sag, no recovery, no oscillation. `booster_train` derives
`stiffness = armature × (2πf)²` at `natural_freq = 4 Hz`, which yields hip gains
of **17.8–30.2** and knee **60.4** — against AGILE's Booster T1 at **100**.

No reward can fix a body that cannot hold itself up. This is what P3's face-first
fall and P4's collapse were both symptoms of, and why training P4 (kick) before
locomotion was backwards: P4's reward peaked at iteration 120 (21.75) and decayed
to 16.8 by 1067 while mean episode length fell 844 → 650.

---

## 3. Framework decision: why AGILE

Four options were evaluated: train a residual over **COMPASS**, adopt **AGILE**
(WBC-AGILE), adopt **ProtoMotions**, or keep a **custom legged policy**.

### The key structural fact

COMPASS does not train walking. It trains a **residual on top of a pretrained
X-Mobility base policy** (`run.py -b <x_mobility_ckpt>`), and its handbook states
the residual "adapts X-Mobility's language-conditioned navigation behaviour to
embodiment-specific dynamics." Its own `robots.py` shows every embodiment needs
its own USD, joint set, actuator gains, and an action term that "maps velocity
commands to joint positions."

**That action term is the walking policy.** COMPASS would leave us building a K1
low-level controller from scratch with no help — i.e. solving the hard problem
first. It becomes the right tool for the *navigation* layer only after a
locomotion policy exists.

### The decision

**Keep a custom legged policy (P2) in our own stack, and use AGILE as a recipe
donor — not as a stack we adopt.**

Reasons, in order of weight:

1. **The P1→P5 contract is the actual asset.** P2 emits 12 leg joint targets at
   50 Hz; P3 (head), P4 (kick) and P5 (HRL) all sit on top of it, gated by
   `tests/test_deployability.py`. Adopting AGILE as the stack puts P2's policy
   in a different action/obs space and requires bridging every downstream policy
   and the safety test. That is permanent debt on the most safety-critical
   interface in the project.

2. **Adopting the framework would not even give us weights.** K1 is a different
   body with different DoF and gains. What AGILE actually offers is *numbers*:
   actuator gains, reward weights, stand-up→walk progression, sim2mujoco
   validation. All of that transplants into our env by reading their T1 config.

3. **The precedent is unusually close.** `booster_train/assets/robots/booster.py`
   already defines **`BOOSTER_T1_CFG`** alongside K1 — same vendor, same actuator
   classes (E6408/E4315/E4310/E6416), same `BoosterDelayedPDActuator` with delay
   2–8, same joint-naming convention. K1 is 6 DoF/leg (3 hip, 1 knee, 2 ankle);
   T1 is 23 DoF. AGILE's T1 velocity task is a near-template, not a foreign body.

4. **ProtoMotions is the wrong tool for this stage.** It is a separate simulator
   stack (Newton / MuJoCo / IsaacGym / Genesis), not Isaac Lab, so adopting it
   abandons the rsl_rl pipeline, the panel recorder and P1–P5. Onboarding also
   wants a K1 MuJoCo MJCF. It is a motion-*imitation* tool — the right answer for
   human-like motion *quality* later, not for velocity-command locomotion now.

### Ordering that follows

1. **AGILE-informed custom locomotion** (now) — P2 velocity, stand → walk
2. **ProtoMotions / AMASS** (later) — human-like motion quality, once it can move
3. **COMPASS** (later) — goal-reaching navigation residual, on top of a base

AGILE also derives from BeyondMimic, and this repo already has a
`booster_train/.../beyond_mimic/robots/k1/` scaffold — so a motion-tracking path
is available later without the stack switch.

---

## 4. What was adopted from AGILE

Read from `agile/rl_env/tasks/locomotion/t1/velocity_env_cfg.py`.

### Rewards

| Term | gait-v2 (old) | AGILE-aligned | Why it matters |
|---|---|---|---|
| `feet_air_time` | **+0.5** | **removed** | AGILE omits it: it peaks on hopping / never loading one foot, and it fought our own `lin_vel_z_l2` |
| `base_height` | absent | **−8.0** | Uprightness was only a *cliff* (termination), not a gradient |
| `flat_orientation_l2` | −1.0 | **−5.0** | A weak tilt penalty is how P3 fell over |
| `track_lin_vel_xy_exp` | 1.5, std 0.5 | **5.0, std 0.25** | Tracking is the task; it should dominate |
| `track_ang_vel_z_exp` | 1.5, std 0.5 | **5.0, std 0.25** | " |
| `joint_pos_limits` | ankles only | **all 12 leg joints** | Hips/knees could hyperextend for free — a real collapse mode |
| foot shaping | — | `feet_roll`, `feet_yaw_diff`, `feet_yaw_mean_vs_base`, `feet_distance_from_ref` | AGILE's substitute for air time |
| limits | — | `joint_vel_limits`, `applied_torque_limits`, `body_lin_acc_l2` | added |

New terms live in `tasks/velocity/gait_rewards.py`; Isaac Lab's locomotion mdp
has no foot-posture or foot-spacing functions.

### Levers that decide whether a biped can stand at all

| Lever | Old | AGILE-aligned | Why |
|---|---|---|---|
| action `scale` | 0.25 | **1.0** (clip ±1.0) | 0.25 rad of authority cannot recover a tilt — the policy could not physically save itself |
| command range | ±1.5 m/s | **±0.5**, 25% stand envs | Asking a biped that cannot stand for 1.5 m/s guarantees collapse |
| tilt termination | 0.8 rad (46°) | **30°** + trunk contact | 46° is well past the point of no return |
| leg stiffness | 17.8–60.4 | **100** (hips, knee) | Measured: 0.076 m vs 0.552 m standing height |

---

## 5. Evidence

**Static audits** (`29` tests, no Isaac Sim required, run on a laptop):

- `tests/test_reward_degeneracy.py` (10) — rejects air-time, requires a dense
  upright reward, requires a fall termination, requires joint-limit coverage,
  pins the P3 masking fix, rejects zero-weight terms
- `tests/test_env_cfg_wiring.py` (19) — every manager group in every
  `k1_velocity` env cfg must carry `@configclass`. **Verified to fail** when the
  fix is reverted.

**Functional probe** (`scripts/probe_rewards.py`), now a pre-flight gate:

- All **23** reward terms compute finite
- Terminations: `['time_out', 'root_height', 'base_orientation', 'illegal_contact']`
- Standing: `end z = 0.552 m`, min over last half `0.548 m` → **STANDS**
- Note: `action_rate_l2`, `dof_vel_limits`, `torque_limits`, `undesired_contacts`
  are legitimately 0 under a *zero* action, and the probe now treats them as
  expected rather than dead. `undesired_contacts` going to 0 is a positive
  signal — the robot is no longer lying on the ground.

**Training** (P2 teacher, W&B `mbd7q7vf`):

```
iter      eplen    reward   std
   0      19.92   -14.13   1.00
 300      12.98    -5.51   0.36
 600      42.20    -4.42   0.23
 900     150.35    -0.10   0.20
1050     215.84     1.51   0.20
1468     349.78     3.39   --
```

---

## 6. Caveats — what this does not prove

- **The leg gains are sim-only and above the hardware values.** They are what
  makes the K1 stand, but the policy is learning against actuators stiffer than
  the real robot's. The standing margin is optimistic. They must be re-tuned
  against measured hardware before any real-robot deployment, and the gains live
  in the velocity task only — `booster.py` keeps the real-robot values and Track
  B is unaffected.
- **"Stands up" is not "walks."** 350 of 1000 steps means the policy survives;
  velocity tracking has not yet been demonstrated.
- The 1200-iteration dip (197 / −1.90) is normal PPO noise, not a regression.

---

## 7. Guard rails added

The recurring failure mode in this project was not a wrong hyperparameter — it
was a **silently inert** piece of the setup (a degenerate reward, a manager
loading zero terms, a body that could not stand). Each now has a check:

- Reward degeneracy → `test_reward_degeneracy.py`
- Manager group not loading → `test_env_cfg_wiring.py`
- Env misconfigured at launch → `probe_rewards.py` as a pre-flight gate in
  `spark_p2_gait_container.sh` (standing + terminations + term health in ~2 min,
  instead of discovering it 3000 iterations in)

---

## 8. Next steps

1. Teacher finishes (~05:25 UTC) → student distillation (~06:45 UTC)
2. Record both with `play_record.py --panel_video`
3. **Verify gait before showing anything** — finite traces, `>0.5 m` root
   displacement, and a frame I have actually looked at. A video is not evidence
   until it has been checked.
4. P4 (kick) only after locomotion is demonstrably walking — training kick
   before it can stand is what made P4 regress
5. P3 retrain, now that it has a fall termination and a masked reward
6. Re-tune gains against hardware before sim2real

---

## Appendix: file map

| Path | What |
|---|---|
| `tasks/velocity/velocity_env_cfg.py` | AGILE-aligned rewards, terminations, commands, action scale, task-local leg gains |
| `tasks/velocity/gait_rewards.py` | New foot-shaping terms (roll, yaw-diff, yaw-vs-base, stance width) |
| `tasks/head/head_mdp.py` | P3 centring-reward visibility mask |
| `tasks/head/head_env_cfg.py` | P3 fall terminations |
| `scripts/probe_rewards.py` | Functional reward/standing/termination probe |
| `scripts/sweep_standing_gains.py` | Stiffness sweep (superseded by the task-local override) |
| `scripts/spark_p2_gait_container.sh` | P2 campaign with pre-flight gate |
| `tests/test_reward_degeneracy.py` | Reward audit |
| `tests/test_env_cfg_wiring.py` | `@configclass` audit |

References: [AGILE](https://github.com/nvidia-isaac/WBC-AGILE) ·
[COMPASS handbook](https://nvlabs.github.io/COMPASS/docs/) ·
[ProtoMotions](https://github.com/NVlabs/ProtoMotions) ·
`PLAN_PHASE6_SOCCER_HRL.md` (architecture and deployability invariant)
