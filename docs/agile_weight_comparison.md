# AGILE weight comparison — and why a weaker policy should run *lighter* early

**Date:** 2026-10-01 · **Config compared:** `velocity_env_cfg.py` @ `bbf01ac` (23 AGILE-parity
terms landed, 31 total) · **Reference:** AGILE `v1.3.x`,
`nvidia-isaac/WBC-AGILE` — `tasks/locomotion/t1/velocity_env_cfg.py` (Booster T1, the
closest published reference to a K1) and `tasks/locomotion/g1/velocity_env_cfg.py`.

Extracted mechanically from the three `RewardsCfg` classes, not transcribed by hand
(`weights.json` via AST, `std` and `func` read alongside each weight).

---

## 1. Where we differ from AGILE T1

Sixteen of T1's 23 terms are **identical** to ours — same weight, same std. Those are
`base_height` (2.0/0.1), `orientation` (-5.0), `torques` (-1e-4), `ankle_torques`
(-1e-4), `ankle_roll_torques` (-2e-3), `lin_vel_z` (-0.5), `ang_vel_xy` (-0.5),
`dof_vel` (-2e-4), `dof_pos_limits` (-1.0), `dof_vel_limits` (-1.0),
`torque_limits` (-0.01), `feet_roll` (-0.1), `feet_yaw_diff` (-0.2), `feet_yaw_mean`
(-4.0), `feet_distance` (-0.2), `root_acc` (-2e-5). The parity work is real.

The seven that diverge:

| Concern | AGILE T1 | K1 now | ratio | note |
|---|---|---|---|---|
| **smoothness** `action_rate` | **-0.5** | **-2.0** | **4.0×** | ours raised from -0.5 in an earlier round |
| **smoothness** `action_rate_rate` | **-0.05** | **-0.5** (`action_jerk_l2`) | **10×** | raised from -0.02 → -0.5 in the 10-01 audit |
| **collision** | **-0.2** (`collision`) | **-1.0** (`undesired_contacts`) | **5×** | ours also watches 4 hip bodies, not just Trunk |
| **foot slip** | **-0.1** (`feet_slip`) | **-0.25** (`feet_slide`) | **2.5×** | |
| fall cost | -100.0 | **-200.0** | 2× | doubled vs G1, which also uses -200 |
| lin-vel tracking | 5.0, std 0.2 | 10.0, std 0.15, **weighted 1–2×** | up to 4× peak | deliberate, this is the speed fix |
| yaw tracking | 5.0, std 0.2 | 5.0, std 0.25, weighted | ~1× | |

**Every term we made heavier is a regularizer or a fall penalty. The task reward is
the only thing that went up.** That is the right direction, but the magnitude is the
question.

## 2. The budget, which is the number that matters

Sum of |per-step penalty weights| (excluding episodic `termination_penalty`):

| Config | terms | penalty budget | non-AGILE extras |
|---|---|---|---|
| AGILE T1 | 23 | **13.36** | 0 |
| AGILE G1 | 22 | **18.62** | 0 |
| **K1 now** | 31 | **35.86** | **19.10 (53%)** |

K1 carries **2.68× AGILE T1's regularization pressure**. And **53% of it is not in
AGILE at all** — it is our own gait shaping: `feet_clearance` -8.0, `stride_length`
-6.0, `phase_swing` -2.0, `feet_alternation` -2.0, `gait_cadence` -1.0.

Those five were added to fix a real measured failure (the 6.68 steps/s shuffle), so
they are not gratuitous. But they were added *statically, at full strength, from
iteration 0*, to a policy that at iteration 12 cannot stand (measured: **99.3% of
episodes terminate on `base_orientation`**).

Note also `feet_clearance` at -8.0 contributed **-0.0000/step** in the last run —
it is a hinge, so it reads zero until the robot actually lifts a foot. That is
correct behaviour for a hinge, but it means a fifth of our stated budget is
invisible in the logs until late.

## 3. Your instinct is right, and AGILE already encodes it

> should we run it lower since our policy is not as good as AGILE's results on G1 and H2

Yes — and AGILE does not do it by picking low constants. It ships a curriculum that
**starts light and ramps regularization up**, from
`agile/rl_env/mdp/curriculums/task_curriculum.py::update_reward_weight_step`:

```python
class update_reward_weight_step(ManagerTermBase):
    def __init__(self, cfg, env):
        self.start_weight = env.reward_manager.get_term_cfg(reward_name).weight
    def __call__(self, env, env_ids, reward_name, start_step, num_steps,
                 terminal_weight, use_log_space=False):
        if env.common_step_counter <= start_step:
            return self.start_weight          # <- OFF until start_step
        ...
        env.reward_manager.get_term_cfg(reward_name).weight = new_weight
```

Wired in AGILE's own T1 task (`CurriculumCfg`):

| term | start weight | `start_step` | ramp | terminal weight |
|---|---|---|---|---|
| `action_rate` | -0.5 | 50 000 | 100 000 | **-2.0** |
| `action_rate_rate` | -0.05 | 60 000 | 100 000 | **-1.0** |

So AGILE deliberately runs **-0.5 / -0.05** while the policy is learning to stand,
then tightens to **-2.0 / -1.0** once it can walk. G1 carries the same idea in a
different place: its `action_rate` is **-0.01**, 50× lighter than T1's, because G1
is the mature policy.

**This reframes the earlier audit.** The 10-01 change raised `action_jerk_l2` from
-0.02 to -0.5 on the reasoning that it was "inert". That was correct *as a static
weight at the end of training*. It was the wrong lever for the beginning of
training, where the problem is the opposite: too much penalty pressure on a policy
that has not learned to stand.

## 4. The catch: our runs are 20× shorter than AGILE's

AGILE's schedule is denominated in `common_step_counter` (control steps, not
iterations). Our `num_steps_per_env = 24` — identical to AGILE's — but:

| | iterations | × 24 | `common_step_counter` reached |
|---|---|---|---|
| AGILE H2 reference | 50 000 | | 1 200 000 |
| our long runs | 3 000 | | 72 000 |
| our budgeted runs | 5 000 | | 120 000 |

AGILE's `action_rate` ramp starts at 50 000 and ends at 150 000. **On a 3000-iteration
run we would reach 72 000 — the ramp would begin near the very end and never
finish.** Copying AGILE's numbers verbatim would leave the term effectively at
-0.5 for the whole run and then jump at the end, i.e. it would do nothing.

Rescaled to the same *fraction* of training as AGILE (start ~4%, finish ~12.5%):

- 3000 iterations: start ≈ **3 000**, finish ≈ **9 000** control steps
- 5000 iterations: start ≈ **5 000**, finish ≈ **15 000**

## 5. Recommendation

Do **not** lower the weights permanently — that trades the "falls over" failure for
the "shuffles with 0.107 jerk" failure, which is the one already measured against
the gait gate's <0.06 bound.

Instead, adopt AGILE's mechanism for the smoothness terms *and* for the five gait
extras that make up the other half of the budget:

1. Set `action_rate_l2` and `action_jerk_l2` (and ideally `feet_clearance`,
   `stride_length`, `phase_swing`, `feet_alternation`, `gait_cadence`) to their
   AGILE *start* weights as the static config value.
2. Add an `update_reward_weight_step`-equivalent curriculum per term, rescaled to
   the run length, ramping to the current values as terminal weights.
3. Gate the ramp on the same upright signal `VelocityRangeCurriculumTerm` already
   uses (`episode_length_buf.mean() / max_episode_length`), not on wall-clock steps
   alone — a policy that is still falling at step 3 000 should not be handed the
   terminal smoothness penalty.

That last point is the one AGILE's own step-based schedule lacks and the one our
measured failure mode most needs: our problem is not "not smooth enough", it is
"falls at 99%". Step 4 above (matching AGILE exactly) is safe; step 3 is better.

**Not implemented yet — this document is the proposal.** The shipped config is the
`bbf01ac` parity baseline, and the zz-bw 3000-iteration run is the control it should
be measured against.