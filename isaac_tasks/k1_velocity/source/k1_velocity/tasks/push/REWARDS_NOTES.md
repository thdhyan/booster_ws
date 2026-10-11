## Why the push robot stands still: reward hacking audit (2026-10-11)

Run `p6_push/2026-10-10_18-59-36` (SG reward set, 5000 iterations). `goal_reached` was 0 % for the whole run.
How to read the numbers: `Episode_Reward/x` is about `weight * mean(func)` per second, so it is directly comparable
between terms.

| term | it200 | it4999 | can it be earned without moving the box? |
|---|---|---|---|
| fast_approach | 0.29 | **1.13** | **yes, it can be farmed** (see 2) |
| etiquette | 0.76 | **0.85** | **yes**: +1 per step for keeping the wrists > 0.5 m from the box (see 1) |
| base_height_command | 0.64 | 0.75 | yes (holding H*) |
| approach_object | 0.47 | 0.53 | yes (hovering near the box) |
| task_box_goal | 0.68 | 0.40 | yes (baseline 0.35/(0.35+d) paid with the box untouched) |
| heading_progress | 0.19 | 0.33 | **yes, sign bug**: pays only while NOT in contact (see 3) |
| stop_bonus | **4.18** | 0.21 | **yes at the start of training** (see 4) |
| track_cmd_lin_vel | 0.15 | 0.12 | yes (command 0, stand) |
| success_bonus / parked | 0 | 0 | never reached |

Only one term pays for the box actually moving toward the goal: the `along / 0.10` part of `etiquette`. Moving
the box needs the wrists on the box, which zeroes the +1 standoff part of the same term and risks a −4 fall
(−200 × dt). So pushing is worth about zero net, and standing near the box is the optimum.

1. **`push_pull_etiquette` pays standoff the whole episode.** `standoff = gap.mean > 0.5` m adds +1 every step,
   which rewards *not touching* the box. It was meant for "withdraw after parking", but it is not gated on success.
   Fix: drop the standoff part, or multiply it by the `goal_reached` hold state.
2. **`fast_approach` can be farmed.** It is `clamp(prev_gap - gap, min=0) / dt` while gap > 0.6 m. Moving the
   arms/body in and out pays on every inward stroke and costs nothing on the outward one. 1.13/1.5 means a mean
   closing speed of 0.75 m/s, far more than a real approach. Fix: make it potential-based and signed,
   `(prev_gap - gap) / dt` without the clamp, which telescopes to a constant over the episode, or delete it
   (`approach_object` already shapes the approach).
3. **`heading_progress_combo` gate is inverted.** `engagement(gap, d) = gap > d` is 1 when FAR from the box, but
   the docstring says "only count while in contact". It pays for facing the goal from a distance. Fix: gate on
   `gap <= contact_dist`.
4. **`stop_at_goal_box_still` pays at reset.** Curriculum `goal_dist` starts at 0.30 m, but the stop radius is
   0.35 m. So the box starts inside the goal and still, and the policy got +5 per step for standing (it200: 4.18 of
   about 137 total). That locked in "do not touch the box" before the curriculum moved the goal out.
   Fix: curriculum start > 0.35 m (e.g. 0.5), or pay only after the box has entered the goal from outside.
5. **No dense progress term.** `box_goal_progress` and `box_vel_toward_goal` (v3, potential-based) are retired in
   the SG set. Fix: re-enable `box_goal_progress` (w 2.0) and `box_vel_toward_goal` (w 0.5).
6. **Success is too hard.** Mean CORNER error < 0.08 m, held for 1 s, with the goal yaw curriculum reaching ±180°,
   means rotating the box while pushing. Fix: centroid-only success first, or turn yaw on only after
   success > 20 %.

Not changed yet: these need a decision and a fresh push run.

Also: the play/eval env (`K1PushSGPlayEnvCfg`) sets `curriculum = None`, so videos always show the 0.30 m start
goal (inside the stop radius). Play should use the final curriculum values.
