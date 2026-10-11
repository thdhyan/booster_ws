# Simplified rewards for kick, track, carry, squat (proposal)

Status: proposal, 2026-10-11. Nothing here is implemented yet; it needs the user's OK first.
Push (single `Isaac-Push-Simple-K1-v0` and MAPPO) already uses the simplified set.

The rule is the same as for push simple:

- 2–3 task terms that say what the task is
- one "do not fall" termination penalty of −500 (× dt = −10)
- light regularisation only

Isaac Lab multiplies every weight by dt = 0.02, so the per-event numbers below are weight × 0.02.

**Residual tasks** (squat residual, kick, track) sit on the frozen velocity student, which already walks with
a clean gait. They do not need the ~25-term AGILE gait stack. `residual_l2` keeps the policy close to the
student's gait.

## Squat residual (`Isaac-Velocity-Squat-Residual-K1-v0`): 33 terms → 8

| term | weight | why |
|---|---|---|
| `track_lin_vel_xy_exp` | 10.0 | follow the velocity command |
| `track_ang_vel_z_exp` | 5.0 | follow the yaw command |
| `base_height` (`base_height_cmd_exp`, std 0.05) | 4.0 (was 2.0) | **the** squat term: trunk at H* |
| `feet_flat` | 1.0 | heels down while squatting (exists) |
| `termination_penalty` | −500 | do not fall |
| `flat_orientation_l2` | −2.0 | trunk upright |
| `residual_l2` | −0.2 | stay a residual |
| `action_rate_l2` | −0.01 | smooth |

Drop: every gait term (feet_* yaw/distance/slide/roll, cadence, phase_*, stride, clearance, alternation, jumping),
the torque/acc/vel L2 terms and `body_acc_l2`, all of which the student already handles.

## Kick (`p9_kick_residual`): 11 terms → 7

| term | weight | why |
|---|---|---|
| `approach_ball` | 1.0 | 1. get to the ball |
| `ball_to_goal_progress` → **rate** (m/s toward goal, like `box_goal_progress_rate`) | 5.0 | 2. ball closer to goal; the current fraction-closed term pays for state, not progress |
| `goal_scored` | 500 | 3. success bonus (× dt = +10) |
| `termination_penalty` | −500 | do not fall |
| `head_track` | 0.5 | keep the ball in view (the student policy will see it through the camera) |
| `residual_l2` | −0.2 | stay a residual |
| `action_rate_l2` | −0.01 | smooth |

Drop: `alive` (it pays for stalling), `kick_toward_goal` (overlaps the progress rate), `align_stance` and
`face_ball` (shaping the policy finds by itself once progress pays), `command_rate_l2`, `flat_orientation_l2`.

## Track (`p8_balltrack_residual`): 10 terms → 5

| term | weight | why |
|---|---|---|
| `ball_distance` (target 0.8 m, std 0.4) | 1.5 | 1. stay at tracking distance |
| `head_track` | 1.0 | 2. look at the ball |
| `termination_penalty` (fall + `ball_lost`) | −500 | do not fall, do not lose the ball |
| `residual_l2` | −0.5 | stay a residual |
| `action_rate_l2` | −0.01 | smooth |

Drop: `alive`, `face_ball` and `ball_visible` (both overlap `head_track`), `command_rate_l2`, `flat_orientation_l2`.

## Carry (`Isaac-Carry-K1-v0`): 40 terms → 9 + 4 leg terms

Carry legs are trained **from scratch** (decoupled actor), not as a residual. So carry still needs a few gait
terms. Option A keeps those 4. Option B switches the legs to a residual over the velocity student (as in the
Carry v2 design) and drops them.

| term | weight | why |
|---|---|---|
| `goal_progress` (new: metres closed toward the goal per s, × `hold_quality`) | 5.0 | 1. carry the box to the goal; replaces goal_coarse/fine tanh + velocity tracking |
| `goal_reached` (× `reach_scale`) | 500 | success bonus (× dt = +10) |
| `stand_hold` (`hold_quality`) | 1.0 | 2. box at the hold spot, trunk upright |
| `termination_penalty` (fall + `weld_broken`) | −500 | do not fall, do not drop the box |
| `weld_error` | −10.0 | box stays in the forearms |
| `box_tilt` | −2.0 | box level |
| `flat_orientation_l2` | −2.0 | trunk upright |
| `action_rate_l2` | −0.05 | smooth |
| `joint_deviation_head` | −0.1 | head still |
| option A only: `feet_slide`, `feet_clearance`, `feet_alternation`, `undesired_contacts` | as now | minimum gait for legs from scratch |

Drop: velocity tracking (the command is derived from the goal, so it duplicates goal progress), `stand_still`,
goal_coarse/fine, and the rest of the gait and torque stack.

Changing rewards on a running policy: relaunch with `--warm_start` from the latest checkpoint. That gives a fresh
optimiser and resets the iteration counter, as for the carry relaunch at 14:22.
