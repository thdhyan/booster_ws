# Base-height parameter: squat, jump, lift — design, not yet implemented

**Date:** 2026-10-01 · **Status:** proposal. The running job (`k1_ramp_1m`,
`2ddec51`) is pure locomotion and does **not** include any of this.

## The request

Add a base-height parameter so the base policy can squat, jump, and lift objects.

## The honest answer to "would these make it easier to learn?"

**No — the opposite, and the evidence is in our own logs.** The custom gait terms
already are in the objective and they are not helping:

| term | weight | measured contribution |
|---|---|---|
| `feet_clearance` | -8.0 | **-0.0000/step** |
| `stride_length` | -6.0 | -0.0015 |
| `feet_alternation` | -2.0 | -0.0004 |
| `phase_swing` | -2.0 | -0.0043 |
| `gait_cadence` | -1.0 | -0.0015 |
| **tracking term** | **10.0** | **+1.18** |

Against a +1.18 task signal they contribute four orders of magnitude less. They
cost 53% of the penalty budget and return almost nothing. Adding *more* gating
terms makes the objective harder to optimize, not easier — which is why this run
ramps them from a fifth of their weight and only charges full price once the
robot can stand.

Adding a height command is a genuinely different proposition from what has failed:
a command *adds* capability the policy can express, while the gait terms only
*restrict* what it may do. But it is not free, and the costs are specific.

## Three conflicts that must be resolved first

1. **`jumping` (-0.5) directly forbids jumping.** We added it for AGILE parity in
   `bbf01ac`; it penalizes both feet leaving the ground. It must become
   *conditional* (only active when commanded to walk) or it will suppress the exact
   behavior being asked for. This is a live contradiction in the current config.

2. **`base_height` (+2.0) pins the trunk to 0.57 m.** Squatting means going *below*
   0.57, so this positive reward actively fights a squat. It has to become a
   *tracking* reward against a commanded height, not a fixed target.

3. **`min_walk_height` / `squatting_threshold` are AGILE's own gates.** AGILE
   already solves (2): `UniformVelocityBaseHeightCommandCfg` scales velocity
   commands down to zero below `min_walk_height` (0.4) and zeroes them on
   transition to a squat below `squatting_threshold` (0.7). That is how AGILE lets
   one policy squat and walk without the two objectives fighting.

## What AGILE already gives us

Verified against AGILE source, not inferred:

- `UniformVelocityBaseHeightCommand` / `Cfg` — adds `base_height` to the command
  tensor, with `min_walk_height`, `squatting_threshold`, `default_height`,
  `random_height_during_walking` (default False: height commanded only for
  standing envs).
- Rewards: `track_base_height` (stance-only), `base_height_in_threshold`,
  `height_reached`, `standing_at_timeout`, `base_height_exp` (already ours).
- Assistance for learning height transitions: `HarnessAction`, `LiftAction` with a
  `remove_harness` curriculum — AGILE ramps harness stiffness to zero rather than
  dropping the robot in.

Lifting objects is **not** in AGILE's locomotion reward set; it needs an object
term and is closer to the existing `k1_kick` task than to locomotion.

## Recommended sequencing

Do not add this to the 1M run. That run is a clean test of one hypothesis (do the
ramps produce a walk?), and mixing in a height command would make its result
unattributable — the mistake AGILE's own Stage 4 warns about.

1. **Now:** let `k1_ramp_1m` run. Gate: does it walk, and do the ramps behave?
2. **Next:** gate `jumping` on the commanded behaviour, and make `base_height`
   track a command. Both are small, both are prerequisites for anything else.
3. **Then:** a height command on a *copy* of the locomotion task — squat
   curriculum from the gait-gate-passing policy via `HarnessAction`/`LiftAction`,
   not from scratch. Warm-starting is what makes this tractable; a from-scratch
   multi-behavior policy on a robot that currently falls 99% of the time will not
   converge.
4. **Lifting:** separate task, warm-started, reusing `kick`'s object plumbing.

**The order matters and it is not the order in the request.** Height *gating* has
to exist before height *commanding*, or the `jumping` and `base_height` terms
actively train against squatting.