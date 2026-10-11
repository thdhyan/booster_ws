## Notes (2026-10-11)

- **New since the run above:** `feet_flat` (`gait_rewards.feet_flat_contact`, weight +1.0, std 0.15) in
  `SquatRewardsCfg`. It pays per loaded foot (contact force > 5 N) for a level sole,
  `exp(-|g_xy in foot frame|^2 / 0.15^2)`: 6° tilt keeps 61 % of the credit, 15° keeps 5 %. The old `feet_roll`
  only charged roll, so heel/toe pitch ("standing on the ankles") was free. The squat students inherit the env, so
  they get it by re-distilling from a teacher trained with it.
- The push frozen base `models/k1_push_base.pt` is a JIT export of the squat teacher (238 inputs, uses the
  privileged height scan).
