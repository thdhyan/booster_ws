# K1 remote-stick to body-velocity calibration

Fitted from the 12-minute, 254 m walk recorded on robot A2 (2026-09-29), bag
`real_robot/bags/k1_walk_20260929_023045`.

## Why this exists

`/remote_controller_state` gives the human's **joystick axes**, not a velocity.
Training behaviour cloning on the raw axes would learn a label in joystick units,
which is meaningless to a policy that has to emit m/s. The odometer gives the
velocity actually achieved, so the mapping can be recovered from the data instead
of guessed.

## Method

1. Decode `Odometer` (float32 x, y, theta) and `RemoteControllerState`
   (uint32 event; float lx, ly, rx, ry; 21 bool; uint8 reserved) from the bag.
2. Unwrap theta, boxcar-smooth the pose over 0.5 s, then central-difference.
   Differencing the raw pose is useless: per-step noise of a few mm at 500 Hz
   appears as apparent speeds over 10 m/s.
3. Reject samples with |v| > 3 m/s or |vyaw| > 4 rad/s, and samples more than
   0.2 s from a fresh joystick reading.
4. Least squares of measured body-frame velocity on [lx, ly, rx, ry] + bias.
   232,828 clean samples survived.

## Result

Coefficients are m/s (or rad/s) per unit of joystick:

| measured | lx | ly | rx | ry | bias | R² |
|---|---|---|---|---|---|---|
| vx_body | 0.066 | **-0.493** | -0.007 | 0.094 | 0.117 | 0.279 |
| vy_body | **-0.231** | 0.074 | 0.077 | -0.006 | -0.020 | 0.280 |
| vyaw | 0.027 | 0.044 | **-0.693** | **-0.378** | 0.007 | 0.371 |

So forward speed is driven by **left-Y** (negative = forward), lateral by
**left-X**, and turning by the **right stick** (mostly right-X).

## Caveats, which matter

- **R² is only 0.28-0.37.** The factory walker has momentum, the odometer is
  noisy, and there is lag between stick and response. Treat this as a rough
  calibration, not a plant model.
- **The relation is not linear at the extremes.** Full stick predicts about
  0.6 m/s but the measured peak was **1.29 m/s**, so the walker gains speed
  beyond the linear fit — there is a deadzone and a saturation. Do not
  extrapolate this fit to command a robot.
- **Achieved speed peaked at ~1.3 m/s** with the stick at full deflection. This
  is the evidence behind the P2 velocity curriculum ceiling of 1.5 m/s, and it is
  why a 3 m/s target is treated as a sim-only stretch goal rather than something
  the hardware has shown.
- **Each bag segment is its own odometry frame.** seg002 starts at y = -29.7 m
  while seg001 ended near y = 0.1, so positions must not be stitched across
  segments. Per-segment path lengths are self-consistent: 106.7 + 132.8 + 23.4 m.

## Reproduce

See `docs/k1_walk_data_plan.md` for the bag layout. The decode and fit are the
steps above; the field orders come from
`sdk/booster_robotics_sdk/include/booster/idl/b1/{Odometer,RemoteControllerState}.h`.
