# cuVSLAM run 2026-09-29 08:39 (K1 A2, head stereo)

- Bag (on robot, 3.9 GB, not copied): `~/k1_bags/k1_stereo_20260929_083924` (rect mode, 242 s)
- EDEX dataset (on robot): `~/k1_bags/edex_083924` — made by `real_robot/bag2edex.py`, then
  `sequence` = per-camera file lists and `frame_metadata: frame_metadata.jsonl` added to the body
  (both now in the script).
- Rectified pair 544x448, fx=fy=207.358, cx=237.196, cy=228.022, baseline 0.0692 m, ~8-10 Hz.
- Run (on Orin, ~/cuvslam/bin, 4 ms/frame):
  `cuvslam_api_launcher -edex stereo.edex -cfg_horizontal -cfg_enable_slam -cfg_enable_export
   -print_format tum -print_odom_poses odom_cuvslam.tum -print_slam_poses slam_cuvslam.tum -print_stats -ignore_tracking_errors`
- `odom_raw.txt`: `/odometer_state` (ns x y theta), every 10th msg.

## Result
- Frames 0-863: robot standing, drift < 2 cm.
- Camera stall (X5) frames 863-866: 41 s gap while robot walked ~16 m -> 2.7 m pose jump.
- After stall (112 s): leg odom 19.7 m; SLAM ATE 0.99 m, VO ATE 1.48 m (rigid-aligned) — mostly from the jump.
- Plot: `traj.png`. Re-open: `MPLBACKEND=TkAgg uv run --no-project --with numpy --with matplotlib python plot_traj.py --show`

## TODO
- Save/load map (`-output_map`, `-loc_input_map`) and load odometry — not done.
- New walk within ~8 min of cold boot (X5 camera dies ~10-12 min after boot).
