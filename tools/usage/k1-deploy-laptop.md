The laptop side: a ROS 2 Jazzy colcon workspace (`src/`), and the push-IK and pad tools (`deploy/`).

```bash
# ROS packages: k1_bringup, k1_control (needs the Booster SDK), k1_fleet_rviz, k1_locomotion, k1_sim_gazebo, k1_wbc
git submodule update --init sdk/booster_robotics_sdk    # Booster SDK, which k1_control's CMake expects here
source /opt/ros/jazzy/setup.bash && colcon build && source install/setup.bash
ros2 launch k1_fleet_rviz fleet_rviz.launch.py
```
| Tool (`deploy/`) | What |
|---|---|
| `pad_udp_send.py --host IP` | DS4 -> robot runner over UDP :9870 |
| `push_ik_web.py [--fake] [--send] --robots A B` | browser tool for the two-robot box push, http://127.0.0.1:8765 |
| `push_ik_sim.py` | CPU MuJoCo preview used by the web tool |
| `gamepad_sim.py` | MuJoCo sim-to-sim of the walk base (same `gamepad_core` as the robot) |
| `k1console/` | decoupled console: truth sources (page / PhaseSpace / own MuJoCo) + policy preview, http://127.0.0.1:8770 |

`k1console` is the simple, source-agnostic console: `python -m k1console --sim` (add `deploy/` and `libs/*/src` to
`PYTHONPATH`), `--viewer` adds its own MuJoCo window (3D, main thread; `--no-viewer` keeps it headless), PhaseSpace via
`python -m k1console.phasespace_bridge --map a=dhyan-hat,box=dhyan-hat-2`.
The page shows live truth and preview replay side by side, adds/removes robots (`/api/agents`), walks a robot to a goal
in the own sim (`/api/goto`) and fills its walk-base / push-policy dropdowns from a scan of the models dir
(`GET /api/policies`). See `deploy/k1console/README.md`.


Robot IPs, safety and step-by-step runs: [docs/ROBOTS.md](https://github.com/thdhyan/k1-workspace/blob/main/docs/ROBOTS.md). The end-to-end test runs without hardware:
`pytest tests` (runs `tests/e2e_push_ik_web.py` and prints `WEB TEST OK`).
