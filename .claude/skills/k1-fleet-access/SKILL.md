---
name: k1-fleet-access
description: Connect to Booster K1 robots (A2, A3, B1, B2, B3) over robotics_5G: IPs, ssh, ROS, ports, and what never to run. Use when asked to reach, check, or run something on a K1.
---

# K1 fleet access

Read `docs/FLEET.md` (IPs) and `docs/ACCESS.md` (how). Registry in `fleet.json`, loader `k1_chat.fleet`.

1. Reachability: `for n in a2 a3 b1 b2 b3; do ssh -o ConnectTimeout=4 boosterk1.$n 'echo $HOSTNAME; grep Version /opt/booster/version.txt | tail -1' ; done`
2. If ping fails: laptop must be on `enp2s0` (192.168.50.x); robot may be off or low battery.
3. ROS on robot: `source /opt/ros/humble/setup.bash; source /opt/booster/ros2/setup.bash`.
4. Never run `booster-cli upgrade` or `upload_log`. Never auto-run policies on hardware.
5. Robot internet needs the laptop NAT (see FLEET.md).
