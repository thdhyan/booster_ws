---
name: k1-robot-discover
description: Find K1 robots on the lab network by UDP heartbeat or ping sweep and check them against fleet.json.
---

# K1 discover

1. Heartbeat: robots broadcast UDP :9000 once a second (serial, model, fw, IP, nickname). Listen 3 s on the robot subnet.
2. Fallback: `nmap -sn 192.168.50.0/24`, then `ssh -i ~/.ssh/booster_k1 booster@<ip> 'cat /opt/booster/robot_info.txt'` (nickname `Booster K1_UMN_<name>`).
3. Compare to `fleet.json`; update `fleet.json`, `docs/FLEET.md` and `~/.ssh/config` together. Identity comes from robot_info.txt, not from the last IP octet.
4. New robot onto robotics_5G (run detached, the ssh drops): `nmcli dev wifi rescan; nmcli dev wifi connect robotics_5G password robotics name robotics_5G`, then pin a static IP with `nmcli con mod robotics_5G ipv4.method manual ipv4.addresses <ip>/24 ipv4.gateway 192.168.50.250 ipv4.dns "8.8.8.8 1.1.1.1" ipv4.ignore-auto-dns yes`.
