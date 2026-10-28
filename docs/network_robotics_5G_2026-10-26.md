# Lab network notes — 2026-10-26

## Facts
- Lab SSID is `robotics_5G` (underscore) and 2.4 GHz `robotics`. Password `robotics` (`robtoics` fails).
- Robot wifi iface `wlP1p1s0`; robots have passwordless `sudo nmcli`. Joined with
  `nmcli dev wifi rescan; nmcli dev wifi connect robotics_5G password robotics name robotics_5G`.
  Run it detached (`nohup ... &`): the SSH session drops when the robot changes network.
  Robots need a `rescan` first, else "No network with SSID found".
- Robotics router (192.168.50.1) has NO internet: its WAN/university jack is not working
  (likely MAC registration needed). Router clock reads 2015 (no NTP).
  Router LAN MAC (ARP): `A0:04:60:8A:68:90`. WAN MAC unknown (needs admin login).
- Router login page looked like ASUS but ARP vendor is Netgear; do not trust either.
- Laptop: wlo1 on eduroam (internet), enp2s0 -> router. NM profile `robotics-lan`:
  DHCP 192.168.50.x + static 192.168.123.222/24 (Unitree G1 subnet), never-default route.
  Old profile "Wired connection 1" left untouched.
- A2, A3, B1, B3 run firmware v1.6.2.2-release.global-02145 (2026-06-03). (First report of 1.6.1.1 was wrong: it read the oldest entry of the version.txt history. Use `grep Version /opt/booster/version.txt | tail -1`.)
- dl and zz-bw reachable once laptop has internet. zz-bw GPU1 was full (94.5/97.9 GB), GPU0 free.
- Hello hotspot = 10.245.87.0/24, gateway 10.245.87.175; robots there are unreachable from the
  laptop while it is on eduroam.

## Robots (identities confirmed from /opt/booster/robot_info.txt nickname, 2026-10-27)
| Robot | SN | MAC | robotics_5G IP (static) | Status |
|---|---|---|---|---|
| A2 | 01022026070048 | 20:bd:1d:d8:31:00 | 192.168.50.215 | firmware update DONE (per user, 2026-10-27); off, charging; version not re-read |
| A3 | 01022026070058 | 20:bd:1d:d8:31:0f | 192.168.50.230 | on, SSH OK |
| B1 | 01022026070044 | 20:bd:1d:d5:3e:5f | 192.168.50.105 | on, SSH OK |
| B3 | 01022026070057 | 20:bd:1d:d8:3c:ea | 192.168.50.248 | on, SSH OK, 1.6.2.2, download started 2026-10-27 |
| B2 | ? | 20:bd:1d:71:ae:53 (guess) | 192.168.50.62 (static set) | off, low battery |
| A1 | 01022026070052 | 20:bd:1d:77:55:b5 | 192.168.50.82 (DHCP, no static yet) | on, SSH OK (found 2026-10-27 via ARP; was listed out of commission) |
Nicknames are `Booster K1_UMN_<name>`. `~/.ssh/config` aliases boosterk1.{a2,a3,b1,b2,b3} point at these IPs
(backup: ~/.ssh/config.bak-20261026). B2 MAC/IP mapping is a guess (it was the .109 Hello robot).

## Tests 2026-10-27 (laptop on ethernet to router)
All 4 live robots: SSH OK, fw v1.6.2.2-release.global-02145, 390-403 GB disk free, ~4.4 GB RAM avail,
wifi robotics_5G signal 72-77, booster-daemon/agent/battery/lui/update-engine services running,
no internet. Ping to robots 24-250 ms avg 67-147 ms (wifi hop; jittery). Robot clocks are CST and
not NTP-synced.  Not tested: DDS/SDK discovery across the router, camera streams.

## Pending
- Test all 5 robots on Hello (mobile AP) once the laptop is on Hello and they have joined.
- Static IP for .230; DHCP reservation in router preferred.
- Robot internet (router uplink) -> then firmware / apt / boosteros updates, devdocs + shared Drive.
- Booster Studio: VS Code fork 1.110.1; uses ROS bridge ws :8765 ("tiered" send); camera RGB+depth
  bandwidth work not started.

## Update status (2026-10-27) — nothing flashed/installed yet
- Robots have no internet. `booster-cli upgrade` (OTA via booster-update-engine) fails: "Failed to get ota dylib metadata".
  Needs router uplink working, or laptop forwarding eduroam over the ethernet link.
- Latest firmware: no public source found (web search empty). B2 was on 1.7.0.7 per the older ROBOTS.md. Ask Booster/vendor for the target version before flashing; flash is hard to undo.
- boosteros: not installed on robots (ModuleNotFoundError). PyPI latest = 1.2.2 (requires Python >=3.10; robots have 3.10); older: 1.2.0, 1.1.3, 1.1.1, 1.0.1, 0.1.0. ROBOTS.md says it needs firmware >=1.7 (unverified).
- Booster Studio: installed 1.110.1-1789720186 (VS Code fork), apt candidate identical, no newer version known; updateUrl https://api.booster.tech (probed path 404).

## Robot internet via laptop (2026-10-27) — NOT persistent on the laptop
- Laptop eduroam (wlo1) NATs to the router LAN: `sysctl net.ipv4.ip_forward=1` + nft table `ip k1nat`
  (postrouting masquerade oifname wlo1, saddr 192.168.50.0/24). Lost on reboot. Undo: `sudo nft delete table ip k1nat`.
- Re-create after a laptop reboot (wlo1 = eduroam, enp2s0 = router link):
  `sudo sysctl -w net.ipv4.ip_forward=1; sudo nft add table ip k1nat;`
  `sudo nft 'add chain ip k1nat post { type nat hook postrouting priority srcnat; policy accept; }';`
  `sudo nft add rule ip k1nat post oifname "wlo1" ip saddr 192.168.50.0/24 masquerade`.
  Check from a robot: `ping -c1 8.8.8.8` and `curl -sI https://pypi.org`. Robots need `ip route` default via 192.168.50.250.
- Laptop enp2s0 has fixed 192.168.50.250/24 (profile robotics-lan) as the robots' gateway.
- Robots A2,A3,B1,B3: nmcli profile robotics_5G gateway 192.168.50.250, DNS 8.8.8.8 1.1.1.1, ignore-auto-dns.
  Router DNS (192.168.50.1) returns 10.0.0.1 for every name while it has no uplink, so do not use it.
- Verified: HTTPS to pypi.org returns 200 from all four.
- Firmware OTA endpoint (from booster-update-engine strings): POST https://pubdataprod.boosterobotics.com/ota/api/get_ota_metadata/
  (GET -> 405). Request fields unknown. `booster-cli upgrade` has no dry-run; it installs "latest".
- Target firmware = B2's version (user request). B2 is off, last known 1.7.0.7 (old ROBOTS.md). Read it when B2 is on.

## Firmware incident + finding (2026-10-27 06:00-06:06 CST)
- `booster-cli upgrade` IGNORES all arguments (`-h`, `--help` included) and, with internet, starts the OTA flow:
  it downloads the LATEST firmware and then installs it. It cannot be pinned to a version, and has no dry-run.
- I ran `booster-cli upgrade -h` on B1 twice while checking help text; with internet newly working it began downloading
  **1.8.1.2-release-02884-2026-10-15** (not B2's 1.7.0.7). Killed both processes and the curl children before the
  download finished; deleted the partial .run and download.toml from /opt/booster/Upgrade. B1 still v1.6.2.2, services active.
  A2, A3, B3: nothing downloaded, no upgrade process.
- OTA source: https://boosterobotics.s3.ap-southeast-1.amazonaws.com/ota_single/f8190bd1/<file>.run
  (md5 for 1.8.1.2 = 2f0089969568ed3f46d9e973ac1c8501). Current latest published = 1.8.1.2-release-02884-2026-10-15.
- Target requested: B2's last recorded version 1.7.0.7 (release-00040), unverified (no file found, B2 off).
  Not installable via booster-cli. NEVER run `booster-cli upgrade` to inspect it; use `booster-cli --version`-style reads of
  /opt/booster/version.txt instead. Firmware flashing is on hold pending decision: 1.8.1.2 (latest) vs hunting the 1.7.0.7 .run.

## Fleet dashboard (2026-10-26)

`uv run tools/fleet_dashboard/dashboard.py [--host 0.0.0.0] [--port 8080]` then open http://localhost:8080 (JSON: `/api/status`).
- Online = UDP :9000 heartbeat seen in last 5 s (robots broadcast each second; laptop must be on the robots' L2 net, `enp2s0`).
- Battery % = first status msg (field1=10000, submsg `{1:'battery',3:f32}`) from `ws://<ip>:51111`, polled every 10 s, socket closed right after (it also streams ~1 Mbit/s JPEG frames, never held open).
- A1/B2 listed offline until they broadcast. Default bind is 127.0.0.1; `--host 0.0.0.0` also exposes it on eduroam (wlo1), no auth.
- Tested: A2 64%, A3 36%, B1 30%, B3 22%, B2 offline.

## Camera / Studio streaming test (2026-10-27, A2 on robotics_5G, fw 1.6.2.2)
- `ws://A2:51111` (Studio websocket) carries only 2 message kinds in 8 s: field1=10001 camera JPEG (~124 KiB, ~10 fps, ~10 Mbit/s,
  steady) and field1=10000 status (~0.9 KiB, battery etc.). No depth frames. RGB only.
- Port 8765 (ROS bridge) is NOT listening on the robot (connection refused). Only :51111 listens. The bridge is not a robot-side service.
- On-robot `ros2 topic hz` (humble): `/boostercamera/head/depth` and `/boostercamera/head/rgb` both ~4.3 Hz, jittery (max gap 1.8 s).
  So A2 camera graph DOES publish now; older doc (`k1_hardware_camera_findings.md`, 2026-09-29, "no sensor fitted") is outdated or covered a
  stalled `booster-video-stream` window. Depth is stereo-derived (StereoNetNode), so it only exists inside the ROS graph.
- Depth path = ROS 2 DDS (domain 0) from laptop, not tested over the router yet. Docs say ROS subscription fails: DDS multicast discovery
  across wifi/AP is filtered, and the laptop is Jazzy vs robot Humble (match QoS RELIABLE/VOLATILE; wired + `ip route add 224.0.0.0/4`).
- 10 fps cap not network-bound (10 Mbit/s over 5 GHz wifi is fine): set by robot encoder/Studio stream.
- `.claude/settings.local.json` valid JSON; allow rules added (gh, ssh boosterk1.*, rtk gh/git). They do not override auto-mode classifier.

## Fleet upgrade run (2026-10-27, user-requested, target = latest 1.8.1.2-release-02884)
- Run in tmux session `k1upgrade` (windows a1, b2, a3b1). `booster-cli upgrade` started on A3, B1, B2, A1. B3 and A2 unreachable (no route / timeout) at the time.
- A1 has no `robotics_5G` profile: DHCP on wlP1p1s0, default gateway router .1 (no internet). B2 default gateway also .1 and DNS wrong (pypi.org -> 10.0.0.1).
  Runtime-only fix (lost on reboot): `sudo ip route replace default via 192.168.50.250 dev wlP1p1s0 metric 20600; sudo resolvectl dns wlP1p1s0 8.8.8.8 1.1.1.1`.
  Persistent `nmcli con mod` on B2 was blocked by the permission classifier; not done.
- `~/.ssh/config`: boosterk1.a1 now -> 192.168.50.82.

### Status 2026-10-27 (update)
| Robot | Firmware | Upgrade state |
|---|---|---|
| A2 | 1.8.1.2 (per user, not re-read) | DONE; charging, off |
| A3 | 1.6.2.2 | 1.8.1.2 .run downloaded (3824 MiB); waiting at install prompt |
| B1 | 1.6.2.2 | download in progress (~1.3 GB at last check) |
| B2 | 1.6.2.2 | download in progress, started via laptop route/DNS |
| A1 | 1.6.2.2 | download in progress, slow (~0.3-1 MB/s) |
| B3 | 1.6.2.2 | download started, tmux window `b3` |
- FINDING: `booster-cli upgrade` pauses after download with "Before installing, please make sure robot is firmly supported. Would you like to install it now? (yes/no)".
  With `nohup ... </dev/null` stdin is EOF, so the process exits after download and nothing installs. Install needs an interactive `yes`
  (run `booster-cli upgrade` again in tmux with a TTY; the .run is already cached in /opt/booster/Upgrade). Robot must be physically supported (hanging/on stand) first.
- Do NOT answer yes remotely until the user confirms each robot is supported and has enough battery.

### Status 2026-10-27 (installs)
- A3, B1, B3: installed, `version.txt` reads 1.8.1.2-release-02884-2026-10-15-global (read over SSH after reboot).
- A2: done per user (version not re-read; off, charging).
- B2, A1: download complete (4010058959 B); installs auto-started in tmux `k1upgrade` (windows b2-install, a1-install), yes answered per user approval.
- Install flow that works: `ssh -t host booster-cli upgrade` in tmux, then type `yes` (piping yes via stdin does NOT work). Robots reset the temporary laptop route (metric 600); re-add with metric 100.

---
> **Network note (2026-10-27):** the robotics router (192.168.50.1) has no internet uplink. The laptop shares its eduroam (`wlo1`) internet to the robots by NAT: `enp2s0` = 192.168.50.250/24 is the robots' gateway, with `ip_forward=1` and nft table `ip k1nat` (masquerade out `wlo1`, source 192.168.50.0/24). It is lost on laptop reboot; robots use DNS 8.8.8.8/1.1.1.1 (router DNS is wrong). Re-create steps: `docs/network_robotics_5G_2026-10-26.md` ("Robot internet via laptop").
