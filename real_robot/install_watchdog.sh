#!/usr/bin/env bash
# Install the K1 stereo watchdog as a systemd service on the robot.
# Run as:  bash install_watchdog.sh
set -o pipefail
SVC=/etc/systemd/system/k1-stereo-watchdog.service
SRC="$(cd "$(dirname "$0")" && pwd)/watchdog_k1_stereo.py"
DST=/home/booster/watchdog_k1_stereo.py

echo "[install] copying $SRC -> $DST"
sudo install -m 0755 "$SRC" "$DST"

echo "[install] writing $SVC"
sudo tee "$SVC" > /dev/null <<'UNIT'
[Unit]
Description=K1 head stereo-pair watchdog
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=booster
# ROS_DOMAIN_ID=0 is what BoosterAgent/start.sh uses; the camera topics live there.
Environment=ROS_DOMAIN_ID=0
Environment=PATH=/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/bin/bash -lc 'source /opt/ros/humble/setup.bash && exec python3 /home/booster/watchdog_k1_stereo.py --interval 30'
Restart=always
RestartSec=15
# The watchdog is small, but cap it so it cannot starve the balance controller.
MemoryMax=300M
Nice=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

# The watchdog logs to this file itself (and to journal); create it up front so
# nothing races on first start.
sudo touch /tmp/watchdog_k1_stereo.log
sudo chmod 0666 /tmp/watchdog_k1_stereo.log
sudo systemctl daemon-reload
sudo systemctl enable k1-stereo-watchdog.service
sudo systemctl restart k1-stereo-watchdog.service
sleep 12
echo "[install] status:"
sudo systemctl status k1-stereo-watchdog.service --no-pager | head -12
echo
echo "[install] log so far:"
sudo tail -14 /tmp/watchdog_k1_stereo.log 2>/dev/null
echo "=== INSTALL_DONE ==="
