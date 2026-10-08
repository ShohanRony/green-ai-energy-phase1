#!/usr/bin/env bash
# One-time installer for the unattended Stage 4b sessions. Run with: sudo bash scripts/install_auto_sessions.sh
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo"; exit 1; }
U=shohan
REPO=/home/$U/green-ai-research/green-ai-energy-phase1
STATE=/home/$U/.stage4b_auto

# 1. Narrow passwordless rule: reboot and enable/disable this one service.
cat > /tmp/stage4b-auto.sudoers <<EOF
$U ALL=(root) NOPASSWD: /usr/bin/systemctl reboot, /usr/bin/systemctl disable stage4b-auto.service, /usr/bin/systemctl enable stage4b-auto.service
EOF
visudo -cf /tmp/stage4b-auto.sudoers
install -m 0440 /tmp/stage4b-auto.sudoers /etc/sudoers.d/stage4b-auto

# 2. Queue: session + not-before (UTC). S5/S6 gated so sessions span 3 days in both UTC and local time.
mkdir -p "$STATE"
if [[ ! -s "$STATE/queue" ]]; then
cat > "$STATE/queue" <<EOF
4 0
5 $(date -u -d 2026-10-08T00:00:00Z +%s)
6 $(date -u -d 2026-10-09T00:00:00Z +%s)
EOF
fi
rm -f "$STATE/HALTED"
chown -R $U:$U "$STATE"

# 3. Boot service (runs as $U, no login needed).
cat > /etc/systemd/system/stage4b-auto.service <<EOF
[Unit]
Description=Stage 4b unattended measurement sessions
After=network-online.target nvidia-persistenced.service multi-user.target
Wants=network-online.target

[Service]
Type=simple
User=$U
Environment=HOME=/home/$U
WorkingDirectory=$REPO
ExecStart=/bin/bash $REPO/scripts/auto_sessions.sh
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable stage4b-auto.service
echo "Installed. Queue:"; cat "$STATE/queue"
echo "Reboot now to start Session 3:  sudo systemctl reboot"
echo "Emergency stop (any time):     sudo systemctl disable --now stage4b-auto.service"
