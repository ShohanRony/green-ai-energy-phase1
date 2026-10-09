#!/bin/bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat << SYSTEMD | sudo tee /etc/systemd/system/p8-auto.service > /dev/null
[Unit]
Description=Green AI Phase 1 Audit P8 Auto Runner
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${DIR}
ExecStart=/bin/bash ${DIR}/scripts/auto_p8.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:${DIR}/p8_auto.log
StandardError=append:${DIR}/p8_auto.log

[Install]
WantedBy=multi-user.target
SYSTEMD

sudo systemctl daemon-reload
sudo systemctl enable p8-auto.service
sudo systemctl start p8-auto.service

echo "p8-auto.service installed and started!"
echo "Check logs with: tail -f ${DIR}/p8_auto.log"
