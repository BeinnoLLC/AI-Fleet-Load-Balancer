#!/bin/bash
# ollama-front-install.sh — install the OpenJev front so it owns the Ollama
# port (:11434) and real Ollama moves just behind it (:11435).
#
# Runs as ROOT (installed passwordless via /usr/local/sbin/ollama-front-install).
# Idempotent: safe to re-run every fleet sync.
set -euo pipefail

DEPLOY="${1:-${JEV_DEPLOY_DIR:-$HOME/workspace/openjev-ollama}}"
VENV="$DEPLOY/.venv/bin/python"
FRONT="$DEPLOY/ollama_front.py"
JEVMODEL="${JEV_OLLAMA_MODEL:-qwen2.5:7b}"

# --- GPU-aware fleet routing -------------------------------------------- #
# The front reads $DEPLOY/fleet_hosts.json for every host on the LAN and
# polls each one's /gpu/status; a brand-new machine becomes routable the
# moment it's added to that file — no redeploy needed on this host.
# Routing is ON by default: it's permanent fleet infrastructure, not an
# opt-in flag. (Note: sudo's NOPASSWD entry for this script does not
# forward arbitrary -E env vars, so these defaults are what actually land
# in the unit — override by editing this file, not via `sudo VAR=x`.)
ENABLE_GPU_ROUTE="${ENABLE_GPU_ROUTE:-1}"
FLEET_SELF_NAME="${FLEET_SELF_NAME:-}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-5}"
GPU_STALE_AFTER="${GPU_STALE_AFTER:-30}"

[ -x "$VENV" ] || { echo "front-install: venv python missing at $VENV"; exit 1; }
[ -f "$FRONT" ] || { echo "front-install: $FRONT missing"; exit 1; }

# 1) move real Ollama to 127.0.0.1:11435 (its other settings stay in override.conf)
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/zz-port-11435.conf <<'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11435"
EOF

# 2) the front system service owns :11434 for every client
cat > /etc/systemd/system/ollama-openjev-front.service <<EOF
[Unit]
Description=OpenJev System One behind Ollama (:11434 front-end)
After=network-online.target ollama.service
Wants=network-online.target ollama.service

[Service]
Type=simple
User=$(stat -c %U "$DEPLOY")
Group=$(stat -c %G "$DEPLOY")
WorkingDirectory=$DEPLOY
Environment=OLLAMA_UPSTREAM=http://127.0.0.1:11435
Environment=JEV_OLLAMA_TAG=openjev:latest
Environment=JEV_OLLAMA_MODEL=$JEVMODEL
Environment=JEV_OLLAMA_CONCURRENCY=4
Environment=JEV_HOST=0.0.0.0
Environment=JEV_PORT=11434
Environment=ENABLE_GPU_ROUTE=$ENABLE_GPU_ROUTE
Environment=FLEET_SELF_NAME=$FLEET_SELF_NAME
Environment=GPU_POLL_INTERVAL=$GPU_POLL_INTERVAL
Environment=GPU_STALE_AFTER=$GPU_STALE_AFTER
Environment=JEV_TLS_PORT=${JEV_TLS_PORT:-}
Environment=JEV_TLS_CERT=${JEV_TLS_CERT:-}
Environment=JEV_TLS_KEY=${JEV_TLS_KEY:-}
ExecStart=$VENV $FRONT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl restart ollama
sleep 3
systemctl enable ollama-openjev-front >/dev/null 2>&1 || true
systemctl restart ollama-openjev-front

# 3) prove it: health on :11434 and the tag visible through the front
for _ in $(seq 1 25); do
  curl -sf --max-time 2 http://127.0.0.1:11434/health >/dev/null 2>&1 && break
  sleep 1
done
if curl -sf --max-time 3 http://127.0.0.1:11434/health >/dev/null 2>&1; then
  echo "front-install: OK ($(curl -s --max-time 3 http://127.0.0.1:11434/api/tags | grep -c openjev) openjev tag(s) on :11434)"
else
  echo "front-install: FAILED — front not answering on :11434" >&2
  systemctl --no-pager status ollama-openjev-front | tail -5 >&2 || true
  exit 1
fi
