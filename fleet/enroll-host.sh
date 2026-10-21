#!/usr/bin/env bash
# enroll-host.sh — enroll a machine into the AI Fleet Load Balancer.
#
# Modes:
#   LAN host (has key SSH access): registered mode=pull — the hub polls
#     its /gpu/status like every other LAN member.
#   Out-of-LAN host (hub cannot reach it): registered mode=push — it runs
#     fleet-agent.py which POSTs samples to the hub's /gpu/ingest with a
#     per-host token.
#
# Usage:
#   enroll-host.sh --name hermes2 --hub http://192.168.1.11:11434 \
#       [--host 10.8.0.5|my.server.example] [--user hazemhagrass] \
#       [--password '…' | --key ~/.ssh/id_rsa] [--push | --pull]
#       [--agent-dir /opt/fleet-agent] [--systemd]
#
# LAN discovery shortcut: --host with key access and no --push → pull mode.
#
# What it does:
#   1. probes SSH reachability (BatchMode first; --password installs a key via sshpass when available)
#   2. for push mode: installs fleet-agent.py + fleet-agent.conf, optionally a systemd unit
#   3. issues a per-host token, writes the host into fleet_hosts.json (JQ-free safe JSON edit via python3)
#   4. prints the hub command to re-check: curl <hub>/gpu/fleet
set -euo pipefail

NAME="" HUB="" HOST="" USER_AT="${SUDO_USER:-$USER}" PASSWORD="" KEY=""
MODE="" AGENT_DIR="/opt/fleet-agent" SYSTEMD=0
REG="${FLEET_REGISTRY:-$HOME/workspace/openjev-ollama/fleet_hosts.json}"
HERE="$(cd "$(dirname "$0")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --hub) HUB="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --user) USER_AT="$2"; shift 2;;
    --password) PASSWORD="$2"; shift 2;;
    --key) KEY="$2"; shift 2;;
    --push) MODE="push"; shift;;
    --pull) MODE="pull"; shift;;
    --agent-dir) AGENT_DIR="$2"; shift 2;;
    --systemd) SYSTEMD=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[ -n "$NAME" ] || { echo "--name required" >&2; exit 1; }
[ -n "$HUB" ]  || HUB="http://127.0.0.1:11434"
[ -f "$REG" ]  || REG="$HOME/workspace/openjev-ollama/fleet_hosts.json"
[ -n "$MODE" ] || MODE="pull"   # default: LAN pull

TOKEN="hft-$(head -c 24 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c 32)"

say() { printf '[enroll] %s\n' "$*"; }

ssh_to() {  # ssh_to <cmd...>
  if [ -n "$PASSWORD" ]; then
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password,keyboard-interactive "$USER_AT@$HOST" "$@"
  elif [ -n "$KEY" ]; then
    ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER_AT@$HOST" "$@"
  else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$USER_AT@$HOST" "$@"
  fi
}

SCP_TO() {
  if [ -n "$PASSWORD" ]; then
    sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password,keyboard-interactive "$@"
  elif [ -n "$KEY" ]; then
    scp -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@"
  else
    scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$@"
  fi
}

# ---------- 1. SSH reachability ----------
if [ -n "$HOST" ]; then
  say "probing ssh $USER_AT@$HOST"
  if ! ssh_to 'echo SSH_OK' 2>/dev/null; then
    if [ -n "$PASSWORD" ] && command -v sshpass >/dev/null 2>&1; then
      say "key auth failed; trying password (sshpass) then installing a key"
      # install key: append pubkey so future logins are key-based
      PUB="${HOME}/.ssh/id_ed25519.pub"
      [ -f "$PUB" ] || ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
      sshpass -p "$PASSWORD" ssh-copy-id -i "$PUB" -o PreferredAuthentications=password,keyboard-interactive -o StrictHostKeyChecking=accept-new "$USER_AT@$HOST" >/dev/null
      ssh_to 'echo SSH_OK' >/dev/null || { echo "FATAL: password auth did not yield key login" >&2; exit 1; }
      say "key installed; password no longer needed"
    else
      echo "FATAL: cannot reach $HOST via ssh (no key, no sshpass+password)" >&2
      exit 1
    fi
  fi
fi

# ---------- 2. guest install: push agent ----------
if [ "$MODE" = "push" ]; then
  say "installing push agent on $HOST:$AGENT_DIR"
  ssh_to "sudo -n mkdir -p '$AGENT_DIR' 2>/dev/null || mkdir -p '$AGENT_DIR'" || true
  SCP_TO "$HERE/../fleet-agent.py" "$USER_AT@$HOST:/tmp/fleet-agent.py"
  SCP_TO /dev/null "$USER_AT@$HOST:/tmp/" >/dev/null 2>&1 || true
  ssh_to "mv /tmp/fleet-agent.py '$AGENT_DIR/fleet-agent.py' && chmod +x '$AGENT_DIR/fleet-agent.py'"
  CONF="$AGENT_DIR/fleet-agent.conf"
  ssh_to "cat > '$CONF' <<EOF
FLEET_HUB_URL=$HUB
FLEET_HOST_NAME=$NAME
FLEET_TOKEN=$TOKEN
PUSH_INTERVAL=5
EOF"
  ssh_to "chmod 600 '$CONF'"
  if [ "$SYSTEMD" = "1" ]; then
    ssh_to "sudo -n tee /etc/systemd/system/fleet-agent.service >/dev/null <<'EOF2'
[Unit]
Description=AI Fleet Load Balancer push agent
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/env python3 $AGENT_DIR/fleet-agent.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF2
sudo -n systemctl daemon-reload && sudo -n systemctl enable --now fleet-agent.service" \
      || say "WARN: systemd unit install failed — agent may need manual start"
  fi
  say "push agent installed (config $CONF)"
fi

# ---------- 3. registry update (python3 for safe JSON, jq-free) ----------
python3 - "$REG" "$NAME" "$HUB" "$MODE" "$TOKEN" <<'PYEOF'
import json, sys, os
reg, name, hub, mode, token = sys.argv[1:6]
with open(reg) as f:
    hosts = json.load(f)
entry = next((h for h in hosts if h.get("name") == name), None)
if entry:
    entry.update({"url": hub, "mode": mode, "token": token})
else:
    hosts.append({"name": name, "url": hub, "mode": mode, "token": token})
tmp = reg + ".tmp"
with open(tmp, "w") as f:
    json.dump(hosts, f, indent=2)
os.replace(tmp, reg)
print("registry updated:", name, "->", mode)
PYEOF

# ---------- 4. where the token lives on the hub ----------
say "token: $TOKEN"
say "verify: curl -s $HUB/gpu/fleet | python3 -m json.tool | grep -A3 '$NAME'"
say "note: hub picks up fleet_hosts.json on next poll cycle (hot-reloaded)"
if [ "$MODE" = "push" ]; then
  say "out-of-LAN transport: agent pushes outbound — works behind NAT; put the hub behind HTTPS for production"
fi
