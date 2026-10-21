#!/usr/bin/env bash
# fleet-sync.sh — converge the Ollama twins to one standard, from any one host.
#   * OpenJev-on-Ollama deployed identically (repo, venv, systemd user unit, linger)
#   * canonical model set pulled on every host
#   * Ollama binary upgraded to the latest release (when JEV_UPDATE_OLLAMA=1)
# Safe to run repeatedly; every step is idempotent. This is what the cron job calls.
#
#   bash fleet-sync.sh            # sync the fleet
#   DRY=1 bash fleet-sync.sh      # show plan, change nothing
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$HERE/fleet.env" ] && . "$HERE/fleet.env"

FLEET_HOSTS="${FLEET_HOSTS:-}"
FLEET_USER="${FLEET_USER:-$USER}"
SSH_KEY="${FLEET_SSH_KEY:-$HOME/.ssh/id_fleet}"
UPDATE_OLLAMA="${JEV_UPDATE_OLLAMA:-1}"
MODELS_FILE="$HERE/models.txt"
DRY="${DRY:-0}"
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new
          -o ConnectTimeout=10)
REMOTE_DIR="workspace/openjev-ollama"
LOG="$HOME/.hermes/fleet-sync.log"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG" >&2; }
[ "${1:-}" = "--quiet" ] || log "=== fleet-sync start (host $(hostname)) ==="

LOCAL_IPS="$(hostname -I 2>/dev/null)"
is_local() { for ip in $LOCAL_IPS; do [ "$1" = "$ip" ] && return 0; done; return 1; }

# rsync payload: code + units + fleet scripts, never the venv or live config
RSYNC_EXCL=(--exclude .venv/ --exclude .env --exclude server.log
            --exclude '*.pyc' --exclude __pycache__/ --exclude fleet/logs/)

sync_host() {
  local h="$1"
  if is_local "$h"; then
    log "[$h] local host (this machine)"
    if [ "$DRY" = "1" ]; then return 0; fi
    ( cd "$REPO" && SUDO_ASKPASS="$HOME/.hermes/askpass.sh" bash "$HERE/fleet-agent-bootstrap.sh" "$HERE" ) 2>&1 | sed "s/^/  [$h] /" | tee -a "$LOG" >&2
  else
    log "[$h] remote: rsync code -> bootstrap"
    if [ "$DRY" = "1" ]; then return 0; fi
    ssh "${SSH_OPTS[@]}" "$FLEET_USER@$h" "mkdir -p ~/$REMOTE_DIR ~/.hermes && chmod 700 ~/.hermes" 2>&1 | tee -a "$LOG" >&2 || { log "[$h] ERROR: ssh unreachable"; return 1; }
    if ! rsync -az --delete "${RSYNC_EXCL[@]}" \
         -e "ssh ${SSH_OPTS[*]}" "$REPO/" "$FLEET_USER@$h:$REMOTE_DIR/"; then
      log "[$h] ERROR: rsync failed"; return 1
    fi
    # deliver the privilege helpers so the remote bootstrap can run root steps
    # unattended (password file 600 + askpass helper), never via the transcript.
    if [ -f "$HOME/.hermes/fleet_pass" ]; then
      scp -q "${SSH_OPTS[@]}" "$HOME/.hermes/fleet_pass" "$FLEET_USER@$h:.hermes/fleet_pass" \
        && ssh "${SSH_OPTS[@]}" "$FLEET_USER@$h" "chmod 600 ~/.hermes/fleet_pass" 2>/dev/null || true
    fi
    if [ -f "$HOME/.hermes/askpass.sh" ]; then
      scp -q "${SSH_OPTS[@]}" "$HOME/.hermes/askpass.sh" "$FLEET_USER@$h:.hermes/askpass.sh" \
        && ssh "${SSH_OPTS[@]}" "$FLEET_USER@$h" "chmod 700 ~/.hermes/askpass.sh" 2>/dev/null || true
    fi
    ssh "${SSH_OPTS[@]}" "$FLEET_USER@$h" \
      "SUDO_ASKPASS=\$HOME/.hermes/askpass.sh bash ~/$REMOTE_DIR/fleet/fleet-agent-bootstrap.sh ~/$REMOTE_DIR/fleet" 2>&1 \
      | sed "s/^/  [$h] /" | tee -a "$LOG" >&2
  fi
}

converge_models() {
  local h="$1" want have line
  _sh() { # run a shell snippet either locally or over ssh
    if is_local "$h"; then bash -c "$1"; else ssh "${SSH_OPTS[@]}" "$FLEET_USER@$h" "$1"; fi
  }
  have="$(_sh 'curl -s --max-time 10 http://127.0.0.1:11434/api/tags | grep -oE "\"name\"[[:space:]]*:[[:space:]]*\"[^\"]+\"" | sed "s/.*\"\([^\"]*\)\"$/\1/"')"
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(printf '%s' "$line" | tr -d '[:space:]')"
    [ -z "$line" ] && continue
    if printf '%s\n' "$have" | grep -qxF "$line"; then
      continue
    fi
    log "[$h] pulling missing model: $line"
    [ "$DRY" = "1" ] && continue
    _sh "ollama pull '$line'" 2>&1 | sed "s/^/  [$h] /" | tee -a "$LOG" >&2 \
      || log "[$h] WARN: pull failed for $line"
  done < "$MODELS_FILE"
}

update_ollama() {
  local h="$1"
  [ "$UPDATE_OLLAMA" = "1" ] || return 0
  local out
  if is_local "$h"; then
    out="$(sudo -n /usr/local/sbin/ollama-update 2>&1)"
  else
    out="$(ssh "${SSH_OPTS[@]}" "$FLEET_USER@$h" 'sudo -n /usr/local/sbin/ollama-update' 2>&1)"
  fi
  log "[$h] ollama-update: $(printf '%s' "$out" | tail -1)"
}

rc=0
for h in $FLEET_HOSTS; do
  sync_host "$h" || rc=1
done
for h in $FLEET_HOSTS; do
  converge_models "$h"
done
for h in $FLEET_HOSTS; do
  update_ollama "$h"
done

[ "${1:-}" = "--quiet" ] || {
  log "=== fleet-sync done (rc=$rc) — model counts ==="
  for h in $FLEET_HOSTS; do
    n=$( { is_local "$h" && curl -s --max-time 5 http://127.0.0.1:11434/api/tags; } \
         || ssh "${SSH_OPTS[@]}" "$FLEET_USER@$h" 'curl -s --max-time 5 http://127.0.0.1:11434/api/tags' 2>/dev/null )
    c=$(printf '%s' "$n" | grep -oE '"name":' | wc -l)
    log "[$h] models present: $c"
  done
}
exit $rc
