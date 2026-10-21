#!/usr/bin/env bash
# fleet-agent-bootstrap.sh — make ONE host identical to the fleet standard.
# Idempotent: safe to run every sync. Invoked on each host by fleet-sync.sh,
# but can be run by hand:   bash fleet-agent-bootstrap.sh [SRC_DIR]
#
# Does: OpenJev repo + venv + systemd user unit + linger + passwordless
#       Ollama updater. Installing the Ollama *binary* is left to
#       fleet-sync.sh so both hosts upgrade in the same run.
set -euo pipefail

SRC_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
DEPLOY_DIR="$HOME/workspace/openjev-ollama"
UNIT="$HOME/.config/systemd/user/openjev-ollama.service"
say() { printf '  [%s] %s\n' "$(hostname)" "$*"; }

# --- pick a python that can build venvs (prefer 3.12, matches upstream) ---
PY=""
for p in python3.12 python3; do
  if command -v "$p" >/dev/null 2>&1; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "FATAL: no python3 found" >&2; exit 1; }
say "python: $($PY --version 2>&1) ($PY)"

# --- OpenJev repo dir ---
mkdir -p "$DEPLOY_DIR"
say "deploy dir: $DEPLOY_DIR"

# --- venv + deps (rebuild venv only if missing/broken) ---
VENV="$DEPLOY_DIR/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  say "creating venv"
  rm -rf "$VENV"
  "$PY" -m venv "$VENV"
fi
say "installing python deps"
"$VENV/bin/python" -m pip install --no-input --quiet --upgrade pip setuptools wheel || true
"$VENV/bin/python" -m pip install --no-input --quiet \
  fastapi httpx typesafe-sdk "uvicorn[standard]" \
  || { echo "FATAL: pip install failed" >&2; exit 1; }

# --- config: seed .env from template once, never overwrite a live one ---
if [ ! -f "$DEPLOY_DIR/.env" ] && [ -f "$DEPLOY_DIR/.env.example" ]; then
  cp "$DEPLOY_DIR/.env.example" "$DEPLOY_DIR/.env"
  say "seeded .env from .env.example"
fi
[ -x "$DEPLOY_DIR/run.sh" ] || chmod +x "$DEPLOY_DIR/run.sh" 2>/dev/null || true
[ -x "$DEPLOY_DIR/jev" ] || chmod +x "$DEPLOY_DIR/jev" 2>/dev/null || true

# --- systemd *user* unit + linger (survives logout, starts at boot) ---
if [ -f "$DEPLOY_DIR/openjev-ollama.service" ]; then
  mkdir -p "$(dirname "$UNIT")"
  cp "$DEPLOY_DIR/openjev-ollama.service" "$UNIT"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  systemctl --user daemon-reload
  systemctl --user enable --now openjev-ollama
  loginctl enable-linger "$USER" 2>/dev/null || true
  say "service: $(systemctl --user is-active openjev-ollama) / $(systemctl --user is-enabled openjev-ollama)"
else
  say "WARN: unit file missing in deploy dir"
fi

# --- privileged install: passwordless updater + OpenJev front on :11434 ---
# Uses an askpass helper when present (works with no TTY, e.g. under cron);
# otherwise pipes the fleet password from the 600 file. Idempotent.
have_sudo() { sudo -n true 2>/dev/null; }

run_root() { # run_root <script-path>
  local s="$1"
  if have_sudo; then
    sudo -n bash "$s"
  elif [ -n "${SUDO_ASKPASS:-}" ] && [ -x "${SUDO_ASKPASS:-}" ]; then
    sudo -A -p '' bash "$s"
  elif [ -f "$HOME/.hermes/fleet_pass" ]; then
    sudo -S -p '' bash "$s" <"$HOME/.hermes/fleet_pass"
  else
    return 3
  fi
}

if command -v sudo >/dev/null 2>&1; then
  WRAP=/usr/local/sbin/ollama-update
  FRONTWRAP=/usr/local/sbin/ollama-front-install
  UPD="$SRC_DIR/ollama-update.sh"
  FRONT="$SRC_DIR/ollama-front-install.sh"

  # stage a root script that installs both wrappers + one scoped sudoers rule
  STAGE="$(mktemp)"
  {
    echo '#!/bin/bash'
    echo 'set -e'
    if [ -f "$UPD" ]; then
      printf 'printf "#!/bin/bash\\nexec bash \\"%s\\" \\"\\$@\\"\\n" > %s; chmod 755 %s\n' "$UPD" "$WRAP" "$WRAP"
    fi
    if [ -f "$FRONT" ]; then
      printf 'printf "#!/bin/bash\\nexec bash \\"%s\\" \\"%s\\" \\"\\$@\\"\\n" > %s; chmod 755 %s\n' "$FRONT" "$DEPLOY_DIR" "$FRONTWRAP" "$FRONTWRAP"
    fi
    echo 'cat > /etc/sudoers.d/ollama-fleet <<EOF'
    [ -f "$UPD" ]   && echo "$USER ALL=(root) NOPASSWD: $WRAP"
    [ -f "$FRONT" ] && echo "$USER ALL=(root) NOPASSWD: $FRONTWRAP"
    echo 'EOF'
    echo 'chmod 440 /etc/sudoers.d/ollama-fleet'
    echo 'visudo -cf /etc/sudoers.d/ollama-fleet >/dev/null'
  } >"$STAGE"

  if run_root "$STAGE"; then
    say "privileged wrappers + sudoers installed"
  else
    say "WARN: no sudo path available (fingerprint-only?); privileged steps skipped"
  fi
  rm -f "$STAGE"

  # apply the front now (moves Ollama to :11435, opens :11434) — needs root
  if [ -f "$FRONT" ]; then
    if have_sudo || [ -n "${SUDO_ASKPASS:-}" ] || [ -f "$HOME/.hermes/fleet_pass" ]; then
      if have_sudo; then sudo -n "$FRONTWRAP"
      elif [ -n "${SUDO_ASKPASS:-}" ]; then sudo -A -p '' "$FRONTWRAP"
      else sudo -S -p '' "$FRONTWRAP" <"$HOME/.hermes/fleet_pass"; fi \
        && say "front on :11434" || say "WARN: front install failed"
    fi
  fi
fi

say "bootstrap done"
