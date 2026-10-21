#!/usr/bin/env bash
# fleet-cron.sh — cron entry point. Runs the fleet sync quietly and prints
# output ONLY when something changed or went wrong (watchdog pattern: an
# empty stdout means "all already current", so the cron job stays silent).
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
[ -f "$HERE/fleet.env" ] && . "$HERE/fleet.env"
HOSTS="${FLEET_HOSTS:-}"

out="$(bash "$HERE/fleet-sync.sh" --quiet 2>&1)"; rc=$?
actionable="$(printf '%s\n' "$out" \
  | grep -E 'pulling missing model|ollama-update: [0-9]|front-install:|front on :11434|ERROR|WARN|FATAL' || true)"

if [ -n "$actionable" ] || [ "$rc" -ne 0 ]; then
  echo "Ollama fleet sync — $(date '+%Y-%m-%d %H:%M')  (hosts: $HOSTS)"
  printf '%s\n' "$out"
  [ "$rc" -ne 0 ] && echo "(exit code $rc)"
fi
exit $rc
