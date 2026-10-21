#!/usr/bin/env bash
# ollama-update.sh — bring the Ollama binary on THIS host to the latest
# official GitHub release. Idempotent: no-ops when already current.
# Installed fleet-wide as /usr/local/sbin/ollama-update (a one-line root
# wrapper) so the cron sync can upgrade without a password prompt.
set -uo pipefail

if [ "$(id -u)" -ne 0 ]; then
  if [ -x /usr/local/sbin/ollama-update ] && command -v sudo >/dev/null 2>&1 && sudo -n /usr/local/sbin/ollama-update --probe 2>/dev/null; then
    exec sudo -n /usr/local/sbin/ollama-update "${@/--probe/}"
  fi
  echo "ollama-update: must be run as root via /usr/local/sbin/ollama-update" >&2
  exit 1
fi
[ "${1:-}" = "--probe" ] && exit 0

case "$(uname -m)" in
  x86_64 | amd64) ASSET=ollama-linux-amd64.tar.zst ;;
  aarch64 | arm64) ASSET=ollama-linux-arm64.tar.zst ;;
  *) echo "ollama-update: unsupported arch $(uname -m)"; exit 0 ;;
esac

TAG="$(curl -fsSL --max-time 20 \
  https://api.github.com/repos/ollama/ollama/releases/latest 2>/dev/null \
  | grep -oE '"tag_name": *"[^"]+"' | head -1 | grep -oE 'v[0-9][^"]*' || true)"
if [ -z "$TAG" ]; then
  echo "ollama-update: cannot reach GitHub (offline?) — skipping"
  exit 0
fi
LATEST="${TAG#v}"

# Read the CLIENT BINARY version, not the server's. `ollama --version` prints
# the running server's version first when one is reachable; if the binary was
# swapped but the service has not restarted, the server still reports the OLD
# version and the script would re-download the same 1.4 GB on every sync.
INST="$(command -v ollama >/dev/null 2>&1 && OLLAMA_HOST=127.0.0.1:1 ollama --version 2>/dev/null \
  | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
if [ "$INST" = "$LATEST" ]; then
  echo "ollama-update: already latest ($INST)"
  exit 0
fi

echo "ollama-update: ${INST:-none} -> $LATEST ($ASSET)"
# Spool the artifact on a REAL filesystem (not /tmp — it is often a small
# tmpfs that cannot hold the 1.4 GB download plus a ~3.2 GB decompressed
# copy). Kept across runs so a dropped download resumes instead of restarting.
SPOOL=/var/cache/ollama-update
mkdir -p "$SPOOL"
# Pre-flight: never download 1.4 GB we cannot install. If under /usr/local the
# free space is below the extract threshold (~3.3 GB) or the spool filesystem
# cannot hold the artifact, bail out BEFORE downloading and leave the spool
# clean — a full-disk host must not be made fuller by an orphaned download.
DEST_FREE_KB=$(df -Pk /usr/local | awk 'NR==2{print $4}')
SPOOL_FREE_KB=$(df -Pk "$SPOOL"        | awk 'NR==2{print $4}')
# Need room for BOTH the ~1.4 GB artifact AND the ~3.2 GB extracted tree on the
# same filesystem (spool and /usr/local both live on /). 5 GB before download
# keeps a near-full host from being pushed to 0 free mid-extract.
if [ "${DEST_FREE_KB:-0}" -lt 5200000 ]; then
  rm -f "$SPOOL/o.tzst"
  echo "ollama-update: not enough free space under /usr/local (${DEST_FREE_KB:-0} KB) — keeping current install"
  exit 0
fi
if [ "${SPOOL_FREE_KB:-0}" -lt 1600000 ]; then
  rm -f "$SPOOL/o.tzst"
  echo "ollama-update: not enough free space on spool ($SPOOL) — keeping current install"
  exit 0
fi
if ! curl -fL --retry 5 --retry-delay 5 --retry-all-errors -C - \
     --connect-timeout 20 --max-time 3600 -o "$SPOOL/o.tzst" \
     "https://github.com/ollama/ollama/releases/download/$TAG/$ASSET"; then
  echo "ollama-update: download failed — keeping current install (partial kept for resume)"
  exit 0
fi
# validate we actually got a full zstd frame before touching the install
if ! zstd -t "$SPOOL/o.tzst" >/dev/null 2>&1; then
  rm -f "$SPOOL/o.tzst"
  echo "ollama-update: download incomplete/corrupt — keeping current install"
  exit 0
fi
# re-check: the download itself consumed spool space; enough room now for the
# extracted tree under /usr/local? (~3.3 GB). Clean up if not — do not orphan.
FREE_KB=$(df -Pk /usr/local | awk 'NR==2{print $4}')
if [ "${FREE_KB:-0}" -lt 3500000 ]; then
  rm -f "$SPOOL/o.tzst"
  echo "ollama-update: not enough free space under /usr/local — keeping current install"
  exit 0
fi
# stream-decompress straight into /usr/local (never a second full copy on disk)
# Ollama must be stopped first: extracting over its running, mmapped binary
# fails with "text file busy".
SYSTEMD_OLLAMA=0
if systemctl list-unit-files 2>/dev/null | grep -q '^ollama\.service'; then
  SYSTEMD_OLLAMA=1
  systemctl stop ollama || true
  # wait for the unit AND the process to actually go away — `stop` returns as
  # soon as the job is queued, and a lingering serve process keeps the old
  # binary mapped, so the tar swap would leave a "(deleted)" inode serving.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    systemctl is-active --quiet ollama || break
    sleep 1
  done
  systemctl kill --signal=SIGKILL ollama 2>/dev/null || true
  sleep 1
fi
if ! zstd -dc "$SPOOL/o.tzst" | tar -xf - -C /usr/local; then
  [ "$SYSTEMD_OLLAMA" = 1 ] && { sleep 1; systemctl start ollama || true; }
  echo "ollama-update: extract failed — keeping current install (artifact kept at $SPOOL/o.tzst)"
  exit 0
fi
rm -f "$SPOOL/o.tzst"
chmod 755 /usr/local/bin/ollama 2>/dev/null || true
if [ "$SYSTEMD_OLLAMA" = 1 ]; then
  # `start` is a no-op if the unit was never observably stopped, and the fresh
  # binary would then stay unmapped — restart to guarantee the swap is live.
  systemctl restart ollama 2>/dev/null || systemctl start ollama || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    systemctl is-active --quiet ollama && break
    sleep 1
  done
fi
sleep 1
echo "ollama-update: now $(OLLAMA_HOST=127.0.0.1:1 ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1) (server: $(curl -s --max-time 5 http://127.0.0.1:11435/api/version 2>/dev/null || echo '?'))"
