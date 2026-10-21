#!/usr/bin/env bash
# volume-watch.sh — ensure "New Volume" (Windows NTFS partition) is mounted.
# Root-side helper invoked by systemd: ollama-volume.service / ollama-volume.timer.
# Logic: mounted? ok. Not mounted? clear the Windows dirty-bit (ntfsfix), then
# mount via fstab entry. If rw refuses due to hibernation/fat-startup, ntfsfix
# -d discards the hibernation volume flag and a follow-up rw mount succeeds.
set -u
VOL="/media/hazemhagrass/New Volume"
DEV="/dev/disk/by-uuid/1050D7C550D7AFA8"

mounted() { findmnt -n -o TARGET "$VOL" >/dev/null 2>&1; }

if mounted; then
  echo "$(date '+%F %T') volume ok (mounted)"
  exit 0
fi

[ -e "$DEV" ] || { echo "$(date '+%F %T') volume device absent (slept/safely removed?)"; exit 0; }

echo "$(date '+%F %T') volume NOT mounted — repairing"
mkdir -p "$VOL" 2>/dev/null || true
ntfsfix -d /dev/nvme0n1p6 2>&1 | tail -1 || echo "(ntfsfix reported; continuing)"
if findmnt -n -o TARGET "$VOL" >/dev/null 2>&1; then
  echo "$(date '+%F %T') volume mounted after ntfsfix"
  exit 0
fi
mount "UUID=1050D7C550D7AFA8" "$VOL" 2>&1 && {
  echo "$(date '+%F %T') volume mounted: $(findmnt -no FSTYPE,OPTIONS "$VOL" | head -1)"
} || {
  echo "$(date '+%F %T') rw mount failed; trying ro fallback"
  mount -o ro "UUID=1050D7C550D7AFA8" "$VOL" 2>&1 \
    && echo "$(date '+%F %T') mounted RO — Windows hibernation likely; reboot to Windows fully then back to fix" \
    || echo "$(date '+%F %T') mount FAILED"
}
