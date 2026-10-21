#!/usr/bin/env bash
# ollama-volume.sh — Ollama <-> NTFS shuttle-volume bridge ("the magic").
#
# Purpose: move Ollama models from this laptop (.11, "home") to the other desk
# (.12) using the internal Windows NTFS partition "New Volume" as the carrier
# folder. Also automates: always-mounted volume + Windows dirty-bit lock repair
# (ntfsfix) so it stays mountable on this laptop.
#
# Volume layout (post 2026-10-09 rename → "llm-models"):
#   /media/hazemhagrass/New Volume/llm-models/            LIVE store on .11 (OLLAMA_MODELS)
#   /media/hazemhagrass/New Volume/llm-models/.12-shuttle/
#     ollama-store/store/blobs/sha256-<digest>            (content blobs)
#     ollama-store/store/manifests/registry.ollama.ai/library/<name>/<tag>
#     ollama-store/store/index.json                       (inventory)
#
# Commands:
#   status                     volume, shuttle folder, local store overview
#   export MODEL[:TAG]...      stage model(s) onto the volume (idempotent)
#   export --all               stage every installed model
#   list                       staged-on-volume vs installed-here
#   install MODEL... --host H  stream staged blobs into host H's store
#                              (H=127.0.0.1 this laptop, 192.168.1.12 desk).
#                              Root ops go via ssh + sudo -A askpass.
#   repair-mount               (root) ntfsfix + mount the volume if missing
#   watch                      (root) ensure-mounted check for the timer
#
# Notes:
#  - Store blobs are content-addressed and immutable: copy once by size, then
#    verify sha256 (the ollama digest IS the sha256 of the file).
#  - openjev:latest is a facade served by ollama_front on each host; its store
#    manifest has no real layers, so export skips it with a note.
set -uo pipefail

VOL="/media/hazemhagrass/New Volume"
TR="$VOL/llm-models/.12-shuttle/ollama-store/store"
TSTORE="$TR"
# .11's live store lives on the volume; .12 still uses the default path.
# Probe candidates instead of hard-coding, so install --host always lands right.
STORE_CANDIDATES=(
  "$VOL/llm-models"
  "$VOL/relocated/llm-models"
  "/media/hazemhagrass/Master/llm-models"
  "/usr/share/ollama/.ollama/models"
)
store_of() { # store_of HOST — first candidate whose manifests/ exists on that host
  local h="$1" c
  for c in "${STORE_CANDIDATES[@]}"; do
    if [ "$h" = "127.0.0.1" ] || [ "$h" = "$(hostname)" ]; then
      [ -d "$c/manifests" ] && { printf '%s\n' "$c"; return; }
    else
      [ "$(run_on "$h" "[ -d '$c/manifests' ] && echo y || echo n" | tail -1)" = "y" ] \
        && { printf '%s\n' "$c"; return; }
    fi
  done
  printf '%s\n' "/usr/share/ollama/.ollama/models"
}
rstore_of() { # rstore_of HOST — resolve the ollama store path on a remote host
  local h="$1" c
  # single reachability probe; offline host => default path, no per-candidate waits
  [ "$(run_on "$h" "true" 2>/dev/null | tail -1)" = "" ] && ! run_on "$h" true 2>/dev/null && {
    printf '%s\n' "/usr/share/ollama/.ollama/models"; return; }
  for c in "${STORE_CANDIDATES[@]}"; do
    [ "$(run_on "$h" "[ -d '$c/manifests' ] && echo y || echo n" 2>/dev/null | tail -1)" = "y" ] \
      && { printf '%s\n' "$c"; return; }
  done
  printf '%s\n' "/usr/share/ollama/.ollama/models"
}
STORE="$(store_of 127.0.0.1)"
LIBDIR="$STORE/manifests/registry.ollama.ai/library"
# volume identity is host-dependent: .11 = "New Volume", .12 desk = "Master"
if grep -q ' /media/hazemhagrass/Master ' /etc/fstab 2>/dev/null; then
  VOL_UUID="B2C053E3C053ABFB"; VOL_DEV="/dev/sdb1"
else
  VOL_UUID="1050D7C550D7AFA8"; VOL_DEV="/dev/nvme0n1p6"
fi
KEY="$HOME/.ssh/id_fleet"
SSHO=(-i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
INCOMING_DIR="ollama_incoming"

log()  { echo "== $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }
human() { awk -v b="$1" 'BEGIN{u[0]="B";u[1]="KB";u[2]="MB";u[3]="GB";s=b;i=0;while(s>=1024&&i<4){s/=1024;i++}printf "%.1f %s",s,u[i]}'; }
bx() { echo "  - $*"; }

# ---------- volume helpers ----------
vol_mounted() { findmnt -n -o TARGET "$VOL" >/dev/null 2>&1; }
vol_ready() {
  vol_mounted || return 1
  mkdir -p "$TSTORE/blobs" 2>/dev/null && touch "$TSTORE/.w" 2>/dev/null && rm -f "$TSTORE/.w"
}

# ---------- remote helpers (all root ops run over ssh, never local sudo) ----------
sudo_on() { # sudo_on HOST "snippet"  (runs snippet as root on HOST)
  local h="$1" cmd="$2" b64 rc
  b64="$(printf '%s' "$cmd" | base64 -w0)"
  ssh "${SSHO[@]}" "hazemhagrass@$h" \
    "printf %s '$b64' | base64 -d > /tmp/.ov-cmd && SUDO_ASKPASS=\$HOME/.hermes/askpass.sh sudo -A bash /tmp/.ov-cmd 2>&1; rc=\$?; rm -f /tmp/.ov-cmd; exit \$rc"
}
run_on() { # run_on HOST "cmd" (user-level)
  local h="$1" cmd="$2"
  ssh "${SSHO[@]}" "hazemhagrass@$h" "$cmd" 2>&1
}

# ---------- manifest helpers ----------
manifest_digests() { # MODEL TAG -> digests (hex) one per line
  local mf="$LIBDIR/$1/$2"; [ -f "$mf" ] || return 1
  python3 - "$mf" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
out = []
c = d.get("config") or {}
if c.get("digest"): out.append(c["digest"].removeprefix("sha256:"))
for l in d.get("layers", []):
    if l.get("digest"): out.append(l["digest"].removeprefix("sha256:"))
print("\n".join(out))
PY
}
manifest_total() { # MODEL TAG -> layer bytes (0 when absent or facade)
  local mf="$LIBDIR/$1/$2"; [ -f "$mf" ] || { echo 0; return; }
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(sum(s.get('size',0) for s in d.get('layers',[])))" "$mf" 2>/dev/null || echo 0
}
stag_manifest_path() { echo "$TSTORE/manifests/registry.ollama.ai/library/$1/$2"; }

# ---------- repair-mount / watch ----------
do_repair_mount() {
  if vol_mounted; then echo "volume already mounted"; return 0; fi
  [ -e "/dev/disk/by-uuid/$VOL_UUID" ] || { echo "volume device not present"; return 1; }
  echo "volume not mounted — repairing Windows lock and mounting"
  ntfsfix -d -q "$VOL_DEV" 2>&1 || echo "(ntfsfix reported; continuing)"
  if ! mount "UUID=$VOL_UUID" "$VOL" 2>&1; then
    echo "rw mount refused; trying read-only fallback"
    mount -o ro "UUID=$VOL_UUID" "$VOL" 2>&1 \
      && echo "WARNING: mounted READ-ONLY (Windows hibernation/Fast Startup) — move staging went to the RW copy below" \
      || { echo "could not mount volume"; return 1; }
  fi
  findmnt -n -o FSTYPE,OPTIONS "$VOL" | head -1
  return 0
}
do_watch() {
  vol_mounted && { echo "ok: mounted"; return 0; }
  do_repair_mount
}

# ---------- export ----------
cmd_export() {
  local models=() verify=0 all=0 a
  while [ $# -gt 0 ]; do case "$1" in
    --verify) verify=1; shift;;
    --all) all=1; shift;;
    -*) fail "export: unknown option $1";;
    *) models+=("$1"); shift;;
  esac; done
  vol_ready || fail "volume not ready (mounted + writable): $VOL"
  mkdir -p "$TSTORE/blobs" "$TSTORE/manifests/registry.ollama.ai/library"

  # --all: expand to every name:tag pair actually installed
  if [ $all = 1 ]; then
    models=()
    while IFS= read -r mf; do
      models+=("$(basename "$(dirname "$mf")"):$(basename "$mf")")
    done < <(find "$LIBDIR" -mindepth 2 -maxdepth 2 -type f 2>/dev/null | sort)
    [ ${#models[@]} -gt 0 ] || fail "no models installed in $LIBDIR"
  fi
  [ ${#models[@]} -gt 0 ] || fail "export: no models given (export MODEL... or export --all)"

  # preflight space
  local need=0 skip="" note=""
  for m in "${models[@]}"; do
    local name="${m%%:*}" tag="${m##*:}"; [ "$tag" = "$name" ] && tag=latest
    local bytes; bytes="$(manifest_total "$name" "$tag")"
    if [ "$bytes" = 0 ]; then note+="$m "; continue; fi
    [ -f "$(stag_manifest_path "$name" "$tag")" ] && continue   # already staged
    need=$((need + bytes))
  done
  if [ "$need" -gt 0 ]; then
    local avail_bytes; avail_bytes=$(df -B1 --output=avail "$VOL" | tail -1 | tr -d ' ')
    [ "$((need + 1073741824))" -le "$avail_bytes" ] || fail "not enough space on volume: need $(human $need), have $(human $avail_bytes)"
  fi
  [ -z "$note" ] || echo "SKIP (facade, served by front on each host; nothing to copy): $note"

  for m in "${models[@]}"; do
    local name="${m%%:*}" tag="${m##*:}"; [ "$tag" = "$name" ] && tag=latest
    [ "$(manifest_total "$name" "$tag")" = 0 ] && continue
    local dstmf; dstmf="$(stag_manifest_path "$name" "$tag")"
    echo ">> export $name:$tag"
    local digs=() d
    mapfile -t digs < <(manifest_digests "$name" "$tag")
    [ ${#digs[@]} -gt 0 ] || fail "no digests in manifest for $name:$tag"
    for d in "${digs[@]}"; do
      local src="$STORE/blobs/sha256-$d" dst="$TSTORE/blobs/sha256-$d"
      [ -f "$src" ] || fail "missing source blob $src"
      if [ -f "$dst" ] && [ "$(stat -c%s "$dst")" = "$(stat -c%s "$src")" ]; then
        bx "blob $d already staged ($(human "$(stat -c%s "$dst")"))"
        if [ $verify = 1 ]; then
          [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$d" ] || fail "staged blob corrupt: $d"
          bx "blob $d verified"
        fi
      else
        cp "$src" "$dst.tmp" 2>/dev/null || fail "copy failed: $d"
        if [ $verify = 1 ]; then
          [ "$(sha256sum "$dst.tmp" | cut -d' ' -f1)" = "$d" ] || fail "digest mismatch staging $d"
        fi
        mv "$dst.tmp" "$dst"
        bx "blob $d staged $(human "$(stat -c%s "$dst")")$([ $verify = 1 ] && echo ' (sha256-verified)')"
      fi
    done
    mkdir -p "$(dirname "$dstmf")"
    install -m 0644 "$LIBDIR/$name/$tag" "$dstmf.tmp" && mv "$dstmf.tmp" "$dstmf"
    echo "   manifest staged"
  done
  rebuild_index
  echo "export complete → $TR"
}

rebuild_index() {
  python3 - "$TSTORE" "$HOSTNAME" <<'PY'
import json, os, glob, sys, datetime
tstore, host = sys.argv[1], sys.argv[2]
idx = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
       "staged_from": host, "store": "ollama", "models": {}}
for mf in glob.glob(tstore + "/manifests/registry.ollama.ai/library/*/*"):
    name = os.path.basename(os.path.dirname(mf)); tag = os.path.basename(mf)
    try: d = json.load(open(mf))
    except Exception: continue
    idx["models"][f"{name}:{tag}"] = {"bytes": sum(s.get("size", 0) for s in d.get("layers", []))}
json.dump(idx, open(tstore + "/index.json", "w"), indent=1)
print(f"index.json rebuilt ({len(idx['models'])} models)")
PY
}

# ---------- list ----------
cmd_list() {
  local s12 up12
  up12=$(run_on 192.168.1.12 "true" 2>/dev/null | tail -1; true)
  if ! run_on 192.168.1.12 true 2>/dev/null; then
    s12=""
  else
    s12="$(rstore_of 192.168.1.12 || echo /usr/share/ollama/.ollama/models)"
  fi
  echo "-- staged on volume ($TR)"
  if [ -d "$TSTORE/manifests/registry.ollama.ai/library" ]; then
    for mf in "$TSTORE/manifests/registry.ollama.ai/library/"*/*; do
      [ -f "$mf" ] || continue
      local name tag bytes st=""
      name=$(basename "$(dirname "$mf")"); tag=$(basename "$mf")
      bytes=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(sum(s.get('size',0) for s in d.get('layers',[])))" "$mf")
      [ -f "$LIBDIR/$name/$tag" ] && st=" [installed here]"
      local h12="" rmd
      if [ -n "$s12" ]; then
        rmd=$(run_on 192.168.1.12 "f='$s12/manifests/registry.ollama.ai/library/$name/$tag'; [ -f \"\$f\" ] && echo yes || echo no" 2>/dev/null | tail -1)
      else
        rmd="offline"
      fi
      [ "$rmd" = "yes" ] && h12=" [on .12]"
      printf "  %-26s %9s  %s%s\n" "$name:$tag" "$(human "$bytes")" "$st" "$h12"
    done
  else echo "  (nothing staged)"; fi
  echo "-- installed here ($(hostname))"
  curl -s --max-time 8 http://127.0.0.1:11434/api/tags \
    | python3 -c "import json,sys;[print(f\"  {m['name']:26s} {m['size']/1e9:6.2f} GB\") for m in sorted(json.load(sys.stdin)['models'],key=lambda x:x['name'])]" \
    2>/dev/null || echo "  (local front unreachable)"
}

# ---------- install: volume -> host ----------
cmd_install() {
  local host="127.0.0.1" models=() all=0
  while [ $# -gt 0 ]; do case "$1" in
    --host) host="$2"; shift 2;;
    --all) all=1; shift;;
    -*) fail "install: unknown option $1";;
    *) models+=("$1"); shift;;
  esac; done
  vol_ready || fail "volume not ready: $VOL"
  if [ $all = 1 ]; then
    models=()
    while IFS= read -r pair; do models+=("$pair"); done < <(
      find "$TSTORE/manifests/registry.ollama.ai/library" -mindepth 2 -maxdepth 2 -type f -printf '%f %h\n' 2>/dev/null |
      while read -r tag dir; do echo "$(basename "$dir"):$tag"; done | sort)
  fi
  [ ${#models[@]} -gt 0 ] || fail "install: no models staged on volume"
  local hstore hlib
  hstore="$(rstore_of "$host")"
  hlib="$hstore/manifests/registry.ollama.ai/library"

  for m in "${models[@]}"; do
    local name="${m%%:*}" tag="${m##*:}"; [ "$tag" = "$name" ] && tag=latest
    local srcmf; srcmf="$(stag_manifest_path "$name" "$tag")"
    [ -f "$srcmf" ] || fail "$name:$tag not staged on volume"
    echo ">> install $name:$tag -> $host"

    # fast path: identical manifest already installed?
    local rmd
    rmd=$(run_on "$host" "f='$hlib/$name/$tag'; [ -f \"\$f\" ] && sha256sum \"\$f\" | cut -d' ' -f1 || echo ABSENT")
    if [ "$rmd" = "$(sha256sum "$srcmf" | cut -d' ' -f1)" ]; then
      bx "$name:$tag already installed on $host (manifest identical) - skip"
      continue
    fi

    run_on "$host" "mkdir -p ~/$INCOMING_DIR" >/dev/null
    local digs=() d
    mapfile -t digs < <(manifest_digests "$name" "$tag")
    # manifest first (streamed pre-install so the root step can move it)
    run_on "$host" "cat > ~/$INCOMING_DIR/manifest.json.tmp" < "$srcmf" || fail "manifest stream failed"
    local lmd5 rmd5
    lmd5=$(sha256sum "$srcmf" | cut -d' ' -f1)
    rmd5=$(run_on "$host" "sha256sum ~/$INCOMING_DIR/manifest.json.tmp | cut -d' ' -f1" | tail -1)
    [ "$lmd5" = "$rmd5" ] || fail "manifest verify failed after stream"
    echo "   manifest streamed + verified"
    for d in "${digs[@]}"; do
      local src="$TSTORE/blobs/sha256-$d" want havesz
      [ -f "$src" ] || fail "staged blob missing on volume: $src"
      want=$(stat -c%s "$src")
      havesz=$(run_on "$host" "f='$hstore/blobs/sha256-$d'; [ -f \"\$f\" ] && stat -c%s \"\$f\" || echo -1" | tail -1)
      if [ "$havesz" = "$want" ]; then bx "blob $d: host already has it ($(human "$want")) - skip"; continue; fi
      echo "   stream $d ($(human "$want"))"
      run_on "$host" "cat > ~/$INCOMING_DIR/sha256-$d" < "$src" || fail "stream failed: $d"
      local got; got=$(run_on "$host" "sha256sum ~/$INCOMING_DIR/sha256-$d | cut -d' ' -f1" | tail -1)
      [ "$got" = "$d" ] || fail "ssh stream digest mismatch for $d (got $got)"
    done

    # root install: verify again + move into store + manifest
    local digs_sp="${digs[*]}"
    sudo_on "$host" "set -e
cd ~/$INCOMING_DIR
ok=0
for d in $digs_sp; do
  [ -f \"sha256-\$d\" ] || continue
  s=\$(sha256sum \"sha256-\$d\" | cut -d' ' -f1)
  [ \"\$s\" = \"\$d\" ] || { echo \"digest bad \$d\"; exit 1; }
  mv -f \"sha256-\$d\" \"$hstore/blobs/sha256-\$d\"
  chown ollama:ollama \"$hstore/blobs/sha256-\$d\" 2>/dev/null || chown ollama \"$hstore/blobs/sha256-\$d\" 2>/dev/null || true
  chmod 0644 \"$hstore/blobs/sha256-\$d\"
  ok=1
done
mkdir -p \"$hlib/$name\"
mv -f ~/$INCOMING_DIR/manifest.json.tmp '$hlib/$name/$tag' 2>/dev/null || true
chown ollama:ollama '$hlib/$name/$tag' 2>/dev/null || chown ollama '$hlib/$name/$tag' 2>/dev/null || true
chmod 0644 '$hlib/$name/$tag'
rm -f ~/$INCOMING_DIR/manifest.json.tmp
[ \"\$ok\" = 1 ] || echo NO_BLOBS_NEEDED
echo INSTALL_OK" || fail "root install step failed on $host"

    local served
    served=$(run_on "$host" "curl -s --max-time 10 http://127.0.0.1:11434/api/tags" | grep -q "$name:$tag" && echo YES || echo NO)
    [ "$served" = "YES" ] && bx "$name:$tag now served on $host:11434" \
      || bx "$name:$tag installed in store (front tag check pending - llm load on next run)"
  done
  echo "install pass complete."
}

cmd_status() {
  echo "-- volume"
  if vol_mounted; then
    echo "  mounted ok: $(findmnt -n -o FSTYPE,OPTIONS "$VOL" | head -1)"
    df -h "$VOL" | tail -1 | sed 's/^/  /'
  else
    echo "  NOT MOUNTED - run (as root): ollama-volume.sh repair-mount"
  fi
  echo "-- local store ($STORE)"
  du -sh "$STORE" 2>/dev/null | sed 's/^/  /'
  echo "-- shuttle folder ($TR)"
  [ -d "$TR" ] && ls -la "$TR" | sed 's/^/  /' || echo "  (not created yet)"
}

case "${1:-}" in
  export) shift; cmd_export "$@";;
  list) cmd_list;;
  status) cmd_status;;
  install) shift; cmd_install "$@";;
  repair-mount) do_repair_mount;;
  watch) do_watch;;
  ""|-h|--help) sed -n '2,32p' "$0";;
  *) fail "unknown command '$1' (see: ollama-volume.sh --help)";;
esac
