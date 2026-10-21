#!/usr/bin/env bash
# Fast LAN replication of the clef models from this host (.11) to .12.
set -euo pipefail
STORE=/usr/share/ollama/.ollama/models
DST=hazemhagrass@192.168.1.12
KEY=$HOME/.ssh/id_fleet
SSHOPT="-i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"

# Every layer digest referenced by the two manifests (models, projectors, config, license, params)
DIGESTS="
7b967ed607c3e1a56d0883e2af077e7cff88f4e8b08ea9f796abcf450d4f8ab2
6c02216a0055c1e1e92994d240e37c440f1c09b30281f538f4a38ebab0743da9
f0e2930e301586f0932262c7d232c74e385d7b4281170111c36a686e16e59ba8
50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32
58e1b82a691f46c155e57be607f1031930ea0cccc22daf4447f91d360f74b2d2
852370b45cde12170b24ca8de600860be94b2faa4b6550911a58066a075ff0bf
c9ac3afdf0b5928b00a5762aed44a338a0096af952194ccdedecdfd46bb92c33
c8a6e9e00e1baf6433c4874dbc73bed2a56e887bcbb582fffcb00b533bc638dd
"

echo "### verify sources exist on .11"
for d in $DIGESTS; do
  f="$STORE/blobs/sha256-$d"
  [ -f "$f" ] || { echo "ABORT: missing $f"; exit 1; }
done
echo "all source blobs present"

echo "### prepare incoming dir on .12"
ssh $SSHOPT "$DST" 'mkdir -p ~/clef_incoming/manifests'

echo "### stream blobs .11 -> .12 home (fast, no sudo)"
for d in $DIGESTS; do
  sz=$(stat -c%s "$STORE/blobs/sha256-$d")
  echo "  -> $d ($sz bytes)"
  ssh $SSHOPT "$DST" "cat > ~/clef_incoming/sha256-$d" < "$STORE/blobs/sha256-$d"
done

echo "### stream manifests"
for m in clef clef-flash; do
  ssh $SSHOPT "$DST" "cat > ~/clef_incoming/manifests/$m.json" \
    < "$STORE/manifests/registry.ollama.ai/library/$m/latest"
done

echo "### install into ollama store on .12 (sudo -A via askpass)"
ssh $SSHOPT "$DST" 'set -e
export SUDO_ASKPASS=$HOME/.hermes/askpass.sh
B=/usr/share/ollama/.ollama/models
sudo -A install -d -o ollama -g ollama "$B/blobs"
sudo -A sh -c "cat ~/clef_incoming/sha256-* > /dev/null" 2>/dev/null || true
for f in ~/clef_incoming/sha256-*; do
  sudo -A install -o ollama -g ollama -m 0644 "$f" "$B/blobs/$(basename "$f")"
done
for m in clef clef-flash; do
  sudo -A install -d -o ollama -g ollama "$B/manifests/registry.ollama.ai/library/$m"
  sudo -A install -o ollama -g ollama -m 0644 ~/clef_incoming/manifests/$m.json "$B/manifests/registry.ollama.ai/library/$m/latest"
done
sudo -A rm -rf /home/hazemhagrass/clef_incoming
echo INSTALL_DONE'

echo "### .12 model list"
ssh $SSHOPT "$DST" 'ollama list | grep -i clef || echo "NO CLEF YET"'
echo "ALL_DONE"
