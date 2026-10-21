#!/usr/bin/env bash
# ollama-update-all.sh — update every model on the local ollama daemon,
# ALWAYS including the engine model behind the openjev facade.
# Usage: setsid bash -c 'exec ~/workspace/openjev-ollama/fleet/ollama-update-all.sh' >/dev/null 2>&1 &
export OLLAMA_HOST=http://127.0.0.1:11435
ENV_FILE="$HOME/workspace/openjev-ollama/.env"
LOG=/var/tmp/ollama-update.log
: > $LOG

# engine model behind the openjev facade (virtual model on :11434)
JEV_MODEL=$(grep -E '^JEV_OLLAMA_MODEL=' "$ENV_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' | xargs)
if [ -n "$JEV_MODEL" ]; then
  echo "openjev engine: $JEV_MODEL" >> $LOG
fi

models=$(curl -s $OLLAMA_HOST/api/tags | python3 -c '
import sys, json
d = json.load(sys.stdin)
skip = {"clef:latest", "clef-flash:latest"}   # custom local clones - never touch
for m in sorted(d["models"], key=lambda x: x["name"]):
    if m["name"] not in skip:
        print(m["name"])
')

# force-include the openjev engine even if it is not currently in /api/tags
if [ -n "$JEV_MODEL" ] && ! grep -qx "$JEV_MODEL" <<< "$models"; then
  models="$models $JEV_MODEL"
fi

for m in $models; do
  echo "=== $m ===" >> $LOG
  status=$(curl -s --max-time 7200 $OLLAMA_HOST/api/pull -d "{\"model\":\"$m\",\"stream\":true}" | python3 -c '
import sys, json
final = {}
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: d = json.loads(line)
    except: continue
    if "error" in d: print("ERROR: " + d["error"]); break
    final = d
print(final.get("status", "unknown"))
')
  echo "  -> $status" >> $LOG
done
echo "ALL_PULLS_DONE" >> $LOG