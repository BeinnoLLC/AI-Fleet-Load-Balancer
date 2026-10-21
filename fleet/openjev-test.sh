#!/usr/bin/env bash
# openjev-test.sh -- smoke-test the OpenJev-on-Ollama facade from any machine.
#
#   ./openjev-test.sh                 # this machine (127.0.0.1:11434)
#   ./openjev-test.sh 192.168.1.11    # the home box, over LAN
#   ./openjev-test.sh 192.168.1.12    # the desk box, over LAN
#
# Tests the facade (= the port your agents use): openjev virtual tag injection,
# the /v1/systemone decision contract (noul / choice / score), native clef-flash
# routing, and the openjev tag as a plain chat model.
set -uo pipefail

HOST="${1:-127.0.0.1}"
B="http://${HOST}:11434"
pass=0; fail=0

say()  { printf '  %-6s %s\n' "$1" "$2"; }
pass_() { say PASS "$1"; pass=$((pass+1)); }
fail_() { say FAIL "$1"; fail=$((fail+1)); }

py() { python3 -c "$1" 2>/dev/null; }

echo "OpenJev facade smoke test -> $B"
echo

# --- reachability -----------------------------------------------------------
if curl -fsS -m 10 "$B/api/tags" >/dev/null 2>&1; then
  pass_ "facade reachable on $B"
else
  fail_ "facade NOT reachable on $B (is the front running? right host/port?)"
  echo; echo "RESULT: $pass passed, $fail failed"; exit 1
fi

# --- 1. openjev virtual tag injected into the model list --------------------
tags=$(curl -fsS -m 10 "$B/api/tags")
n=$(printf '%s' "$tags" | py 'import sys,json;print(len(json.load(sys.stdin)["models"]))')
if printf '%s' "$tags" | py 'import sys,json;sys.exit(0 if any(m["name"]=="openjev:latest" for m in json.load(sys.stdin)["models"]) else 1)'; then
  pass_ "openjev:latest present in /api/tags  (models: ${n:-?})"
else
  fail_ "openjev:latest missing from /api/tags"
fi

# --- 2. /v1/models lists models ---------------------------------------------
if curl -fsS -m 10 "$B/v1/models" | py 'import sys,json;d=json.load(sys.stdin);sys.exit(0 if (d.get("data") or d.get("models")) else 1)'; then
  pass_ "/v1/models returns a model list"
else
  fail_ "/v1/models did not return a list"
fi

# --- 3. noul (yes/no probability) -------------------------------------------
r=$(curl -fsS -m 120 "$B/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"openjev:latest","state":"User requests a refund for an order delivered 40 days ago; policy window is 30 days.","questions":{"approve":{"type":"noul","instructions":"Approve the refund?","criteria":{"true":"Approve","false":"Deny"}}}}')
v=$(printf '%s' "$r" | py 'import sys,json;print(json.load(sys.stdin)["answers"]["approve"]["noul"])')
if [ -n "${v:-}" ]; then pass_ "noul answered (noul=$v)"; else fail_ "noul call failed -> $r"; fi

# --- 4. choice (pick a label) -----------------------------------------------
r=$(curl -fsS -m 120 "$B/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"openjev:latest","state":"Customer complains about an unexpected invoice charge.","questions":{"label":{"type":"choice","instructions":"Pick the category.","criteria":{"billing":"invoice or charge issue","technical":"bug or outage","account":"login or profile issue"}}}}')
v=$(printf '%s' "$r" | py 'import sys,json;print(json.load(sys.stdin)["answers"]["label"]["choice"])')
if [ -n "${v:-}" ]; then pass_ "choice answered (choice=$v)"; else fail_ "choice call failed -> $r"; fi

# --- 5. score (0-based severity scale) --------------------------------------
r=$(curl -fsS -m 120 "$B/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"openjev:latest","state":"Server is down and all users are affected.","questions":{"urgency":{"type":"score","instructions":"Rate severity.","criteria":["trivial","low","medium","high","critical"]}}}')
v=$(printf '%s' "$r" | py 'import sys,json;d=json.load(sys.stdin)["answers"]["urgency"];print("%s (legend levels: %d)"%(d["score"],len(d.get("legend",{}))))')
if [ -n "${v:-}" ]; then pass_ "score answered (score=$v)"; else fail_ "score call failed -> $r"; fi

# --- 6. native clef-flash routed through the facade (proxied to real ollama) -
r=$(curl -fsS -m 120 "$B/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"clef-flash:latest","state":"Refund request 40 days after delivery, outside the 30-day window.","questions":{"approve":{"type":"noul","instructions":"Approve?","criteria":{"true":"Approve","false":"Deny"}}}}')
m=$(printf '%s' "$r" | py 'import sys,json;print(json.load(sys.stdin)["model"])')
if [ "$m" = "clef-flash:latest" ]; then pass_ "clef-flash:latest answered natively through the facade"; else fail_ "clef-flash routing broken -> $r"; fi

# --- 7. openjev tag usable as a chat model on the same port -----------------
if curl -fsS -m 120 "$B/api/chat" -d '{"model":"openjev:latest","messages":[{"role":"user","content":"Should we approve a refund 40 days after delivery?"}],"stream":false}' | py 'import sys,json;sys.exit(0 if json.load(sys.stdin).get("message",{}).get("content") else 1)'; then
  pass_ "openjev:latest answered on /api/chat"
else
  fail_ "/api/chat with openjev:latest returned nothing"
fi

echo
echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
