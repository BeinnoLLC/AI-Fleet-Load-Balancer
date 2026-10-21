#!/usr/bin/env bash
# Start the OpenJev-on-Ollama facade. Loads .env if present.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
exec .venv/bin/python ollama_jev.py
