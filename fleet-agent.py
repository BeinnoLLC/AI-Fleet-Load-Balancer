#!/usr/bin/env python3
"""Fleet push agent — for out-of-LAN fleet members.

Reads GPUs locally with nvidia-smi (same shape as the hub's read_local_gpus)
and POSTs {host_name, gpus, token} to the hub's POST /gpu/ingest every
PUSH_INTERVAL seconds, then sleeps as instructed by the hub response.

Config via env or a small config file (/etc/fleet-agent.conf or next to this
script, KEY=VALUE lines):
  FLEET_HUB_URL   e.g. https://cloudbox.example.com  (or http://192.168.1.11:11434)
  FLEET_HOST_NAME this machine's registry name (must exist in hub fleet_hosts.json, mode=push)
  FLEET_TOKEN     the hft-… token issued at enrollment
  PUSH_INTERVAL   seconds between pushes (default 5)

Stdlib only (urllib) — no pip installs needed on fleet members.
Fails soft: any error just logs to stderr and retries next interval.
"""

import json
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error

CONFIG_CANDIDATES = ["/etc/fleet-agent.conf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet-agent.conf")]


def load_config() -> dict:
    cfg = dict(os.environ)
    for path in CONFIG_CANDIDATES:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        cfg.setdefault(k.strip(), v.strip())
                break
            except OSError:
                pass
    return cfg


def read_local_gpus() -> list:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for row in out.stdout.strip().splitlines():
        # match the hub's read_local_gpus schema exactly (6 fields):
        # index,name,utilization.gpu,memory.used,memory.total,temperature.gpu
        parts = [p.strip() for p in row.split(",")]
        if len(parts) >= 6:
            try:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_pct": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                    "temperature_c": float(parts[5]),
                })
            except (ValueError, IndexError):
                continue
    return gpus


def push(cfg: dict, gpus: list) -> dict:
    hub = cfg.get("FLEET_HUB_URL", "").rstrip("/")
    name = cfg.get("FLEET_HOST_NAME", "")
    token = cfg.get("FLEET_TOKEN", "")
    if not hub or not name or not token:
        raise RuntimeError("FLEET_HUB_URL / FLEET_HOST_NAME / FLEET_TOKEN not configured")
    url = f"{hub}/gpu/ingest"
    payload = {"host_name": name, "gpus": gpus, "token": token}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode() or "{}")


def main() -> int:
    cfg = load_config()
    interval = float(cfg.get("PUSH_INTERVAL", "5"))
    print(f"fleet-agent: pushing {cfg.get('FLEET_HOST_NAME', '?')} -> {cfg.get('FLEET_HUB_URL', '?')} every {interval}s", file=sys.stderr)
    while True:
        try:
            gpus = read_local_gpus()
            result = push(cfg, gpus)
            nxt = result.get("next_s")
            if isinstance(nxt, (int, float)) and nxt > 0:
                interval = float(nxt)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:200]
            except Exception:
                pass
            print(f"fleet-agent: hub rejected ({e.code}) {body}", file=sys.stderr)
            if e.code in (401, 404, 403):
                print("fleet-agent: fatal config error (unknown host/token) — fix fleet-agent.conf", file=sys.stderr)
                return 2
        except Exception as e:
            print(f"fleet-agent: push failed: {e}", file=sys.stderr)
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
