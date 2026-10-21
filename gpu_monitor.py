"""gpu_monitor.py — fleet-wide GPU awareness for ollama_front.py.

Generalized to N hosts, not a hardcoded pair. The source of truth is a
JSON file (default: fleet_hosts.json next to this module) listing every
front on the LAN:

    [
      {"name": "raider", "url": "http://192.168.1.11:11434"},
      {"name": "hermes", "url": "http://192.168.1.12:11434"}
    ]

Add a machine by adding a line to that file — no service restart, no code
change. The file is re-read on every poll cycle (mtime-cached), so a new
entry is picked up within one `poll_interval`.

Each cycle, every host in the file is probed:
  * If the URL points at this machine (matched against local IPs), GPU
    stats come from a local `nvidia-smi` call — zero network round trip.
  * Otherwise `{url}/gpu/status` is fetched over HTTP with a short timeout.
    A host that doesn't answer is marked unreachable/offline; its last
    known-good reading (if any) is kept but flagged stale, never used for
    routing.

This means:
  * GPU count/model changes (user adds/removes a card) are picked up
    automatically on the next nvidia-smi call — no config needed.
  * A machine going down/booting is detected within one poll interval.
  * A brand-new machine becomes routable as soon as it's added to
    fleet_hosts.json and its front is reachable — no other change needed.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import time
from typing import Optional

HISTORY_MS = 10 * 60 * 1000  # 10-minute server-side history window


async def read_local_gpus() -> list[dict]:
    """Return per-GPU stats from `nvidia-smi` on THIS machine, or [] if unavailable."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _err = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode != 0:
            return []
        gpus = []
        for line in out.decode().strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            idx, name, util, mem_used, mem_total, temp = parts
            gpus.append({
                "index": int(idx),
                "name": name,
                "utilization_pct": float(util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
                "temperature_c": float(temp),
            })
        return gpus
    except Exception:
        return []


def _temp_factor(temp_c: float) -> float:
    """Capacity multiplier that derates GPUs running hot (1.0 below 75C,
    sliding toward 0.4 at 90C+) so routing steers away from near-throttle cards."""
    if temp_c <= 75:
        return 1.0
    return max(0.4, 1.0 - (temp_c - 75) * 0.04)


def _gpu_power(name: str, total_mb: float) -> float:
    """Relative compute weight for a GPU model name; null where unknown,
    so the caller can fall back to a VRAM-proportional weight."""
    lname = (name or "").lower()
    for key, w in (
        # User preference: rank GPUs by generation, best first — 40-series and
        # newer go first; 3090 Ti deliberately LAST (idle reserve, not workhorse).
        ("h100", 3.0), ("b200", 3.5), ("a100", 1.6),
        ("5090", 2.2), ("4090", 2.0), ("5080", 1.4), ("4080", 1.3),
        ("l40", 1.5), ("5000", 1.4),
        ("4070 ti", 0.9), ("4070", 0.75), ("5070 ti", 0.9), ("5070", 0.7),
        ("3070 ti", 0.55), ("3070", 0.45), ("3080 ti", 0.5), ("3080", 0.42),
        ("4060", 0.4), ("4050", 0.3), ("3060", 0.3), ("3050", 0.2),
        ("3090 ti", 0.08),  # dead last: only used when nothing else fits
    ):
        if key in lname:
            return w
    # Unknown model: assume capability roughly tracks VRAM (24GB card ≈ 0.85).
    return min(1.2, max(0.15, (total_mb or 8000) / 28000.0))


def _score(gpus: list[dict]) -> dict:
    """Summarize a host's GPUs into one comparable capacity score.

    `capacity` = sum over GPUs of  compute_power × free_VRAM(GB) ×
    idle_headroom × thermal_factor, i.e. how much *useful work* this host
    could take on right now. A 3090 Ti with 20GB free idles worth ~2x a
    4090 laptop with 2GB free and a busy card."""
    if not gpus:
        return {"free_mb": 0.0, "avg_util": 100.0, "gpu_count": 0, "capacity": 0.0,
                "max_temp": 0.0}
    free_mb = sum(g["memory_total_mb"] - g["memory_used_mb"] for g in gpus)
    avg_util = sum(g["utilization_pct"] for g in gpus) / len(gpus)
    max_temp = max(g.get("temperature_c", 0.0) for g in gpus)
    capacity = 0.0
    for g in gpus:
        power = _gpu_power(g.get("name", ""), g.get("memory_total_mb", 0.0))
        free_g = max(0.0, (g["memory_total_mb"] - g["memory_used_mb"]) / 1024.0)
        headroom = 1.0 - 0.5 * (g.get("utilization_pct", 0.0) / 100.0)
        factor = _temp_factor(g.get("temperature_c", 0.0))
        # User preference: the 4090 is the preferred workhorse. When it is
        # genuinely free (>= 8GB VRAM open, low load), boost it so it wins
        # the queue over nominally-higher-capacity hosts. Busy/hot 4090
        # keeps no boost and loses to idle hosts as usual.
        lname = (g.get("name", "") or "").lower()
        if "4090" in lname and free_g >= 8 and g.get("utilization_pct", 0) <= 50:
            factor *= 1.5
        capacity += power * free_g * headroom * factor
    return {"free_mb": free_mb, "avg_util": avg_util, "gpu_count": len(gpus),
            "capacity": round(capacity, 2), "max_temp": max_temp}


def _local_ips() -> set[str]:
    """Every IP this machine answers to, best-effort, for self-detection."""
    ips = {"127.0.0.1", "localhost"}
    try:
        ips.add(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass
    try:
        # Doesn't actually send a packet — just asks the kernel to pick the
        # outbound-route source IP, which is the LAN IP on a normal box.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(info[4][0])
    except Exception:
        pass
    return ips


def _url_host(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0].split(":")[0]


class HostsFile:
    """Hot-reloading reader for the fleet host registry.

    Each entry: {"name": ..., "url": ..., "mode": "pull"|"push"}.
    `pull` (LAN default): the hub polls {url}/gpu/status as always.
    `push` (out-of-LAN): the remote agent POSTs samples to our /gpu/ingest;
      `url` is optional for push hosts (used only for display/deep-links).
    """

    def __init__(self, path: str):
        self.path = path
        self._mtime: float = -1.0
        self._hosts: list[dict] = []
        self._pollable: list[dict] = []

    def load(self) -> list[dict]:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return self._hosts
        if mtime == self._mtime:
            return self._hosts
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            hosts = [h for h in data if h.get("name") and (h.get("url") or h.get("mode") == "push")]
            self._hosts = hosts
            self._pollable = [h for h in hosts if h.get("url")]
            self._mtime = mtime
        except Exception:
            pass  # keep last-known-good list rather than go blind on a bad edit
        return self._hosts

    def pull_hosts(self) -> list[dict]:
        """Only hosts this machine should actively poll (mode=pull, default)."""
        self.load()
        return self._pollable


class FleetMonitor:
    """Polls every host in fleet_hosts.json; local host is read via nvidia-smi,
    remote hosts via their front's /gpu/status. Feeds routing decisions."""

    def __init__(self, hosts_file: str, self_name: Optional[str] = None,
                 poll_interval: float = 5.0, stale_after: float = 30.0):
        self.hosts = HostsFile(hosts_file)
        self.self_name = self_name
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        self.state: dict[str, dict] = {}  # name -> {url, gpus, ts, reachable, is_self}
        self._last_pick: Optional[str] = None  # routing stickiness (anti-flapping)
        self._client = None
        self._task: Optional[asyncio.Task] = None
        self._local_ips = _local_ips()
        # Server-side utilization history (10 min window), survives page close.
        self.hist_path = os.path.join(os.path.dirname(os.path.abspath(hosts_file)), ".gpu_history.json")
        self.history: dict[str, dict] = {}  # key "{host}·GPU{i}" -> {t:[ms], v:[GB]}
        self._hist_loaded = False
        self._hist_flush_at = 0.0
        self._load_history()

    def _load_history(self) -> None:
        """Restore previous history window from disk (fills the gap after a restart)."""
        try:
            with open(self.hist_path, "r") as f:
                data = json.load(f)
            now_ms = time.time() * 1000.0
            self.history = {
                k: s for k, s in (data or {}).items()
                if isinstance(s, dict) and s.get("t") and now_ms - s["t"][-1] <= HISTORY_MS * 1000.0
            }
        except Exception:
            self.history = {}
        self._hist_loaded = True

    def _record_history(self) -> None:
        """Append one sample per online GPU; prune the 10-minute window; flush periodically."""
        now_ms = time.time() * 1000.0
        for name, entry in self.state.items():
            if not entry.get("reachable") or not entry.get("gpus"):
                continue
            for g in entry["gpus"]:
                k = f"{name}·GPU{g.get('index', 0)}"
                s = self.history.setdefault(k, {"t": [], "v": []})
                s["t"].append(now_ms)
                s["v"].append(round(g.get("memory_used_mb", 0) / 1024.0, 2))
                while s["t"] and now_ms - s["t"][0] > HISTORY_MS * 1000.0:
                    s["t"].pop(0)
                    s["v"].pop(0)
        # prune series not seen in the window
        for k in list(self.history.keys()):
            s = self.history[k]
            if not s["t"] or now_ms - s["t"][-1] > HISTORY_MS * 1000.0:
                del self.history[k]
        # flush to disk at most every 30s so a service restart loses <30s
        if time.time() >= self._hist_flush_at:
            self._hist_flush_at = time.time() + 30.0
            try:
                tmp = self.hist_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(self.history, f)
                os.replace(tmp, self.hist_path)
            except Exception:
                pass

    def history_snapshot(self) -> dict:
        return {"window_s": HISTORY_MS, "series": self.history}

    def _is_self(self, host: dict) -> bool:
        if self.self_name and host["name"] == self.self_name:
            return True
        return _url_host(host["url"]) in self._local_ips

    async def start(self, client) -> None:
        self._client = client
        await self._poll_once()  # warm cache before serving traffic
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.poll_interval)
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.poll_interval)

    async def _poll_once(self) -> None:
        hosts = self.hosts.pull_hosts()  # push-mode hosts arrive via ingest_sample() instead
        for host in hosts:
            name, url = host["name"], host["url"].rstrip("/")
            is_self = self._is_self(host)
            prev = self.state.get(name, {})
            if is_self:
                gpus = await read_local_gpus()
                self.state[name] = {"url": url, "gpus": gpus, "ts": time.time(),
                                     "reachable": True, "is_self": True}
                continue
            try:
                r = await self._client.get(f"{url}/gpu/status", timeout=3.0)
                data = r.json()
                self.state[name] = {"url": url, "gpus": data.get("gpus", []),
                                     "ts": time.time(), "reachable": True, "is_self": False}
            except Exception:
                # Keep last-known gpu reading (for display) but mark unreachable now;
                # freshness check in snapshot() prevents stale data from being routed to.
                self.state[name] = {**prev, "url": url, "reachable": False,
                                     "is_self": False, "ts": prev.get("ts", 0.0)}
        # Drop hosts removed from the registry so the dashboard doesn't show ghosts.
        known = {h["name"] for h in hosts}
        for stale_name in list(self.state.keys()):
            if stale_name not in known:
                del self.state[stale_name]
        self._record_history()

    def ingest_sample(self, name: str, gpus: list[dict]) -> bool:
        """Accept a pushed sample from a push-mode remote agent (out-of-LAN host).

        Mirrors what _poll_once does for pull hosts: update state so routing,
        history and the dashboard see it exactly like a polled host. Rejects
        names not present in fleet_hosts.json with mode=push.
        """
        entry = next((h for h in self.hosts.load()
                      if h.get("name") == name and h.get("mode") == "push"), None)
        if entry is None:
            return False
        self.state[name] = {"url": entry.get("url"), "gpus": gpus if isinstance(gpus, list) else [],
                            "ts": time.time(), "reachable": True, "is_self": False,
                            "mode": "push"}
        self._record_history()
        return True

    def _fresh(self, entry: dict) -> bool:
        return entry.get("reachable", False) and (time.time() - entry.get("ts", 0.0)) <= self.stale_after

    def snapshot(self) -> list[dict]:
        """Full table for the dashboard / /gpu/fleet endpoint."""
        out = []
        for name, entry in sorted(self.state.items()):
            fresh = self._fresh(entry)
            score = _score(entry.get("gpus", [])) if fresh else None
            out.append({
                "name": name,
                "url": entry.get("url"),
                "mode": entry.get("mode", "pull"),
                "is_self": entry.get("is_self", False),
                "online": fresh,
                "last_seen_s_ago": round(time.time() - entry["ts"], 1) if entry.get("ts") else None,
                "gpus": entry.get("gpus", []),
                "score": score,
            })
        return out

    def pick_host(self) -> Optional[str]:
        """Return the front URL of the host that should take the next job.

        Ranking is by `capacity` (from _score): a composite of GPU model
        power, free VRAM, current utilization headroom and thermals — so a
        3090 Ti idling at 38C beats a 4090 running at 95% and 80C, but a
        busy high-end card still beats an idle weak one when it has room.
        Hysteresis: once a host wins, it keeps the job unless a challenger
        beats its capacity by >= 15%, to stop load flapping between hosts.
        """
        candidates = [h for h in self.snapshot()
                      if h["online"] and h["score"] and h["score"]["gpu_count"] > 0
                      and h["score"]["capacity"] > 0]
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda h: h["score"]["capacity"], reverse=True)
        best = ranked[0]
        holder = next((c for c in ranked if c["url"] == self._last_pick), None)
        if holder is not None and holder is not best \
                and best["score"]["capacity"] < holder["score"]["capacity"] * 1.15:
            self._last_pick = holder["url"]
            return holder["url"]
        self._last_pick = best["url"]
        return best["url"]


DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Ollama Fleet — GPU Command Center</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<style>
:root{--bg:#0b0e14;--card:#131822;--line:#1f2632;--fg:#e6edf3;--mut:#8b949e;
--blue:#58a6ff;--green:#3fb950;--red:#f85149;--amber:#d29922;--purple:#bc8cff}
*{box-sizing:border-box}
body{font-family:-apple-system,'Segoe UI',system-ui,sans-serif;background:
radial-gradient(1200px 600px at 70% -10%,#16202e 0%,var(--bg) 60%);color:var(--fg);
margin:0;padding:2rem 2.5rem}
h1{font-size:1.45rem;margin:0;letter-spacing:.01em}
.sub{color:var(--mut);font-size:.83rem;margin:.3rem 0 1.6rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1.1rem;max-width:1200px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.2rem;min-width:0}
.card h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin:0 0 .8rem;font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.online-dot{background:var(--green);box-shadow:0 0 6px var(--green)}
.offline-dot{background:var(--red);box-shadow:0 0 6px var(--red)}
.hosthead{display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem}
.hostname{font-weight:700;font-size:1rem}
.selftag{color:var(--blue);font-size:.72rem;border:1px solid var(--blue);border-radius:10px;padding:1px 8px;margin-left:6px}
.routehost{color:var(--green);font-size:.72rem;border:1px solid var(--green);border-radius:10px;padding:1px 8px;margin-left:6px}
.chartbox{position:relative;height:200px}
canvas{max-width:100%}
.fleetwrap{position:relative;height:230px;max-width:280px;margin:0 auto}
.fleetcenter{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;pointer-events:none}
.fleetcenter b{font-size:1.35rem;display:block}
.fleetcenter span{color:var(--mut);font-size:.68rem;text-transform:uppercase;letter-spacing:.06em}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{padding:.3rem .4rem;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em}
tr:last-child td{border-bottom:none}
.chip{display:inline-block;padding:1px 7px;border-radius:8px;font-size:.7rem;font-weight:600}
.c-ok{background:rgba(63,185,80,.15);color:var(--green)}
.c-warn{background:rgba(210,153,34,.18);color:var(--amber)}
.c-hot{background:rgba(248,81,73,.18);color:var(--red)}
.utilbar{width:100%;height:6px;background:#232b38;border-radius:3px;overflow:hidden;margin-top:3px}
.utilbar>i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));transition:width .6s}
.foot{color:var(--mut);font-size:.72rem;margin-top:1.3rem}
.route-banner{max-width:1200px;margin:0 auto;background:linear-gradient(90deg,rgba(63,185,80,.08),rgba(88,166,255,.06));
border:1px solid var(--line);border-radius:12px;padding:.7rem 1.1rem;margin-bottom:1.1rem;
display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem}
.route-banner .lbl{color:var(--mut);font-size:.78rem;text-transform:uppercase;letter-spacing:.07em}
.route-banner .tgt{font-weight:700;color:var(--green)}
.score{font-size:.72rem;color:var(--mut)}
.tabs{display:flex;gap:.4rem;max-width:1200px;margin:0 auto 1.1rem;border-bottom:1px solid var(--line)}
.tab{background:none;border:none;color:var(--mut);font-size:.9rem;font-weight:600;padding:.55rem 1.1rem;cursor:pointer;border-bottom:2px solid transparent}
.tab.active{color:var(--fg);border-bottom-color:var(--acc,#58a6ff)}
.tabpane{max-width:1200px;margin:0 auto}
.mtest{background:var(--blue);color:#0b0e14;border:none;border-radius:8px;font-size:.72rem;font-weight:700;padding:.35rem .8rem;cursor:pointer}
.mtest:disabled{opacity:.5;cursor:wait}
.mtest.err{background:var(--red)}
.mtest.ok{background:var(--green)}
.mrow{display:flex;justify-content:space-between;align-items:center;padding:.35rem .2rem;border-bottom:1px solid var(--line);gap:.6rem}
.mrow:last-child{border-bottom:none}
.mname{font-size:.85rem;font-weight:600}
.mmeta{font-size:.7rem;color:var(--mut)}
.mres{font-size:.75rem;margin-top:.35rem;padding:.5rem .7rem;background:var(--line,rgba(128,128,128,.12));border-radius:8px;line-height:1.4;display:none}
.mres.show{display:block}
.mres .mres-head{display:flex;gap:.8rem;flex-wrap:wrap;font-size:.68rem;color:var(--mut);margin-bottom:.3rem}
.mres .mres-reply{font-family:ui-monospace,monospace;font-size:.78rem;color:var(--fg);white-space:pre-wrap;word-break:break-word}
.mres.err .mres-reply{color:var(--red)}
.mtoolbar{display:flex;gap:.6rem;flex-wrap:wrap;align-items:center;margin-bottom:1rem}
.mtoolbar input,.mtoolbar select{background:var(--card);border:1px solid var(--line);border-radius:8px;color:var(--fg);font-size:.85rem;padding:.5rem .7rem;min-width:180px}
.mtoolbar input#modelsearch{max-width:280px}
.mtoolbar input#testprompt{flex:1;min-width:220px}
.mtest.primary{background:var(--blue);color:#0b0e14;padding:.5rem .9rem;font-size:.8rem}
.mtest.busy{background:var(--red)}
.hbadge{display:inline-block;padding:.15rem .5rem;border-radius:999px;background:rgba(88,166,255,.15);color:var(--blue);font-size:.68rem;font-weight:700;letter-spacing:.02em;white-space:nowrap}
#mtab td{vertical-align:middle}
#mtab td.num{text-align:right;font-variant-numeric:tabular-nums;font-size:.8rem}
#mtab .mname-cell{font-weight:600;font-size:.82rem}
#mtab tr[data-row]:hover td{background:rgba(88,166,255,.05)}
.mresrow td{padding:0;border-top:none}
.mresrow .mres{margin:.1rem 0 .7rem;border-left:2px solid var(--line)}
.mres.ok{border-left-color:var(--green)}
.mres.err{border-left-color:var(--red)}
.mres-io{display:grid;grid-template-columns:1fr;gap:.15rem}
.mres-label{font-size:.6rem;letter-spacing:.08em;color:var(--mut);font-weight:700;margin-top:.35rem}
.mres-text{font-size:.78rem;color:var(--mut);white-space:pre-wrap;word-break:break-word;font-style:italic}
.tag{font-size:.62rem;padding:.1rem .45rem;border-radius:999px;font-weight:700}
.tag.ok{background:rgba(63,185,80,.15);color:var(--green)}
.tag.err{background:rgba(248,81,73,.15);color:var(--red)}
.spin{display:inline-block;animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.mtrend{display:inline-flex;align-items:center;gap:.3rem;height:24px}
.mtrend-bar{display:inline-block;width:3px;height:100%;background:var(--blue);opacity:.6;margin:0 1px}
footer{position:sticky;bottom:0;z-index:50;border-top:1px solid var(--line);padding:.8rem 1.5rem;margin-top:2rem;text-align:center;color:var(--mut);font-size:.85rem;background:var(--bg);backdrop-filter:blur(8px);background:color-mix(in srgb,var(--bg) 88%,transparent)}
footer a{color:var(--blue);text-decoration:none;transition:color .2s}
footer a:hover{color:var(--fg)}
#visgraph{filter:drop-shadow(0 2px 8px rgba(0,0,0,0.5));border-radius:10px}
/* ---- toasts ---- */
#toasts{position:fixed;top:1rem;right:1rem;z-index:9999;display:flex;flex-direction:column;gap:.6rem;max-width:420px}
.toast{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:10px;padding:.7rem .9rem;box-shadow:0 6px 24px rgba(0,0,0,.45);opacity:0;transform:translateX(24px);transition:opacity .25s,transform .25s;font-size:.85rem}
.toast.show{opacity:1;transform:translateX(0)}
.toast.done{border-left-color:var(--ok,#22c55e)}
.toast.err{border-left-color:#ef4444}
.toast .t-title{display:flex;align-items:center;gap:.5rem;font-weight:600;color:var(--fg)}
.toast .t-sub{color:var(--mut);margin-top:.25rem;font-size:.8rem}
</style></head>
<body>
<div id="toasts"></div>
<script>
// ---- toast + sound system ----
let audioCtx=null,lastEventId=0,inflightT={};
function ensureAudio(){ if(!audioCtx){ try{audioCtx=new (window.AudioContext||window.webkitAudioContext)();}catch(e){} } if(audioCtx&&audioCtx.state==='suspended') audioCtx.resume(); return audioCtx; }
function beep(kind){
  const ac=ensureAudio(); if(!ac) return;
  const seq = kind==='done' ? [660,880] : kind==='err' ? [330,220] : [520,780];
  seq.forEach((f,i)=>{
    const o=ac.createOscillator(), g=ac.createGain();
    o.type= kind==='err' ? 'sawtooth':'sine'; o.frequency.value=f;
    g.gain.setValueAtTime(0.0001,ac.currentTime+i*0.14);
    g.gain.exponentialRampToValueAtTime(0.18,ac.currentTime+i*0.14+0.02);
    g.gain.exponentialRampToValueAtTime(0.0001,ac.currentTime+i*0.14+0.13);
    o.connect(g); g.connect(ac.destination);
    o.start(ac.currentTime+i*0.14); o.stop(ac.currentTime+i*0.14+0.14);
  });
}
function toast(kind,title,sub,ttl){
  const wrap=document.getElementById('toasts'); if(!wrap) return;
  const el=document.createElement('div'); el.className='toast'+(kind==='done'?' done':kind==='err'?' err':'');
  el.innerHTML=`<div class="t-title">${kind==='done'?'✅':kind==='err'?'⛔':'📨'} ${title}</div>${sub?`<div class="t-sub">${sub}</div>`:''}`;
  wrap.appendChild(el);
  requestAnimationFrame(()=>el.classList.add('show'));
  setTimeout(()=>{el.classList.remove('show'); setTimeout(()=>el.remove(),300);},ttl||6000);
  while(wrap.children.length>6) wrap.firstChild.remove();
}
</script>
<h1>⚙️ Ollama Fleet — GPU Command Center</h1>
<div class="sub">Live · refreshes every 3s · smart routing by GPU power, free VRAM, utilization & thermals</div>
<div style="display:flex;justify-content:space-between;align-items:center;max-width:1200px;margin:0 auto 1.1rem">
  <div class="tabs">
    <button class="tab active" data-tab="fleet">Fleet</button>
    <button class="tab" data-tab="visualization">Visualization</button>
    <button class="tab" data-tab="models">Models</button>
    <button class="tab" data-tab="events">Events</button>
  </div>
  <button id="refreshbtn" style="background:none;border:none;color:var(--mut);cursor:pointer;padding:.5rem;display:flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:8px;transition:all .2s" title="Refresh all data"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15.36-6.36L21 8"/><path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15.36 6.36L3 16"/></svg></button>
</div>
<div id="tab-fleet" class="tabpane">
<div class="route-banner">
  <div><span class="lbl">Next request routes to</span><br><span class="tgt" id="route">…</span></div>
  <div><span class="lbl">Decision basis</span><br><span class="score" id="why">—</span></div>
</div>
<div class="grid">
  <div class="card"><h2>Fleet capacity share</h2>
    <div class="fleetwrap">
      <canvas id="pie"></canvas>
      <div class="fleetcenter"><b id="fleetcap">—</b><span>total capacity</span></div>
    </div>
  </div>
  <div class="card" style="grid-column:span 2"><h2>All GPUs — utilization history (10 min)</h2>
    <div class="chartbox" style="height:260px"><canvas id="allgpu"></canvas></div>
  </div>
  <div class="card" style="grid-column:1/-1"><h2>Per-GPU detail</h2>
    <table id="t"><thead><tr><th>Host</th><th>GPU</th><th>Utilization</th><th>VRAM</th><th>Temp</th><th>Last seen</th></tr></thead><tbody></tbody></table>
  </div>
</div>
</div>
<div id="tab-visualization" class="tabpane" style="display:none">
<div class="card"><h2>Router Network — Machines & Active Models</h2>
  <svg id="visgraph" style="border:1px solid var(--line);border-radius:10px;display:block;margin:0 auto;background:var(--bg);position:relative"></svg>
  <p style="text-align:center;color:var(--mut);margin-top:.5rem;font-size:.85rem\">🔷 Router · 🔵 Machine · 🟢 Model (size = usage frequency)</p>
</div>
</div>
<div id="tab-models" class="tabpane" style="display:none">
  <div class="mtoolbar">
    <input type="text" id="modelsearch" placeholder="Search models by name…">
    <select id="machinefilter"><option value="">All machines</option></select>
    <input type="text" id="testprompt" placeholder="Test prompt" value="Reply with exactly: OK">
    <button id="testall" class="mtest primary">Test all</button>
    <span id="testprogress" class="mmeta"></span>
  </div>
  <div class="card" style="overflow-x:auto">
    <table id="mtab">
      <thead><tr><th>Machine</th><th>Model</th><th>Details</th><th class="num">Calls</th><th class="num">Avg</th><th class="num">Success</th><th>Test</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>
<div id="tab-events" class="tabpane" style="display:none">
  <div class="card"><h2>Event log — routing decisions &amp; requests</h2>
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:.7rem" id="evstats"></div>
    <table id="ev"><thead><tr><th>Time</th><th>Kind</th><th>Path/Model</th><th>Upstream</th><th>Why</th><th>Status</th><th>Duration</th></tr></thead><tbody></tbody></table>
  </div>
</div>
<div class="foot" id="err"></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js"></script>
<script>
const colors=['#58a6ff','#3fb950','#bc8cff','#d29922','#f85149','#39c5cf'];
let pie=null, allGpuChart=null, allGpuHist={};   // GPU history ring buffer (10 min)
function mkPie(){
  if(typeof Chart==='undefined')return;
  pie=new Chart(document.getElementById('pie'),{
    type:'doughnut',
    data:{labels:[],datasets:[{data:[],backgroundColor:[],borderColor:'#131822',borderWidth:3}]},
    options:{cutout:'68%',plugins:{legend:{position:'bottom',labels:{color:'#c9d1d9',boxWidth:10,font:{size:11}}},tooltip:{callbacks:{label:c=>` ${c.label}: ${c.parsed.toFixed(1)} pts (${(c.parsed/ (pie._total||1) *100).toFixed(0)}%)`}}}}
  });
}
const HIST_MS=10*60*1000;
let histSeeded=false, histLastTs=0;
function seedHistory(){ // pull server-side 10-min window so a fresh page shows full history
  fetch('/gpu/history').then(r=>r.json()).then(h=>{
    allGpuHist={};
    for(const [k,s] of Object.entries(h.series||{})) allGpuHist[k]={t:[...s.t], v:[...s.v]};
    if(Object.keys(allGpuHist).length){
      histLastTs=Math.max(...Object.values(allGpuHist).map(s=>s.t[s.t.length-1]||0));
      drawAllGpuChart(null);
    }
    histSeeded=true;
  }).catch(()=>{histSeeded=true;});
}
function drawAllGpuChart(hosts){
  if(typeof Chart==='undefined')return;
  const now=Date.now();
  if(hosts){ // live tick: append sample for each online GPU
    for(const h of (hosts||[])){
      if(!h.online||!h.gpus) continue;
      for(const g of h.gpus){
        const k=`${h.name}·GPU${g.index}`;
        const s=allGpuHist[k]||(allGpuHist[k]={t:[],v:[]});
        s.t.push(now); s.v.push(+(g.memory_used_mb/1024).toFixed(1));
        while(s.t.length&&now-s.t[0]>HIST_MS){s.t.shift();s.v.shift();}
      }
    }
  }
  // drop series for GPUs not seen in the window
  for(const k of Object.keys(allGpuHist)){
    const s=allGpuHist[k];
    if(!s.t.length||now-s.t[s.t.length-1]>HIST_MS) delete allGpuHist[k];
  }
  const keys=Object.keys(allGpuHist);
  const labels=keys.length?allGpuHist[keys[0]].t.map(t=>new Date(t).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'})):[];
  const datasets=keys.map((k,i)=>({
    label:k, data:allGpuHist[k].v,
    borderColor:(colors[i%colors.length]||'#58a6ff'), backgroundColor:'transparent',
    borderWidth:1.6, pointRadius:0, tension:.25, fill:false}));
  if(!allGpuChart){
    allGpuChart=new Chart(document.getElementById('allgpu'),{
      type:'line',
      data:{labels,datasets},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:true,labels:{color:'#8b949e',boxWidth:10,font:{size:10}}}},
        scales:{x:{ticks:{color:'#8b949e',font:{size:9},maxTicksLimit:8},grid:{color:'#1f2632'}},
                y:{beginAtZero:true,ticks:{color:'#8b949e',font:{size:10},callback:v=>v+'G'},grid:{color:'#1f2632'},
                   title:{display:true,text:'VRAM used (GB)',color:'#8b949e',font:{size:10}}}}}
    });
  }else{
    allGpuChart.data.labels=labels;
    allGpuChart.data.datasets=datasets;
    allGpuChart.update('none');
  }
}
function tick(){
  fetch('/gpu/fleet').then(r=>r.json()).then(d=>{
    document.getElementById('err').textContent='';
    if(!pie && window.Chart) mkPie();
    // route banner
    const route=d.would_route_to||'(local fallback — routing off or no candidates)';
    const el=document.getElementById('route');
    const match=(d.hosts||[]).find(h=>route.includes(h.url&&!h.is_self?h.url:h.url));
    el.textContent=match? (match.name+(match.is_self?' (this host)':'')) : route;
    // pie of capacity share
    if(pie){
      const hosts=(d.hosts||[]).filter(h=>h.online&&h.score);
      pie._total=hosts.reduce((s,h)=>s+h.score.capacity,0)||1;
      pie.data.labels=hosts.map(h=>h.name);
      pie.data.datasets[0].data=hosts.map(h=>h.score.capacity);
      pie.data.datasets[0].backgroundColor=hosts.map((h,i)=>colors[i%colors.length]);
      const rr=hosts.map(h=>h.url===d.would_route_to);
      pie.update('none');
      document.getElementById('fleetcap').textContent=pie._total.toFixed(1);
      document.getElementById('why').textContent=hosts.map(h=>`${h.name} ${h.score.capacity.toFixed(1)}pts (free ${(h.score.free_mb/1024).toFixed(1)}GB, util ${h.score.avg_util}%, ${h.score.max_temp}°C)`).join(' · ');
    }
    drawAllGpuChart(d.hosts||[]);
    // detail table
    const tb=document.querySelector('#t tbody');tb.innerHTML='';
    for(const h of (d.hosts||[])){
      const gpus=h.gpus&&h.gpus.length?h.gpus:[{name:'—',utilization_pct:0,memory_used_mb:0,memory_total_mb:0,temperature_c:0}];
      for(const [i,g] of gpus.entries()){
        const tr=document.createElement('tr');
        const utilCls=g.utilization_pct>90?'c-hot':(g.utilization_pct>60?'c-warn':'c-ok');
        const tCls=g.temperature_c>=85?'c-hot':(g.temperature_c>=75?'c-warn':'c-ok');
        const tBad = h.mode==='push' ? ' <span class="hbadge" style="background:rgba(163,113,247,.18);color:#a371f7" title="push-mode: agent POSTs samples to hub">📡 PUSH</span>' : (h.is_self?' <span style="color:#58a6ff">·</span>':'');
        tr.innerHTML=`<td>${i===0?h.name+tBad:''}</td>
          <td>${g.name}</td>
          <td><span class="chip ${utilCls}">${g.utilization_pct}%</span><div class="utilbar"><i style="width:${g.utilization_pct}%"></i></div></td>
          <td>${(g.memory_used_mb/1024).toFixed(1)} / ${(g.memory_total_mb/1024).toFixed(1)} GB</td>
          <td><span class="chip ${tCls}">${g.temperature_c}°C</span></td>
          <td>${h.last_seen_s_ago!=null?h.last_seen_s_ago.toFixed(0)+'s':'—'}</td>`;
        tb.appendChild(tr);
      }
    }
  }).catch(e=>{document.getElementById('err').textContent='poll failed: '+e;});
}
function drawEvents(){
  fetch('/gpu/events?limit=40').then(r=>r.json()).then(d=>{
    const st=document.getElementById('evstats');
    const by=d.stats&&d.stats.by_upstream?d.stats.by_upstream:[];
    st.innerHTML=`<span class="score">total events: <b style="color:var(--fg)">${d.stats.total}</b></span>`+
      by.map(b=>`<span class="score"><span class="chip ${b.upstream&&b.upstream.includes('11435')?'c-ok':'c-warn'}">${b.upstream}</span> <b style="color:var(--fg)">${b.count}</b> reqs</span>`).join(' ')+
      (d.stats.avg_duration_ms?`<span class="score">avg request: <b style="color:var(--fg)">${(d.stats.avg_duration_ms/1000).toFixed(1)}s</b></span>`:'');
    const tb=document.querySelector('#ev tbody');
    tb.innerHTML=(d.events||[]).map(e=>{
      // toast watcher: fire once per event id
      const eid=e.id!=null?e.id:(e.ts_iso+e.kind);
      if(lastEventId && eid>lastEventId){
        const upRaw=String(e.chosen_upstream||'');
        const gpuHost=(e.why||'').trim()||upRaw.replace(/^https?:\/\//,'').replace(/:1143[45]$/,'');
        if(e.kind==='request'&&e.duration_ms!=null){
          toast('done',(e.model||e.path||'request')+' — done',
            `${((e.duration_ms/1000).toFixed(1))}s · GPU: ${gpuHost||'?'} · status ${e.status||e.status_code||'200'}`,7000);
          beep('done');
        } else if(e.kind==='request'){
          toast('',(e.model||e.path||'Inference request')+' — started',`GPU: ${gpuHost||'routing…'}`,5000);
          beep('start');
        } else if(e.kind==='error'){
          toast('err',(e.model||e.path||'request')+' — failed',(e.why||e.message||''),7000);
          beep('err');
        }
      }
      if(typeof eid==='number') lastEventId=Math.max(lastEventId,eid);
      const kindChip = e.kind==='routing'?'<span class="chip c-warn">routing</span>':
                      e.kind==='error'?'<span class="chip c-hot">error</span>':
                      '<span class="chip c-ok">request</span>';
      const up=e.chosen_upstream||'';
      const upShort=up.replace('http://','').replace(':11434','').replace(':11435','');
      const when=e.ts_iso?e.ts_iso.slice(11,19):'';
      return `<tr><td>${when}</td><td>${kindChip}</td>
        <td>${e.model?e.model:(e.path||'')}</td>
        <td>${upShort}</td>
        <td class="score" style="white-space:normal">${e.why||''}</td>
        <td>${e.status||''}</td>
        <td>${e.duration_ms!=null?(e.duration_ms/1000).toFixed(2)+'s':''}</td></tr>`;
    }).join('');
  }).catch(()=>{});
}
tick();setInterval(tick,3000);
seedHistory();
setInterval(drawEvents,5000);
window.addEventListener('load',drawEvents);
// ---- refresh button ----
const refreshBtn=document.getElementById('refreshbtn');
refreshBtn.addEventListener('click',()=>{
  refreshBtn.style.opacity='.5';refreshBtn.style.transform='rotate(180deg)';
  tick();drawEvents();
  const tab=document.querySelector('.tab.active').dataset.tab;
  if(tab==='models')drawModels();
  setTimeout(()=>{refreshBtn.style.opacity='1';refreshBtn.style.transform='rotate(0)'},400);
});
// ---- tabs ----
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabpane').forEach(p=>p.style.display='none');
  b.classList.add('active');
  const pane=document.getElementById('tab-'+b.dataset.tab); pane.style.display='block';
  if(b.dataset.tab==='models') drawModels();
  if(b.dataset.tab==='events') drawEvents();
  if(b.dataset.tab==='visualization') startVisualizationRefresh();
}));
// ---- models tab (single table, machine badge, sequential test-all) ----
let modelTimer=null, testAbort=false, modelsCache=null;
const esc=s=>String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const stripProto=u=>u.split('//')[1]||u;
function rowKey(host,model){return btoa(unescape(encodeURIComponent(host+'|'+model))).replace(/=/g,'');}
function currentPrompt(){
  const el=document.getElementById('testprompt');
  const v=el&&el.value.trim();
  return v||'Reply with exactly: OK';
}
function renderModels(){
  const q=((document.getElementById('modelsearch')||{}).value||'').toLowerCase().trim();
  const mf=(document.getElementById('machinefilter')||{}).value||'';
  const tb=document.querySelector('#mtab tbody');
  if(!tb||!modelsCache)return;
  const rows=[];
  (modelsCache.hosts||[]).forEach(h=>{
    (h.models||[]).forEach(m=>{
      if(mf && h.name!==mf)return;
      if(q && !m.name.toLowerCase().includes(q))return;
      rows.push({h,m});
    });
  });
  rows.sort((a,b)=>(a.h.name+a.m.name).localeCompare(b.h.name+b.m.name));
  tb.innerHTML=rows.map(({h,m})=>{
    const key=rowKey(h.url,m.name);
    const sr=m.success_rate, srColor=sr==null?'var(--muted)':(sr>=90?'var(--green)':sr>=70?'var(--warn,#e3b341)':'var(--red)');
    return `<tr data-row="${key}" data-host="${esc(h.url)}" data-model="${esc(m.name)}" data-machine="${esc(h.name)}">
      <td><span class="hbadge">${esc(h.name)}</span>${h.mode==='push'?' <span class="hbadge" style="background:rgba(163,113,247,.18);color:#a371f7" title="push-mode: agent POSTs samples to hub">📡 PUSH</span>':''}</td>
      <td class="mname-cell">${esc(m.name)}</td>
      <td class="mmeta">${esc(m.params||'?')} · ${esc(m.quant||'?')} · ${m.size_gb}GB</td>
      <td class="num">${m.total_calls||0}</td>
      <td class="num">${m.avg_ms!=null?(m.avg_ms/1000).toFixed(2)+'s':'—'}</td>
      <td class="num" style="color:${srColor}">${sr!=null?sr+'%':'—'}</td>
      <td><button class="mtest" onclick="testOne('${key}')">Test</button></td>
    </tr>
    <tr class="mresrow" id="rw-${key}"><td colspan="7"><div class="mres" id="res-${key}"></div></td></tr>`;
  }).join('') || `<tr><td colspan="7" class="mmeta" style="padding:1rem">No models match the filter.</td></tr>`;
  const pr=document.getElementById('testprogress');
  if(pr && !pr.dataset.busy) pr.textContent=rows.length+' model(s)';
  return rows;
}
async function drawModels(){
  try{
    const d=await (await fetch('/gpu/models')).json();
    modelsCache=d;
    const sel=document.getElementById('machinefilter');
    if(sel){
      const cur=sel.value;
      sel.innerHTML='<option value="">All machines</option>'+(d.hosts||[]).map(h=>`<option value="${esc(h.name)}">${esc(h.name)} (${esc(stripProto(h.url))})</option>`).join('');
      sel.value=cur;
    }
    renderModels();
  }catch(e){}
}
async function testOne(key){
  const tr=document.querySelector(`tr[data-row="${key}"]`);
  if(!tr) return;
  const host=tr.dataset.host, model=tr.dataset.model;
  const btn=tr.querySelector('.mtest');
  const res=document.getElementById('res-'+key);
  const prompt=currentPrompt();
  btn.disabled=true; btn.textContent='Testing…'; btn.classList.remove('ok','err');
  res.classList.add('show'); res.classList.remove('err','ok');
  res.innerHTML=`<div class="mres-head"><span class="spin">⏳</span> running…</div>`;
  try{
    const r=await fetch('/gpu/test',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({host, model, prompt})});
    const j=await r.json();
    const ts=new Date().toLocaleTimeString();
    const lat=j.latency_ms!=null?(j.latency_ms/1000).toFixed(2)+'s':'—';
    if(j.ok){
      btn.textContent='OK '+lat; btn.classList.add('ok');
      res.innerHTML=`<div class="mres-head"><span><b>${esc(model)}</b> @ ${esc(stripProto(host))} · ${lat} · ${ts}</span><span class="tag ok">200 OK</span></div>
        <div class="mres-io"><div class="mres-label">INPUT</div><div class="mres-text">${esc(prompt)}</div>
        <div class="mres-label">OUTPUT</div><div class="mres-reply">${esc(j.reply||'(empty reply)')}</div></div>`;
      res.classList.add('ok');
    }else{
      btn.textContent='Failed'; btn.classList.add('err');
      res.innerHTML=`<div class="mres-head"><span><b>${esc(model)}</b> @ ${esc(stripProto(host))} · ${lat} · ${ts}</span><span class="tag err">${j.capabilities?esc(j.capabilities.join(', ')):'error'}</span></div>
        <div class="mres-io"><div class="mres-label">INPUT</div><div class="mres-text">${esc(prompt)}</div>
        <div class="mres-label">ERROR</div><div class="mres-reply">${esc(j.error||'unknown error')}</div></div>`;
      res.classList.add('err');
    }
    return j.ok;
  }catch(e){
    btn.textContent='Failed'; btn.classList.add('err');
    res.innerHTML=`<div class="mres-head"><span>network error</span></div><div class="mres-reply">${esc(e)}</div>`;
    res.classList.add('show','err');
    return false;
  }finally{
    btn.disabled=false;
  }
}
async function testAll(){
  const btn=document.getElementById('testall');
  const pr=document.getElementById('testprogress');
  if(testAbort){ testAbort=false; btn.textContent='Test all'; return; }
  testAbort=false;
  const rows=[...document.querySelectorAll('#mtab tbody tr[data-row]')];
  if(!rows.length) return;
  btn.textContent='Stop'; btn.classList.add('busy');
  pr.dataset.busy='1';
  let ok=0, fail=0, i=0;
  for(const tr of rows){
    if(testAbort) break;
    i++;
    pr.textContent=`Testing ${i}/${rows.length} — ${tr.dataset.machine} ${tr.dataset.model} (${ok} ok, ${fail} failed)`;
    tr.scrollIntoView({block:'nearest'});
    const passed=await testOne(tr.dataset.row);
    passed?ok++:fail++;
    await new Promise(r=>setTimeout(r,150));
  }
  delete pr.dataset.busy;
  pr.textContent = testAbort ? `Stopped — ${ok} ok, ${fail} failed` : `Done — ${ok} ok, ${fail} failed of ${rows.length}`;
  btn.textContent='Test all'; btn.classList.remove('busy'); testAbort=false;
}
document.getElementById('modelsearch').addEventListener('input',()=>renderModels());
document.getElementById('machinefilter').addEventListener('change',()=>renderModels());
document.getElementById('testall').addEventListener('click',testAll);
// ---- Visualization Tab: Force-Directed Graph (models starfield + host routing) ----
const HOST_COLORS=['#58a6ff','#a371f7','#f0883e','#3fb950','#f778ba','#d29922'];
let visGraph=null, visSimulation=null;
async function initVisualization(){
  const svg=document.getElementById('visgraph');
  if(!svg||svg.tagName!=='svg') return;
  try{
    const resp=await fetch('/gpu/models');
    const data=await resp.json();
    drawVisualization(svg,data);
  }catch(e){ console.error('vis init',e); }
}
function drawVisualization(svg,data){
  if(!svg||!window.d3) return;
  const rect=svg.parentElement.getBoundingClientRect();
  const W=Math.max(760,rect.width-36), H=W*0.72;
  svg.setAttribute('width',W); svg.setAttribute('height',H);
  svg.innerHTML='';
  
  const hosts=(data&&data.hosts)?data.hosts:[];
  
  // Build nodes: 1 router hub + N hosts + all unique models as starfield
  const modelSet=new Set();
  hosts.forEach(h=>{(h.models||[]).forEach(m=>modelSet.add(m.name||''));});
  const models=Array.from(modelSet).slice(0,120); // cap at 120 for perf
  
  const nodes=[];
  const links=[];
  
  // Central router hub
  const router={id:'router',name:'router',type:'hub',calls:0,x:W/2,y:H/2,fx:W/2,fy:H/2};
  nodes.push(router);
  
  // Hosts
  const hostNodes=hosts.map((h,i)=>({
    id:h.name,name:h.name,type:'host',calls:h.models?.reduce((s,m)=>s+(m.total_calls||0),0)||0,
    color:HOST_COLORS[i%HOST_COLORS.length],size:h.models?.length||0
  }));
  nodes.push(...hostNodes);
  
  // Models as low-opacity background scatter
  const modelNodes=models.map((m,i)=>({
    id:m,name:m.split(':')[0],type:'model',calls:0,opacity:0.15
  }));
  nodes.push(...modelNodes);
  
  // Links: router↔hosts, hosts→models (weighted by call count)
  hostNodes.forEach(h=>{
    links.push({source:'router',target:h.id,type:'primary'});
    hosts.find(x=>x.name===h.id)?.models?.forEach(m=>{
      const idx=models.indexOf(m.name);
      if(idx>=0) links.push({source:h.id,target:m.name,type:'model',weight:Math.max(1,Math.log(1+(m.total_calls||0)))});
    });
  });
  
  // Force simulation
  const simulation=d3.forceSimulation(nodes)
    .force('link',d3.forceLink(links).id(d=>d.id).distance(d=>d.type==='primary'?100:120).strength(d=>d.type==='primary'?0.8:0.2))
    .force('charge',d3.forceManyBody().strength(-400).distanceMax(500))
    .force('collide',d3.forceCollide().radius(d=>d.type==='hub'?30:(d.type==='host'?25:12)))
    .force('center',d3.forceCenter(W/2,H/2).strength(0.4))
    .on('tick',()=>tick());
  visSimulation=simulation;
  
  // Canvas overlay for rendering (faster than SVG for dense graphs)
  const canvas=document.createElement('canvas');
  canvas.width=W; canvas.height=H;
  canvas.style.display='block'; canvas.style.position='absolute'; canvas.style.top='0'; canvas.style.left='0';
  svg.appendChild(canvas);
  const ctx=canvas.getContext('2d');
  
  function tick(){
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='rgba(0,0,0,0)'; ctx.fillRect(0,0,W,H);
    
    // Draw links first (behind nodes)
    links.forEach(l=>{
      const s=typeof l.source==='string'?nodes.find(n=>n.id===l.source):l.source;
      const t=typeof l.target==='string'?nodes.find(n=>n.id===l.target):l.target;
      if(!s||!t) return;
      ctx.beginPath(); ctx.moveTo(s.x,s.y); ctx.lineTo(t.x,t.y);
      if(l.type==='primary'){
        ctx.strokeStyle='rgba(255,255,255,0.2)'; ctx.lineWidth=2;
      }else{
        ctx.strokeStyle=`rgba(255,255,255,${0.05*(l.weight||1)})`; ctx.lineWidth=0.8;
      }
      ctx.stroke();
    });
    
    // Draw nodes
    nodes.forEach(n=>{
      if(n.type==='hub'){
        // Router hub: green glow
        const g=ctx.createRadialGradient(n.x,n.y,5,n.x,n.y,30);
        g.addColorStop(0,'rgba(76,175,80,0.3)'); g.addColorStop(1,'rgba(76,175,80,0)');
        ctx.fillStyle=g; ctx.beginPath(); ctx.arc(n.x,n.y,30,0,2*Math.PI); ctx.fill();
        ctx.fillStyle='#4caf50'; ctx.beginPath(); ctx.arc(n.x,n.y,16,0,2*Math.PI); ctx.fill();
        ctx.fillStyle='#fff'; ctx.font='700 11px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText('ROUTER',n.x,n.y);
      }else if(n.type==='host'){
        // Host nodes: colored
        ctx.fillStyle=n.color+'44'; ctx.beginPath(); ctx.arc(n.x,n.y,20,0,2*Math.PI); ctx.fill();
        ctx.strokeStyle=n.color; ctx.lineWidth=2; ctx.stroke();
        ctx.fillStyle='#fff'; ctx.font='600 9px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(n.name.slice(0,8),n.x,n.y-4);
        ctx.fillStyle='#aaa'; ctx.font='7px sans-serif'; ctx.fillText(n.size+' m',n.x,n.y+6);
      }else{
        // Models: tiny dim dots (starfield)
        const op=n.opacity||0.15;
        ctx.fillStyle=`rgba(200,200,200,${op})`; ctx.beginPath(); ctx.arc(n.x,n.y,3,0,2*Math.PI); ctx.fill();
      }
    });
  }
  tick(); // render initial layout
}
async function startVisualizationRefresh(){
  await initVisualization();
  setInterval(initVisualization,5000);
}
// ---- end visualization ----
</script>
<footer>
  <p>AI Fleet Load Balancer — <a href="https://beinno.com" target="_blank" rel="noopener noreferrer">beinno.com</a></p>
</footer>
</body></html>"""
