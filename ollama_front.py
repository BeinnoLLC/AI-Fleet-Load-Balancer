"""ollama_front.py — put OpenJev "System One" BEHIND the Ollama endpoint.

Agents face ONE endpoint (this server, :11434) exactly as they do today. Real
Ollama runs just behind it on :11435. This front:

  * injects an `openjev:latest` tag into GET /api/tags  -> shows in `ollama list`
    and in GET /v1/models, side by side with every real model;
  * when a request selects that tag, answers it with the OpenJev decision engine
    (the same engine as ollama_jev.py, backed by JEV_OLLAMA_MODEL via Ollama);
  * passes EVERY other request straight through to real Ollama, streaming and
    all, so nothing else changes for existing clients.

So an agent keeps using http://<host>:11434 and simply adds one more model id:

      openjev:latest          (aliases: openjev, jev:latest, systemone:latest)

OpenJev-native callers can also POST the full {state, questions} body to
/v1/systemone through this same port.
"""
from __future__ import annotations

import json
import os
import socket
import time

import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

from gpu_monitor import FleetMonitor, read_local_gpus, DASHBOARD_HTML

# Ollama real server (the thing we front). The front owns :11434.
UPSTREAM = os.environ.get("OLLAMA_UPSTREAM", "http://127.0.0.1:11435").rstrip("/")
# --- GPU-aware fleet routing -------------------------------------------- #
# Plain (non-strata/non-jev) chat/generate/embeddings calls are routed to
# whichever host in fleet_hosts.json currently has the most GPU headroom.
# Add a new machine to that JSON file — no code or env change needed, no
# restart required; it's picked up within one poll cycle. Agents always
# talk to *this* front; they never need to know other hosts exist.
ENABLE_GPU_ROUTE = os.environ.get("ENABLE_GPU_ROUTE", "0") == "1"
FLEET_HOSTS_FILE = os.environ.get(
    "FLEET_HOSTS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet_hosts.json"),
)
FLEET_SELF_NAME = os.environ.get("FLEET_SELF_NAME") or None
GPU_POLL_INTERVAL = float(os.environ.get("GPU_POLL_INTERVAL", "5"))
GPU_STALE_AFTER = float(os.environ.get("GPU_STALE_AFTER", "30"))
# the model id the fleet advertises for the decision engine
JEV_TAG = os.environ.get("JEV_OLLAMA_TAG", "openjev:latest")
# every name an agent may use for the engine (all served by JEV_OLLAMA_MODEL)
JEV_ALIASES = {
    JEV_TAG,
    "openjev",
    "openjev:latest",
    "jev:latest",
    "systemone:latest",
    "systemone",
}
JEV_MODEL = os.environ.get("JEV_OLLAMA_MODEL", "qwen2.5:7b")
# Strata big-model server (OpenAI-compatible on :8090). The front advertises
# it as `strata:latest` so agents call it like any Ollama model.
def _strata_upstream_default():
    # 1) env var, 2) override file (works under systemd units that cannot read ~/.env), 3) local default
    import os as _os
    try:
        with open(_os.path.expanduser("~/workspace/openjev-ollama/strata_upstream.txt")) as _f:
            _u = _f.read().strip()
            if _u:
                return _u.rstrip("/")
    except OSError:
        pass
    return "http://127.0.0.1:8090"

STRATA_UPSTREAM = os.environ.get("STRATA_UPSTREAM", _strata_upstream_default()).rstrip("/")
STRATA_MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next")
STRATA_TAG = os.environ.get("STRATA_OLLAMA_TAG", "strata:latest")
STRATA_ALIASES = {STRATA_TAG, "strata", "strata:latest"}
# question set used when an agent calls the engine *as a chat model* (no explicit
# questions). JSON object of OpenJev Questions; override via env.
DEFAULT_QUESTIONS = {
    "verdict": {
        "type": "noul",
        "instructions": "Decide whether the content satisfies the request or is true.",
        "criteria": {"true": "it does", "false": "it does not"},
    }
}

HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
       "upgrade", "proxy-authenticate", "proxy-authorization", "te", "trailer"}


def _load_questions() -> dict:
    raw = os.environ.get("JEV_DEFAULT_QUESTIONS")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return DEFAULT_QUESTIONS


def _is_strata_model(name: str | None) -> bool:
    return bool(name) and name.strip().lower() in {a.lower() for a in STRATA_ALIASES}

def _is_jev_model(name: str | None) -> bool:
    return bool(name) and name.strip().lower() in {a.lower() for a in JEV_ALIASES}


def _last_user_text(payload: dict) -> str:
    """State = the last user message (chat) or the prompt (generate)."""
    msgs = payload.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") in (None, "user"):
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return " ".join(
                        p.get("text", "") for p in c if isinstance(p, dict)
                    ).strip()
    for key in ("prompt", "input", "state"):
        v = payload.get(key)
        if isinstance(v, str):
            return v
    return ""


def _build_app() -> FastAPI:
    # import the proven engine (it reads OLLAMA_URL, so point it at the upstream)
    os.environ.setdefault("OLLAMA_URL", UPSTREAM)
    from ollama_jev import build_prompt, ollama_first_token, score_question  # type: ignore

    app = FastAPI(title="OpenJev-behind-Ollama", version="0.1.0")
    state: dict = {}
    from event_log import EventLog
    events = EventLog()

    @app.on_event("startup")
    async def _up():
        state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        state["sem"] = asyncio.Semaphore(int(os.environ.get("JEV_OLLAMA_CONCURRENCY", "4")))
        if ENABLE_GPU_ROUTE:
            state["gpu"] = FleetMonitor(FLEET_HOSTS_FILE, self_name=FLEET_SELF_NAME,
                                         poll_interval=GPU_POLL_INTERVAL, stale_after=GPU_STALE_AFTER)
            await state["gpu"].start(state["client"])
        else:
            state["gpu"] = None

    @app.on_event("shutdown")
    async def _down():
        if state.get("gpu"):
            await state["gpu"].stop()
        await state["client"].aclose()
        events.close()

    def _pick_upstream() -> str:
        """Which host's front should answer the next plain request.

        Returns UPSTREAM (this host's raw Ollama, :11435) for local, or the
        best peer FRONT url (:11434) for a remote host — the peer's own
        front then recurses this same decision on its end, so routing never
        needs this host to know the peer's raw-Ollama port."""
        t0 = time.monotonic()
        gpu = state.get("gpu")
        if gpu is None:
            events.log("routing", chosen_upstream="local",
                       why="gpu routing disabled (ENABLE_GPU_ROUTE=0)")
            return UPSTREAM
        best_url = gpu.pick_host()
        snap = gpu.snapshot()
        scores = {h["name"]: h.get("score") for h in snap if h.get("score")}
        if best_url is None:
            events.log("routing", chosen_upstream="local", scores=scores,
                       why="no valid candidates (all stale/offline) — failing soft to local")
            return UPSTREAM
        # If the winning host is this machine, use the local raw upstream
        # directly rather than looping back through our own front.
        for h in snap:
            if h["url"] == best_url and h.get("is_self"):
                why = (f"local wins: capacity {h['score']['capacity']} pts "
                       f"(free {h['score']['free_mb']/1024:.1f}GB, util {h['score']['avg_util']}%, "
                       f"max {h['score']['max_temp']}°C) — {h['score'].get('gpu_count', 0)} GPU(s)")
                events.log("routing", chosen_upstream="local", why=why,
                           route_latency_ms=round((time.monotonic() - t0) * 1000, 1), scores=scores)
                return UPSTREAM
        winner = next((h for h in snap if h["url"] == best_url), {})
        ws = winner.get("score") or {}
        ls = scores.get(socket.gethostname()) or scores.get(FLEET_SELF_NAME or "") or {}
        why = (f"remote wins: {winner.get('name', best_url)} capacity {ws.get('capacity')} pts "
               f"(free {ws.get('free_mb', 0)/1024:.1f}GB, util {ws.get('avg_util')}%, "
               f"max {ws.get('max_temp')}°C) vs local {ls.get('capacity', 'n/a')} pts")
        events.log("routing", chosen_upstream=best_url, why=why,
                   route_latency_ms=round((time.monotonic() - t0) * 1000, 1), scores=scores)
        return best_url

    async def _run_engine(payload: dict) -> dict:
        """Answer DEFAULT_QUESTIONS about the request's state; return answers."""
        client = state["client"]
        sem = state["sem"]
        st = _last_user_text(payload)
        questions = _load_questions()

        async def one(name: str, qd: dict):
            prompt = build_prompt(qd, st)
            async with sem:
                r = await ollama_first_token(client, JEV_MODEL, prompt)
            return name, score_question(qd, r["top_logprobs"])

        results = await asyncio.gather(*(one(n, q) for n, q in questions.items()))
        return dict(results)

    def _render(answers: dict) -> str:
        """Human+JSON summary an agent reads as the assistant message."""
        lines = []
        for name, a in answers.items():
            if a.get("type") == "noul":
                p = a["noul"]
                lines.append(f"{name}: {'yes' if p >= 0.5 else 'no'} (p={p:.3f})")
            elif a.get("type") == "choice":
                lines.append(f"{name}: {a['choice']} (conf={a['confidence']:.3f})")
            else:
                lines.append(f"{name}: score={a['score']:.3f} (conf={a['confidence']:.3f})")
        return "\n".join(lines) + "\n\n" + json.dumps({"openjev": answers}, ensure_ascii=False)


    async def _strata_chat(payload: dict, want_model: str, stream: bool, openai: bool):
        """Forward a chat payload to the Strata server (OpenAI-compatible)."""
        t0 = time.monotonic()
        if not state.get("client"):
            state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        body = {
            "model": STRATA_MODEL,
            "messages": payload.get("messages", []),
            "stream": bool(stream),
        }
        # tools/tool_choice must pass through or agents get "no tool available" replies
        for k in ("temperature", "top_p", "max_tokens", "stop", "seed", "tools", "tool_choice",
                  "parallel_tool_calls", "response_format", "stream_options"):
            if payload.get(k) is not None:
                body[k] = payload[k]
        # Ollama-style options object -> flat OpenAI fields
        opts = payload.get("options") or {}
        if opts.get("temperature") is not None: body["temperature"] = opts["temperature"]
        if opts.get("top_p") is not None: body["top_p"] = opts["top_p"]
        if opts.get("num_predict") is not None: body["max_tokens"] = opts["num_predict"]
        headers = {"content-type": "application/json"}
        url = f"{STRATA_UPSTREAM}/v1/chat/completions"
        if not stream:
            try:
                r = await state["client"].post(url, json=body, headers=headers)
                r.raise_for_status()
            except Exception as e:
                msg = f"strata server unreachable at {STRATA_UPSTREAM}: {type(e).__name__}"
                if openai:
                    return JSONResponse({"error": {"message": msg, "type": "strata_down"}},
                                        status_code=502)
                return JSONResponse({"error": msg, "model": want_model}, status_code=502)
            data = r.json()
            events.log("request", path="/v1/chat/completions" if openai else "/api/chat",
                       model=want_model, stream=False, chosen_upstream="strata",
                       status=r.status_code,
                       duration_ms=round((time.monotonic() - t0) * 1000, 1))
            if openai:
                data["model"] = want_model
                return JSONResponse(data)
            # OpenAI -> Ollama /api/chat shape
            msg = (data.get("choices") or [{}])[0].get("message", {})
            out_msg = {"role": msg.get("role", "assistant"), "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                tcs = []
                for tc in msg["tool_calls"]:
                    fn = dict(tc.get("function") or {})
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                        except ValueError:
                            pass
                    tcs.append({"function": fn})
                out_msg["tool_calls"] = tcs
            return JSONResponse({
                "model": want_model,
                "created_at": "2026-01-01T00:00:00Z",
                "message": out_msg,
                "done": True,
                "done_reason": "stop",
                "total_duration": data.get("usage", {}).get("total_tokens", 0),
            })
        # streaming: re-label chunks so the agent still sees `strata:latest`
        try:
            req = state["client"].build_request("POST", url, json=body, headers=headers)
            resp = await state["client"].send(req, stream=True)
        except Exception as e:
            msg = f"strata server unreachable at {STRATA_UPSTREAM}: {type(e).__name__}"
            return JSONResponse({"error": {"message": msg, "type": "strata_down"}}, status_code=502)

        async def sse():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                events.log("request", path="/v1/chat/completions" if openai else "/api/chat",
                           model=want_model, stream=True, chosen_upstream="strata",
                           status=resp.status_code,
                           duration_ms=round((time.monotonic() - t0) * 1000, 1))

        return StreamingResponse(sse(), status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type", "text/event-stream"),
                                 headers={"x-strata-relabeled": want_model})

    # ------------------------- model discovery ------------------------- #
    @app.get("/api/tags")
    async def tags():
        try:
            r = await state["client"].get(f"{UPSTREAM}/api/tags")
            data = r.json()
        except Exception:
            data = {"models": []}
        models = data.get("models") or []
        if not any((m.get("name") or m.get("model")) == STRATA_TAG for m in models):
            models.append({
                "name": STRATA_TAG,
                "model": STRATA_TAG,
                "modified_at": "2026-01-01T00:00:00Z",
                "size": 0,
                "digest": "strata0big0model0server",
                "details": {"family": "strata", "parameter_size": "125B-moe",
                            "quantization_level": "gsq-rco"},
            })
        if not any((m.get("name") or m.get("model")) == JEV_TAG for m in models):
            models.append({
                "name": JEV_TAG,
                "model": JEV_TAG,
                "modified_at": "2026-01-01T00:00:00Z",
                "size": 0,
                "digest": "openjev0system0one0decision0engine",
                "details": {"family": "openjev", "parameter_size": f"backed by {JEV_MODEL}",
                            "quantization_level": "system-one"},
            })
        data["models"] = models
        return JSONResponse(data)

    @app.get("/v1/models")
    async def v1_models():
        out = []
        try:
            r = await state["client"].get(f"{UPSTREAM}/v1/models")
            out = (r.json() or {}).get("data", []) or []
        except Exception:
            pass
        ids = {m.get("id") for m in out}
        out.append({"id": STRATA_TAG, "object": "model", "created": 1767225600, "owned_by": "strata"})
        for alias in ("openjev:latest", "openjev", "jev:latest", "systemone:latest"):
            if alias not in ids:
                out.append({"id": alias, "object": "model", "created": 1767225600, "owned_by": "openjev"})
        return {"object": "list", "data": out}

    @app.get("/api/show")
    async def show(name: str = ""):
        if _is_jev_model(name):
            return {
                "license": "MIT",
                "modelfile": f"# OpenJev System One, backed by {JEV_MODEL}",
                "details": {"family": "openjev", "parameter_size": f"backed by {JEV_MODEL}",
                            "quantization_level": "system-one"},
                "capabilities": ["completion"],
            }
        r = await state["client"].get(f"{UPSTREAM}/api/show", params={"name": name})
        return JSONResponse(r.json(), status_code=r.status_code)

    @app.get("/api/version")
    async def version():
        try:
            r = await state["client"].get(f"{UPSTREAM}/api/version")
            return r.json()
        except Exception:
            return {"version": "openjev-front"}

    # ------------------------- engine, as a model ------------------------- #
    def _chat_reply(answers: dict, model_echo: str, stream: bool, openai: bool):
        content = _render(answers)
        if stream:
            if openai:
                def gen():
                    yield ("data: " + json.dumps({
                        "id": "chatcmpl-openjev", "object": "chat.completion.chunk",
                        "created": 1767225600, "model": model_echo,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                                     "finish_reason": None}],
                    }) + "\n\n").encode()
                    yield ("data: " + json.dumps({
                        "id": "chatcmpl-openjev", "object": "chat.completion.chunk",
                        "created": 1767225600, "model": model_echo,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }) + "\n\n").encode()
                    yield b"data: [DONE]\n\n"
                return StreamingResponse(gen(), media_type="text/event-stream")
            def gen2():
                yield (json.dumps({"model": JEV_TAG, "created_at": "2026-01-01T00:00:00Z",
                                   "message": {"role": "assistant", "content": content},
                                   "done": False}) + "\n").encode()
                yield (json.dumps({"model": JEV_TAG, "created_at": "2026-01-01T00:00:00Z",
                                   "message": {"role": "assistant", "content": ""},
                                   "done": True, "done_reason": "stop"}) + "\n").encode()
            return StreamingResponse(gen2(), media_type="application/x-ndjson")
        if openai:
            return JSONResponse({
                "id": "chatcmpl-openjev", "object": "chat.completion", "created": 1767225600,
                "model": model_echo,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        return JSONResponse({
            "model": JEV_TAG, "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": content}, "done": True,
            "done_reason": "stop",
        })

    @app.post("/api/chat")
    async def api_chat(request: Request):
        payload = await request.json()
        if _is_strata_model(payload.get("model")):
            return await _strata_chat(payload, STRATA_TAG, bool(payload.get("stream")), openai=False)
        if _is_jev_model(payload.get("model")):
            answers = await _run_engine(payload)
            return _chat_reply(answers, JEV_TAG, bool(payload.get("stream")), openai=False)
        return await _proxy(request)

    @app.post("/api/generate")
    async def api_generate(request: Request):
        payload = await request.json()
        if _is_strata_model(payload.get("model")):
            if not payload.get("messages") and payload.get("prompt") is not None:
                payload = dict(payload)
                payload["messages"] = [{"role": "user", "content": payload["prompt"]}]
            return await _strata_chat(payload, STRATA_TAG, bool(payload.get("stream")), openai=False)
        if _is_jev_model(payload.get("model")):
            answers = await _run_engine(payload)
            return _chat_reply(answers, JEV_TAG, bool(payload.get("stream")), openai=False)
        return await _proxy(request)

    @app.post("/v1/chat/completions")
    async def v1_chat(request: Request):
        payload = await request.json()
        if _is_strata_model(payload.get("model")):
            return await _strata_chat(payload, payload.get("model"), bool(payload.get("stream")), openai=True)
        if _is_jev_model(payload.get("model")):
            answers = await _run_engine(payload)
            return _chat_reply(answers, payload.get("model"), bool(payload.get("stream")), openai=True)
        return await _proxy(request)

    @app.post("/v1/systemone")
    async def v1_systemone(request: Request):
        """OpenJev-native: full {state, model, questions} answered by the engine.

        If the body names a real upstream decision model (e.g. clef, clef-flash
        — Ollama's own decision-capable models), pass it straight through
        so Ollama's native /v1/systemone answers it. Only a model absent from
        the alias set falls back to this engine."""
        payload = await request.json()
        if payload.get("model") and not _is_jev_model(payload.get("model")):
            return await _proxy(request)
        state_in = payload.get("state", "")
        questions = payload.get("questions") or _load_questions()
        client = state["client"]
        sem = state["sem"]

        async def one(name: str, qd: dict):
            prompt = build_prompt(qd, state_in)
            async with sem:
                r = await ollama_first_token(client, JEV_MODEL, prompt)
            return name, score_question(qd, r["top_logprobs"]), r["usage"]

        results = await asyncio.gather(*(one(n, q) for n, q in questions.items()))
        answers = {n: a for n, a, _ in results}
        return {
            "model": "openjev-0.1",
            "answers": answers,
            "usage": {"input_tokens": sum(u["input"] for _, _, u in results),
                      "output_tokens": sum(u["output"] for _, _, u in results)},
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "upstream": UPSTREAM, "jev_tag": JEV_TAG}

    @app.get("/gpu/status")
    async def gpu_status():
        """Raw local GPU snapshot — what a peer front polls to route traffic here."""
        gpus = await read_local_gpus()
        return {"host": socket.gethostname(), "gpus": gpus}

    @app.get("/gpu/fleet")
    async def gpu_fleet():
        """Agent/human-facing view of every host in fleet_hosts.json and current routing."""
        gpu = state.get("gpu")
        if gpu is None:
            local_gpus = await read_local_gpus()
            return {"routing_enabled": False,
                    "hosts": [{"name": socket.gethostname(), "url": None, "is_self": True,
                                "online": True, "last_seen_s_ago": 0, "gpus": local_gpus,
                                "score": None}],
                    "would_route_to": None}
        return {
            "routing_enabled": True,
            "hosts": gpu.snapshot(),
            "would_route_to": gpu.pick_host(),
        }

    @app.get("/gpu/history")
    async def gpu_history():
        """Server-side 10-min GPU memory-usage history — keeps recording with the page closed."""
        gpu = state.get("gpu")
        if gpu is None:
            return {"window_s": 600, "series": {}}
        return gpu.history_snapshot()

    @app.get("/gpu/dashboard")
    async def gpu_dashboard():
        """Live HTML page: every fleet host, online/offline, GPUs, utilization."""
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/gpu/events")
    async def gpu_events(limit: int = 50, kind: str = ""):
        """Recent request/routing events from the SQLite log."""
        rows = events.recent(limit=min(limit, 500), kind=kind or None)
        return JSONResponse({"stats": events.stats(), "events": rows})

    async def _fetch_tags(url: str):
        try:
            r = await state["client"].get(f"{url}/api/tags")
            r.raise_for_status()
            return url, r.json().get("models", [])
        except Exception as exc:
            return url, []

    @app.get("/gpu/models")
    async def gpu_models():
        """Models available on every fleet host (aggregate of /api/tags)."""
        gpu = state.get("gpu")
        urls = [h["url"] for h in gpu.snapshot()] if gpu else [f"http://127.0.0.1:{os.environ.get('JEV_PORT','11434')}"]
        import asyncio as _aio
        pairs = await _aio.gather(*[_fetch_tags(u) for u in urls])
        snap = gpu.snapshot() if gpu else []
        names = {h["url"]: h["name"] for h in snap}
        selfurls = {h["url"] for h in snap if h.get("is_self")}
        hosts = []
        for url, models in pairs:
            hosts.append({"url": url, "name": names.get(url, url), "is_self": url in selfurls,
                          "models": [{"name": m.get("name"), "size_gb": round((m.get("size") or 0)/1e9, 1),
                                      "params": (m.get("details") or {}).get("parameter_size"),
                                      "quant": (m.get("details") or {}).get("quantization_level"),
                                      "family": (m.get("details") or {}).get("family")} for m in models]})
        # Per-model request trends + avg response time from event log
        recent = events.recent(limit=200, kind="request")
        model_stats = {}
        for e in recent:
            if e.get("kind") == "test":
                continue  # dashboard smoke-tests must not pollute traffic stats
            if e.get("model"):
                key = e["model"]
                ms = e.get("duration_ms") or 0
                s = model_stats.setdefault(key, {"count": 0, "dur_sum": 0.0, "dur_n": 0, "ok": 0})
                s["count"] += 1
                if e.get("duration_ms") and e.get("status") == 200:
                    s["dur_sum"] += e["duration_ms"]; s["dur_n"] += 1
                if e.get("status") == 200:
                    s["ok"] += 1
        for h in hosts:
            for m in h["models"]:
                s = model_stats.get(m["name"])
                m["request_count_20"] = s["count"] if s else 0
                m["total_calls"] = s["count"] if s else 0
                m["avg_ms"] = round(s["dur_sum"]/s["dur_n"], 0) if s and s["dur_n"] else None
                m["success_rate"] = round(100*s["ok"]/s["count"]) if s and s["count"] else None
        return JSONResponse({"hosts": hosts})

    @app.post("/gpu/test")
    async def gpu_test(request: Request):
        """Quick smoke-test: send 'Reply with OK only' to a model on a host.

        Falls back chat -> generate; if the model supports neither (decision/
        embedding-only models), probes /api/show and reports capabilities.
        """
        try:
            p = await request.json()
        except Exception:
            return JSONResponse({"error": "bad json"}, status_code=400)
        host = p.get("host"); model = p.get("model")
        prompt = (p.get("prompt") or "Reply with exactly: OK")[:2000]
        if not host or not model:
            return JSONResponse({"error": "host and model required"}, status_code=400)
        base = host.rstrip("/")
        t0 = time.monotonic()
        reply = None
        err = None
        status = None
        # 1) /api/chat
        try:
            r = await state["client"].post(base + "/api/chat", json={
                "model": model, "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"num_predict": 128}})
            status = r.status_code
            if r.status_code < 400:
                j = r.json()
                msg = j.get("message") or {}
                reply = msg.get("content") or ""
                if msg.get("thinking"):
                    reply = (msg["thinking"] + ("\n---\n" if reply else "")) + reply
            else:
                err = r.text[:300]
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        # 2) /api/generate fallback
        if reply is None:
            try:
                r = await state["client"].post(base + "/api/generate", json={
                    "model": model, "stream": False, "prompt": prompt,
                    "options": {"num_predict": 128}})
                status = r.status_code
                if r.status_code < 400:
                    reply = r.json().get("response", "")
                else:
                    err = r.text[:300]
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        dur = round((time.monotonic() - t0) * 1000, 1)
        # 3) neither endpoint works -> capability probe
        if reply is None:
            caps = None
            try:
                r = await state["client"].post(base + "/api/show",
                                               json={"model": model})
                if r.status_code < 400:
                    caps = r.json().get("capabilities")
            except Exception:
                pass
            events.log("test", path="/gpu/test", model=model,
                       client=request.client.host if request.client else None,
                       chosen_upstream=host, status=status or 0,
                       prompt=prompt, reply=None, duration_ms=dur,
                       why=(err or "")[:500])
            return JSONResponse({
                "ok": False, "host": host, "model": model,
                "capabilities": caps,
                "error": (f"model does not support chat/generate"
                          + (f" (capabilities: {', '.join(caps)})" if caps else "")
                          + (f" — {err}" if err else "")),
                "latency_ms": dur})
        ok = bool(reply.strip())
        events.log("test", path="/gpu/test", model=model,
                   client=request.client.host if request.client else None,
                   chosen_upstream=host, status=status, prompt=prompt,
                   reply=reply[:4000], duration_ms=dur)
        return JSONResponse({"ok": ok, "host": host, "model": model,
                             "reply": reply, "latency_ms": dur})

    # ------------------------- transparent passthrough ------------------------- #
    async def _proxy(request: Request) -> StreamingResponse:
        path = request.url.path
        t0 = time.monotonic()
        routed = path in ("/api/chat", "/api/generate", "/v1/chat/completions",
                          "/v1/completions", "/v1/embeddings", "/api/embed",
                          "/api/embeddings")
        target = _pick_upstream() if routed else UPSTREAM
        url = f"{target}{path}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        body = await request.body()
        model = None
        stream = False
        if routed:
            with_suppress = None
            try:
                with_suppress = json.loads(body)
                model = with_suppress.get("model")
                stream = bool(with_suppress.get("stream"))
            except Exception:
                pass
        req = state["client"].build_request(
            request.method, url, headers=headers, content=body,
            params=dict(request.query_params),
        )
        status = None
        try:
            resp = await state["client"].send(req, stream=True)
            status = resp.status_code
            fwd = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}
            out = StreamingResponse(resp.aiter_raw(), status_code=resp.status_code,
                                    headers=fwd, media_type=resp.headers.get("content-type"))
            if routed:
                events.log("request", path=path, model=model, stream=stream,
                           client=request.client.host if request.client else None,
                           chosen_upstream=target,
                           status=status,
                           duration_ms=round((time.monotonic() - t0) * 1000, 1))
            return out
        except Exception as exc:
            if routed:
                events.log("error", path=path, model=model, client=request.client.host if request.client else None,
                           chosen_upstream=target, why=f"proxy failure: {type(exc).__name__}: {exc}",
                           duration_ms=round((time.monotonic() - t0) * 1000, 1))
            raise

    @app.api_route("/{path:path}",
                   methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catchall(path: str, request: Request):
        return await _proxy(request)

    return app


app = _build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("JEV_HOST", "0.0.0.0"),
                port=int(os.environ.get("JEV_PORT", "11434")), log_level="warning")
