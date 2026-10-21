"""OpenJev-compatible "System One" decision server backed by a local Ollama server.

OpenJev (https://github.com/razorback16/openjev) exposes POST /v1/systemone and
GET /v1/models and answers `noul` (yes/no), `choice` and `score` questions about a
piece of state. Its own backends are vLLM/MLX DiffusionGemma and small encoder
heads -- there is no Ollama backend upstream. This module keeps the exact same
wire contract but answers each question with a local Ollama model, using the
per-token logprobs Ollama returns on /api/chat.

Why logprobs and not generated text: the contract wants calibrated probabilities
(a float in [0,1] for noul, a probability per choice/level, and an expected score),
not a parsed string. Ollama gives us the top_logprobs for the first generated
token, so we can read the probability mass on each answer label directly.

Contract (TypeSafe OpenAPI 0.2.0), so the TypeSafe SDKs work by pointing
TYPESAFE_BASE_URL at this server:

    POST /v1/systemone
      { "state": <str|obj|list>, "model": <str>, "questions": {name: Question} }
      -> { "model": str, "answers": {name: Answer},
           "usage": {"input_tokens": int, "output_tokens": int} }

    Question: {type: "noul",  instructions, criteria:{true,false}}
              {type: "choice", instructions, criteria:{label: desc, ...}}
              {type: "score",  instructions, criteria:[desc, ...]}   # ordered, 0-based

    Answer:   {type: "noul",  noul: float}
              {type: "choice", choice: str, confidence: float, probabilities: {label: float}}
              {type: "score",  score: float, confidence: float,
                               legend: {str(level): desc}, probabilities: {str(level): float}}

Everything is read from the first generated token's logprobs, so a whole request
costs one forward pass per question and never depends on parsing model prose.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from typing import Annotated, Any, Literal, Union

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
# default model used when the request asks for an alias this server advertises
DEFAULT_MODEL = os.environ.get("JEV_OLLAMA_MODEL", "qwen2.5:7b")
# how many questions to run against Ollama at once (Ollama serializes per model,
# but concurrent calls still overlap queueing/loading and different models in parallel)
MAX_CONCURRENCY = int(os.environ.get("JEV_OLLAMA_CONCURRENCY", "4"))
# a single first-token read; generous enough for a cold model load on the first call
REQUEST_TIMEOUT = float(os.environ.get("JEV_OLLAMA_TIMEOUT", "180"))
TOP_LOGPROBS = int(os.environ.get("JEV_OLLAMA_TOP_LOGPROBS", "20"))

JSONContent = Union[str, dict[str, Any], list[Any]]
Described = Union[str, dict[str, Any], list[Any], None]


# --------------------------------------------------------------------------- #
# Request models (mirror OpenJev's api.py so validation errors look the same)
# --------------------------------------------------------------------------- #
class NoulCriteria(BaseModel):
    true: Described = None
    false: Described = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Described = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Described = None
    criteria: dict[str, Described]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Described = None
    criteria: list[JSONContent] = Field(min_length=1)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)


def as_text(value: Described) -> str:
    """Described fields may be a string, a nested object or a list; flatten to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def state_text(state: JSONContent) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #
def build_prompt(q: dict, state: JSONContent) -> str:
    """Ask the model to reply with exactly one label token for this question."""
    kind = q["type"]
    instr = as_text(q.get("instructions")) or {
        "noul": "Decide whether the statement is true.",
        "choice": "Select the best matching option.",
        "score": "Rate the content using the levels.",
    }[kind]
    parts = [instr, ""]
    if kind == "noul":
        crit = q.get("criteria") or {}
        t, f = as_text(crit.get("true")), as_text(crit.get("false"))
        if t:
            parts.append(f"yes = {t}")
        if f:
            parts.append(f"no = {f}")
        parts.append("Answer only yes or no.")
        labels = "yes / no"
    elif kind == "choice":
        parts.append("Options:")
        for name, desc in q["criteria"].items():
            d = as_text(desc)
            parts.append(f"  {name}: {d}" if d else f"  {name}")
        labels = " / ".join(q["criteria"].keys())
        parts.append("Answer only with one option name.")
    else:  # score
        parts.append("Levels (0-based):")
        for i, desc in enumerate(q["criteria"]):
            d = as_text(desc)
            parts.append(f"  {i}: {d}" if d else f"  {i}")
        labels = " / ".join(str(i) for i in range(len(q["criteria"])))
        parts.append("Answer only with one level number.")
    parts += ["", "Content:", state_text(state), "", "Answer:"]
    return "\n".join(parts)


async def ollama_first_token(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    """One deterministic forward pass; return the top-logprob list for token #1."""
    body = {
        "model": model,
        "stream": False,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "options": {"temperature": 0, "num_predict": 1, "seed": 0},
        "messages": [{"role": "user", "content": prompt}],
    }
    r = await client.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data.get("logprobs"):
        raise RuntimeError(
            f"Ollama returned no logprobs for model {model!r}; "
            "this server needs logprobs support (Ollama >= 0.12.11)."
        )
    step = data["logprobs"][0]
    usage = {"input": data.get("prompt_eval_count") or 0, "output": data.get("eval_count") or 0}
    return {"top_logprobs": step.get("top_logprobs", []), "usage": usage}


# --------------------------------------------------------------------------- #
# Scoring: turn the first-token distribution into contract answers
# --------------------------------------------------------------------------- #
def _tok_pieces(top: list[dict]) -> list[tuple[str, float]]:
    out = []
    for t in top:
        tok = (t.get("token") or "").strip().lower()
        if tok:
            out.append((tok, float(t.get("logprob", -100.0))))
    return out


def _normalize(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not pairs:
        return []
    m = max(p for _, p in pairs)
    weights = [(k, math.exp(p - m)) for k, p in pairs]
    total = sum(w for _, w in weights) or 1.0
    return [(k, w / total) for k, w in weights]


def answer_noul(top: list[dict]) -> dict:
    pairs = _tok_pieces(top)
    yes = sum(w for k, w in _normalize(pairs) if k.startswith("y"))
    no = sum(w for k, w in _normalize(pairs) if k.startswith("n"))
    if yes + no == 0:  # model answered neither: fall back to a neutral read
        return {"type": "noul", "noul": 0.5}
    return {"type": "noul", "noul": round(yes / (yes + no), 6)}


def answer_choice(q: dict, top: list[dict]) -> dict:
    names = list(q["criteria"].keys())
    pairs = _normalize(_tok_pieces(top))
    probs: dict[str, float] = {}
    for name in names:
        key = name.strip().lower()
        # full-name match plus first-token/prefix match for multi-word names
        mass = sum(w for k, w in pairs if k == key or (len(k) >= 2 and (key.startswith(k) or k.startswith(key))))
        probs[name] = mass
    total = sum(probs.values())
    if total <= 0:
        probs = {n: 1.0 / len(names) for n in names}
    else:
        probs = {n: v / total for n, v in probs.items()}
    best = max(probs, key=probs.get)
    return {
        "type": "choice",
        "choice": best,
        "confidence": round(probs[best], 6),
        "probabilities": {n: round(v, 6) for n, v in probs.items()},
    }


def answer_score(q: dict, top: list[dict]) -> dict:
    levels = [str(i) for i in range(len(q["criteria"]))]
    pairs = _normalize(_tok_pieces(top))
    probs = {}
    for lvl in levels:
        mass = sum(w for k, w in pairs if k == lvl)
        probs[lvl] = mass
    total = sum(probs.values())
    if total <= 0:
        probs = {lvl: 1.0 / len(levels) for lvl in levels}
    else:
        probs = {lvl: v / total for lvl, v in probs.items()}
    expected = sum(int(lvl) * p for lvl, p in probs.items())
    legend = {str(i): as_text(d) for i, d in enumerate(q["criteria"])}
    best = max(probs, key=probs.get)
    return {
        "type": "score",
        "score": round(expected, 6),
        "confidence": round(probs[best], 6),
        "legend": legend,
        "probabilities": {lvl: round(p, 6) for lvl, p in probs.items()},
    }


def score_question(q: dict, top: list[dict]) -> dict:
    kind = q["type"]
    if kind == "noul":
        return answer_noul(top)
    if kind == "choice":
        return answer_choice(q, top)
    return answer_score(q, top)


# --------------------------------------------------------------------------- #
# HTTP app
# --------------------------------------------------------------------------- #
def advertised_models() -> list[dict]:
    """Names clients can ask for. Aliases map to DEFAULT_MODEL."""
    return [
        {"name": "openjev-latest", "description": f"Ollama-backed System One ({DEFAULT_MODEL})", "release_date": "2026-01-01"},
        {"name": "openjev-0.1", "description": f"Ollama-backed System One ({DEFAULT_MODEL})", "release_date": "2026-01-01"},
        {"name": "jev-latest", "description": "Alias for openjev-latest (TypeSafe SDK default)", "release_date": "2026-01-01"},
        {"name": "jev-preview", "description": "Alias for openjev-latest (TypeSafe SDK default)", "release_date": "2026-01-01"},
    ]


def create_app() -> FastAPI:
    app = FastAPI(title="OpenJev-on-Ollama", version="0.1.0")
    names = {m["name"] for m in advertised_models()}

    @app.on_event("startup")
    async def _startup():
        app.state.client = httpx.AsyncClient()
        app.state.sem = asyncio.Semaphore(MAX_CONCURRENCY)

    @app.on_event("shutdown")
    async def _shutdown():
        await app.state.client.aclose()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"models": advertised_models()}

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest, request: Request):
        if req.model not in names:
            return JSONResponse(
                {"detail": {"error_type": "api_usage_error", "message": f"Unknown model: {req.model}"}},
                status_code=400,
            )
        model = DEFAULT_MODEL
        client: httpx.AsyncClient = request.app.state.client
        sem: asyncio.Semaphore = request.app.state.sem

        async def one(name: str, q: Question) -> tuple[str, dict, dict]:
            qd = q.model_dump()
            prompt = build_prompt(qd, req.state)
            async with sem:
                r = await ollama_first_token(client, model, prompt)
            return name, score_question(qd, r["top_logprobs"]), r["usage"]

        try:
            results = await asyncio.gather(*(one(n, q) for n, q in req.questions.items()))
        except httpx.HTTPError as e:
            return JSONResponse(
                {"detail": {"error_type": "api_error", "message": f"Ollama unavailable: {type(e).__name__}"}},
                status_code=503, headers={"retry-after": "2"},
            )
        except Exception as e:  # noqa: BLE001 - surface any scoring/back end failure
            return JSONResponse(
                {"detail": {"error_type": "api_error", "message": f"{type(e).__name__}: {e}"}},
                status_code=503,
            )

        answers = {n: a for n, a, _ in results}
        input_tokens = sum(u["input"] for _, _, u in results)
        output_tokens = sum(u["output"] for _, _, u in results)
        return {
            "model": "openjev-0.1",
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("JEV_HOST", "127.0.0.1"),
                port=int(os.environ.get("JEV_PORT", "8080")))
