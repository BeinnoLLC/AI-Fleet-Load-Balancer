# OpenJev-on-Ollama

Use your local [Ollama](https://ollama.com) models as an
[OpenJev](https://github.com/razorback16/openjev) "System One" decision server.

OpenJev answers small, structured questions about a piece of state — yes/no
(`noul`), pick-one (`choice`), rate-on-a-scale (`score`) — and returns calibrated
*probabilities*, not free text. Its own backends are vLLM (DiffusionGemma 26B,
needs ~24 GB+ VRAM) and Apple-silicon MLX. **There is no Ollama backend upstream.**

This project adds one. It is a small FastAPI server that speaks OpenJev's exact
wire contract (`POST /v1/systemone`, `GET /v1/models`) but answers each question
with a local Ollama model, reading the per-token `logprobs` Ollama returns on
`/api/chat` to turn "what would the model say" into a probability distribution.

Result: the official `typesafe-sdk` (and anything else that speaks the OpenJev
API) runs against your Ollama box, with no GPU upgrade and no model download.

## Why logprobs, not generated text

The contract asks for calibrated probabilities (`noul` -> a float in `[0,1]`,
`choice`/`score` -> a probability per label plus an expected score). Parsing
model prose ("I think it's probably billing") gives you none of that. Ollama's
`logprobs` on the first generated token give the answer-label mass directly, so
one forward pass per question yields a proper distribution. Verified working on
Ollama `0.12.x`+ on both `/api/chat` and `/v1/chat/completions`.

## Install

```bash
cd openjev-ollama
python3.12 -m venv .venv
.venv/bin/pip install fastapi httpx "uvicorn[standard]" typesafe-sdk
cp .env.example .env      # then edit if you want a different model/port
./run.sh                  # or: .venv/bin/python ollama_jev.py
```

Make sure Ollama is running and you have a model pulled:

```bash
ollama serve           # if not already running
ollama pull qwen2.5:7b
```

## Installed state on this machine

- Runs as a **systemd user service** (`openjev-ollama.service`, enabled, and
  lingering is on — it starts at boot and survives logouts).
- The **`jev` CLI** is installed at `~/.local/bin/jev`.
- Config lives in `~/workspace/openjev-ollama/.env`.

```bash
systemctl --user status openjev-ollama     # is it running?
systemctl --user restart openjev-ollama    # after editing .env
journalctl --user -u openjev-ollama -f     # logs
```

## The `jev` CLI (fastest way to use it)

```bash
jev health                                  # is the facade up?
jev models                                  # what the server advertises

jev ask "<text>"                            # billing? outage? tone?
jev ask "<text>" --urgency                  # + urgency score 0..4

# custom questions: any noul / choice / score block per question
jev custom "<text>" '{"refund": {"type": "noul",
  "instructions": "Does the customer want a refund?"}}'

jev chat                                    # interactive: paste context, then
                                            # ask yes/no questions about it
```

`jev ask` prints one line per question, e.g.:

```
billing: yes  (p=0.99997)
outage: no  (p=0.000003)
tone: angry  (conf 0.904493)
urgency: 3.999372 / 4  (conf 0.999372)
```

## Use it

Point the TypeSafe SDK at the facade and call it exactly as you would OpenJev:

```python
import os
os.environ["TYPESAFE_BASE_URL"] = "http://127.0.0.1:8080"
os.environ["TYPESAFE_API_KEY"] = "local-no-auth"   # SDK needs a key; facade ignores it

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    r = client.system_one(
        state="I was charged twice and I'm cancelling my subscription.",
        questions={
            "billing": Noul(instructions="Is this a billing problem?"),
            "tone": Choice(instructions="Tone?", criteria={
                "calm": "...", "angry": "...", "threatening": "..."}),
            "urgency": Score(instructions="How urgent?",
                             criteria=["none", "low", "moderate", "high", "critical"]),
        },
    )
    print(r.nouls["billing"].noul)          # 1.0
    print(r.choices["tone"].choice)         # 'threatening'
    print(r.scores["urgency"].score)        # 4.0
```

`demo_sdk.py` in this repo is that exact script, runnable:

```bash
.venv/bin/python demo_sdk.py
```

Raw HTTP works too:

```bash
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "The server is down for everyone.",
  "model": "openjev-latest",
  "questions": {"outage": {"type": "noul", "instructions": "Is this a service outage?"}}
}'
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Your Ollama server |
| `JEV_OLLAMA_MODEL` | `qwen2.5:7b` | Which local model answers |
| `JEV_HOST` / `JEV_PORT` | `127.0.0.1` / `8080` | Where the facade listens |
| `JEV_OLLAMA_CONCURRENCY` | `4` | Questions answered in parallel |
| `JEV_OLLAMA_TIMEOUT` | `180` | First-token read timeout (s) |
| `JEV_OLLAMA_TOP_LOGPROBS` | `20` | Candidate tokens read per question |

Advertised model names (all answer via `JEV_OLLAMA_MODEL`): `openjev-latest`,
`openjev-0.1`, `jev-latest`, `jev-preview`. `jev-latest` is the SDK's default, so
a bare `TypeSafeClient()` works.

## Run as a service

Already installed on this machine (see "Installed state" above). To do it by hand:

```bash
cp openjev-ollama.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now openjev-ollama
journalctl --user -u openjev-ollama -f
```

## Model choice

Any Ollama chat model works. Small instruction-tuned models are enough for
label picking and are fastest; larger ones are better calibrated on subtle
rubrics. On a 16 GB card this box runs `qwen2.5:7b` comfortably; `gemma3:4b` /
`llama3.2:3b` are snappier, `qwen3:14b` sharper if you have the room.

## Scope and caveats

- This is OpenJev's **System One API contract**, faithfully implemented, not
  OpenJev's DiffusionGemma/MLX models. You get the interface and calibrated
  label probabilities from a standard LLM — not the diffusion decoder's outputs.
- The facade ignores the `Authorization` header; it is meant for localhost or a
  trusted LAN. Do not expose it publicly without putting auth in front.
- Calibration quality is the underlying model's, not DiffusionGemma's; treat the
  numbers as model beliefs, and sanity-check on your own data.
