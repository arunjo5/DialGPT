# DialGPT

Connects a Twilio phone number to the OpenAI Realtime API so callers can talk to a GPT voice model. Handles interruptions (barge-in), replies in the caller's language, and can answer from a PDF you provide.

## How it works

When someone calls, Twilio sends a POST to `/incoming-call`, and the app answers with TwiML that opens a WebSocket from Twilio to `/media-stream`. From there a single call is driven by a small state machine split across two layers:

- **`app/domain/`** is a pure, synchronous core: a `CallSession` finite state machine (`GREETING → LISTENING → SPEAKING → INTERRUPTED → CLOSING`) plus the barge-in timing logic. It does no I/O. `handle(event) -> [command]` takes a transport-agnostic event and returns intents.
- **`app/transport/`** holds the I/O: a Twilio adapter and an OpenAI adapter (each translating wire frames to/from events and commands), and an orchestrator that merges both sockets into one `asyncio.Queue` and runs a single consumer over the session.

Because one object owns all call state and a single consumer mutates it, the turn-taking logic is race-free by construction and unit-testable without a network or a phone. Server-side VAD drives turn detection; if the caller talks over the assistant for long enough, the app truncates the in-flight reply and clears buffered audio.

## Layout

```
app/
  domain/      # pure FSM + barge-in, no I/O: state.py, events.py, commands.py, session.py
  transport/   # Twilio + OpenAI adapters and the orchestrator (all the async I/O)
  config.py    # env + prompt assembly
  twiml.py     # inbound-call TwiML
  pdf.py       # optional grounding-doc loader
main.py        # thin FastAPI entrypoint (main:app)
tests/         # pure domain tests + async orchestrator tests + route tests
```

## Setup

Needs Python 3.11+, an OpenAI API key with Realtime access, a Twilio voice number, and ngrok (or another tunnel).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

## Run

```bash
./run.sh
```

Then expose it with `ngrok http 5050` and point the number's "A Call Comes In" webhook at `https://<ngrok-domain>/incoming-call` (POST). Call the number and talk.

If `--reload` throws `ModuleNotFoundError`, run with the venv Python directly: `.venv/bin/python -m uvicorn main:app --reload` (what `run.sh` does).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The domain tests exercise the state machine and the 300ms/200ms barge-in debounce with no network. The orchestrator tests cover the consumer loop and teardown using in-memory fake adapters.

## Metrics

Prometheus latency metrics are exposed at `/metrics`:

- `voice_response_latency_seconds`: caller stops speaking to first assistant audio
- `voice_barge_in_latency_seconds`: caller barge-in start to assistant cut off
- `voice_calls_total`, `voice_turns_total`, `voice_barge_ins_total`

Each call also logs a one-line latency summary when it ends.

## Config

Set in `.env`:

- `OPENAI_API_KEY` (required)
- `PORT` (default 5050)
- `PDF_PATH` (optional grounding doc, ignored if the file is missing)
- `GREET_ON_CONNECT` (default true; assistant speaks first. Set false for caller-first)
- `OPENAI_MODEL` (default gpt-realtime), `VOICE` (default alloy)
