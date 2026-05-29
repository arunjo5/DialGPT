# Twilio + OpenAI Realtime Voice Assistant

Connects a Twilio phone number to the OpenAI Realtime API so callers can talk to a GPT voice model. Handles interruptions, replies in the caller's language, and can answer from a PDF you provide.

## How it works

When someone calls, Twilio sends a POST to `/incoming-call`, and the app answers with TwiML that opens a WebSocket from Twilio to `/media-stream`. The app forwards the caller's audio over a second WebSocket to OpenAI's Realtime API and streams the reply back. OpenAI tracks when the caller starts and stops talking, so it knows when to answer. If the caller speaks while the assistant is talking, the app cuts the reply off.

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

## Config

Set in `.env`:

- `OPENAI_API_KEY` (required)
- `PORT` (default 5050)
- `PDF_PATH` (optional grounding doc, ignored if the file is missing)
