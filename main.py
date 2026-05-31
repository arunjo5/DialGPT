"""FastAPI entrypoint. Routes delegate to the app package; kept at the repo root
so `uvicorn main:app` is unchanged.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse

from app import config
from app.transport.orchestrator import run_call
from app.twiml import build_incoming_call_twiml


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.require_api_key()  # fail fast at startup, not at import
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    twiml = build_incoming_call_twiml(
        request.url.hostname, greet_on_connect=config.GREET_ON_CONNECT
    )
    return HTMLResponse(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await run_call(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
