"""HTTP surface tests via FastAPI's TestClient (no real call, no OpenAI socket)."""
from fastapi.testclient import TestClient

import main


def test_app_is_fastapi():
    assert type(main.app).__name__ == "FastAPI"


def test_index_route_reports_running():
    with TestClient(main.app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Twilio Media Stream Server is running!"}


def test_incoming_call_returns_twiml_that_opens_the_stream():
    with TestClient(main.app, base_url="http://example.com") as client:
        resp = client.post("/incoming-call")
    assert resp.status_code == 200
    assert "<Connect>" in resp.text
    assert "wss://example.com/media-stream" in resp.text
