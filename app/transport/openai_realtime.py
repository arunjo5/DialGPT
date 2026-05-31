"""OpenAI Realtime API adapter: connection plus wire events and commands."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import websockets

from app import config
from app.domain import events as E

_LOG_EVENT_TYPES = {
    "error", "response.content.done", "rate_limits.updated", "response.done",
    "input_audio_buffer.committed", "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started", "session.created", "session.updated",
}


class OpenAIRealtime:
    """Owns the OpenAI realtime WebSocket. Use as an async context manager."""

    def __init__(self, *, api_key: str, model: str, voice: str, instructions: str) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._ws = None

    async def __aenter__(self) -> "OpenAIRealtime":
        self._ws = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={self._model}",
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def initialize_session(self) -> None:
        await self._send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self._model,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "turn_detection": {"type": "server_vad"},
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": self._voice,
                    },
                },
                "instructions": self._instructions,
            },
        })

    async def start_assistant_turn(self) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": config.GREETING_INSTRUCTION}],
            },
        })
        await self._send({"type": "response.create"})

    async def forward_caller_audio(self, payload_b64: str) -> None:
        if self._ws is None or self._ws.state.name != "OPEN":
            return
        await self._send({"type": "input_audio_buffer.append", "audio": payload_b64})

    async def truncate_assistant(self, item_id: str, audio_end_ms: int) -> None:
        await self._send({
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms,
        })

    async def events(self) -> AsyncIterator[E.Event]:
        try:
            async for message in self._ws:
                data = json.loads(message)
                etype = data.get("type")
                if etype in _LOG_EVENT_TYPES:
                    print(f"Received event: {etype}", data)
                if etype == "response.output_audio.delta" and "delta" in data:
                    yield E.AssistantAudioDelta(item_id=data.get("item_id"), payload_b64=data["delta"])
                elif etype == "response.done":
                    yield E.AssistantResponseDone()
                elif etype == "input_audio_buffer.speech_started":
                    yield E.CallerSpeechStarted()
                elif etype == "input_audio_buffer.speech_stopped":
                    yield E.CallerSpeechStopped()
                elif etype == "session.updated":
                    yield E.SessionConfigured()
        except websockets.ConnectionClosed:
            return

    async def _send(self, payload: dict) -> None:
        await self._ws.send(json.dumps(payload))
