"""Twilio Media Streams adapter: wire frames to and from domain events and commands.

Holds no domain state; it only caches the stream_sid for addressing outgoing frames.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.domain import events as E

_MARK_NAME = "responsePart"


class TwilioMediaStream:
    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._stream_sid: str | None = None

    async def events(self) -> AsyncIterator[E.Event]:
        try:
            async for message in self._ws.iter_text():
                data = json.loads(message)
                event = data.get("event")
                if event == "media":
                    media = data["media"]
                    yield E.CallerAudioReceived(
                        payload_b64=media["payload"],
                        timestamp_ms=int(media["timestamp"]),
                    )
                elif event == "start":
                    self._stream_sid = data["start"]["streamSid"]
                    yield E.CallStarted(stream_sid=self._stream_sid)
                elif event == "mark":
                    yield E.PlaybackMarkAcked(name=data["mark"]["name"])
        except WebSocketDisconnect:
            return

    async def play(self, payload_b64: str) -> None:
        if self._stream_sid is None:
            return
        await self._ws.send_json({
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {"payload": payload_b64},
        })
        await self._ws.send_json({
            "event": "mark",
            "streamSid": self._stream_sid,
            "mark": {"name": _MARK_NAME},
        })

    async def clear(self) -> None:
        if self._stream_sid is None:
            return
        await self._ws.send_json({"event": "clear", "streamSid": self._stream_sid})

    async def aclose(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass
