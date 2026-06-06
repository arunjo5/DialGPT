"""In-process media-stream simulator: drives both wire protocols against the real
run_call pipeline with fake sockets, so the whole call path is testable with no
network or keys. Driver methods auto-settle, so events process in call order.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from app import config
from app.rag.service import set_retriever
from app.transport.orchestrator import run_call

_SENTINEL = object()
_SETTLE = 0.01


class FakeTwilioWebSocket:
    """Stands in for the Starlette WebSocket that TwilioMediaStream wraps."""

    def __init__(self) -> None:
        self._inbound: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []

    async def accept(self) -> None:
        pass

    async def iter_text(self):
        while True:
            item = await self._inbound.get()
            if item is _SENTINEL:
                return
            yield item

    async def send_json(self, obj: dict) -> None:
        self.sent.append(obj)

    async def close(self) -> None:
        self._inbound.put_nowait(_SENTINEL)

    def push(self, frame: dict) -> None:
        self._inbound.put_nowait(json.dumps(frame))

    def end(self) -> None:
        self._inbound.put_nowait(_SENTINEL)


class FakeOpenAIConnection:
    """Stands in for the websockets client connection OpenAIRealtime wraps."""

    def __init__(self) -> None:
        self._inbound: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.state = SimpleNamespace(name="OPEN")

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._inbound.get()
        if item is _SENTINEL:
            raise StopAsyncIteration
        return item

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.state = SimpleNamespace(name="CLOSED")
        self._inbound.put_nowait(_SENTINEL)

    def push(self, event: dict) -> None:
        self._inbound.put_nowait(json.dumps(event))

    def end(self) -> None:
        self._inbound.put_nowait(_SENTINEL)


class CallSimulator:
    def __init__(self, *, greet_on_connect: bool = False, retriever=None) -> None:
        self._greet = greet_on_connect
        self._retriever = retriever
        self.twilio = FakeTwilioWebSocket()
        self.openai = FakeOpenAIConnection()
        self._task: asyncio.Task | None = None
        self._saved_greet = config.GREET_ON_CONNECT

    async def _connect(self, *args, **kwargs):
        return self.openai

    @asynccontextmanager
    async def run(self):
        self._saved_greet = config.GREET_ON_CONNECT
        config.GREET_ON_CONNECT = self._greet
        set_retriever(self._retriever)
        self._task = asyncio.create_task(run_call(self.twilio, openai_connect=self._connect))
        try:
            await self.settle()
            yield self
        finally:
            self.twilio.end()
            self.openai.end()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            config.GREET_ON_CONNECT = self._saved_greet
            set_retriever(None)

    async def settle(self) -> None:
        await asyncio.sleep(_SETTLE)

    async def caller_start(self, stream_sid: str = "STREAM") -> None:
        self.twilio.push({"event": "start", "start": {"streamSid": stream_sid}})
        await self.settle()

    async def caller_audio(self, payload: str, ts: int) -> None:
        self.twilio.push({"event": "media", "media": {"payload": payload, "timestamp": str(ts)}})
        await self.settle()

    async def caller_mark(self, name: str = "responsePart") -> None:
        self.twilio.push({"event": "mark", "mark": {"name": name}})
        await self.settle()

    async def session_configured(self) -> None:
        self.openai.push({"type": "session.updated"})
        await self.settle()

    async def assistant_audio(self, item_id: str, payload: str) -> None:
        self.openai.push({"type": "response.output_audio.delta", "item_id": item_id, "delta": payload})
        await self.settle()

    async def assistant_done(self) -> None:
        self.openai.push({"type": "response.done"})
        await self.settle()

    async def caller_speech_started(self) -> None:
        self.openai.push({"type": "input_audio_buffer.speech_started"})
        await self.settle()

    async def caller_speech_stopped(self) -> None:
        self.openai.push({"type": "input_audio_buffer.speech_stopped"})
        await self.settle()

    async def assistant_tool_call(self, call_id: str, query: str) -> None:
        self.openai.push({
            "type": "response.function_call_arguments.done",
            "call_id": call_id,
            "name": "search_document",
            "arguments": json.dumps({"query": query}),
        })
        await self.settle()

    async def sent_to_openai(self, *, type: str, timeout: float = 1.0) -> dict:
        return await self._await_match(
            self.openai.sent, lambda m: m.get("type") == type, timeout, f"openai message type={type}"
        )

    async def sent_to_caller(self, *, event: str, timeout: float = 1.0) -> dict:
        return await self._await_match(
            self.twilio.sent, lambda m: m.get("event") == event, timeout, f"caller frame event={event}"
        )

    async def _await_match(self, log, predicate, timeout, what):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            for item in log:
                if predicate(item):
                    return item
            if self._task is not None and self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc is not None:
                    raise exc  # surface a run_call crash instead of timing out
            if loop.time() >= deadline:
                raise AssertionError(f"never saw {what}; saw {log}")
            await asyncio.sleep(0.005)
