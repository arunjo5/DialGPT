"""The async shell: merge both sockets into one queue, run a single consumer over
CallSession, and execute the commands it returns.

The single consumer means handle() is never concurrent, so call state can't race.
Dispatch is serial, so a slow send on one socket can briefly delay the other; we
accept that and bound the queue so a stalled consumer can't grow memory.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import WebSocket

from app import config
from app.domain import commands as C
from app.domain import events as E
from app.domain.session import CallSession
from app.domain.state import CallState
from app.metrics import CALLS, RETRIEVAL_LATENCY, CallLatency
from app.rag.service import get_retriever
from app.transport.openai_realtime import OpenAIRealtime
from app.transport.twilio_stream import TwilioMediaStream

_QUEUE_MAXSIZE = 512  # ~10s of 20ms frames

SEARCH_DOCUMENT_TOOL = {
    "type": "function",
    "name": "search_document",
    "description": (
        "Search the caller's document for passages relevant to a question. "
        "Call this whenever the caller asks something the document might answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, usually the caller's question.",
            },
        },
        "required": ["query"],
    },
}


async def run_call(twilio_ws: WebSocket) -> None:
    await twilio_ws.accept()
    twilio = TwilioMediaStream(twilio_ws)
    monitor = CallLatency()
    CALLS.inc()
    retriever = get_retriever()
    has_document = bool(retriever and retriever.chunk_count)
    try:
        async with OpenAIRealtime(
            api_key=config.require_api_key(),
            model=config.MODEL,
            voice=config.VOICE,
            instructions=config.build_instructions(has_document),
            tools=[SEARCH_DOCUMENT_TOOL] if has_document else None,
        ) as openai:
            await openai.initialize_session()

            queue: asyncio.Queue[E.Event] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
            session = CallSession(greet_on_connect=config.GREET_ON_CONNECT)
            producers = [
                asyncio.create_task(_pump(twilio.events(), queue, E.CallerDisconnected())),
                asyncio.create_task(_pump(openai.events(), queue, E.OpenAIClosed())),
            ]
            try:
                await _drive(session, queue, twilio, openai, monitor)
            finally:
                for task in producers:
                    task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
    finally:
        monitor.log_summary()
        await twilio.aclose()


async def _pump(source: AsyncIterator[E.Event], queue: asyncio.Queue, terminal: E.Event) -> None:
    try:
        async for event in source:
            await queue.put(event)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass
    await queue.put(terminal)


async def _drive(
    session: CallSession,
    queue: asyncio.Queue,
    twilio: TwilioMediaStream,
    openai: OpenAIRealtime,
    observer: CallLatency | None = None,
) -> None:
    while True:
        event = await queue.get()
        if isinstance(event, E.AssistantToolCall):
            await _handle_tool_call(event, openai, get_retriever())
            continue
        commands = session.handle(event)
        if observer is not None:
            observer.observe(event, commands)
        for command in commands:
            await _dispatch(command, twilio, openai)
        if session.state is CallState.CLOSING:
            return


async def _dispatch(command: C.Command, twilio: TwilioMediaStream, openai: OpenAIRealtime) -> None:
    if isinstance(command, C.ForwardCallerAudio):
        await openai.forward_caller_audio(command.payload_b64)
    elif isinstance(command, C.PlayAudioToCaller):
        await twilio.play(command.payload_b64)
    elif isinstance(command, C.TruncateAssistant):
        await openai.truncate_assistant(command.item_id, command.audio_end_ms)
    elif isinstance(command, C.ClearCallerPlayback):
        await twilio.clear()
    elif isinstance(command, C.StartAssistantTurn):
        await openai.start_assistant_turn()
    elif isinstance(command, C.HangUp):
        pass


async def _handle_tool_call(event: E.AssistantToolCall, openai: OpenAIRealtime, retriever) -> None:
    query = _tool_query(event.arguments)
    if retriever is None or not query:
        await openai.submit_tool_result(event.call_id, "No document is available.")
        return
    start = time.monotonic()
    chunks = await retriever.search(query)
    RETRIEVAL_LATENCY.observe(time.monotonic() - start)
    output = "\n\n---\n\n".join(chunks) if chunks else "No relevant passages found."
    await openai.submit_tool_result(event.call_id, output)


def _tool_query(arguments: str) -> str:
    try:
        return json.loads(arguments).get("query", "")
    except (ValueError, AttributeError):
        return ""
