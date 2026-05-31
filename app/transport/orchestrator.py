"""The async shell: merge both sockets into one queue, run a single consumer over
CallSession, and execute the commands it returns.

The single consumer means handle() is never concurrent, so call state can't race.
Dispatch is serial, so a slow send on one socket can briefly delay the other; we
accept that and bound the queue so a stalled consumer can't grow memory.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import WebSocket

from app import config
from app.domain import commands as C
from app.domain import events as E
from app.domain.session import CallSession
from app.domain.state import CallState
from app.transport.openai_realtime import OpenAIRealtime
from app.transport.twilio_stream import TwilioMediaStream

_QUEUE_MAXSIZE = 512  # ~10s of 20ms frames


async def run_call(twilio_ws: WebSocket) -> None:
    await twilio_ws.accept()
    twilio = TwilioMediaStream(twilio_ws)
    try:
        async with OpenAIRealtime(
            api_key=config.require_api_key(),
            model=config.MODEL,
            voice=config.VOICE,
            instructions=config.build_instructions(),
        ) as openai:
            await openai.initialize_session()

            queue: asyncio.Queue[E.Event] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
            session = CallSession(greet_on_connect=config.GREET_ON_CONNECT)
            producers = [
                asyncio.create_task(_pump(twilio.events(), queue, E.CallerDisconnected())),
                asyncio.create_task(_pump(openai.events(), queue, E.OpenAIClosed())),
            ]
            try:
                await _drive(session, queue, twilio, openai)
            finally:
                for task in producers:
                    task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
    finally:
        await twilio.aclose()


async def _pump(source: AsyncIterator[E.Event], queue: asyncio.Queue, terminal: E.Event) -> None:
    # Emit the terminal when the source ends or errors so the consumer learns the
    # stream is gone, but not on cancellation, where we're already tearing down.
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
) -> None:
    while True:
        event = await queue.get()
        for command in session.handle(event):
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
