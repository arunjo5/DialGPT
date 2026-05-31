"""Orchestrator (imperative shell) tests: async, but no network.

In-memory fake adapters record dispatched commands; scripted async generators
stand in for the sockets. Covers the consumer loop, dispatch routing, terminal-
event injection, and the cancellation path.
"""
import asyncio

import pytest

from app.domain import commands as C
from app.domain import events as E
from app.domain.session import CallSession
from app.domain.state import CallState
from app.transport import orchestrator as orch


class FakeTwilio:
    def __init__(self):
        self.calls = []

    async def play(self, payload_b64):
        self.calls.append(("play", payload_b64))

    async def clear(self):
        self.calls.append(("clear",))

    async def aclose(self):
        self.calls.append(("aclose",))


class FakeOpenAI:
    def __init__(self):
        self.calls = []

    async def forward_caller_audio(self, payload_b64):
        self.calls.append(("forward", payload_b64))

    async def truncate_assistant(self, item_id, audio_end_ms):
        self.calls.append(("truncate", item_id, audio_end_ms))

    async def start_assistant_turn(self):
        self.calls.append(("start_turn",))

    async def submit_tool_result(self, call_id, output):
        self.calls.append(("tool_result", call_id, output))


async def _scripted(events):
    for event in events:
        yield event


async def test_dispatch_routes_each_command_to_the_right_adapter():
    twilio, openai = FakeTwilio(), FakeOpenAI()
    await orch._dispatch(C.ForwardCallerAudio("aaa"), twilio, openai)
    await orch._dispatch(C.PlayAudioToCaller("bbb"), twilio, openai)
    await orch._dispatch(C.TruncateAssistant("X", 42), twilio, openai)
    await orch._dispatch(C.ClearCallerPlayback(), twilio, openai)
    await orch._dispatch(C.StartAssistantTurn(), twilio, openai)
    await orch._dispatch(C.HangUp(), twilio, openai)  # no-op
    assert ("forward", "aaa") in openai.calls
    assert ("truncate", "X", 42) in openai.calls
    assert ("start_turn",) in openai.calls
    assert ("play", "bbb") in twilio.calls
    assert ("clear",) in twilio.calls


async def test_consumer_drives_session_and_stops_on_closing():
    twilio, openai = FakeTwilio(), FakeOpenAI()
    session = CallSession(greet_on_connect=True)
    queue = asyncio.Queue()
    for event in (E.SessionConfigured(), E.CallerAudioReceived("p", 100), E.CallerDisconnected()):
        queue.put_nowait(event)
    await orch._drive(session, queue, twilio, openai)
    assert session.state is CallState.CLOSING
    assert ("start_turn",) in openai.calls
    assert ("forward", "p") in openai.calls


async def test_full_barge_in_flow_through_the_shell():
    twilio, openai = FakeTwilio(), FakeOpenAI()
    session = CallSession(greet_on_connect=False)
    queue = asyncio.Queue()
    script = [
        E.CallerAudioReceived("c", 1000),
        E.AssistantAudioDelta("A", "x"),   # SPEAKING, response_start=1000, marks=1
        E.CallerAudioReceived("c", 1500),
        E.CallerSpeechStarted(),
        E.CallerAudioReceived("c", 1900),
        E.CallerSpeechStopped(),           # sustained barge-in -> truncate + clear
        E.CallerDisconnected(),
    ]
    for event in script:
        queue.put_nowait(event)
    await orch._drive(session, queue, twilio, openai)
    assert ("truncate", "A", 900) in openai.calls
    assert ("clear",) in twilio.calls
    assert session.state is CallState.CLOSING


async def test_pump_appends_terminal_on_normal_end():
    queue = asyncio.Queue()
    source = _scripted([E.CallerAudioReceived("p", 1), E.CallerAudioReceived("p", 2)])
    await orch._pump(source, queue, E.CallerDisconnected())
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert len(drained) == 3
    assert isinstance(drained[-1], E.CallerDisconnected)


async def test_pump_appends_terminal_even_when_source_errors():
    async def boom():
        yield E.CallerAudioReceived("p", 1)
        raise RuntimeError("socket exploded")

    queue = asyncio.Queue()
    await orch._pump(boom(), queue, E.OpenAIClosed())
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert isinstance(drained[-1], E.OpenAIClosed)


async def test_pump_cancelled_does_not_emit_terminal():
    async def forever():
        while True:
            await asyncio.sleep(0.005)
            yield E.CallerAudioReceived("p", 1)

    queue = asyncio.Queue()
    task = asyncio.create_task(orch._pump(forever(), queue, E.CallerDisconnected()))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert all(isinstance(i, E.CallerAudioReceived) for i in drained)  # no terminal


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    async def search(self, query, k=None):
        return self._chunks


async def test_tool_call_runs_retrieval_and_submits_result():
    openai = FakeOpenAI()
    retriever = FakeRetriever(["passage about cats", "more cat facts"])
    await orch._handle_tool_call(
        E.AssistantToolCall(call_id="c1", name="search_document", arguments='{"query": "cats"}'),
        openai,
        retriever,
    )
    submitted = [c for c in openai.calls if c[0] == "tool_result"]
    assert submitted and submitted[0][1] == "c1" and "passage about cats" in submitted[0][2]


async def test_tool_call_without_retriever_returns_placeholder():
    openai = FakeOpenAI()
    await orch._handle_tool_call(
        E.AssistantToolCall(call_id="c2", name="search_document", arguments='{"query": "x"}'),
        openai,
        None,
    )
    assert ("tool_result", "c2", "No document is available.") in openai.calls
