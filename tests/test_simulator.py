"""End-to-end pipeline tests via the in-process media-stream simulator (no network)."""
from tests.simulator import CallSimulator


async def test_session_is_configured_on_connect():
    sim = CallSimulator()
    async with sim.run():
        await sim.sent_to_openai(type="session.update")


async def test_caller_audio_is_forwarded_to_openai():
    sim = CallSimulator()
    async with sim.run():
        await sim.caller_start()
        await sim.caller_audio("AAAA", ts=1000)
        msg = await sim.sent_to_openai(type="input_audio_buffer.append")
        assert msg["audio"] == "AAAA"


async def test_assistant_audio_reaches_caller():
    sim = CallSimulator()
    async with sim.run():
        await sim.caller_start()
        await sim.assistant_audio("item1", "REPLY")
        frame = await sim.sent_to_caller(event="media")
        assert frame["media"]["payload"] == "REPLY"


async def test_sustained_barge_in_truncates_and_clears():
    sim = CallSimulator()
    async with sim.run():
        await sim.caller_start()
        await sim.caller_audio("a", ts=1000)
        await sim.assistant_audio("item1", "REPLY")
        await sim.sent_to_caller(event="media")
        await sim.caller_speech_started()
        await sim.caller_audio("b", ts=1500)
        await sim.caller_audio("c", ts=1900)  # caller spoke ~900ms over the assistant
        await sim.caller_speech_stopped()
        truncate = await sim.sent_to_openai(type="conversation.item.truncate")
        assert truncate["item_id"] == "item1"
        await sim.sent_to_caller(event="clear")


async def test_short_noise_does_not_interrupt():
    sim = CallSimulator()
    async with sim.run():
        await sim.caller_start()
        await sim.caller_audio("a", ts=1000)
        await sim.assistant_audio("item1", "REPLY")
        await sim.sent_to_caller(event="media")
        await sim.caller_speech_started()
        await sim.caller_audio("b", ts=1150)  # 150ms of speech, below the 300ms floor
        await sim.caller_speech_stopped()
        await sim.settle()
        assert not any(m.get("type") == "conversation.item.truncate" for m in sim.openai.sent)


async def test_greeting_fires_on_session_configured():
    sim = CallSimulator(greet_on_connect=True)
    async with sim.run():
        await sim.sent_to_openai(type="session.update")
        await sim.session_configured()
        created = await sim.sent_to_openai(type="conversation.item.create")
        assert "Greet the user" in created["item"]["content"][0]["text"]
        await sim.sent_to_openai(type="response.create")


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.chunk_count = len(chunks)

    async def search(self, query, k=None):
        return self._chunks


async def test_tool_call_returns_retrieved_passages():
    sim = CallSimulator(retriever=_FakeRetriever(["the answer is 42"]))
    async with sim.run():
        await sim.sent_to_openai(type="session.update")
        await sim.assistant_tool_call("call1", "what is the answer")
        out = await sim.sent_to_openai(type="conversation.item.create")
        assert out["item"]["type"] == "function_call_output"
        assert "42" in out["item"]["output"]
        await sim.sent_to_openai(type="response.create")
