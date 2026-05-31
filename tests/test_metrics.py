"""CallLatency tests: the latency math, driven by a fake clock (no real time)."""
import pytest

from app.domain import commands as C
from app.domain import events as E
from app.metrics import CallLatency


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_response_latency_from_speech_stopped_to_first_audio():
    clk = _Clock()
    m = CallLatency(clock=clk)
    clk.t = 10.0
    m.observe(E.CallerSpeechStopped(), [])
    clk.t = 10.8
    m.observe(E.AssistantAudioDelta("A", "x"), [C.PlayAudioToCaller("x")])
    assert m.response_latencies == [pytest.approx(0.8)]


def test_only_first_audio_of_a_turn_is_measured():
    clk = _Clock()
    m = CallLatency(clock=clk)
    clk.t = 1.0
    m.observe(E.CallerSpeechStopped(), [])
    clk.t = 1.5
    m.observe(E.AssistantAudioDelta("A", "x"), [])
    clk.t = 1.9
    m.observe(E.AssistantAudioDelta("A", "y"), [])  # later audio is not re-counted
    assert m.response_latencies == [pytest.approx(0.5)]


def test_barge_in_latency_from_speech_started_to_truncate():
    clk = _Clock()
    m = CallLatency(clock=clk)
    clk.t = 5.0
    m.observe(E.CallerSpeechStarted(), [])
    clk.t = 5.4
    m.observe(
        E.CallerSpeechStopped(),
        [C.TruncateAssistant("A", 100), C.ClearCallerPlayback()],
    )
    assert m.barge_in_latencies == [pytest.approx(0.4)]


def test_greeting_audio_is_not_a_response_turn():
    clk = _Clock()
    m = CallLatency(clock=clk)
    clk.t = 2.0
    m.observe(E.AssistantAudioDelta("G", "greet"), [C.PlayAudioToCaller("greet")])
    assert m.response_latencies == []


def test_summary_is_quiet_with_no_data():
    CallLatency().log_summary()  # must not raise on an empty call
