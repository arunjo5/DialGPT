"""Barge-in / interruption debounce tests (300ms caller, 200ms assistant).

These are the hard real-time-timing rules, now testable with plain function
calls, with no event loop, sockets, or phone.
"""
from app.domain import commands as C
from app.domain import events as E
from app.domain.session import (
    MIN_ASSISTANT_STARTED_MS,
    MIN_CALLER_SPEECH_MS,
    CallSession,
)
from app.domain.state import CallState


def _speaking_at(s: CallSession, t: int) -> None:
    """Drive a fresh session into SPEAKING with the assistant turn started at time ``t``."""
    s.handle(E.CallerAudioReceived("p", t))         # clock = t
    s.handle(E.AssistantAudioDelta("A", "audio"))   # response_start = t, marks = 1, SPEAKING


def test_sustained_speech_interrupts():
    s = CallSession(greet_on_connect=False)
    _speaking_at(s, 1000)                            # assistant turn starts at 1000
    s.handle(E.CallerAudioReceived("p", 1500))       # clock 1500 (assistant elapsed 500 > 200)
    s.handle(E.CallerSpeechStarted())                # caller starts at 1500
    s.handle(E.CallerAudioReceived("p", 1900))       # clock 1900 (caller spoke 400 >= 300)
    out = s.handle(E.CallerSpeechStopped())
    assert out == [C.TruncateAssistant("A", 900), C.ClearCallerPlayback()]
    assert s.state is CallState.LISTENING
    assert s.outstanding_marks == 0
    assert s.last_assistant_item is None
    assert s.response_start_timestamp_ms is None


def test_short_noise_does_not_interrupt():
    s = CallSession(greet_on_connect=False)
    _speaking_at(s, 1000)
    s.handle(E.CallerAudioReceived("p", 1500))
    s.handle(E.CallerSpeechStarted())                # caller starts 1500
    s.handle(E.CallerAudioReceived("p", 1650))       # caller spoke 150 < 300
    out = s.handle(E.CallerSpeechStopped())
    assert out == []
    assert s.state is CallState.SPEAKING
    assert s.last_assistant_item == "A"


def test_assistant_just_started_no_interrupt():
    s = CallSession(greet_on_connect=False)
    # Caller starts speaking BEFORE the assistant audio, so the assistant's
    # elapsed time stays at the 200ms boundary even though the caller spoke long.
    s.handle(E.CallerAudioReceived("p", 900))
    s.handle(E.CallerSpeechStarted())                # caller starts 900
    s.handle(E.CallerAudioReceived("p", 1200))       # clock 1200
    s.handle(E.AssistantAudioDelta("A", "audio"))    # response_start = 1200
    s.handle(E.CallerAudioReceived("p", 1400))       # clock 1400
    out = s.handle(E.CallerSpeechStopped())
    # caller_duration = 500 >= 300, but elapsed = 1400-1200 = 200, not > 200
    assert out == []
    assert s.state is CallState.SPEAKING


def test_no_speech_start_marker_ignored():
    s = CallSession(greet_on_connect=False)
    _speaking_at(s, 1000)
    s.handle(E.CallerAudioReceived("p", 2000))
    out = s.handle(E.CallerSpeechStopped())          # caller_speech_start is None
    assert out == []
    assert s.state is CallState.SPEAKING


def test_speech_stopped_in_listening_never_truncates():
    s = CallSession(greet_on_connect=False)          # LISTENING, no audio in flight
    s.handle(E.CallerAudioReceived("p", 1000))
    s.handle(E.CallerSpeechStarted())
    s.handle(E.CallerAudioReceived("p", 2000))       # caller spoke 1000ms
    out = s.handle(E.CallerSpeechStopped())
    assert out == []
    assert s.state is CallState.LISTENING


def test_gates_pass_but_no_marks_is_noop_and_stays_speaking():
    s = CallSession(greet_on_connect=False)
    _speaking_at(s, 1000)
    s.handle(E.PlaybackMarkAcked("responsePart"))    # marks now 0
    s.handle(E.CallerAudioReceived("p", 1500))
    s.handle(E.CallerSpeechStarted())
    s.handle(E.CallerAudioReceived("p", 1900))       # caller 400 >= 300, elapsed 900 > 200
    out = s.handle(E.CallerSpeechStopped())
    assert out == []                                 # original no-ops when no unacked marks
    assert s.state is CallState.SPEAKING
    assert s.last_assistant_item == "A"              # NOT reset


def test_resumes_to_speaking_after_interrupt():
    s = CallSession(greet_on_connect=False)
    _speaking_at(s, 1000)
    s.handle(E.CallerAudioReceived("p", 1500))
    s.handle(E.CallerSpeechStarted())
    s.handle(E.CallerAudioReceived("p", 1900))
    s.handle(E.CallerSpeechStopped())                # interrupt, back to LISTENING
    assert s.state is CallState.LISTENING
    s.handle(E.CallerAudioReceived("p", 2000))
    out = s.handle(E.AssistantAudioDelta("B", "audio2"))
    assert out == [C.PlayAudioToCaller("audio2")]
    assert s.state is CallState.SPEAKING
    assert s.last_assistant_item == "B"
    assert s.response_start_timestamp_ms == 2000


def test_thresholds_are_the_preserved_values():
    assert MIN_CALLER_SPEECH_MS == 300
    assert MIN_ASSISTANT_STARTED_MS == 200
