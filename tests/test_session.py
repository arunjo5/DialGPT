"""Pure FSM tests: no network, no async, no mocks."""
from app.domain import commands as C
from app.domain import events as E
from app.domain.session import CallSession
from app.domain.state import CallState


def test_default_starts_in_greeting():
    assert CallSession().state is CallState.GREETING


def test_greet_off_starts_in_listening():
    assert CallSession(greet_on_connect=False).state is CallState.LISTENING


def test_session_configured_triggers_greeting():
    s = CallSession()
    assert s.handle(E.SessionConfigured()) == [C.StartAssistantTurn()]
    assert s.state is CallState.GREETING


def test_session_configured_noop_when_greet_off():
    s = CallSession(greet_on_connect=False)
    assert s.handle(E.SessionConfigured()) == []
    assert s.state is CallState.LISTENING


def test_session_configured_noop_outside_greeting():
    s = CallSession()
    s.handle(E.AssistantAudioDelta("A", "a"))  # now SPEAKING
    assert s.state is CallState.SPEAKING
    assert s.handle(E.SessionConfigured()) == []  # no re-greet
    assert s.state is CallState.SPEAKING


def test_call_started_stores_sid_and_resets_counters():
    s = CallSession(greet_on_connect=False)
    s.handle(E.CallerAudioReceived("p", 500))
    s.handle(E.AssistantAudioDelta("A", "a"))
    assert s.handle(E.CallStarted("STREAM123")) == []
    assert s.stream_sid == "STREAM123"
    assert s.latest_media_timestamp == 0
    assert s.last_assistant_item is None
    assert s.response_start_timestamp_ms is None
    assert s.outstanding_marks == 0


def test_caller_audio_forwarded_and_advances_clock():
    s = CallSession(greet_on_connect=False)
    assert s.handle(E.CallerAudioReceived("payload", 120)) == [C.ForwardCallerAudio("payload")]
    assert s.latest_media_timestamp == 120
    assert s.state is CallState.LISTENING


def test_assistant_delta_listening_to_speaking():
    s = CallSession(greet_on_connect=False)
    s.handle(E.CallerAudioReceived("p", 1000))
    assert s.handle(E.AssistantAudioDelta("A", "audio")) == [C.PlayAudioToCaller("audio")]
    assert s.state is CallState.SPEAKING
    assert s.last_assistant_item == "A"
    assert s.response_start_timestamp_ms == 1000
    assert s.outstanding_marks == 1


def test_assistant_delta_same_item_does_not_restamp():
    s = CallSession(greet_on_connect=False)
    s.handle(E.CallerAudioReceived("p", 1000))
    s.handle(E.AssistantAudioDelta("A", "a1"))
    s.handle(E.CallerAudioReceived("p", 1100))
    assert s.handle(E.AssistantAudioDelta("A", "a2")) == [C.PlayAudioToCaller("a2")]
    assert s.response_start_timestamp_ms == 1000  # unchanged for same item
    assert s.outstanding_marks == 2


def test_assistant_delta_new_item_restamps():
    s = CallSession(greet_on_connect=False)
    s.handle(E.CallerAudioReceived("p", 1000))
    s.handle(E.AssistantAudioDelta("A", "a1"))
    s.handle(E.CallerAudioReceived("p", 1100))
    s.handle(E.AssistantAudioDelta("B", "b1"))
    assert s.last_assistant_item == "B"
    assert s.response_start_timestamp_ms == 1100


def test_response_done_speaking_to_listening():
    s = CallSession(greet_on_connect=False)
    s.handle(E.CallerAudioReceived("p", 1000))
    s.handle(E.AssistantAudioDelta("A", "a"))
    assert s.handle(E.AssistantResponseDone()) == []
    assert s.state is CallState.LISTENING
    assert s.last_assistant_item is None
    assert s.response_start_timestamp_ms is None


def test_response_done_in_listening_is_noop():
    s = CallSession(greet_on_connect=False)
    assert s.handle(E.AssistantResponseDone()) == []
    assert s.state is CallState.LISTENING


def test_greeting_delta_to_speaking_then_done_to_listening():
    s = CallSession()  # GREETING
    assert s.handle(E.AssistantAudioDelta("G", "greet")) == [C.PlayAudioToCaller("greet")]
    assert s.state is CallState.SPEAKING
    s.handle(E.AssistantResponseDone())
    assert s.state is CallState.LISTENING


def test_mark_ack_decrements_and_floors_at_zero():
    s = CallSession(greet_on_connect=False)
    s.handle(E.CallerAudioReceived("p", 10))
    s.handle(E.AssistantAudioDelta("A", "a"))  # marks = 1
    assert s.handle(E.PlaybackMarkAcked("responsePart")) == []
    assert s.outstanding_marks == 0
    s.handle(E.PlaybackMarkAcked("responsePart"))  # floored, never negative
    assert s.outstanding_marks == 0


def test_disconnect_to_closing():
    s = CallSession()
    assert s.handle(E.CallerDisconnected()) == [C.HangUp()]
    assert s.state is CallState.CLOSING


def test_openai_closed_to_closing():
    s = CallSession(greet_on_connect=False)
    s.handle(E.AssistantAudioDelta("A", "a"))
    assert s.handle(E.OpenAIClosed()) == [C.HangUp()]
    assert s.state is CallState.CLOSING


def test_closing_absorbs_all_events():
    s = CallSession()
    s.handle(E.CallerDisconnected())
    for ev in (
        E.CallerAudioReceived("p", 1),
        E.AssistantAudioDelta("A", "a"),
        E.AssistantResponseDone(),
        E.SessionConfigured(),
        E.PlaybackMarkAcked("x"),
    ):
        assert s.handle(ev) == []
        assert s.state is CallState.CLOSING


def test_speech_stopped_in_greeting_gap_is_noop():
    # Caller talks before the first greeting delta lands: no audio in flight,
    # so the barge-in check safely no-ops and we stay in GREETING.
    s = CallSession()  # GREETING
    s.handle(E.CallerAudioReceived("p", 100))
    s.handle(E.CallerSpeechStarted())
    s.handle(E.CallerAudioReceived("p", 600))
    assert s.handle(E.CallerSpeechStopped()) == []
    assert s.state is CallState.GREETING
