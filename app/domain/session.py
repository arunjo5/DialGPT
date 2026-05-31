"""The call state machine: pure, synchronous, no I/O.

CallSession owns all per-call state and mutates it only in handle(). A single
consumer feeds it events, so handle() is never re-entrant and call state can't
race.
"""
from __future__ import annotations

from app.domain import commands as C
from app.domain import events as E
from app.domain.state import CallState

MIN_CALLER_SPEECH_MS = 300       # caller must speak this long for a barge-in
MIN_ASSISTANT_STARTED_MS = 200   # assistant must be this far in to be interruptible


class CallSession:
    def __init__(self, *, greet_on_connect: bool = True) -> None:
        self.greet_on_connect = greet_on_connect
        self.state: CallState = (
            CallState.GREETING if greet_on_connect else CallState.LISTENING
        )
        self.stream_sid: str | None = None
        # Only CallerAudioReceived writes this; everything else reads it.
        self.latest_media_timestamp: int = 0
        self.last_assistant_item: str | None = None
        self.response_start_timestamp_ms: int | None = None
        self.caller_speech_start_ms: int | None = None
        self.outstanding_marks: int = 0

    def handle(self, event: E.Event) -> list[C.Command]:
        if self.state is CallState.CLOSING:
            return []
        if isinstance(event, (E.CallerDisconnected, E.OpenAIClosed)):
            self.state = CallState.CLOSING
            return [C.HangUp()]
        handler = self._HANDLERS.get(type(event))
        if handler is None:
            return []  # unmodelled (state, event) pairs are no-ops
        return handler(self, event)

    def _on_call_started(self, event: E.CallStarted) -> list[C.Command]:
        self.stream_sid = event.stream_sid
        self.latest_media_timestamp = 0
        self.response_start_timestamp_ms = None
        self.last_assistant_item = None
        self.caller_speech_start_ms = None
        self.outstanding_marks = 0
        return []

    def _on_caller_audio(self, event: E.CallerAudioReceived) -> list[C.Command]:
        self.latest_media_timestamp = event.timestamp_ms
        return [C.ForwardCallerAudio(event.payload_b64)]

    def _on_mark_acked(self, event: E.PlaybackMarkAcked) -> list[C.Command]:
        self.outstanding_marks = max(0, self.outstanding_marks - 1)
        return []

    def _on_assistant_audio(self, event: E.AssistantAudioDelta) -> list[C.Command]:
        # Re-stamp on each new item so audio_end_ms is measured from the current
        # item, matching the original.
        if event.item_id and event.item_id != self.last_assistant_item:
            self.response_start_timestamp_ms = self.latest_media_timestamp
            self.last_assistant_item = event.item_id
        if self.state in (CallState.GREETING, CallState.LISTENING):
            self.state = CallState.SPEAKING
        self.outstanding_marks += 1
        return [C.PlayAudioToCaller(event.payload_b64)]

    def _on_assistant_response_done(self, event: E.AssistantResponseDone) -> list[C.Command]:
        if self.state in (CallState.GREETING, CallState.SPEAKING):
            self.state = CallState.LISTENING
        self.last_assistant_item = None
        self.response_start_timestamp_ms = None
        return []

    def _on_caller_speech_started(self, event: E.CallerSpeechStarted) -> list[C.Command]:
        self.caller_speech_start_ms = self.latest_media_timestamp
        return []

    def _on_caller_speech_stopped(self, event: E.CallerSpeechStopped) -> list[C.Command]:
        return self._maybe_barge_in()

    def _on_session_configured(self, event: E.SessionConfigured) -> list[C.Command]:
        if self.state is CallState.GREETING and self.greet_on_connect:
            return [C.StartAssistantTurn()]
        return []

    def _maybe_barge_in(self) -> list[C.Command]:
        start = self.caller_speech_start_ms
        self.caller_speech_start_ms = None
        if start is None:
            return []
        caller_duration = self.latest_media_timestamp - start
        audio_in_flight = (
            self.last_assistant_item is not None
            and self.response_start_timestamp_ms is not None
        )
        if not (caller_duration >= MIN_CALLER_SPEECH_MS and audio_in_flight):
            return []
        elapsed = self.latest_media_timestamp - self.response_start_timestamp_ms
        if elapsed <= MIN_ASSISTANT_STARTED_MS:
            return []
        # With no audio still queued at Twilio the original leaves the assistant
        # talking, so do nothing here too.
        if self.outstanding_marks == 0:
            return []
        self.state = CallState.INTERRUPTED
        commands: list[C.Command] = [
            C.TruncateAssistant(self.last_assistant_item, elapsed),
            C.ClearCallerPlayback(),
        ]
        self.outstanding_marks = 0
        self.last_assistant_item = None
        self.response_start_timestamp_ms = None
        self.state = CallState.LISTENING
        return commands

    _HANDLERS = {
        E.CallStarted: _on_call_started,
        E.CallerAudioReceived: _on_caller_audio,
        E.PlaybackMarkAcked: _on_mark_acked,
        E.AssistantAudioDelta: _on_assistant_audio,
        E.AssistantResponseDone: _on_assistant_response_done,
        E.CallerSpeechStarted: _on_caller_speech_started,
        E.CallerSpeechStopped: _on_caller_speech_stopped,
        E.SessionConfigured: _on_session_configured,
    }
