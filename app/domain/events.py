"""Inbound events. Each adapter turns one wire message into one of these."""
from __future__ import annotations

from dataclasses import dataclass


class Event:
    """Marker base for inbound events."""


# from Twilio
@dataclass(frozen=True)
class CallStarted(Event):
    stream_sid: str


@dataclass(frozen=True)
class CallerAudioReceived(Event):
    payload_b64: str
    timestamp_ms: int


@dataclass(frozen=True)
class PlaybackMarkAcked(Event):
    name: str


@dataclass(frozen=True)
class CallerDisconnected(Event):
    pass


# from OpenAI
@dataclass(frozen=True)
class AssistantAudioDelta(Event):
    item_id: str | None
    payload_b64: str


@dataclass(frozen=True)
class AssistantResponseDone(Event):
    pass


@dataclass(frozen=True)
class CallerSpeechStarted(Event):
    pass


@dataclass(frozen=True)
class CallerSpeechStopped(Event):
    pass


@dataclass(frozen=True)
class SessionConfigured(Event):
    pass


@dataclass(frozen=True)
class AssistantToolCall(Event):
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class OpenAIClosed(Event):
    pass
