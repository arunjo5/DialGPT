"""Output intents from CallSession.handle(); the orchestrator runs each one."""
from __future__ import annotations

from dataclasses import dataclass


class Command:
    """Marker base for output intents."""


# run by the OpenAI adapter
@dataclass(frozen=True)
class ForwardCallerAudio(Command):
    payload_b64: str


@dataclass(frozen=True)
class TruncateAssistant(Command):
    item_id: str
    audio_end_ms: int


@dataclass(frozen=True)
class StartAssistantTurn(Command):
    pass


# run by the Twilio adapter
@dataclass(frozen=True)
class PlayAudioToCaller(Command):
    payload_b64: str


@dataclass(frozen=True)
class ClearCallerPlayback(Command):
    pass


# run by the orchestrator
@dataclass(frozen=True)
class HangUp(Command):
    pass
