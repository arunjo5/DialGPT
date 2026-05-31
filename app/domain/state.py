"""Call lifecycle states."""
from __future__ import annotations

from enum import Enum, auto


class CallState(Enum):
    GREETING = auto()
    LISTENING = auto()
    SPEAKING = auto()
    # Transient: a barge-in enters and leaves INTERRUPTED within one handle()
    # call, so a session is never seen resting in it.
    INTERRUPTED = auto()
    CLOSING = auto()
