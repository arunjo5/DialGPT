"""Latency metrics. Lives in the shell, not the pure domain, since it reads the
clock. CallLatency is fed (event, commands) per handle() and records to Prometheus.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from prometheus_client import Counter, Histogram

from app.domain import commands as C
from app.domain import events as E

log = logging.getLogger("voice.latency")

RESPONSE_LATENCY = Histogram(
    "voice_response_latency_seconds",
    "Caller finishing speaking to the first assistant audio.",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0),
)
BARGE_IN_LATENCY = Histogram(
    "voice_barge_in_latency_seconds",
    "Caller starting a barge-in to the assistant being cut off.",
    buckets=(0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
)
RETRIEVAL_LATENCY = Histogram(
    "voice_retrieval_latency_seconds",
    "Document retrieval time for a search_document tool call.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
CALLS = Counter("voice_calls", "Calls handled.")
TURNS = Counter("voice_turns", "Assistant response turns.")
BARGE_INS = Counter("voice_barge_ins", "Barge-in interruptions.")


class CallLatency:
    """Per-call observer. Response latency: caller speech-stopped to first
    assistant audio. Barge-in latency: caller speech-started to the truncate.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._speech_started_at: float | None = None
        self._response_pending_at: float | None = None
        self.response_latencies: list[float] = []
        self.barge_in_latencies: list[float] = []

    def observe(self, event: E.Event, commands: Sequence[C.Command]) -> None:
        now = self._clock()
        if isinstance(event, E.CallerSpeechStarted):
            self._speech_started_at = now
        elif isinstance(event, E.CallerSpeechStopped):
            self._response_pending_at = now
        elif isinstance(event, E.AssistantAudioDelta) and self._response_pending_at is not None:
            latency = now - self._response_pending_at
            self._response_pending_at = None
            self.response_latencies.append(latency)
            RESPONSE_LATENCY.observe(latency)
            TURNS.inc()
        for command in commands:
            if isinstance(command, C.TruncateAssistant) and self._speech_started_at is not None:
                latency = now - self._speech_started_at
                self._speech_started_at = None
                self.barge_in_latencies.append(latency)
                BARGE_IN_LATENCY.observe(latency)
                BARGE_INS.inc()

    def log_summary(self) -> None:
        responses = self.response_latencies
        if not responses and not self.barge_in_latencies:
            return
        log.info(
            "call ended: turns=%d avg_response=%.3fs max_response=%.3fs barge_ins=%d",
            len(responses),
            sum(responses) / len(responses) if responses else 0.0,
            max(responses, default=0.0),
            len(self.barge_in_latencies),
        )
