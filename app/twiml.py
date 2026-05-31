"""TwiML for the inbound-call webhook."""
from __future__ import annotations

from twilio.twiml.voice_response import Connect, VoiceResponse

_VOICE = "Google.en-US-Chirp3-HD-Aoede"


def build_incoming_call_twiml(host: str, *, greet_on_connect: bool = True) -> str:
    response = VoiceResponse()
    response.say(
        "Please wait while we connect your call to the A. I. voice assistant ",
        voice=_VOICE,
    )
    if not greet_on_connect:
        # With greet-on-connect the assistant speaks first, so skip this line.
        response.pause(length=1)
        response.say("O.K. you can start talking!", voice=_VOICE)
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return str(response)
