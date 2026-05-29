import os
import json
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv

load_dotenv()

# Load PDF document into memory (simple approach)
PDF_PATH = os.getenv('PDF_PATH', 'document.pdf')


def load_pdf_text(path: str, max_chars: int = 20000) -> str:
    """Load text from a PDF file and return up to max_chars characters.

    Uses PyPDF2 to extract text from pages. If the library or file is
    unavailable, returns an empty string and logs the error.
    """
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        joined = "\n\n".join(pages)
        return joined[:max_chars]
    except Exception as e:
        print(f"Could not load PDF at {path}: {e}")
        return ""


# Read the PDF into memory at startup. Keep it reasonably sized.
PDF_TEXT = load_pdf_text(PDF_PATH)

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
SYSTEM_MESSAGE = (
    "You are a helpful and adaptable AI assistant. Default language: English (en-US). "
    "Detect the language the user speaks in each turn and reply in that same language. "
    "If the user speaks in English reply in English; if they speak in Spanish reply in Spanish, etc. "
    "Do not switch languages mid-response — keep each spoken response entirely in a single language. "
    "If you are uncertain which language the user used, ask a short clarification question in the user's last-used language. "
    "Keep spoken responses uninterrupted unless the caller actually begins speaking."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'session.updated'
]
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say(
        "Please wait while we connect your call to the A. I. voice assistant ",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    response.pause(length=1)
    response.say(   
        "O.K. you can start talking!",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-realtime",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        caller_speech_start = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp, response_start_timestamp_twilio, last_assistant_item
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.state.name == 'OPEN':
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.state.name == 'OPEN':
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio, caller_speech_start
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    if response.get('type') == 'response.output_audio.delta' and 'delta' in response:
                        audio_payload = response['delta']
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)


                        if response.get("item_id") and response["item_id"] != last_assistant_item:
                            response_start_timestamp_twilio = latest_media_timestamp
                            last_assistant_item = response["item_id"]
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        await send_mark(websocket, stream_sid)

                    # Use caller speech started/stopped events to decide when to interrupt.
                    # We record when the caller starts speaking and only interrupt after
                    # the caller has spoken for a short duration (debounce). This avoids
                    # cutting the assistant off on very brief VAD blips.
                    evt_type = response.get('type')
                    if evt_type == 'input_audio_buffer.speech_started':
                        # record when caller began speaking (timestamp comes from Twilio)
                        caller_speech_start = latest_media_timestamp
                        if SHOW_TIMING_MATH:
                            print(f"Caller speech started at {caller_speech_start}ms")

                    elif evt_type == 'input_audio_buffer.speech_stopped':
                        # determine how long the caller spoke for
                        if caller_speech_start is None:
                            # no start timestamp — ignore
                            continue
                        caller_duration = latest_media_timestamp - caller_speech_start
                        if SHOW_TIMING_MATH:
                            print(f"Caller spoke for {caller_duration}ms")

                        # Only interrupt if caller spoke for a minimum duration to
                        # avoid truncating assistant audio for very short noises.
                        MIN_CALLER_SPEECH_MS = 300
                        MIN_ASSISTANT_STARTED_MS = 200
                        if caller_duration >= MIN_CALLER_SPEECH_MS and last_assistant_item and response_start_timestamp_twilio is not None:
                            assistant_elapsed = latest_media_timestamp - response_start_timestamp_twilio
                            if assistant_elapsed > MIN_ASSISTANT_STARTED_MS:
                                print(f"Interrupting response with id: {last_assistant_item} (assistant_elapsed={assistant_elapsed}ms, caller_duration={caller_duration}ms)")
                                await handle_speech_event()

                        # reset caller speech start marker after processing
                        caller_speech_start = None
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        async def handle_speech_event():
            """Handle interruption when the caller's speech is detected (stopped/started).

            This truncates the assistant audio at the correct timestamp and clears
            internal state so the assistant can resume normally.
            """
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech event (possible interruption).")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    # Append the loaded PDF text to the system instructions so the realtime
    # model can answer questions based on the local document. This is a
    # simple approach: we provide the document as context in the session
    # instructions. For larger documents, consider a retrieval+RAG approach.
    instructions = SYSTEM_MESSAGE
    if PDF_TEXT:
        instructions = f"{instructions}\n\nUse the following document as the primary source when answering user questions:\n{PDF_TEXT}"

    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"}
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE
                }
            },
            "instructions": instructions,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
