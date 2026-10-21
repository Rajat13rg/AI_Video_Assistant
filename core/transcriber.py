import os
import tempfile
import time

import requests
from pydub import AudioSegment

# Keep API requests below Sarvam's 30-second limit.
SARVAM_PIECE_SECONDS = 25

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_SARVAM_MODEL = "saaras:v4"

# English is transcribed as spoken. Hinglish is translated into English,
# so the summarization and RAG steps continue receiving English text.
TRANSCRIPTION_OPTIONS = {
    "english": {"language_code": "en-IN", "mode": "transcribe"},
    "hinglish": {"language_code": "unknown", "mode": "translate"},
}


def _send_to_sarvam(piece_path: str, language_code: str, mode: str) -> str:
    """Send one short WAV clip to Sarvam and return the transcript."""
    # Read keys at runtime so local .env loading works even if this module
    # was imported before load_dotenv() was called by the Streamlit app.
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is missing. Add it in Render's Environment settings."
        )

    model = os.getenv("SARVAM_STT_MODEL", DEFAULT_SARVAM_MODEL).strip()
    headers = {"api-subscription-key": api_key}
    data = {
        "model": model,
        "language_code": language_code,
        "mode": mode,
    }

    for attempt in range(3):
        try:
            with open(piece_path, "rb") as audio_file:
                files = {
                    "file": (os.path.basename(piece_path), audio_file, "audio/wav")
                }
                response = requests.post(
                    SARVAM_STT_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120,
                )

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < 2:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait_seconds = float(retry_after) if retry_after else 2 ** (attempt + 1)
                    except ValueError:
                        wait_seconds = 2 ** (attempt + 1)
                    wait_seconds = min(wait_seconds, 15)
                    print(
                        f"Sarvam returned HTTP {response.status_code}; "
                        f"retrying in {wait_seconds:g}s."
                    )
                    time.sleep(wait_seconds)
                    continue

            if not response.ok:
                raise RuntimeError(
                    f"Sarvam speech-to-text failed (HTTP {response.status_code}): "
                    f"{response.text[:1000]}"
                )

            transcript = response.json().get("transcript", "")
            if not isinstance(transcript, str) or not transcript.strip():
                raise RuntimeError(
                    "Sarvam returned an empty transcript for an audio segment."
                )
            return transcript.strip()

        except requests.Timeout as exc:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(
                "Sarvam speech-to-text timed out after multiple attempts."
            ) from exc
        except requests.RequestException as exc:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(
                f"Could not connect to Sarvam speech-to-text: {exc}"
            ) from exc

    raise RuntimeError("Sarvam transcription failed after multiple attempts.")


def transcribe_chunk_sarvam(chunk_path: str, language: str = "english") -> str:
    """Split a WAV chunk into short pieces and transcribe each with Sarvam."""
    if not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError(
            "SARVAM_API_KEY is missing. Add it in Render's Environment settings."
        )

    language_key = language.lower().strip()
    options = TRANSCRIPTION_OPTIONS.get(language_key)
    if options is None:
        raise ValueError(
            f"Unsupported language '{language}'. Choose 'english' or 'hinglish'."
        )

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    transcript_parts = []

    for piece_index, start in enumerate(range(0, len(audio), piece_ms), start=1):
        piece = audio[start : start + piece_ms]
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            piece.export(temp_path, format="wav")
            print(f"Sending audio piece {piece_index}/{total_pieces} to Sarvam...")

            text = _send_to_sarvam(
                temp_path,
                language_code=options["language_code"],
                mode=options["mode"],
            )
            transcript_parts.append(text)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return " ".join(transcript_parts).strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Transcribe one audio chunk using Sarvam's hosted API."""
    return transcribe_chunk_sarvam(chunk_path, language=language)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe all supplied audio chunks and return a combined transcript."""
    if not chunks:
        raise ValueError("No audio chunks were provided for transcription.")

    language_key = language.lower().strip()
    if language_key not in TRANSCRIPTION_OPTIONS:
        raise ValueError(
            f"Unsupported language '{language}'. Choose 'english' or 'hinglish'."
        )

    if not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError(
            "SARVAM_API_KEY is missing. Add it in Render's Environment settings."
        )

    print(f"Using Sarvam AI for {language_key} transcription.")
    transcript_parts = []

    for index, chunk_path in enumerate(chunks, start=1):
        print(f"Transcribing audio chunk {index}/{len(chunks)}...")
        transcript = transcribe_chunk(chunk_path, language=language_key)
        if transcript:
            transcript_parts.append(transcript)

    print("Transcription complete.")
    return " ".join(transcript_parts).strip()
