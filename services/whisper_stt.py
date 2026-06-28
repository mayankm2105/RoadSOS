import os
import psutil
import asyncio
import time
from typing import Optional
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

# Whisper model singleton — loaded once, reused for every request
_whisper_model = None
_model_loading = False

# Map our lang codes to Whisper language codes
WHISPER_LANG_MAP = {
    "en": "en",
    "hi": "hi",    # Hindi
    "pa": "pa",    # Punjabi
    "hw": "hi",    # Haryanvi — closest to Hindi for Whisper
}

# Max audio duration in seconds Whisper should handle
MAX_AUDIO_DURATION_SECONDS = 300  # 5 minutes


def _get_available_ram_mb() -> int:
    """Return available system RAM in MB."""
    try:
        return psutil.virtual_memory().available // (1024 * 1024)
    except Exception:
        return 999  # assume enough if psutil fails

def get_whisper_model():
    """
    Lazy-load Whisper model only when actually needed.
    Returns None if not enough RAM is available.
    """
    global _whisper_model
    
    if _whisper_model is not None:
        return _whisper_model
    
    available_mb = _get_available_ram_mb()
    required_mb = 200  # minimum safe threshold
    
    if available_mb < required_mb:
        logger.warning(
            f"⚠️  Insufficient RAM for Whisper: {available_mb}MB available, "
            f"{required_mb}MB required. Voice input unavailable."
        )
        return None
    
    try:
        import whisper
        from config import settings
        logger.info(f"⏳ Loading Whisper model '{settings.WHISPER_MODEL}' on demand...")
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
        logger.info(f"✅ Whisper model loaded successfully")
        return _whisper_model
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
        return None


def transcribe_audio_sync(
    audio_path: str,
    lang: str = "en"
) -> dict:
    """
    Synchronous Whisper transcription. Must be called via asyncio.to_thread()
    from async contexts — Whisper inference is CPU-bound and blocks.

    Args:
        audio_path: Absolute path to the audio file
        lang: Language hint code ("en", "hi", "pa", "hw")

    Returns dict with:
        {
          "text": "transcribed text here",
          "language": "hi",       # detected language (from Whisper)
          "duration_seconds": 12  # audio duration
        }

    Raises:
        RuntimeError: If Whisper model not loaded
        TimeoutError: If audio is too long (> MAX_AUDIO_DURATION_SECONDS)
        Exception: If transcription fails
    """
    model = get_whisper_model()
    if model is None:
        raise RuntimeError(
            "Whisper model is not available. "
            "Check server logs for loading errors."
        )

    # Map lang code to Whisper format
    whisper_lang = WHISPER_LANG_MAP.get(lang, None)
    # None means auto-detect — let Whisper figure it out

    try:
        import whisper

        # Load audio and check duration BEFORE transcribing
        logger.debug(f"Loading audio file: {audio_path}")
        audio = whisper.load_audio(audio_path)
        duration_seconds = len(audio) / 16000  # Whisper uses 16kHz

        logger.debug(f"Audio duration: {duration_seconds:.1f}s")

        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            raise TimeoutError(
                f"Audio too long ({duration_seconds:.0f}s). "
                f"Maximum is {MAX_AUDIO_DURATION_SECONDS}s (5 minutes)."
            )

        # Run transcription
        logger.info(
            f"Transcribing {duration_seconds:.1f}s audio "
            f"(lang hint: {whisper_lang or 'auto'})"
        )
        start = time.time()

        transcribe_options = {
            "fp16": False,  # fp16 only works on GPU; CPU must use fp32
        }
        if whisper_lang:
            transcribe_options["language"] = whisper_lang

        result = model.transcribe(audio, **transcribe_options)

        elapsed = time.time() - start
        text = result["text"].strip()
        detected_lang = result.get("language", lang)

        logger.info(
            f"Transcription complete in {elapsed:.1f}s: "
            f"'{text[:80]}...' (lang: {detected_lang})"
        )

        return {
            "text": text,
            "language": detected_lang,
            "duration_seconds": round(duration_seconds, 1)
        }

    except (TimeoutError, RuntimeError):
        raise
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise Exception(f"Transcription failed: {str(e)}")


async def transcribe_audio(
    audio_path: str,
    lang: str = "en"
) -> dict:
    """
    Async wrapper around transcribe_audio_sync().
    Runs Whisper in a thread pool so the FastAPI event loop stays free.

    This is the function called by the endpoint — always use this,
    never call transcribe_audio_sync() directly from async code.
    """
    model = get_whisper_model()
    
    if model is None:
        return {
            "success": False,
            "error": "Voice input temporarily unavailable (insufficient server memory). Please type your message instead.",
            "transcription": None
        }

    return await asyncio.to_thread(
        transcribe_audio_sync,
        audio_path,
        lang
    )
