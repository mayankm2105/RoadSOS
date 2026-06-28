from typing import List
"""
Pre-flight startup checks for RoadSoS.

Called once during FastAPI lifespan startup (before the server
starts accepting requests). Logs the status of every dependency
so you can catch misconfigurations immediately on boot rather
than at runtime when a user hits the failing endpoint.

Checks performed (in order):
  1. SQLite database — tables exist and are accessible
  2. API keys — which are set vs missing (never logs the key values)
  3. Whisper model — pre-load into memory to avoid cold-start delay
     on first /chat/voice request (takes 5–30 sec depending on model)
  4. Gemini API — quick connectivity ping (list models)
  5. Google Places API — quick connectivity ping (geocode a known location)

All checks are non-blocking: a failed check logs a WARNING but does
NOT crash the server. The health endpoint will reflect the failure.
This lets the server start even with partial config (useful in dev).

Returns a dict with check results that is stored on app.state so
the /health endpoint can read it without re-running the checks.
"""

import asyncio
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


async def run_preflight_checks() -> dict:
    """
    Run all startup checks. Returns status dict:
    {
      "sqlite": "ok" | "error: <msg>",
      "gemini": "ok" | "error: <msg>" | "not_configured",
      "google_places": "ok" | "error: <msg>" | "not_configured",
      "google_geocoding": "ok" | "error: <msg>" | "not_configured",
      "whisper": "ok" | "error: <msg>" | "not_loaded",
      "api_keys": {
        "GEMINI_API_KEY": "set" | "missing",
        "GOOGLE_PLACES_API_KEY": "set" | "missing",
        "GOOGLE_GEOCODING_API_KEY": "set" | "missing",
        "ADMIN_KEY": "set" | "missing",
        "SOS_BASE_URL": "set" | "missing"
      }
    }
    """
    results = {}

    # ── Check 1: SQLite ──────────────────────────────────────────────────
    try:
        from database import SessionLocal, engine, Base
        from models.cache import CacheEntry

        # Create all tables if they don't exist (needed on Render's ephemeral filesystem)
        Base.metadata.create_all(bind=engine)

        db = SessionLocal()
        # Simple query to verify table exists and DB is reachable
        count = db.query(CacheEntry).count()
        db.close()
        results["sqlite"] = f"ok ({count} cache entries)"
        logger.info(f"✅ SQLite: OK — {count} cache entries")
    except Exception as e:
        results["sqlite"] = f"error: {str(e)}"
        logger.error(f"❌ SQLite check failed: {e}")

    

    # ── Check 2: API Keys presence ───────────────────────────────────────
    key_checks = {
        "GEMINI_API_KEY":          bool(settings.GEMINI_API_KEY),
        "GOOGLE_PLACES_API_KEY":   bool(settings.GOOGLE_PLACES_API_KEY),
        "GOOGLE_GEOCODING_API_KEY": bool(
            settings.GOOGLE_GEOCODING_API_KEY
            or settings.GOOGLE_PLACES_API_KEY
        ),
        "ADMIN_KEY":               bool(settings.ADMIN_KEY),
        "SOS_BASE_URL":            bool(settings.SOS_BASE_URL),
    }
    results["api_keys"] = {
        k: ("set" if v else "missing") for k, v in key_checks.items()
    }
    for key, present in key_checks.items():
        if present:
            logger.info(f"✅ {key}: configured")
        else:
            logger.warning(f"⚠️  {key}: NOT SET — dependent features will fail")

    # ── Check 3: Whisper model — lazy load only, don't preload at startup ────
    # On low-memory environments (Render free tier, 512MB), importing torch
    # at startup causes OOM. Whisper is loaded on first actual /chat/voice
    # request instead. This trades cold-start latency for server stability.
    try:
        import whisper  # just check it's installed, don't load model
        results["whisper"] = f"ok (model={settings.WHISPER_MODEL}, lazy-load enabled)"
        logger.info(f"✅ Whisper package available — model will load on first voice request")
    except ImportError:
        results["whisper"] = "not_loaded (whisper not installed)"
        logger.warning("⚠️  whisper package not installed — voice input disabled")

    # ── Check 4: Gemini API ping ─────────────────────────────────────────
    if settings.GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            # List models is a lightweight call (no token usage)
            models = [m.name for m in genai.list_models()]
            flash_available = any("flash" in m for m in models)
            results["gemini"] = (
                "ok" if flash_available
                else "ok (flash model not listed — check model name)"
            )
            logger.info("✅ Gemini API: reachable")
        except Exception as e:
            results["gemini"] = f"error: {str(e)}"
            logger.error(f"❌ Gemini API ping failed: {e}")
    else:
        results["gemini"] = "not_configured"
        logger.warning("⚠️  Gemini: GEMINI_API_KEY not set")

    # ── Check 5: Google Places API ping ─────────────────────────────────
    if settings.GOOGLE_PLACES_API_KEY:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/geocode/json",
                    params={
                        "latlng": "28.6139,77.2090",
                        "key": settings.GOOGLE_PLACES_API_KEY
                    }
                )
            data = resp.json()
            status = data.get("status", "UNKNOWN")
            if status in ["OK", "ZERO_RESULTS"]:
                results["google_places"] = "ok"
                logger.info("✅ Google Places API: reachable and key valid")
            else:
                results["google_places"] = (
                    f"error: API returned status={status}"
                )
                logger.warning(
                    f"⚠️  Google Places API returned status: {status} "
                    f"— {data.get('error_message', '')}"
                )
        except Exception as e:
            results["google_places"] = f"error: {str(e)}"
            logger.error(f"❌ Google Places API ping failed: {e}")
    else:
        results["google_places"] = "not_configured"
        logger.warning("⚠️  Google Places: GOOGLE_PLACES_API_KEY not set")

    logger.info("🚀 Pre-flight checks complete. Server is starting.")
    return results
