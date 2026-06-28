from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Middleware imports
from middleware.rate_limiter import limiter, rate_limit_exceeded_handler
from middleware.security import SecurityHeadersMiddleware
from slowapi.errors import RateLimitExceeded

# Startup checks
from scripts.preflight import run_preflight_checks

# Config
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

from routers.services import router as services_router
from routers.chat import router as chat_router
from routers.triage import router as triage_router
from routers.sos import router as sos_router
from routers.report import router as report_router
from routers.geo import router as geo_router
from routers.offline import router as offline_router
from routers.i18n import router as i18n_router
from routers.health import router as health_router

# ── Lifespan: startup + shutdown ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Replaces the deprecated @app.on_event("startup") pattern.
    Code before `yield` runs on startup.
    Code after `yield` runs on shutdown.

    Startup sequence:
    1. Log that the server is booting
    2. Run pre-flight checks (DB, API keys, Whisper model load)
    3. Store results on app.state for /health to read
    4. Yield (server is now live and accepting requests)
    
    Shutdown sequence:
    5. Log graceful shutdown
    6. Close any resources if needed
    """
    # ── STARTUP ──────────────────────────────────────────────────────────
    logger.info(
        f"🚀 RoadSoS v{settings.VERSION} starting "
        f"[{settings.ENVIRONMENT}]"
    )
    logger.info(f"   Coverage: Haryana (HR) + Delhi (DL)")
    logger.info(f"   Whisper model: {settings.WHISPER_MODEL}")

    # Run all pre-flight checks and store results
    preflight_results = await run_preflight_checks()
    app.state.preflight_results = preflight_results

    logger.info("✅ Server is ready to accept requests")

    yield   # Server runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info("🛑 RoadSoS shutting down gracefully...")


# ── App creation ──────────────────────────────────────────────────────────
app = FastAPI(
    title="RoadSoS API",
    description=(
        "Emergency assistance for road accidents in Haryana and Delhi. "
        "Provides nearest emergency services, AI chatbot, medical triage, "
        "SOS sharing, and FIR report generation."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/api/v1/docs",       # Swagger UI
    redoc_url="/api/v1/redoc",     # ReDoc
    openapi_url="/api/v1/openapi.json"
)


# ── Middleware (order matters — added in reverse order of execution) ──────

# 1. Rate limiter state (must be set before adding exception handler)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# 2. Security headers (runs last in chain = wraps everything)
app.add_middleware(SecurityHeadersMiddleware)

# 3. CORS (must come AFTER security headers middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RoadSoS-Version", "Retry-After"]
)


# ── Routers ───────────────────────────────────────────────────────────────
# Health must be first (Railway hits it during deploy to confirm readiness)
app.include_router(health_router, prefix="/api/v1")

# All your existing routers — DO NOT REMOVE ANY
app.include_router(services_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(triage_router, prefix="/api/v1")
app.include_router(sos_router, prefix="/api/v1")
app.include_router(report_router, prefix="/api/v1")
app.include_router(geo_router, prefix="/api/v1")
app.include_router(offline_router, prefix="/api/v1")
app.include_router(i18n_router, prefix="/api/v1")


# ── Root redirect ────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/docs")
