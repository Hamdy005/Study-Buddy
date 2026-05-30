import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.materials.routes import router as materials_router
from src.summary_generator.routes import router as summary_router
from src.rag.routes import router as tutor_router
from src.quiz_generator.routes import router as quiz_router
from src.auth.routes import router as auth_router
from src.store import get_usage
from src.dependencies import get_current_user_id
from src.config import settings 

# ── Logging Setup ──────────────────────────────────────
os.makedirs("logs", exist_ok=True)

log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File handler — rotates at 5MB, keeps 3 backups
file_handler = logging.handlers.RotatingFileHandler(
    "logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Apply to root logger so all src.* modules inherit it
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from src.rag.rag import get_embedder
        get_embedder()
        logger.info("Embedder loaded successfully.")
    except Exception as e:
        logger.warning(f"Embedder failed to load: {e}")

    try:
        from src.materials.validator import warmup_validation_models
        warmup_validation_models()
        logger.info("Validation models loaded and warmed up successfully.")
    except Exception as e:
        logger.warning(f"Validation models failed to load: {e}")

    from src.rag.batch_workers import start_workers
    start_workers()

    yield


app = FastAPI(
    title="AI Tutor API",
    description="Backend API for the AI Tutor for Students application",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def normalize_path(request, call_next):
    # Fix double slashes in paths (e.g., //api/usage -> /api/usage)
    path = request.scope.get("path")
    if path and "//" in path:
        request.scope["path"] = path.replace("//", "/")
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(materials_router)
app.include_router(summary_router)
app.include_router(tutor_router)
app.include_router(quiz_router)
app.include_router(auth_router)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "AI Tutor API"}


@app.get("/api/usage")
async def get_user_usage(user_id: str = Depends(get_current_user_id)):
    return get_usage(user_id)
