"""Main FastAPI application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import requests

from src.api.routers import extraction
from src.api.exceptions import (
    database_exception_handler,
    generic_exception_handler,
    requests_exception_handler,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("=" * 80)
    logger.info("Claim Extraction API Starting")
    logger.info("=" * 80)
    logger.info(f"Host: {settings.api_host}:{settings.api_port}")
    logger.info(f"Timeout: {settings.api_timeout}s (0 = no timeout)")
    logger.info(f"Database: {settings.database_url}")
    logger.info(f"Ollama: {settings.ollama_url}")
    logger.info(f"Ollama Embedding: {settings.ollama_embedding_url}")
    logger.info(
        f"Reranker: {settings.reranker_url} (enabled={settings.enable_reranker})"
    )
    logger.info(f"Quote Processing: {settings.enable_quote_processing}")
    logger.info("=" * 80)
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("Claim Extraction API shutting down")


# Create FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Claim Extraction API",
    description="""
    API for extracting claims and supporting quotes from podcast transcripts using DSPy.

    ## Features
    - Extract claims from single episodes
    - Batch processing across multiple podcasts
    - Automatic quote finding and validation
    - Confidence scoring and filtering
    - Health checks for all services

    ## Processing
    - Claims are extracted using DSPy LLM models
    - Quotes are found via semantic search and validated for entailment
    - Results are returned in the API response (not saved to database)
    - Average processing time: ~6 minutes per episode

    ## Authentication
    Currently no authentication is required. Add API key authentication as needed.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS (needed for Swagger UI and external API access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production with API key auth)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key Authentication Middleware
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Verify API key for all endpoints except docs."""
    # Allow access to documentation without authentication
    public_paths = ["/", "/docs", "/redoc", "/openapi.json"]
    if request.url.path in public_paths:
        return await call_next(request)

    # Check for API key in header
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Missing API key. Include 'X-API-Key' header in your request."
            }
        )

    # Validate API key
    if api_key != settings.api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Invalid API key"
            }
        )

    # API key is valid, proceed with request
    return await call_next(request)


# Register routers
app.include_router(extraction.router)

# Register exception handlers
app.add_exception_handler(SQLAlchemyError, database_exception_handler)
app.add_exception_handler(requests.RequestException, requests_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")
