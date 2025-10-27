"""
Configuration settings for the claim-quote extraction pipeline.

Loads configuration from environment variables using Pydantic Settings.
All settings can be overridden via .env file or environment variables.

Usage:
    from src.config.settings import settings

    print(settings.database_url)
    print(settings.ollama_url)
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/podcast_db",
        description="PostgreSQL connection string"
    )

    # Ollama
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API endpoint for LLM operations (qwen)"
    )
    ollama_embedding_url: str = Field(
        default="http://localhost:11435",
        description="Ollama API endpoint for embedding operations"
    )
    ollama_model: str = Field(
        default="qwen2.5:7b-instruct-q4_0",
        description="LLM model for claim extraction and entailment"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model (768 dimensions)"
    )

    # Reranker
    enable_reranker: bool = Field(
        default=True,
        description="Enable reranker service for high-precision scoring"
    )
    reranker_url: str = Field(
        default="http://localhost:8080",
        description="Reranker service endpoint"
    )
    reranker_timeout: int = Field(
        default=5000,
        description="Reranker request timeout in milliseconds"
    )

    # Chunking
    chunk_size: int = Field(
        default=16000,
        description="Maximum chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=1000,
        description="Overlap between chunks in characters"
    )

    # Processing
    parallel_batch_size: int = Field(
        default=3,
        description="Number of chunks to process in parallel"
    )

    # Deduplication Thresholds
    embedding_similarity_threshold: float = Field(
        default=0.85,
        description="Cosine similarity threshold for claim deduplication"
    )
    reranker_verification_threshold: float = Field(
        default=0.9,
        description="Reranker score threshold for duplicate verification"
    )
    string_similarity_threshold: float = Field(
        default=0.95,
        description="Jaccard similarity threshold for quote deduplication"
    )
    vector_distance_threshold: float = Field(
        default=0.15,
        description="pgvector L2 distance threshold for database search"
    )

    # Scoring
    min_confidence: float = Field(
        default=0.3,
        description="Minimum confidence score to save claims"
    )
    max_quotes_per_claim: int = Field(
        default=10,
        description="Maximum number of quotes to link per claim"
    )
    min_quote_relevance: float = Field(
        default=0.85,
        description="Minimum relevance score to link quote to claim"
    )

    # Caching
    cache_max_size: int = Field(
        default=10000,
        description="Maximum number of entries in LRU caches"
    )
    cache_ttl_hours: int = Field(
        default=1,
        description="Cache entry time-to-live in hours"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_file: str = Field(
        default="logs/extraction.log",
        description="Log file path"
    )


# Global settings instance
settings = Settings()
