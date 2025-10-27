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

    # Ollama - Dual Instance Configuration
    # Separate instances prevent pinned memory conflicts when switching models
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API endpoint for LLM (Qwen 2.5 7B)"
    )
    ollama_embedding_url: str = Field(
        default="http://localhost:11435",
        description="Ollama API endpoint for embeddings (nomic-embed-text) - separate instance"
    )
    ollama_model: str = Field(
        default="qwen2.5:7b-instruct-q4_0",
        description="LLM model for claim extraction and entailment"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model (768 dimensions)"
    )
    ollama_num_ctx: int = Field(
        default=16384,
        description="Context window size in tokens (8192=~5GB VRAM, 16384=~6GB VRAM, 32768=~7.6GB VRAM per instance)"
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

    # Quote Optimization (Coarse Chunking + DSPy)
    coarse_chunk_size: int = Field(
        default=3000,
        description="Coarse chunk size in tokens for quote finding (reduces embeddings by 50x)"
    )
    coarse_chunk_overlap: int = Field(
        default=500,
        description="Overlap between coarse chunks in tokens"
    )
    top_k_chunks: int = Field(
        default=4,
        description="Number of top relevant chunks to retrieve per claim"
    )
    quote_verification_min_confidence: float = Field(
        default=0.90,
        description="Minimum confidence threshold for quote verification (0.0-1.0)"
    )
    quote_finder_model_path: str = Field(
        default="models/quote_finder_v1.json",
        description="Path to optimized DSPy QuoteFinder model (set to empty string to use zero-shot baseline)"
    )
    enable_entailment_validation: bool = Field(
        default=True,
        description="Enable entailment validation to filter non-SUPPORTS quotes (disable for debugging)"
    )

    # DSPy Optimization (shared across all training: claims, entailment, quotes)
    dspy_max_bootstrapped_demos: int = Field(
        default=4,
        description="Number of bootstrapped few-shot examples to include in DSPy prompts"
    )
    dspy_max_labeled_demos: int = Field(
        default=2,
        description="Number of seed examples to include in DSPy prompts (internally set to match bootstrapped_demos)"
    )
    dspy_max_rounds: int = Field(
        default=3,
        description="Number of DSPy optimization rounds (more rounds = better quality, longer time)"
    )
    dspy_max_errors: int = Field(
        default=5,
        description="Maximum errors tolerated during DSPy optimization before failing"
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
