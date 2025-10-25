"""
Embedding service with Ollama integration and LRU caching.

Provides text embedding generation using Ollama's nomic-embed-text model (768 dimensions).
Includes LRU caching, batch processing, retry logic, and cosine similarity calculation.

Usage:
    from src.infrastructure.embedding_service import EmbeddingService

    embedder = EmbeddingService()
    embedding = await embedder.embed_text("Bitcoin is a cryptocurrency")
    print(f"Embedding dimension: {len(embedding)}")  # 768

    # Batch processing
    embeddings = await embedder.embed_texts(["text1", "text2", "text3"])

    # Cosine similarity
    sim = embedder.cosine_similarity(emb1, emb2)
    print(f"Similarity: {sim:.3f}")
"""

import asyncio
import hashlib
import time
from functools import lru_cache
from typing import List, Tuple, Optional
import math

import requests

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Embedding service using Ollama with LRU caching.

    Features:
    - Generates 768-dimensional embeddings using nomic-embed-text
    - LRU cache (10,000 entries, 1 hour TTL)
    - Batch processing (10 texts at a time)
    - Exponential backoff retry logic
    - Cosine similarity calculation
    """

    def __init__(
        self,
        cache_max_size: int = None,
        cache_ttl_hours: int = None,
        batch_size: int = 10,
    ):
        """
        Initialize the embedding service.

        Args:
            cache_max_size: Maximum cache entries (default from settings)
            cache_ttl_hours: Cache TTL in hours (default from settings)
            batch_size: Number of texts to process in parallel (default: 10)
        """
        self.ollama_url = settings.ollama_url
        self.model = settings.ollama_embedding_model
        self.batch_size = batch_size

        # Cache configuration
        self.cache_max_size = cache_max_size or settings.cache_max_size
        self.cache_ttl_seconds = (cache_ttl_hours or settings.cache_ttl_hours) * 3600

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Simple in-memory cache with timestamps
        # Format: {text_hash: (embedding, timestamp)}
        self._cache: dict[str, Tuple[List[float], float]] = {}

        logger.info(
            f"Initialized EmbeddingService: model={self.model}, "
            f"cache_size={self.cache_max_size}, ttl={cache_ttl_hours}h"
        )

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available and not expired."""
        cache_key = self._get_cache_key(text)

        if cache_key in self._cache:
            embedding, timestamp = self._cache[cache_key]

            # Check if expired
            if time.time() - timestamp < self.cache_ttl_seconds:
                self.cache_hits += 1
                return embedding
            else:
                # Remove expired entry
                del self._cache[cache_key]

        self.cache_misses += 1
        return None

    def _put_in_cache(self, text: str, embedding: List[float]):
        """Put embedding in cache, enforcing size limit."""
        cache_key = self._get_cache_key(text)

        # Enforce cache size limit (simple FIFO eviction)
        if len(self._cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = (embedding, time.time())

    async def embed_text(self, text: str, retry_count: int = 3) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            retry_count: Number of retries on failure

        Returns:
            768-dimensional embedding vector

        Raises:
            Exception: If all retries fail
        """
        # Check cache first
        cached = self._get_from_cache(text)
        if cached is not None:
            logger.debug(f"Cache hit for text ({len(text)} chars)")
            return cached

        # Generate embedding with retry logic
        for attempt in range(retry_count):
            try:
                # Call Ollama API
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30,
                )

                response.raise_for_status()
                data = response.json()
                embedding = data["embedding"]

                # Validate embedding dimension
                if len(embedding) != 768:
                    raise ValueError(f"Expected 768 dimensions, got {len(embedding)}")

                # Cache the result
                self._put_in_cache(text, embedding)

                logger.debug(f"Generated embedding for text ({len(text)} chars)")
                return embedding

            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Embedding generation failed (attempt {attempt + 1}/{retry_count}): {e}"
                )

                if attempt < retry_count - 1:
                    logger.debug(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {retry_count} attempts failed for embedding generation")
                    raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batch processing.

        Args:
            texts: List of texts to embed

        Returns:
            List of 768-dimensional embedding vectors

        Example:
            ```python
            embedder = EmbeddingService()
            embeddings = await embedder.embed_texts([
                "Bitcoin is a cryptocurrency",
                "Ethereum enables smart contracts",
                "NFTs are non-fungible tokens"
            ])
            print(len(embeddings))  # 3
            print(len(embeddings[0]))  # 768
            ```
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")

        # Process in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Process batch in parallel
            batch_embeddings = await asyncio.gather(
                *[self.embed_text(text) for text in batch]
            )

            embeddings.extend(batch_embeddings)

            logger.debug(
                f"Processed batch {i // self.batch_size + 1} "
                f"({len(batch)} texts)"
            )

        logger.info(
            f"Generated {len(embeddings)} embeddings "
            f"(cache hits: {self.cache_hits}, misses: {self.cache_misses})"
        )

        return embeddings

    @staticmethod
    def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity score (0.0 to 1.0)

        Example:
            ```python
            sim = EmbeddingService.cosine_similarity(embedding1, embedding2)
            print(f"Similarity: {sim:.3f}")
            ```
        """
        # Dot product
        dot_product = sum(a * b for a, b in zip(emb1, emb2))

        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in emb1))
        mag2 = math.sqrt(sum(b * b for b in emb2))

        # Cosine similarity
        if mag1 == 0 or mag2 == 0:
            return 0.0

        similarity = dot_product / (mag1 * mag2)

        # Clamp to [0, 1] (should already be in this range)
        return max(0.0, min(1.0, similarity))

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache hits, misses, hit rate, and size

        Example:
            ```python
            stats = embedder.get_cache_stats()
            print(f"Cache hit rate: {stats['hit_rate']:.1%}")
            ```
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_size": self.cache_max_size,
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Embedding cache cleared")
