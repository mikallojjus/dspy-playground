"""
Reranker service for high-precision semantic relevance scoring.

Uses BGE reranker v2-m3 via HTTP API (Docker container on localhost:8080).
Provides batch reranking with LRU caching for performance.

Usage:
    from src.infrastructure.reranker_service import RerankerService

    reranker = RerankerService()
    await reranker.wait_for_ready()

    results = await reranker.rerank_quotes(
        claim="Bitcoin reached $69,000",
        quotes=["BTC hit $69k", "Crypto was volatile", ...],
        top_k=10
    )
"""

import asyncio
import hashlib
import time
from typing import List, Dict, Tuple, Optional

import requests

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class RerankerService:
    """
    Reranker service using BGE reranker v2-m3.

    Features:
    - HTTP API client for Docker reranker service
    - Batch reranking (30-50 pairs per call)
    - LRU cache (10,000 entries, 1 hour TTL)
    - Retry logic with exponential backoff
    - Mandatory dependency (throws error if unavailable)
    """

    def __init__(
        self,
        cache_max_size: Optional[int] = None,
        cache_ttl_hours: Optional[int] = None
    ):
        """
        Initialize the reranker service.

        Args:
            cache_max_size: Maximum cache entries (default from settings)
            cache_ttl_hours: Cache TTL in hours (default from settings)
        """
        self.reranker_url = settings.reranker_url
        self.timeout = settings.reranker_timeout / 1000
        self.enabled = settings.enable_reranker

        self.cache_max_size = cache_max_size or settings.cache_max_size
        self.cache_ttl_seconds = (cache_ttl_hours or settings.cache_ttl_hours) * 3600

        self.cache_hits = 0
        self.cache_misses = 0

        self._cache: dict[str, Tuple[List[Dict], float]] = {}

        logger.info(
            f"Initialized RerankerService: url={self.reranker_url}, "
            f"enabled={self.enabled}, cache_size={self.cache_max_size}"
        )

    def _get_cache_key(self, claim: str, quotes: List[str]) -> str:
        """Generate cache key from claim and quotes."""
        combined = f"{claim}::{'||'.join(quotes[:5])}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Get reranker results from cache if available and not expired."""
        if cache_key in self._cache:
            results, timestamp = self._cache[cache_key]

            if time.time() - timestamp < self.cache_ttl_seconds:
                self.cache_hits += 1
                return results
            else:
                del self._cache[cache_key]

        self.cache_misses += 1
        return None

    def _put_in_cache(self, cache_key: str, results: List[Dict]):
        """Put reranker results in cache, enforcing size limit."""
        if len(self._cache) >= self.cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = (results, time.time())

    async def wait_for_ready(self, max_attempts: int = 5, delay: int = 2):
        """
        Wait for reranker service to be ready.

        Args:
            max_attempts: Maximum number of connection attempts
            delay: Delay between attempts in seconds

        Raises:
            RuntimeError: If service unavailable after all attempts
        """
        if not self.enabled:
            logger.warning("Reranker service is disabled in settings")
            return

        logger.info(f"Checking reranker service at {self.reranker_url}...")

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(
                    f"{self.reranker_url}/health",
                    timeout=2
                )

                if response.status_code == 200:
                    logger.info("✅ Reranker service is ready")
                    return

                logger.warning(
                    f"Reranker health check returned {response.status_code} "
                    f"(attempt {attempt}/{max_attempts})"
                )

            except Exception as e:
                logger.warning(
                    f"Reranker connection failed (attempt {attempt}/{max_attempts}): {e}"
                )

                if attempt < max_attempts:
                    logger.debug(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"❌ Reranker service unavailable at {self.reranker_url}. "
            f"Ensure Docker container is running: "
            f"docker-compose -f docker-compose.reranker.yml up -d"
        )

    async def rerank_quotes(
        self,
        claim: str,
        quotes: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank quotes by relevance to claim.

        Args:
            claim: The claim text
            quotes: List of quote texts to rank
            top_k: Number of top results to return (default: all)

        Returns:
            List of dicts with keys: text, score, index
            Sorted by score (highest first)

        Raises:
            RuntimeError: If reranker API call fails after retries

        Example:
            ```python
            reranker = RerankerService()
            await reranker.wait_for_ready()

            results = await reranker.rerank_quotes(
                "Bitcoin reached $69,000",
                ["BTC hit $69k", "Crypto was volatile", "..."],
                top_k=10
            )

            for r in results:
                print(f"[{r['score']:.3f}] {r['text']}")
            ```
        """
        if not self.enabled:
            raise RuntimeError(
                "Reranker service is disabled. "
                "Set ENABLE_RERANKER=true in .env to enable."
            )

        if not quotes:
            return []

        cache_key = self._get_cache_key(claim, quotes)
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            logger.debug(f"Cache hit for reranking ({len(quotes)} quotes)")
            return cached[:top_k] if top_k else cached

        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.reranker_url}/rerank",
                    json={
                        "query": claim,
                        "texts": quotes,
                        "truncate": True
                    },
                    timeout=self.timeout
                )

                response.raise_for_status()
                data = response.json()

                scored = [
                    {
                        "text": quotes[item["index"]],
                        "score": item["score"],
                        "index": item["index"]
                    }
                    for item in data
                ]

                scored.sort(key=lambda x: x["score"], reverse=True)

                self._put_in_cache(cache_key, scored)

                logger.debug(
                    f"Reranked {len(quotes)} quotes "
                    f"(score range: {scored[0]['score']:.3f} - {scored[-1]['score']:.3f})"
                )

                return scored[:top_k] if top_k else scored

            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Reranker API call failed (attempt {attempt + 1}/3): {e}"
                )

                if attempt < 2:
                    logger.debug(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Reranker API call failed after 3 attempts: {e}"
                    ) from e

        # This line should never be reached, but satisfies type checker
        raise RuntimeError("Reranker failed - all retries exhausted")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache hits, misses, hit rate, and size
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
        """Clear the reranker cache."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Reranker cache cleared")
