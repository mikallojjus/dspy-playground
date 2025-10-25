"""
Claim deduplicator using embedding similarity and reranker verification.

Deduplicates claims within single episode (batch level) using multi-method
approach: embedding similarity for filtering + reranker for verification.

Usage:
    from src.deduplication.claim_deduplicator import ClaimDeduplicator

    deduplicator = ClaimDeduplicator(embedder, reranker)
    deduplicated = await deduplicator.deduplicate_batch(claims_with_quotes)

    print(f"Deduplicated: {len(claims_with_quotes)} → {len(deduplicated)}")
"""

from typing import List, Tuple

from src.extraction.quote_finder import ClaimWithQuotes
from src.infrastructure.embedding_service import EmbeddingService
from src.infrastructure.reranker_service import RerankerService
from src.deduplication.quote_deduplicator import QuoteDeduplicator
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ClaimDeduplicator:
    """
    Deduplicate claims within episode using embedding similarity and reranker.

    Features:
    - Group similar claims using embedding similarity (>0.85 threshold)
    - Verify duplicates using reranker (>0.9 score = duplicate)
    - Merge duplicate groups intelligently
    - Deduplicate quotes within merged claims
    - Track merge metadata

    Example:
        ```python
        deduplicator = ClaimDeduplicator(embedder, reranker)

        claims = [
            ClaimWithQuotes("Bitcoin reached $69,000", ...),
            ClaimWithQuotes("BTC hit $69k in November 2021", ...),
            ClaimWithQuotes("Bitcoin's price peaked at sixty-nine thousand", ...),
        ]

        deduplicated = await deduplicator.deduplicate_batch(claims)
        # Result: 1 merged claim with combined quotes
        ```
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        reranker: RerankerService,
        embedding_threshold: float = None,
        reranker_threshold: float = None
    ):
        """
        Initialize the claim deduplicator.

        Args:
            embedder: Embedding service for similarity calculation
            reranker: Reranker service for verification
            embedding_threshold: Cosine similarity threshold (default from settings)
            reranker_threshold: Reranker score threshold (default from settings)
        """
        self.embedder = embedder
        self.reranker = reranker
        self.quote_deduplicator = QuoteDeduplicator()

        self.embedding_threshold = embedding_threshold or settings.embedding_similarity_threshold
        self.reranker_threshold = reranker_threshold or settings.reranker_verification_threshold

        logger.info(
            f"Initialized ClaimDeduplicator: "
            f"embedding_threshold={self.embedding_threshold}, "
            f"reranker_threshold={self.reranker_threshold}"
        )

    async def deduplicate_batch(
        self,
        claims_with_quotes: List[ClaimWithQuotes]
    ) -> List[ClaimWithQuotes]:
        """
        Deduplicate claims within episode.

        Algorithm:
        1. Generate embeddings for all claims
        2. Find candidate pairs (cosine similarity > 0.85)
        3. Verify with reranker (batch scoring, threshold > 0.9)
        4. Build groups from verified pairs (union-find)
        5. Merge each group

        Args:
            claims_with_quotes: List of claims with quotes

        Returns:
            List of deduplicated claims

        Example:
            ```python
            deduplicator = ClaimDeduplicator(embedder, reranker)

            claims = [...]  # 15 claims
            deduplicated = await deduplicator.deduplicate_batch(claims)  # 12 claims

            merged_claims = [c for c in deduplicated if c.metadata.get('merged_from_claims')]
            print(f"Merged {len(merged_claims)} claim groups")
            ```
        """
        if len(claims_with_quotes) <= 1:
            logger.info("Only 1 claim, skipping deduplication")
            return claims_with_quotes

        logger.info(f"Deduplicating {len(claims_with_quotes)} claims...")

        claim_texts = [c.claim_text for c in claims_with_quotes]
        embeddings = await self.embedder.embed_texts(claim_texts)

        candidate_pairs = self._find_similar_pairs(embeddings)

        logger.debug(f"Found {len(candidate_pairs)} candidate pairs")

        if not candidate_pairs:
            logger.info("No candidate pairs found, no deduplication needed")
            return claims_with_quotes

        verified_pairs = await self._verify_with_reranker(candidate_pairs, claim_texts)

        logger.debug(f"Verified {len(verified_pairs)} duplicate pairs with reranker")

        if not verified_pairs:
            logger.info("No duplicates verified by reranker")
            return claims_with_quotes

        groups = self._build_duplicate_groups(verified_pairs, len(claims_with_quotes))

        deduplicated = []
        for group in groups:
            if len(group) == 1:
                deduplicated.append(claims_with_quotes[group[0]])
            else:
                group_claims = [claims_with_quotes[i] for i in group]
                merged = self._merge_claim_group(group_claims)
                deduplicated.append(merged)

        logger.info(
            f"Deduplicated {len(claims_with_quotes)} → {len(deduplicated)} claims "
            f"({len(claims_with_quotes) - len(deduplicated)} duplicates removed, "
            f"{100 * (len(claims_with_quotes) - len(deduplicated)) / len(claims_with_quotes):.1f}% reduction)"
        )

        return deduplicated

    def _find_similar_pairs(
        self,
        embeddings: List[List[float]]
    ) -> List[Tuple[int, int]]:
        """
        Find pairs with cosine similarity above threshold.

        Args:
            embeddings: List of claim embeddings

        Returns:
            List of (index1, index2) pairs
        """
        pairs = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self.embedder.cosine_similarity(embeddings[i], embeddings[j])

                if similarity >= self.embedding_threshold:
                    pairs.append((i, j))
                    logger.debug(
                        f"Candidate pair: claims {i} and {j} "
                        f"(similarity: {similarity:.3f})"
                    )

        return pairs

    async def _verify_with_reranker(
        self,
        candidate_pairs: List[Tuple[int, int]],
        claim_texts: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Verify candidate pairs using reranker.

        Args:
            candidate_pairs: List of (index1, index2) candidate pairs
            claim_texts: List of claim texts

        Returns:
            List of verified duplicate pairs
        """
        verified = []

        for i, j in candidate_pairs:
            results = await self.reranker.rerank_quotes(
                claim_texts[i],
                [claim_texts[j]],
                top_k=1
            )

            if results and results[0]["score"] >= self.reranker_threshold:
                verified.append((i, j))
                logger.debug(
                    f"Verified duplicate: claims {i} and {j} "
                    f"(reranker score: {results[0]['score']:.3f})"
                )
            else:
                score = results[0]["score"] if results else 0.0
                logger.debug(
                    f"Not duplicate: claims {i} and {j} "
                    f"(reranker score: {score:.3f} < {self.reranker_threshold})"
                )

        return verified

    def _build_duplicate_groups(
        self,
        verified_pairs: List[Tuple[int, int]],
        num_claims: int
    ) -> List[List[int]]:
        """
        Build groups from pairwise duplicate relationships using union-find.

        Args:
            verified_pairs: List of verified duplicate pairs
            num_claims: Total number of claims

        Returns:
            List of groups (each group is list of claim indices)

        Example:
            ```python
            pairs = [(0, 1), (1, 2), (3, 4)]
            groups = self._build_duplicate_groups(pairs, 5)
            # Result: [[0, 1, 2], [3, 4]]
            # Claims 0, 1, 2 form one group, claims 3, 4 form another
            ```
        """
        parent = list(range(num_claims))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in verified_pairs:
            union(i, j)

        groups_dict = {}
        for i in range(num_claims):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(i)

        groups = list(groups_dict.values())

        logger.debug(
            f"Built {len(groups)} groups from {len(verified_pairs)} pairs: "
            f"{[len(g) for g in groups]}"
        )

        return groups

    def _merge_claim_group(
        self,
        group: List[ClaimWithQuotes]
    ) -> ClaimWithQuotes:
        """
        Merge duplicate claims.

        Merge strategy:
        - Keep claim text with highest confidence
        - Combine all quotes from all claims
        - Deduplicate combined quotes
        - Track merge metadata

        Args:
            group: List of duplicate claims to merge

        Returns:
            Merged claim with combined quotes

        Example:
            ```python
            group = [
                ClaimWithQuotes("Bitcoin reached $69,000", confidence=0.85, quotes=[...]),
                ClaimWithQuotes("BTC hit $69k", confidence=0.78, quotes=[...]),
            ]

            merged = self._merge_claim_group(group)
            # Result: claim text from first (higher confidence),
            #         combined and deduplicated quotes from both
            ```
        """
        best_claim = max(group, key=lambda c: c.confidence)

        all_quotes = []
        for claim in group:
            all_quotes.extend(claim.quotes)

        logger.debug(f"Merging {len(group)} claims with {len(all_quotes)} total quotes...")

        unique_quotes = self.quote_deduplicator.deduplicate(all_quotes)

        logger.debug(f"Deduplicated quotes: {len(all_quotes)} → {len(unique_quotes)}")

        merged = ClaimWithQuotes(
            claim_text=best_claim.claim_text,
            source_chunk_id=best_claim.source_chunk_id,
            quotes=unique_quotes,
            confidence=best_claim.confidence,
            metadata={
                "merged_from_claims": len(group),
                "original_quote_count": len(all_quotes),
                "deduplicated_quote_count": len(unique_quotes)
            }
        )

        logger.debug(
            f"Merged {len(group)} claims: "
            f"'{best_claim.claim_text[:60]}...' "
            f"({len(all_quotes)} quotes → {len(unique_quotes)} unique)"
        )

        return merged
