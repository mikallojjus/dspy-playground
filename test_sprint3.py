"""
Comprehensive tests for Sprint 3: Deduplication & Reranker

Tests all Sprint 3 components:
- RerankerService
- QuoteDeduplicator
- ClaimDeduplicator
- ConfidenceCalculator
- Full pipeline integration

Usage:
    uv run python test_sprint3.py
"""

import asyncio

from src.infrastructure.reranker_service import RerankerService
from src.deduplication.quote_deduplicator import QuoteDeduplicator
from src.deduplication.claim_deduplicator import ClaimDeduplicator
from src.scoring.confidence_calculator import ConfidenceCalculator
from src.infrastructure.embedding_service import EmbeddingService
from src.extraction.quote_finder import Quote, ClaimWithQuotes
from src.pipeline.extraction_pipeline import ExtractionPipeline
from src.database.connection import get_db_session
from src.database.models import PodcastEpisode


async def test_reranker_service():
    """Test reranker service API calls, caching, and error handling."""
    print("\n" + "="*60)
    print("TEST 1: RerankerService")
    print("="*60)

    reranker = RerankerService()

    await reranker.wait_for_ready()
    print("✓ Reranker service is ready")

    claim = "Bitcoin reached $69,000 in November 2021"
    quotes = [
        "Bitcoin hit its all-time high of $69,000 in November 2021",
        "Cryptocurrency markets were very volatile in 2021",
        "Many investors lost money during the crash",
        "Bitcoin is a decentralized digital currency"
    ]

    results = await reranker.rerank_quotes(claim, quotes, top_k=2)

    assert len(results) == 2, "Should return top 2 results"
    assert results[0]["score"] > results[1]["score"], "Results should be sorted by score"

    print(f"\nTop 2 results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.3f}] {r['text'][:50]}...")

    results2 = await reranker.rerank_quotes(claim, quotes, top_k=2)
    assert results == results2, "Second call should hit cache"

    stats = reranker.get_cache_stats()
    print(f"\nCache stats: {stats['hits']} hits, {stats['misses']} misses")
    assert stats['hits'] >= 1, "Should have at least 1 cache hit"

    print("\n✅ RerankerService test passed!")


async def test_quote_deduplication():
    """Test quote deduplication with position overlap and text similarity."""
    print("\n" + "="*60)
    print("TEST 2: QuoteDeduplicator")
    print("="*60)

    deduplicator = QuoteDeduplicator()

    quotes = [
        Quote("Bitcoin reached $69,000", 0.90, 1500, 1525, "Speaker_1", 90),
        Quote("Bitcoin reached $69,000 in November", 0.92, 1500, 1540, "Speaker_1", 90),
        Quote("BTC hit sixty-nine thousand", 0.88, 1520, 1550, "Speaker_1", 92),
        Quote("Ethereum also increased", 0.85, 2000, 2025, "Speaker_2", 120),
        Quote("ETH went up as well", 0.83, 2005, 2025, "Speaker_2", 121),
    ]

    print(f"\nBefore deduplication: {len(quotes)} quotes")

    unique = deduplicator.deduplicate(quotes)

    print(f"After deduplication: {len(unique)} quotes")

    assert len(unique) < len(quotes), "Should remove duplicates"
    assert all(q.relevance_score > 0 for q in unique), "All quotes should have relevance scores"

    print(f"\nDuplicate reduction: {len(quotes) - len(unique)} removed "
          f"({100 * (len(quotes) - len(unique)) / len(quotes):.1f}%)")

    for i, quote in enumerate(unique[:3], 1):
        print(f"  {i}. [{quote.relevance_score:.3f}] {quote.quote_text[:40]}...")

    print("\n✅ QuoteDeduplicator test passed!")


async def test_claim_deduplication():
    """Test claim deduplication with embedding similarity and reranker verification."""
    print("\n" + "="*60)
    print("TEST 3: ClaimDeduplicator")
    print("="*60)

    embedder = EmbeddingService()
    reranker = RerankerService()
    await reranker.wait_for_ready()

    deduplicator = ClaimDeduplicator(embedder, reranker)

    claims = [
        ClaimWithQuotes(
            "Bitcoin reached $69,000 in November 2021",
            source_chunk_id=0,
            quotes=[
                Quote("BTC hit $69k", 0.90, 1500, 1520, "Speaker_1", 90),
                Quote("Bitcoin peaked at 69000", 0.88, 1550, 1575, "Speaker_1", 92)
            ],
            confidence=0.85
        ),
        ClaimWithQuotes(
            "BTC hit $69k in Nov 2021",
            source_chunk_id=1,
            quotes=[
                Quote("Bitcoin reached all-time high", 0.87, 1600, 1630, "Speaker_2", 95)
            ],
            confidence=0.78
        ),
        ClaimWithQuotes(
            "Ethereum enables smart contracts",
            source_chunk_id=2,
            quotes=[
                Quote("ETH supports programmable money", 0.92, 2000, 2030, "Speaker_3", 120)
            ],
            confidence=0.90
        ),
    ]

    print(f"\nBefore deduplication: {len(claims)} claims")

    deduplicated = await deduplicator.deduplicate_batch(claims)

    print(f"After deduplication: {len(deduplicated)} claims")

    assert len(deduplicated) <= len(claims), "Should not increase number of claims"

    if len(deduplicated) < len(claims):
        print(f"\nDuplicate reduction: {len(claims) - len(deduplicated)} claims merged")

        merged_claims = [c for c in deduplicated if c.metadata.get('merged_from_claims', 0) > 1]
        if merged_claims:
            for claim in merged_claims:
                print(f"\nMerged claim: '{claim.claim_text[:60]}...'")
                print(f"  Merged from: {claim.metadata['merged_from_claims']} claims")
                print(f"  Total quotes: {len(claim.quotes)}")

    print("\n✅ ClaimDeduplicator test passed!")


async def test_confidence_calculator():
    """Test confidence calculation with various quote configurations."""
    print("\n" + "="*60)
    print("TEST 4: ConfidenceCalculator")
    print("="*60)

    calculator = ConfidenceCalculator()

    print("\nScenario 1: High-quality claim (8 quotes, high relevance)")
    quotes1 = [
        Quote("", 0.92, 0, 10, "", 0),
        Quote("", 0.89, 10, 20, "", 0),
        Quote("", 0.87, 20, 30, "", 0),
        Quote("", 0.85, 30, 40, "", 0),
        Quote("", 0.82, 40, 50, "", 0),
        Quote("", 0.80, 50, 60, "", 0),
        Quote("", 0.78, 60, 70, "", 0),
        Quote("", 0.75, 70, 80, "", 0),
    ]

    comp1 = calculator.calculate(quotes1)
    print(f"  Avg relevance: {comp1.avg_relevance:.3f}")
    print(f"  Max relevance: {comp1.max_relevance:.3f}")
    print(f"  Quote count: {comp1.quote_count}")
    print(f"  Count score: {comp1.count_score:.3f}")
    print(f"  → Confidence: {comp1.final_confidence:.3f}")

    assert 0.8 <= comp1.final_confidence <= 1.0, "High-quality claim should have high confidence"

    print("\nScenario 2: Medium-quality claim (3 quotes, medium relevance)")
    quotes2 = [
        Quote("", 0.72, 0, 10, "", 0),
        Quote("", 0.68, 10, 20, "", 0),
        Quote("", 0.65, 20, 30, "", 0),
    ]

    comp2 = calculator.calculate(quotes2)
    print(f"  Avg relevance: {comp2.avg_relevance:.3f}")
    print(f"  Max relevance: {comp2.max_relevance:.3f}")
    print(f"  Quote count: {comp2.quote_count}")
    print(f"  Count score: {comp2.count_score:.3f}")
    print(f"  → Confidence: {comp2.final_confidence:.3f}")

    assert 0.5 <= comp2.final_confidence <= 0.8, "Medium-quality claim should have medium confidence"

    print("\nScenario 3: Low-quality claim (1 quote, low relevance)")
    quotes3 = [Quote("", 0.58, 0, 10, "", 0)]

    comp3 = calculator.calculate(quotes3)
    print(f"  Avg relevance: {comp3.avg_relevance:.3f}")
    print(f"  Max relevance: {comp3.max_relevance:.3f}")
    print(f"  Quote count: {comp3.quote_count}")
    print(f"  Count score: {comp3.count_score:.3f}")
    print(f"  → Confidence: {comp3.final_confidence:.3f}")

    assert 0.3 <= comp3.final_confidence <= 0.6, "Low-quality claim should have low confidence"

    print("\n✅ ConfidenceCalculator test passed!")


async def test_end_to_end_with_dedup():
    """Test full pipeline with deduplication on real episode."""
    print("\n" + "="*60)
    print("TEST 5: End-to-End Pipeline with Deduplication")
    print("="*60)

    session = get_db_session()
    episode = session.query(PodcastEpisode).filter(
        PodcastEpisode.transcript.isnot(None)
    ).first()
    session.close()

    assert episode, "No episodes with transcripts found in database"

    print(f"\nProcessing episode: {episode.name}")
    print(f"Transcript length: {len(episode.transcript):,} chars")

    pipeline = ExtractionPipeline()
    result = await pipeline.process_episode(episode.id)

    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)

    print(f"\nClaims:")
    print(f"  Extracted: {result.stats.claims_extracted}")
    print(f"  After dedup: {result.stats.claims_after_dedup}")
    print(f"  With quotes: {result.stats.claims_with_quotes}")
    if result.stats.claims_extracted > 0:
        claim_reduction = result.stats.claims_extracted - result.stats.claims_after_dedup
        print(f"  Reduction: {claim_reduction} "
              f"({100 * claim_reduction / result.stats.claims_extracted:.1f}%)")

    print(f"\nQuotes:")
    print(f"  Before dedup: {result.stats.quotes_before_dedup}")
    print(f"  After dedup: {result.stats.quotes_after_dedup}")
    if result.stats.quotes_before_dedup > 0:
        quote_reduction = result.stats.quotes_before_dedup - result.stats.quotes_after_dedup
        print(f"  Reduction: {quote_reduction} "
              f"({100 * quote_reduction / result.stats.quotes_before_dedup:.1f}%)")

    print(f"\nFinal results:")
    print(f"  Claims with quotes: {len(result.claims)}")
    print(f"  Total quotes: {result.stats.total_quotes}")
    print(f"  Avg quotes/claim: {result.stats.avg_quotes_per_claim:.1f}")

    print(f"\nProcessing time: {result.stats.processing_time_seconds:.1f}s")

    assert result.stats.claims_after_dedup <= result.stats.claims_extracted, \
        "Deduplicated claims should not exceed extracted claims"
    assert result.stats.quotes_after_dedup <= result.stats.quotes_before_dedup, \
        "Deduplicated quotes should not exceed original quotes"
    assert all(0 <= c.confidence <= 1.0 for c in result.claims), \
        "All confidence scores should be in [0, 1]"
    assert all(c.confidence_components is not None for c in result.claims), \
        "All claims should have confidence components"

    print("\n" + "="*60)
    print("SAMPLE CLAIMS")
    print("="*60)

    for i, claim in enumerate(result.claims[:3], 1):
        print(f"\n{i}. {claim.claim_text}")
        print(f"   Confidence: {claim.confidence:.3f} "
              f"(avg_rel={claim.confidence_components.avg_relevance:.3f}, "
              f"max_rel={claim.confidence_components.max_relevance:.3f}, "
              f"count={claim.confidence_components.quote_count})")

        if claim.metadata.get('merged_from_claims'):
            print(f"   [Merged from {claim.metadata['merged_from_claims']} duplicate claims]")

        print(f"   Quotes ({len(claim.quotes)}):")
        for j, quote in enumerate(claim.quotes[:2], 1):
            print(f"     {j}. [{quote.relevance_score:.3f}] {quote.quote_text[:60]}...")

    print("\n✅ End-to-End Pipeline test passed!")
    print("\n" + "="*60)
    print("ALL SPRINT 3 TESTS PASSED!")
    print("="*60)


async def main():
    """Run all Sprint 3 tests."""
    try:
        await test_reranker_service()
        await test_quote_deduplication()
        await test_claim_deduplication()
        await test_confidence_calculator()
        await test_end_to_end_with_dedup()

        print("\n" + "="*60)
        print("✅ SPRINT 3 ACCEPTANCE TESTS: ALL PASSED")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
