# Production Claim-Quote Extraction System - BACKLOG

**Last Updated:** 2025-10-25
**Status:** Planning → Implementation
**Goal:** Build production-ready claim-quote extraction pipeline in Python with DSPy optimization

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What We're Building](#what-were-building)
3. [What We've Proven (Experiments)](#what-weve-proven-experiments)
4. [Sprint-by-Sprint Backlog](#sprint-by-sprint-backlog)
5. [Technical Architecture](#technical-architecture)
6. [Database Schema](#database-schema)
7. [Configuration](#configuration)
8. [Success Criteria](#success-criteria)

---

## Executive Summary

### The Mission

Build production software that extracts high-quality factual claims from podcast transcripts, finds supporting quotes, validates them with entailment checking, and stores everything in PostgreSQL with comprehensive deduplication.

### Key Differences from Old Implementation

| Old (TypeScript) | New (Python + DSPy) |
|-----------------|---------------------|
| Hardcoded prompts | DSPy-optimized modules |
| Pattern matching quality checks | LLM-as-judge metrics |
| Manual prompt tuning | Data-driven optimization |
| ~40% low-quality claims | ~9% low-quality claims (proven) |
| No entailment validation | LLM-based entailment filtering |
| Flies blind on quality | Measurable, improvable |

### What We're Keeping from Old Architecture

✅ **Two-pass extraction** (LLM extract → global quote search)
✅ **Three-tier deduplication** (quotes, batch claims, DB claims)
✅ **Reranker service** for high-precision duplicate detection
✅ **Semantic search** with embeddings
✅ **Position-based quote deduplication**
✅ **Weighted confidence scoring**
✅ **Speaker-aware parsing**
✅ **Chunking with overlap**
✅ **LRU caching** for embeddings/reranker
✅ **PostgreSQL with pgvector**
✅ **Many-to-many claim-quote relationships**

### Development Approach

**Agile & Incremental:**

- 5 weekly sprints, each delivering working software
- Test with real database after each sprint
- Use proven DSPy models from experiments
- Start simple, add complexity incrementally
- Production-quality code from day one

---

## What We're Building

### Full Pipeline Flow

```
┌────────────────────────────────────────────────────────────────┐
│ INPUT: Episode ID                                              │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 1. FETCH EPISODE FROM POSTGRESQL                               │
│    - Load transcript                                           │
│    - Check if already processed (unless --force)               │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 2. PREPROCESSING                                               │
│    - Parse speakers & timestamps (TranscriptParser)           │
│    - Chunk transcript (16K chars, 1K overlap)                 │
│    - Build semantic search index (embeddings)                 │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 3. PASS 1: CLAIM EXTRACTION (DSPy Optimized)                  │
│    - Process chunks in parallel (3 at a time)                 │
│    - Use optimized DSPy model:                                │
│      claim_extractor_llm_judge_v1.json                       │
│    - Extract claims from each chunk                           │
│    - Track source chunk IDs                                   │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 4. PASS 2: GLOBAL QUOTE SEARCH                                │
│    For each claim:                                             │
│    - Semantic search → top 30 candidates                      │
│    - Filter questions (SentenceClassifier)                    │
│    - Rerank with reranker service                            │
│    - Select top 10 most relevant quotes                       │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 5. ENTAILMENT VALIDATION (DSPy Optimized)                     │
│    For each claim-quote pair:                                  │
│    - Use optimized DSPy entailment model                      │
│    - Classify: SUPPORTS/RELATED/NEUTRAL/CONTRADICTS          │
│    - Filter out non-SUPPORTS quotes                          │
│    - Keep only quotes that genuinely support claims          │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 6. DEDUPLICATION - TIER 1: Quotes                             │
│    - Collect all quotes from all claims                       │
│    - Position-based deduplication (50% overlap)               │
│    - Text similarity fallback (80% threshold)                 │
│    - Merge duplicates (keep longest, highest relevance)      │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 7. DEDUPLICATION - TIER 2: Batch Claims (Within Episode)      │
│    - Embed all claims                                         │
│    - Group similar (embedding similarity > 0.85)              │
│    - Verify with reranker (score > 0.9 = duplicate)         │
│    - Merge duplicate groups (keep best, combine quotes)      │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 8. CONFIDENCE CALCULATION                                      │
│    For each claim:                                             │
│    - confidence = (avgRelevance × 0.6) +                      │
│                   (maxRelevance × 0.2) +                      │
│                   (quoteCount/5 × 0.2)                        │
│    - Store confidence components for debugging                │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 9. DEDUPLICATION - TIER 3: Database (Cross-Episode)           │
│    For each claim:                                             │
│    - pgvector search: similar claims (L2 distance < 0.15)    │
│    - Reranker verify: score > 0.9 = duplicate                │
│    - If duplicate:                                            │
│      - Compare confidence → keep better claim text           │
│      - Add new episode's quotes to existing claim            │
│      - Update confidence with merged quotes                  │
│    - If unique: prepare for insertion                        │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ 10. SAVE TO POSTGRESQL                                         │
│     - Insert/update claims with embeddings                    │
│     - Insert unique quotes                                    │
│     - Create claim_quote junction records                     │
│     - Store relevance scores per pair                         │
│     - Commit transaction                                      │
└────────────┬───────────────────────────────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────────────┐
│ OUTPUT: Claims, Quotes, and ClaimQuote records in database    │
└────────────────────────────────────────────────────────────────┘
```

---

## What We've Proven (Experiments)

From our successful experiments, we know:

### ✅ Claim Extraction Works (exp_3_1c)

- **Model:** `claim_extractor_llm_judge_v1.json`
- **Quality:** 90.9% (9.1% low-quality claims, target <15%)
- **Improvement:** +21.2 percentage points vs baseline
- **Optimizer:** BootstrapFewShot with 4 few-shot examples
- **Metric:** LLM-as-judge (semantic quality evaluation)
- **Training data:** 14 positive examples (20 good claims total)

**Key learnings:**

- LLM-as-judge > pattern matching (understands context)
- BootstrapFewShot works great (10 min runs vs hours for MIPROv2)
- 20 good examples sufficient (don't need 200+)
- Positive-only training effective

### 🔄 Entailment Validation Needs Building

- **Status:** Not yet optimized
- **Plan:** Create dataset similar to claims_manual_review.json
- **Target:** <10% false positive rate (RELATED → SUPPORTS)
- **Approach:** Same as claim extraction (LLM judge + BootstrapFewShot)

### 🎯 Production Integration Path

Experiments proved DSPy works. Now we build production software:

1. ✅ DSPy claim extraction → works, use in production
2. 🔄 DSPy entailment validation → need to build
3. 🔄 Database integration → need to build
4. 🔄 Two-pass pipeline → need to build
5. 🔄 Deduplication (3-tier) → need to build
6. 🔄 Reranker integration → need to build

---

## Sprint-by-Sprint Backlog

### SPRINT 1: Foundation & Data Layer (Week 1)

**Goal:** Database connectivity, basic models, DSPy integration ready

#### Tasks

**1.1: PostgreSQL Connection & Models**

- [ ] Create `src/database/` module
- [ ] Implement PostgreSQL connection with `psycopg2`
- [ ] Create Pydantic models:
  - `PodcastEpisode` (id, transcript, name, etc.)
  - `Claim` (id, episode_id, claim_text, confidence, embedding, etc.)
  - `Quote` (id, episode_id, quote_text, start_position, end_position, speaker, etc.)
  - `ClaimQuote` (claim_id, quote_id, relevance_score, match_type, etc.)
- [ ] Add pgvector support for embeddings
- [ ] Test connection to existing database

**Acceptance Criteria:**

```python
# This should work:
from src.database.models import PodcastEpisode, Claim, Quote
from src.database.connection import get_db_session

session = get_db_session()
episodes = session.query(PodcastEpisode).filter(
    PodcastEpisode.transcript.isnot(None)
).limit(5).all()
print(f"Found {len(episodes)} episodes with transcripts")
```

**1.2: DSPy Model Loader**

- [ ] Create `src/dspy_models/` module
- [ ] Implement `ClaimExtractorModel` class
  - Loads `models/claim_extractor_llm_judge_v1.json`
  - Wraps DSPy ChainOfThought(ClaimExtraction)
  - Handles errors gracefully
- [ ] Create `EntailmentValidatorModel` class (placeholder for now)
- [ ] Test claim extraction with optimized model

**Acceptance Criteria:**

```python
from src.dspy_models.claim_extractor import ClaimExtractorModel

extractor = ClaimExtractorModel()
claims = extractor.extract_claims("Bitcoin hit $69,000 in November 2021.")
assert len(claims) > 0
assert "Bitcoin" in claims[0]
```

**1.3: Embedding Service with Caching**

- [ ] Create `src/infrastructure/embedding_service.py`
- [ ] Implement `EmbeddingService` class:
  - Uses Ollama nomic-embed-text (768 dims)
  - LRU cache (10,000 entries, 1 hour TTL)
  - Batch processing (10 texts at a time)
  - Exponential backoff retry logic
  - Cosine similarity helper
- [ ] Test with sample texts
- [ ] Verify cache hit rates

**Acceptance Criteria:**

```python
from src.infrastructure.embedding_service import EmbeddingService

embedder = EmbeddingService()
emb = await embedder.embed_text("Bitcoin is a cryptocurrency")
assert len(emb) == 768
assert 0.0 <= abs(emb[0]) <= 1.0

# Second call should hit cache
emb2 = await embedder.embed_text("Bitcoin is a cryptocurrency")
assert emb == emb2  # Exact same result from cache
```

**1.4: Configuration Management**

- [ ] Create `src/config/settings.py`
- [ ] Load configuration from environment variables (.env)
- [ ] Define settings for:
  - Database connection
  - Ollama URL/models
  - Reranker URL
  - Chunking parameters
  - Deduplication thresholds
  - Parallel processing settings
- [ ] Create `.env.example` template

**Acceptance Criteria:**

```python
from src.config.settings import settings

assert settings.ollama_url == "http://localhost:11434"
assert settings.chunk_size == 16000
assert settings.reranker_threshold == 0.9
```

**Sprint 1 Deliverable:**

- Database connection working
- Can load episodes from PostgreSQL
- DSPy claim extractor loaded and ready
- Embedding service functional with caching
- Configuration system in place

---

### SPRINT 2: Core Extraction Pipeline (Week 2)

**Goal:** Two-pass extraction working end-to-end (no dedup yet)

#### Tasks

**2.1: Transcript Preprocessing**

- [ ] Create `src/preprocessing/transcript_parser.py`
- [ ] Implement `TranscriptParser` class:
  - Parse speaker format: `1 (21m 33s): text`
  - Extract speakers, timestamps, positions
  - Return clean text (no timestamps) for LLM
  - Return segments with positions for quote extraction
- [ ] Handle multi-line speaker segments
- [ ] Test with real podcast transcript

**Acceptance Criteria:**

```python
from src.preprocessing.transcript_parser import TranscriptParser

parser = TranscriptParser()
result = parser.parse(transcript)
assert len(result.segments) > 0
assert result.segments[0].speaker == "Speaker_1"
assert result.segments[0].clean_text  # No timestamps
assert result.segments[0].start_position >= 0
```

**2.2: Chunking Service**

- [ ] Create `src/preprocessing/chunking_service.py`
- [ ] Implement `ChunkingService` class:
  - Split text into chunks (16K chars default)
  - 1K overlap between chunks
  - Find sentence boundaries (don't cut mid-sentence)
  - Track chunk positions in original text
- [ ] Test with various transcript sizes

**Acceptance Criteria:**

```python
from src.preprocessing.chunking_service import ChunkingService

chunker = ChunkingService(max_chunk_size=16000, overlap=1000)
chunks = chunker.chunk_text(long_transcript)
assert all(len(c.text) <= 16500 for c in chunks)  # Some margin
assert chunks[1].text[:1000] == chunks[0].text[-1000:]  # Overlap
```

**2.3: Pass 1 - Claim Extraction**

- [ ] Create `src/extraction/claim_extractor.py`
- [ ] Implement `ClaimExtractor` class:
  - Takes chunks as input
  - Uses DSPy `ClaimExtractorModel`
  - Processes chunks in parallel (3 at a time)
  - Tracks source chunk for each claim
  - Returns raw claims (no dedup yet)
- [ ] Add error handling per chunk
- [ ] Test with multi-chunk transcript

**Acceptance Criteria:**

```python
from src.extraction.claim_extractor import ClaimExtractor

extractor = ClaimExtractor()
claims = await extractor.extract_from_chunks(chunks)
assert len(claims) > 0
assert all(hasattr(c, 'source_chunk_id') for c in claims)
```

**2.4: Semantic Search Index**

- [ ] Create `src.search/transcript_search_index.py`
- [ ] Implement `TranscriptSearchIndex` class:
  - Build index from transcript segments
  - Create windowed segments (2-3 sentences)
  - Generate embeddings for all segments
  - Semantic search: find top K similar segments
  - Return segments with positions for quote extraction
- [ ] Optimize for speed (in-memory, no database)

**Acceptance Criteria:**

```python
from src.search.transcript_search_index import TranscriptSearchIndex

index = await TranscriptSearchIndex.build_from_transcript(
    transcript, parser, embedder
)
candidates = await index.find_quotes_for_claim(
    "Bitcoin reached $69,000",
    top_k=30
)
assert len(candidates) == 30
assert all(hasattr(c, 'similarity_score') for c in candidates)
```

**2.5: Pass 2 - Global Quote Search**

- [ ] Create `src/extraction/quote_finder.py`
- [ ] Implement `QuoteFinder` class:
  - Takes claims + search index
  - For each claim:
    - Semantic search → top 30 candidates
    - Filter questions (simple heuristic for now)
    - Score with embeddings (reranker in Sprint 3)
    - Select top 10 quotes
  - Calculate initial relevance scores
  - Return claims with quotes

**Acceptance Criteria:**

```python
from src.extraction.quote_finder import QuoteFinder

finder = QuoteFinder(search_index, embedder)
claims_with_quotes = await finder.find_quotes_for_claims(claims)
assert all(len(c.quotes) <= 10 for c in claims_with_quotes)
assert all(q.relevance_score > 0 for c in claims_with_quotes for q in c.quotes)
```

**2.6: End-to-End Pipeline (No Dedup/Entailment)**

- [ ] Create `src/pipeline/extraction_pipeline.py`
- [ ] Implement `ExtractionPipeline` class:
  - Orchestrates: parse → chunk → extract → search
  - Takes episode as input
  - Returns claims with quotes
  - No deduplication yet
  - No entailment filtering yet
- [ ] Test with 1 real episode from database

**Acceptance Criteria:**

```python
from src.pipeline.extraction_pipeline import ExtractionPipeline

pipeline = ExtractionPipeline()
result = await pipeline.process_episode(episode)
assert len(result.claims) > 0
assert all(len(c.quotes) > 0 for c in result.claims)
print(f"Extracted {len(result.claims)} claims with quotes")
```

**Sprint 2 Deliverable:**

- Full two-pass extraction working
- Claims extracted from real transcripts
- Global quote search functional
- End-to-end pipeline (without dedup/entailment)
- Can process 1 episode successfully

---

### SPRINT 3: Deduplication & Reranker (Week 3)

**Goal:** Three-tier deduplication working with reranker service

#### Tasks

**3.1: Reranker Service Integration**

- [ ] Create `src/infrastructure/reranker_service.py`
- [ ] Implement `RerankerService` class:
  - HTTP client for reranker API (localhost:8080)
  - Batch reranking (30-50 pairs per call)
  - LRU cache (10,000 entries)
  - Fallback to embedding similarity if unavailable
  - Retry logic with timeout
- [ ] Verify reranker Docker container running
- [ ] Test with sample claim-quote pairs

**Acceptance Criteria:**

```python
from src.infrastructure.reranker_service import RerankerService

reranker = RerankerService()
await reranker.wait_for_ready()  # Ensure service is up

scores = await reranker.rerank_quotes(
    claim="Bitcoin reached $69,000",
    quotes=["BTC hit $69k", "Crypto was volatile", "..."],
    top_k=10
)
assert len(scores) == 10
assert scores[0].score > scores[-1].score  # Sorted by score
```

**3.2: Quote Deduplicator (Position-Based)**

- [ ] Create `src/deduplication/quote_deduplicator.py`
- [ ] Implement `QuoteDeduplicator` class:
  - Sort quotes by transcript position
  - Detect position overlap (>50% = duplicate)
  - Text similarity fallback (Jaccard 80%)
  - Merge duplicates (keep longest, highest relevance)
  - Text normalization helper
- [ ] Test with overlapping quotes

**Acceptance Criteria:**

```python
from src.deduplication.quote_deduplicator import QuoteDeduplicator

deduplicator = QuoteDeduplicator()
unique_quotes = deduplicator.deduplicate(all_quotes)
assert len(unique_quotes) < len(all_quotes)  # Should remove some
# Verify no position overlaps in result
for i, q1 in enumerate(unique_quotes):
    for q2 in unique_quotes[i+1:]:
        assert not deduplicator.has_position_overlap(q1, q2)
```

**3.3: Claim Deduplicator - Batch Level**

- [ ] Create `src/deduplication/claim_deduplicator.py`
- [ ] Implement `ClaimDeduplicator` class (batch level):
  - Group claims by embedding similarity (>0.85)
  - Verify duplicates with reranker (>0.9)
  - Merge duplicate groups:
    - Keep highest confidence claim text
    - Combine all quotes from all claims
    - Deduplicate combined quotes
    - Recalculate confidence
  - Track merge metadata
- [ ] Test with similar claims

**Acceptance Criteria:**

```python
from src.deduplication.claim_deduplicator import ClaimDeduplicator

deduplicator = ClaimDeduplicator(embedder, reranker)
deduplicated = await deduplicator.deduplicate_batch(claims)
assert len(deduplicated) <= len(claims)
# Verify merged claims have combined quotes
merged = [c for c in deduplicated if c.metadata.get('merged_from_claims')]
assert all(len(c.quotes) >= 2 for c in merged)
```

**3.4: Confidence Calculator**

- [ ] Create `src/scoring/confidence_calculator.py`
- [ ] Implement `ConfidenceCalculator` class:
  - Weighted formula:
    - avgRelevance × 0.6
    - maxRelevance × 0.2
    - quoteCount/5 × 0.2
  - Store confidence components
  - Clamp to [0, 1]
- [ ] Test with various quote configurations

**Acceptance Criteria:**

```python
from src.scoring.confidence_calculator import ConfidenceCalculator

calc = ConfidenceCalculator()
confidence = calc.calculate(
    avg_relevance=0.85,
    max_relevance=0.92,
    quote_count=8
)
assert 0.0 <= confidence <= 1.0
assert confidence > 0.8  # Should be high with these numbers
```

**3.5: Integrate Deduplication into Pipeline**

- [ ] Update `ExtractionPipeline` to use:
  - Quote deduplication (after Pass 2)
  - Batch claim deduplication (within episode)
  - Confidence calculation
- [ ] Add reranker scoring to quote selection
- [ ] Test with episode that has duplicate claims/quotes

**Acceptance Criteria:**

```python
# Pipeline now produces deduplicated claims
result = await pipeline.process_episode(episode)
# Should have fewer claims than before dedup
# All quotes should be unique by position
```

**Sprint 3 Deliverable:**

- Reranker service integrated
- Quote deduplication working (position-based)
- Batch claim deduplication working
- Confidence scoring implemented
- Pipeline produces high-quality deduplicated claims

---

### SPRINT 4: Entailment & Database Deduplication (Week 4)

**Goal:** Entailment filtering + cross-episode deduplication

#### Tasks

**4.1: Create Entailment Dataset**

- [ ] Extract 20-30 claim-quote pairs from existing DB
- [ ] Manually label each pair:
  - SUPPORTS: Quote directly validates claim
  - RELATED: Topically related but doesn't validate
  - NEUTRAL: Unrelated
  - CONTRADICTS: Contradicts claim
- [ ] Save in `evaluation/entailment_manual_review.json`
  - Same format as claims_manual_review.json
  - Include claim, quote, relationship, reasoning
- [ ] Split into train (70%) and val (30%)

**Example format:**

```json
{
  "examples": [
    {
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quote": "Bitcoin hit its all-time high of $69,000 in November",
      "relationship": "SUPPORTS",
      "reasoning": "Quote directly states the claim with exact figures",
      "confidence": 1.0
    },
    {
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quote": "Cryptocurrency markets were volatile in 2021",
      "relationship": "RELATED",
      "reasoning": "Topically related but doesn't mention Bitcoin's price",
      "confidence": 0.9
    }
  ]
}
```

**4.2: Build LLM-as-Judge Metric for Entailment**

- [ ] Create `src/metrics/entailment_metrics.py`
- [ ] Implement `entailment_llm_judge_metric`:
  - Uses LLM to evaluate relationship
  - Returns 1.0 if correct, 0.0 if wrong
  - Extra penalty (-2.0) for RELATED→SUPPORTS (false positive)
- [ ] Test on validation set

**Acceptance Criteria:**

```python
from src.metrics.entailment_metrics import entailment_llm_judge_metric

# Test with labeled example
example = dspy.Example(
    claim="Bitcoin reached $69,000",
    quote="Bitcoin hit $69k in November",
    relationship="SUPPORTS"
).with_inputs('claim', 'quote')

pred = dspy.Example(relationship="SUPPORTS")
score = entailment_llm_judge_metric(example, pred)
assert score == 1.0  # Correct classification
```

**4.3: Optimize Entailment with DSPy**

- [ ] Create `src/experiments/exp_4_1_optimize_entailment.py`
- [ ] Define `EntailmentValidation` signature
- [ ] Load entailment train/val datasets
- [ ] Run BootstrapFewShot optimization
- [ ] Target: <10% false positive rate
- [ ] Save optimized model: `models/entailment_validator_v1.json`

**Acceptance Criteria:**

```python
# After optimization:
# - Model saved to models/entailment_validator_v1.json
# - Validation accuracy > 90%
# - False positive rate < 10%
# - Ready for production use
```

**4.4: Entailment Validator Module**

- [ ] Create `src/dspy_models/entailment_validator.py`
- [ ] Implement `EntailmentValidatorModel` class:
  - Loads `models/entailment_validator_v1.json`
  - Takes claim-quote pairs
  - Returns relationship classification
  - Filters out non-SUPPORTS quotes
- [ ] Integrate into pipeline

**Acceptance Criteria:**

```python
from src.dspy_models.entailment_validator import EntailmentValidatorModel

validator = EntailmentValidatorModel()
result = validator.validate(
    claim="Bitcoin reached $69,000",
    quote="Bitcoin hit $69k in November"
)
assert result.relationship == "SUPPORTS"
assert result.confidence > 0.8
```

**4.5: Database-Level Claim Deduplication**

- [ ] Update `ClaimDeduplicator` with `deduplicate_against_database` method:
  - pgvector similarity search (L2 distance < 0.15)
  - Returns top 10 similar claims
  - Verify with reranker (score > 0.9 = duplicate)
  - If duplicate found:
    - Compare confidence scores
    - Merge quotes (add new episode's quotes to existing claim)
    - Update confidence
  - If unique: return None (will be inserted as new)
- [ ] Test with claims already in database

**Acceptance Criteria:**

```python
from src.deduplication.claim_deduplicator import ClaimDeduplicator

deduplicator = ClaimDeduplicator(embedder, reranker, db_session)
result = await deduplicator.deduplicate_against_database(
    claim=new_claim,
    episode_id=current_episode_id
)
if result.is_duplicate:
    print(f"Duplicate of claim {result.existing_claim_id}")
    print(f"Will add {len(new_claim.quotes)} quotes to existing claim")
else:
    print("Unique claim, will insert as new")
```

**4.6: Database Persistence Layer**

- [ ] Create `src/database/claim_repository.py`
- [ ] Implement `ClaimRepository` class:
  - `save_claims(claims, episode_id)`: Insert/update claims
  - `save_quotes(quotes, episode_id)`: Insert unique quotes
  - `create_claim_quote_links(claim_id, quote_ids, relevance_scores)`
  - Handle duplicates (reuse existing quotes when possible)
  - Transaction management
  - Error handling
- [ ] Test with real claims

**Acceptance Criteria:**

```python
from src.database.claim_repository import ClaimRepository

repo = ClaimRepository(db_session)
claim_ids = await repo.save_claims(claims, episode_id)
assert len(claim_ids) == len(claims)

# Verify claims in database
saved = db_session.query(Claim).filter(Claim.id.in_(claim_ids)).all()
assert all(c.embedding is not None for c in saved)
```

**4.7: Full Pipeline with Entailment & DB Dedup**

- [ ] Update `ExtractionPipeline`:
  - Add entailment validation step
  - Add database deduplication step
  - Save to PostgreSQL
  - Return saved claim/quote IDs
- [ ] Test end-to-end with episode
- [ ] Verify deduplication works across episodes

**Acceptance Criteria:**

```python
# Process episode 1
result1 = await pipeline.process_episode(episode1)
print(f"Episode 1: {len(result1.claims)} claims saved")

# Process episode 2 (may have duplicate claims)
result2 = await pipeline.process_episode(episode2)
print(f"Episode 2: {len(result2.claims)} new claims, {result2.duplicates_found} duplicates")

# Verify duplicate detection
if result2.duplicates_found > 0:
    # Check that quotes were added to existing claims
    for dup in result2.duplicate_details:
        existing_claim = db_session.query(Claim).get(dup.existing_claim_id)
        assert existing_claim.quotes.count() > dup.original_quote_count
```

**Sprint 4 Deliverable:**

- Entailment validation optimized and integrated
- False positive rate <10% for quote filtering
- Database deduplication working across episodes
- Full pipeline saving to PostgreSQL
- Can process multiple episodes with dedup

---

### SPRINT 5: Production Readiness & CLI (Week 5)

**Goal:** Production-quality features, CLI, monitoring, testing

#### Tasks

**5.1: Sentence Classifier (Question Filtering)**

- [ ] Create `src/preprocessing/sentence_classifier.py`
- [ ] Implement `SentenceClassifier` class:
  - Classify: STATEMENT, QUESTION, RHETORICAL, OTHER
  - Question markers: ?, who, what, when, where, why, how
  - Rhetorical patterns: "isn't it", "don't you think"
  - Statement patterns: "What happened was" (not a question)
- [ ] Integrate into quote selection
- [ ] Filter out questions before entailment check

**Acceptance Criteria:**

```python
from src.preprocessing.sentence_classifier import SentenceClassifier

classifier = SentenceClassifier()
assert classifier.classify("Bitcoin is great") == "STATEMENT"
assert classifier.classify("Is Bitcoin great?") == "QUESTION"
assert classifier.classify("Isn't Bitcoin great?") == "RHETORICAL"
assert classifier.classify("What happened was Bitcoin crashed") == "STATEMENT"
```

**5.2: Error Handling & Retries**

- [ ] Add retry logic to:
  - Ollama API calls (3 retries, exponential backoff)
  - Reranker API calls (3 retries, timeout 5s)
  - Database operations (deadlock retry)
  - Embedding generation (individual failures don't block batch)
- [ ] Add error logging throughout pipeline
- [ ] Graceful degradation:
  - If reranker unavailable → fallback to embeddings
  - If chunk fails → log and continue
  - If claim has no quotes → skip with warning

**Acceptance Criteria:**

```python
# Simulate reranker failure
reranker_service.stop()
result = await pipeline.process_episode(episode)
# Should complete with warnings, using embedding fallback
assert len(result.claims) > 0
assert result.warnings.count("reranker unavailable") > 0
```

**5.3: Logging & Monitoring**

- [ ] Set up structured logging (`src/infrastructure/logging.py`)
- [ ] Log levels:
  - DEBUG: Detailed execution info
  - INFO: Major pipeline steps
  - WARNING: Fallbacks, skipped items
  - ERROR: Failures that need attention
- [ ] Log important metrics:
  - Claims extracted per episode
  - Quotes found per claim
  - Deduplication stats (% removed)
  - Processing time per episode
  - Cache hit rates
- [ ] Save logs to file + console

**Acceptance Criteria:**

```python
# After processing episode:
# Logs should include:
# INFO: Processing episode 123 (50KB transcript)
# INFO: Extracted 15 claims from 3 chunks
# INFO: Found average 6.2 quotes per claim
# INFO: Batch dedup: 15 → 12 claims (20% reduction)
# INFO: DB dedup: 2 duplicates found, added quotes to existing
# INFO: Saved 10 new claims, 45 unique quotes
# INFO: Processing time: 45.3s
```

**5.4: CLI Interface**

- [ ] Create `src/cli/extract_claims.py`
- [ ] Implement CLI with arguments:
  - `--episode <id>`: Process specific episode
  - `--limit <n>`: Process N unprocessed episodes
  - `--all`: Process all unprocessed episodes
  - `--force`: Reprocess episode (delete existing claims)
  - `--stats`: Show database statistics
  - `--test-ollama`: Test Ollama connection
  - `--test-reranker`: Test reranker service
  - `--config`: Show current configuration
- [ ] Add progress bars for multi-episode processing
- [ ] Graceful shutdown (Ctrl+C handling)

**Acceptance Criteria:**

```bash
# Process specific episode
uv run python -m src.cli.extract_claims --episode 123

# Process 10 unprocessed episodes
uv run python -m src.cli.extract_claims --limit 10

# Show stats
uv run python -m src.cli.extract_claims --stats
# Output:
# Total episodes: 500
# Processed: 250
# Total claims: 3,750
# Avg claims per episode: 15.0

# Test services
uv run python -m src.cli.extract_claims --test-ollama
# ✅ Ollama connection successful!
# Available models: qwen2.5:7b-instruct, nomic-embed-text
```

**5.5: Testing & Validation**

- [ ] Test on 5-10 diverse episodes:
  - Short (~20KB), medium (~50KB), long (~100KB)
  - Different topics (crypto, tech, general)
  - Episodes with likely duplicate claims
- [ ] Verify output quality:
  - Claims are factual and self-contained
  - Quotes genuinely support claims
  - No duplicate claims in database
  - Confidence scores make sense
- [ ] Performance benchmarks:
  - 50KB episode: ~1-2 min target
  - Cache hit rate: >70%
  - Deduplication reduction: 20-40%
- [ ] Create test report

**5.6: Documentation**

- [ ] Create `PRODUCTION_USAGE.md`:
  - How to run the pipeline
  - Configuration options
  - Troubleshooting common issues
  - Expected performance
- [ ] Update `README.md`:
  - Project overview
  - Setup instructions
  - Quick start guide
- [ ] Code documentation:
  - Docstrings for all classes/methods
  - Type hints throughout
  - Inline comments for complex logic

**5.7: Feedback Collection System**

- [ ] Create `src/cli/review_claims.py`:
  - Fetch recent claims from database
  - Display claim + quotes
  - User labels: GOOD/BAD with reasons
  - Save feedback to `evaluation/production_feedback.json`
  - Format similar to claims_manual_review.json
- [ ] Plan for periodic re-optimization:
  - Collect 20-30 new examples
  - Re-run DSPy optimization
  - Deploy updated models

**Acceptance Criteria:**

```bash
# Review recent claims
uv run python -m src.cli.review_claims --limit 10

# Shows:
# Claim: "Bitcoin reached $69,000 in November 2021"
# Quotes:
#   1. "Bitcoin hit its all-time high of $69,000..."
#   2. "In November 2021, BTC peaked at 69k..."
# Quality (good/bad): good
# Notes: Factual, self-contained, well-supported

# Saves to evaluation/production_feedback.json
```

**Sprint 5 Deliverable:**

- Production-ready pipeline with error handling
- CLI for easy operation
- Logging and monitoring
- Tested on real episodes
- Documentation complete
- Feedback collection system

---

## Technical Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│  src/cli/extract_claims.py    src/cli/review_claims.py    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Pipeline Orchestrator                    │
│              src/pipeline/extraction_pipeline.py            │
│                                                             │
│  - Coordinates all components                              │
│  - Manages transaction boundaries                          │
│  - Error handling & logging                                │
└─────┬──────────┬──────────┬──────────┬─────────────────────┘
      │          │          │          │
┌─────▼──────────▼──────────▼──────────▼─────────────────────┐
│              Core Processing Components                     │
├─────────────────────────────────────────────────────────────┤
│ Preprocessing:                    Extraction:               │
│ - TranscriptParser                - ClaimExtractor (DSPy)   │
│ - ChunkingService                 - QuoteFinder             │
│ - SentenceClassifier              - EntailmentValidator     │
│                                                             │
│ Search:                           Deduplication:            │
│ - TranscriptSearchIndex           - QuoteDeduplicator       │
│                                   - ClaimDeduplicator       │
│                                                             │
│ Scoring:                                                    │
│ - ConfidenceCalculator                                      │
└─────┬──────────┬──────────┬──────────┬─────────────────────┘
      │          │          │          │
┌─────▼──────────▼──────────▼──────────▼─────────────────────┐
│              Infrastructure Services                        │
├─────────────────────────────────────────────────────────────┤
│ - EmbeddingService (Ollama, LRU cache)                     │
│ - RerankerService (HTTP API, LRU cache, fallback)         │
│ - DSPy Model Loaders (claim_extractor, entailment)        │
└─────┬───────────────────────────────────────┬──────────────┘
      │                                       │
┌─────▼───────────────────────────────────────▼──────────────┐
│              External Dependencies                          │
├─────────────────────────────────────────────────────────────┤
│ - PostgreSQL + pgvector (Database)                         │
│ - Ollama (LLM + Embeddings)                                │
│ - Reranker Service (Docker container)                      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Episode (DB) → Parse → Chunk → Extract (DSPy) → Search Index
                                      ↓
                          Find Quotes (Semantic Search)
                                      ↓
                          Validate Entailment (DSPy)
                                      ↓
                          Deduplicate Quotes (Position)
                                      ↓
                          Deduplicate Claims (Batch)
                                      ↓
                          Calculate Confidence
                                      ↓
                          Deduplicate Against DB (pgvector + Reranker)
                                      ↓
                          Save Claims + Quotes + ClaimQuotes (DB)
```

### Directory Structure

```
dspy-playground/
├── src/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── extract_claims.py          # Main CLI
│   │   └── review_claims.py           # Feedback collection
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── extraction_pipeline.py     # Main orchestrator
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── claim_extractor.py         # Pass 1: Claim extraction
│   │   └── quote_finder.py            # Pass 2: Quote search
│   ├── dspy_models/
│   │   ├── __init__.py
│   │   ├── claim_extractor.py         # Loads DSPy model
│   │   └── entailment_validator.py    # Loads DSPy model
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── transcript_parser.py       # Speaker parsing
│   │   ├── chunking_service.py        # Chunk with overlap
│   │   └── sentence_classifier.py     # Question filtering
│   ├── search/
│   │   ├── __init__.py
│   │   └── transcript_search_index.py # Semantic search
│   ├── deduplication/
│   │   ├── __init__.py
│   │   ├── quote_deduplicator.py      # Position-based
│   │   └── claim_deduplicator.py      # Batch + DB level
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── confidence_calculator.py   # Weighted scoring
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── embedding_service.py       # Ollama + cache
│   │   ├── reranker_service.py        # HTTP + cache
│   │   └── logging.py                 # Structured logging
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py              # DB connection
│   │   ├── models.py                  # SQLAlchemy models
│   │   └── claim_repository.py        # CRUD operations
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                # Configuration
│   └── metrics/
│       ├── __init__.py
│       ├── claim_metrics.py           # LLM-as-judge for claims
│       └── entailment_metrics.py      # LLM-as-judge for entailment
├── experiments/                        # Keep for reference only
│   └── (existing experiment files)
├── evaluation/
│   ├── claims_manual_review.json
│   ├── claims_train.json
│   ├── claims_val.json
│   ├── entailment_manual_review.json  # Sprint 4
│   ├── entailment_train.json
│   ├── entailment_val.json
│   └── production_feedback.json       # Sprint 5
├── models/
│   ├── claim_extractor_llm_judge_v1.json     # From experiments
│   └── entailment_validator_v1.json          # Sprint 4
├── results/
│   └── (experiment results)
├── specs/
│   └── OLD_ARCHITECTURE.md
├── BACKLOG.md                          # This file
├── CLAUDE.md
├── README.md
└── .env                                # Configuration
```

---

## Database Schema

### Entities

**PodcastEpisode**

```sql
CREATE TABLE podcast_episodes (
    id BIGSERIAL PRIMARY KEY,
    podcast_id BIGINT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    episode_number INTEGER,
    duration INTEGER,
    published_at DATE,
    transcript TEXT,  -- Our input
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Claim**

```sql
CREATE TABLE claims (
    id BIGSERIAL PRIMARY KEY,
    episode_id BIGINT NOT NULL REFERENCES podcast_episodes(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,  -- 0.0-1.0
    embedding VECTOR(768),  -- pgvector for similarity search
    metadata JSONB,  -- speaker, timestamp, etc.
    confidence_components JSONB,  -- avg_relevance, max_relevance, quote_count, etc.
    reranker_scores JSONB,  -- debugging info
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_claims_episode ON claims(episode_id);
CREATE INDEX idx_claims_confidence ON claims(confidence);
CREATE INDEX idx_claims_embedding_vector ON claims USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
```

**Quote**

```sql
CREATE TABLE quotes (
    id BIGSERIAL PRIMARY KEY,
    episode_id BIGINT NOT NULL REFERENCES podcast_episodes(id) ON DELETE CASCADE,
    quote_text TEXT NOT NULL,
    start_position INTEGER,  -- Character position in transcript
    end_position INTEGER,
    speaker VARCHAR(255),
    timestamp_seconds INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(quote_text, start_position, end_position, episode_id)
);

CREATE INDEX idx_quotes_episode ON quotes(episode_id);
CREATE INDEX idx_quotes_position ON quotes(start_position, end_position);
```

**ClaimQuote** (Many-to-Many Junction Table)

```sql
CREATE TABLE claim_quotes (
    claim_id BIGINT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    quote_id BIGINT NOT NULL REFERENCES quotes(id) ON DELETE CASCADE,
    relevance_score DOUBLE PRECISION NOT NULL,  -- How well quote supports claim
    match_confidence DOUBLE PRECISION NOT NULL,
    match_type VARCHAR(20),  -- 'semantic', 'reranked', 'entailment'
    entailment_score DOUBLE PRECISION,  -- From DSPy entailment model
    entailment_relationship VARCHAR(20),  -- 'SUPPORTS', 'RELATED', etc.
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (claim_id, quote_id)
);

CREATE INDEX idx_claim_quotes_claim ON claim_quotes(claim_id);
CREATE INDEX idx_claim_quotes_quote ON claim_quotes(quote_id);
CREATE INDEX idx_claim_quotes_relevance ON claim_quotes(relevance_score);
```

---

## Configuration

### Environment Variables (.env)

```bash
#──────────────────────────────────────────────────────────────
# Database
#──────────────────────────────────────────────────────────────
DATABASE_URL=postgresql://user:password@localhost:5432/podcast_db

#──────────────────────────────────────────────────────────────
# Ollama
#──────────────────────────────────────────────────────────────
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_0
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

#──────────────────────────────────────────────────────────────
# Reranker
#──────────────────────────────────────────────────────────────
ENABLE_RERANKER=true
RERANKER_URL=http://localhost:8080
RERANKER_TIMEOUT=5000  # milliseconds

#──────────────────────────────────────────────────────────────
# Chunking
#──────────────────────────────────────────────────────────────
CHUNK_SIZE=16000  # characters
CHUNK_OVERLAP=1000

#──────────────────────────────────────────────────────────────
# Processing
#──────────────────────────────────────────────────────────────
PARALLEL_BATCH_SIZE=3  # Chunks processed in parallel

#──────────────────────────────────────────────────────────────
# Deduplication Thresholds
#──────────────────────────────────────────────────────────────
EMBEDDING_SIMILARITY_THRESHOLD=0.85  # Cosine similarity
RERANKER_VERIFICATION_THRESHOLD=0.9  # For duplicates
STRING_SIMILARITY_THRESHOLD=0.95     # Fallback
VECTOR_DISTANCE_THRESHOLD=0.15       # pgvector L2 distance

#──────────────────────────────────────────────────────────────
# Scoring
#──────────────────────────────────────────────────────────────
MIN_CONFIDENCE=0.3  # Don't save claims below this
MAX_QUOTES_PER_CLAIM=10
MIN_QUOTE_RELEVANCE=0.85  # Don't link quotes below this

#──────────────────────────────────────────────────────────────
# Caching
#──────────────────────────────────────────────────────────────
CACHE_MAX_SIZE=10000
CACHE_TTL_HOURS=1

#──────────────────────────────────────────────────────────────
# Logging
#──────────────────────────────────────────────────────────────
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/extraction.log
```

---

## Success Criteria

### Sprint 1 Success

- [ ] Can connect to PostgreSQL and query episodes
- [ ] Can load optimized DSPy claim extractor
- [ ] Embedding service working with >70% cache hit rate
- [ ] Configuration system functional

### Sprint 2 Success

- [ ] Can extract claims from transcript using DSPy model
- [ ] Two-pass pipeline working (extract + quote search)
- [ ] Can process 1 episode end-to-end (no dedup yet)
- [ ] Claims have quotes with relevance scores

### Sprint 3 Success

- [ ] Reranker service integrated and tested
- [ ] Quote deduplication reduces duplicates by 20-40%
- [ ] Batch claim deduplication working within episode
- [ ] Confidence scores calculated correctly

### Sprint 4 Success

- [ ] Entailment model optimized (false positive rate <10%)
- [ ] Entailment filtering integrated (only SUPPORTS quotes kept)
- [ ] Database deduplication working across episodes
- [ ] Can process multiple episodes with cross-episode dedup
- [ ] Claims, quotes, claim_quotes saved to PostgreSQL

### Sprint 5 Success

- [ ] CLI working for all operations
- [ ] Tested on 5-10 diverse episodes
- [ ] Processing time: ~1-2 min per 50KB episode
- [ ] Error handling and logging comprehensive
- [ ] Documentation complete
- [ ] Feedback collection system ready

### Overall Production Readiness

- [ ] Claim quality: <15% low-quality claims (using LLM-as-judge)
- [ ] Entailment accuracy: <10% false positive rate
- [ ] Deduplication: 20-40% reduction in raw claims
- [ ] Performance: 30-60 episodes per hour
- [ ] No duplicate claims in database (verified)
- [ ] Can run continuously without crashes
- [ ] Logs provide clear visibility into processing
- [ ] Feedback loop enables continuous improvement

---

## Next Actions

1. **Read this backlog carefully**
2. **Set up development environment**
   - Python virtual environment
   - Install dependencies (DSPy, psycopg2, sqlalchemy, etc.)
   - Verify Ollama running
   - Verify reranker Docker container running
   - Verify PostgreSQL connection
3. **Start Sprint 1, Task 1.1**
   - Create src/database/ module
   - Implement models
   - Test database connection

**Let's build production software!** 🚀
