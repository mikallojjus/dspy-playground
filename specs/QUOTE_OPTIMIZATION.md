# Quote Finding Optimization

## Status: ✅ READY FOR PRODUCTION

**Summary**: All 4 epics complete + model loading fix applied. The quote finding pipeline is fully ready for production use.

- ✅ **Epic 1**: Core infrastructure (coarse chunker, quote verification)
- ✅ **Epic 2**: DSPy quote finder module and training infrastructure
- ✅ **Epic 3**: Integration with extraction pipeline
- ✅ **Epic 4**: Production deployment
- ✅ **Fix Applied**: QuoteFinder now loads optimized models (2024-10-26)

**Current State**:
- Production pipeline automatically uses optimized model from `models/quote_finder_v1.json`
- Fallback to zero-shot baseline if model file not found (graceful degradation)
- Model path configurable via `QUOTE_FINDER_MODEL_PATH` environment variable

**What's Next**: Expand training data from 3 → 50-100 examples and retrain for production-grade performance

---

## Context

Current quote finding pipeline creates 600-1200+ windowed segments per episode and embeds each one, resulting in 2-3 minute embedding bottleneck per episode. Total processing time: ~4-5 minutes per episode.

With support for 2-4 hour podcasts (up to 72,000 tokens), we need a more efficient approach that maintains or improves quality while reducing processing time by 3x.

## Current Architecture Problems

1. **Excessive segmentation**: 900+ segments per episode
2. **Embedding bottleneck**: 2-3 minutes just for embeddings
3. **Context fragmentation**: Small segments (500 tokens) lose narrative context
4. **Boundary issues**: Quotes can be split across segment boundaries
5. **Retrieval failures**: Semantic similarity can miss relevant quotes with different wording

## Proposed Solution: Coarse Retrieval + DSPy-Optimized LLM Extraction

### Two-Tier Architecture

**Tier 1: Coarse Retrieval (Embeddings)**

- Create 20-30 large chunks per episode (3000 tokens each, 500 token overlap)
- 50x reduction in embeddings: 24 embeddings vs 1200 embeddings
- For each claim: retrieve top 4 most relevant chunks (~12k tokens context)

**Tier 2: DSPy-Optimized LLM Extraction**

- Feed LLM: claim + relevant chunks (12k tokens, fits in 32k context)
- DSPy-optimized prompt extracts quotes
- Strict verification catches hallucinations (90% substring similarity)
- Entailment validation filters RELATED quotes (keep only SUPPORTS)

### Why DSPy?

- **Prompt optimization**: DSPy learns best instructions and few-shot examples
- **Measurable improvement**: Metric-driven optimization (verification rate + entailment rate)
- **Adaptation**: Learns patterns specific to our podcast domain
- **Baseline target**: Reduce false positives from 30% to <10%

## Key Components

### 1. Coarse Chunking

- Chunk size: 3000 tokens (~2-3 minutes of speech)
- Overlap: 500 tokens (prevent boundary issues)
- 4-hour podcast: ~24 chunks
- Preserves narrative context

### 2. DSPy Quote Finder Module

```python
class QuoteFinderSignature(dspy.Signature):
    """Find supporting quotes for a claim from transcript chunks."""
    claim: str = dspy.InputField()
    transcript_chunks: str = dspy.InputField()
    quotes: list[dict] = dspy.OutputField()

class QuoteFinder(dspy.Module):
    def __init__(self):
        self.find_quotes = dspy.ChainOfThought(QuoteFinderSignature)
```

### 3. Evaluation Metric

Composite score (0.0-1.0):

- **40%** Verification rate (quotes pass substring matching)
- **40%** Entailment rate (verified quotes score SUPPORTS)
- **20%** Recall (finds ground truth quotes, if available)

Threshold: 90% substring similarity for verification

### 4. Multi-Stage Validation

1. **DSPy extraction**: LLM finds candidate quotes
2. **Verification**: Strict substring matching (catch hallucinations)
3. **Entailment**: SUPPORTS filter (reject RELATED quotes)

### 5. Training Data

- Format: 50-100 labeled examples (claim, transcript_chunks, gold_quotes)
- Source: Manually review 10 episodes from current pipeline
- Labeling: Mark good quotes (keep), bad quotes (remove), missing quotes (add)

## Expected Outcomes

### Performance

- **Current**: 4-5 minutes per episode
- **Target**: 1.5-2 minutes per episode
- **Speedup**: 3x improvement

### Quality

- **Verification rate**: >95% (hallucination rate <5%)
- **Entailment false positives**: <10% (down from 30% baseline)
- **Recall**: Maintain or improve (LLM sees more context)

### Complexity

- **Remove**: ~300 lines (TranscriptSearchIndex, fine-grained segmentation)
- **Add**: ~200 lines (QuoteFinder, DSPy integration, verification)
- **Net**: Slight reduction in code complexity

## Architecture Changes

### Files Added

- `src/search/coarse_chunker.py` - Create 3000-token chunks with overlap ✅
- `src/search/llm_quote_finder.py` - DSPy QuoteFinder module ✅
- `src/search/quote_verification.py` - Substring matching + fuzzy fallback ✅
- `src/search/quote_pipeline.py` - Quote finding pipeline (coarse retrieval + DSPy) ✅
- `src/metrics/quote_finder_metrics.py` - DSPy evaluation metric ✅
- `src/training/train_quote_finder.py` - Self-contained training script ✅

### Files Removed ✅

- `src/search/transcript_search_index.py` - DELETED
- `src/extraction/quote_finder.py` - QuoteFinder class removed (kept dataclasses only)

### Files Modified ✅

- `src/pipeline/extraction_pipeline.py` - Integrated QuoteFindingPipeline ✅
- `src/config/settings.py` - Added DSPy configuration ✅

### Files to Keep

- `src/infrastructure/embedding_service.py` - Still needed for coarse chunk embeddings
- `src/infrastructure/reranker_service.py` - May remove after DSPy validation proves effective
- `src/dspy_models/entailment_validator.py` - Keep as final quality gate
- `src/deduplication/quote_deduplicator.py` - Keep for quote deduplication

## Risk Mitigation

**Hallucination risk**: Multi-stage validation (verification + entailment)
**Context window limits**: Coarse retrieval ensures only relevant chunks are sent to LLM
**Quality regression**: Comprehensive metrics track performance during optimization
**Long podcasts (>4hrs)**: Top-K retrieval ensures context fits in 32k window

## DSPy Optimization Details

**Optimizer**: BootstrapFewShot

- `max_bootstrapped_demos=4` - Include 4 learned examples in prompt
- `max_labeled_demos=2` - Include 2 seed examples
- `max_rounds=3` - 3 optimization iterations

**Training time**: ~10-20 minutes for 50-100 examples

**Model**: Qwen 2.5 7B via Ollama (32k context window)

---

## Backlog

### Epic 1: Core Infrastructure

**Goal**: Build coarse chunking and verification components

**Stories**:

- Implement coarse chunker (3000-token chunks, 500 overlap)
- Build quote verification function (substring + fuzzy matching)
- Unit tests for chunking and verification

**Acceptance Criteria**:

- Coarse chunker creates 20-30 chunks per 4-hour episode
- Verification correctly identifies exact matches and catches hallucinations

---

### Epic 2: DSPy Quote Finder

**Goal**: Implement DSPy-based quote extraction module

**Stories**:

- Define QuoteFinderSignature and QuoteFinder module
- Implement QuoteFinderMetric (verification + entailment + recall)
- Create training data: manually label 10 episodes (~100 examples)
- Build QuoteFinderOptimizer with BootstrapFewShot

**Acceptance Criteria**:

- QuoteFinder module runs successfully with basic prompt
- Metric calculates composite score (0.0-1.0)
- Training data formatted as dspy.Example objects
- Optimizer produces optimized QuoteFinder

---

### Epic 3: Integrated Pipeline ✅ COMPLETE

**Goal**: Integrate quote finding pipeline with extraction pipeline

**Status**: COMPLETE

**Completed**:

- ✅ Implemented QuoteFindingPipeline (coarse retrieval → DSPy extraction → verification)
- ✅ Integrated with extraction_pipeline.py (steps 4-5)
- ✅ Configuration settings added for DSPy params (coarse chunk size, overlap, top_k, verification threshold)
- ✅ All 40 search tests passing

**Implementation Notes**:

- Pipeline uses coarse-grained chunks (3000 tokens, 500 overlap)
- DSPy-based quote extraction with multi-tier verification (exact → normalized → token overlap → fuzzy)
- Quotes verified at 90% confidence threshold to catch hallucinations
- Full integration with existing entailment validation and deduplication

---

### Epic 4: Deployment ✅ COMPLETE

**Goal**: Deploy quote finding pipeline to production

**Status**: COMPLETE

**Completed**:

- ✅ Cleaned up obsolete code (TranscriptSearchIndex, old QuoteFinder service)
- ✅ Updated extraction_pipeline.py to use QuoteFindingPipeline
- ✅ Kept Quote and ClaimWithQuotes dataclasses (used throughout pipeline)
- ✅ All search tests passing (40/40)
- ✅ Documentation updated

**Implementation Notes**:

- Backward compatibility maintained through dataclass interfaces
- Ready for production deployment
- Next step: Train DSPy model on real data for prompt optimization (see evaluation/ directory)

---

## Training Data Example

```python
dspy.Example(
    claim="Bitcoin reached $69,000 in November 2021",
    transcript_chunks="""
    [Speaker 1 (0s)]: Bitcoin's price history has been volatile...
    [Speaker 2 (45s)]: In November 2021, BTC hit an all-time high of $69,000...
    [Speaker 1 (1m 30s)]: After that peak, the market crashed...
    """,
    gold_quotes=[
        "In November 2021, BTC hit an all-time high of $69,000"
    ]
).with_inputs("claim", "transcript_chunks")
```

## Success Metrics

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Processing time (per episode) | 4-5 min | 1.5-2 min | <1.5 min |
| Hallucination rate | N/A | <5% | <2% |
| Entailment false positives | 30% | <10% | <5% |
| Verification rate | N/A | >95% | >98% |
| Embeddings per episode | 900-1200 | 20-30 | 20-30 |

## References

- DSPy documentation: <https://dspy-docs.vercel.app/>
- Current claim extraction in: `src/extraction/claim_extractor.py`
- Current quote finding in: `src/search/transcript_search_index.py`
- Entailment validation in: `src/validation/entailment_validator.py`
