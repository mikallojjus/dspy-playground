# DSPy Optimization Initiative - Backlog

**Last Updated:** 2025-10-25
**Status:** Planning / Pre-Implementation
**Goal:** Use DSPy to systematically optimize claim extraction and entailment validation prompts with measurable improvements

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Current State Analysis](#current-state-analysis)
4. [DSPy Solution Overview](#dspy-solution-overview)
5. [Architecture Comparison](#architecture-comparison)
6. [Success Criteria](#success-criteria)
7. [Phase-by-Phase Backlog](#phase-by-phase-backlog)
8. [Code Templates & Examples](#code-templates--examples)
9. [Metrics Definitions](#metrics-definitions)
10. [Key Decisions & Trade-offs](#key-decisions--trade-offs)
11. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
12. [Resource Requirements](#resource-requirements)
13. [References & Documentation](#references--documentation)

---

## Executive Summary

### The Core Problem

We have been manually tuning prompts for claim extraction and entailment validation in isolation, with no systematic way to measure improvements. This has led to:

- 40% of extracted claims being low-quality (vague, opinion-based, or using pronouns)
- 30% false positive rate in entailment validation (RELATED misclassified as SUPPORTS)
- Flying blind on whether prompt changes actually improve results

### The DSPy Solution

DSPy is a framework for programming language models that enables:

1. **Declarative signatures** - Type-safe prompt definitions instead of string templates
2. **Metrics** - Quantifiable measurements of quality (e.g., "% of claims without pronouns")
3. **Optimizers (Teleprompters)** - Automatic prompt improvement via algorithms like MIPROv2
4. **Evaluation** - Systematic testing on labeled ground truth data

### Expected Outcomes

- **Claim Extraction:** Reduce low-quality claims from 40% → <15%
- **Entailment Validation:** Reduce false positives from 30% → <10%
- **Process:** Shift from manual prompt tweaking to data-driven optimization
- **Deliverable:** Two optimized prompt modules ready for production integration

### Timeline Estimate

- **Minimum Viable:** 2 weeks (30-50 labeled examples, basic optimization)
- **Production Ready:** 4-6 weeks (100-200 labeled examples, comprehensive evaluation)
- **Ongoing:** Continuous improvement as more data becomes available

---

## Problem Statement

### What We're Building

A podcast processing pipeline that extracts meaningful information from podcast transcripts:

1. **Input:** Raw podcast transcript (50-100KB text)
2. **Step 1:** Extract factual claims from the transcript
3. **Step 2:** Find supporting quotes for each claim using semantic search
4. **Step 3:** Validate that quotes actually support claims (entailment)
5. **Output:** Database of claims with verified supporting evidence

### Current Pain Points

#### Pain Point 1: Claim Extraction Quality Issues

**Current approach:** Hardcoded prompt with manual rules
**Problems:**

- 40% of claims are low-quality:
  - Use pronouns ("he said", "they announced") without context
  - Vague statements ("it was amazing", "very interesting")
  - Opinions disguised as facts ("I think Bitcoin is great")
  - Missing entity names ("the CEO announced" without naming who)

**Example bad claim:**

```
"He said it was a groundbreaking achievement"
```

**What we want:**

```
"Elon Musk described SpaceX's Starship launch as a groundbreaking achievement"
```

#### Pain Point 2: Entailment False Positives

**Current approach:** Hardcoded prompt asking "does quote support claim?"
**Problems:**

- 30% false positive rate - quotes marked as SUPPORTS when they're only RELATED
- Example:
  - **Claim:** "Bitcoin reached $69,000 in November 2021"
  - **Quote:** "Cryptocurrency markets were very volatile in 2021"
  - **Current system:** SUPPORTS (WRONG - this is RELATED)
  - **Should be:** RELATED (topically connected but doesn't validate the specific claim)

#### Pain Point 3: No Systematic Optimization

**Current approach:** Manually tweak prompts, deploy, hope for the best
**Problems:**

- No ground truth dataset to test against
- No metrics to measure if changes help or hurt
- Prompt changes affect edge cases unpredictably
- Can't compare different prompt variations objectively

---

## Current State Analysis

### Technology Stack

- **LLM:** Qwen 2.5 7B (q4_K_M quantized) via Ollama (localhost:11434)
- **Embeddings:** nomic-embed-text (768 dimensions) via Ollama
- **Reranker:** BGE reranker v2-m3 (Docker container, localhost:8080)
- **Database:** PostgreSQL with pgvector
- **Original Implementation:** TypeScript/TypeORM (see `specs/OLD_ARCHITECTURE.md`)

### Current Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. TRANSCRIPT CHUNKING                                          │
│    - 16K char chunks with 1K overlap                            │
│    - Handles Ollama context window limitations                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 2. CLAIM EXTRACTION (HARDCODED PROMPT)                          │
│    - Ollama API call with fixed prompt string                   │
│    - Parallel processing (3 chunks at a time)                   │
│    - Returns: claims + initial supporting quotes                │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 3. GLOBAL QUOTE SEARCH                                          │
│    - Semantic search using embeddings (cosine similarity)       │
│    - Top 30 candidates per claim                                │
│    - Reranker scores candidates                                 │
│    - Select top 10 quotes                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 4. ENTAILMENT VALIDATION (HARDCODED PROMPT)                     │
│    - For each claim-quote pair                                  │
│    - Returns: SUPPORTS/RELATED/NEUTRAL/CONTRADICTS              │
│    - Filter out quotes with entailment_score < 0.7              │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│ 5. DEDUPLICATION & PERSISTENCE                                  │
│    - Quote deduplication (position-based)                       │
│    - Claim deduplication (embedding + reranker)                 │
│    - Database storage with pgvector                             │
└─────────────────────────────────────────────────────────────────┘
```

### Evaluation Data Status

**Current state:** **NO LABELED EVALUATION DATA EXISTS BUT WE CAN PROVIDE IT**

**What we need to create:**

1. 30-50 labeled examples of good vs bad claims
2. 50-100 labeled examples of claim-quote pairs with correct entailment labels

---

## DSPy Solution Overview

### What is DSPy?

DSPy is a framework that treats prompts as **optimizable programs** rather than static strings. Think of it as "PyTorch for prompts" - instead of manually tuning neural network weights, you define what you want and let the framework optimize how to get it.

**Key DSPy Concepts:**

#### 1. Signatures (Type-Safe Prompt Definitions)

Instead of string templates, you define input/output types:

```python
class ClaimExtraction(dspy.Signature):
    """Extract factual claims from transcript text."""
    transcript_chunk: str = dspy.InputField()
    claims: list[str] = dspy.OutputField()
```

DSPy turns this into a prompt automatically, but can optimize the exact wording.

#### 2. Modules (Prompt Execution Strategies)

Wrap signatures in execution strategies:

```python
# Simple: just ask the LLM
extractor = dspy.Predict(ClaimExtraction)

# Chain of Thought: ask LLM to reason first
extractor = dspy.ChainOfThought(ClaimExtraction)

# ReAct: reasoning + actions
extractor = dspy.ReAct(ClaimExtraction)
```

#### 3. Metrics (Quantifiable Quality)

Define functions that score predictions (0.0 to 1.0):

```python
def claim_quality_metric(example, pred, trace=None):
    """Return 1.0 if claim has no pronouns, 0.0 if it does."""
    has_pronouns = any(p in pred.claim.lower() for p in ['he', 'she', 'they'])
    return 0.0 if has_pronouns else 1.0
```

#### 4. Optimizers (Teleprompters)

Algorithms that improve prompts based on metrics:

```python
optimizer = MIPROv2(metric=claim_quality_metric)
optimized = optimizer.compile(
    student=extractor,
    trainset=labeled_examples,
    valset=validation_examples
)
```

The optimizer will:

- Try different prompt wordings
- Add few-shot examples
- Test on validation set
- Keep the best-performing version

### Why DSPy for This Project?

| Manual Prompt Engineering | DSPy Approach |
|---------------------------|---------------|
| Write prompt string | Define signature (types + docstring) |
| Deploy and hope | Measure with metrics |
| Manually tweak wording | Optimizer tests variations |
| Subjective assessment | Quantitative scores |
| One prompt at a time | Systematic comparison |
| Changes break edge cases | Validated on test set |

**Specific benefits for claim extraction:**

- **Metric:** "% of claims without pronouns" - directly measures your goal
- **Optimization:** DSPy will find prompt wording that maximizes this metric
- **Few-shot learning:** Automatically selects best examples to include
- **Validation:** Ensures improvements generalize to unseen data

**Specific benefits for entailment:**

- **Metric:** "accuracy on SUPPORTS vs RELATED" - measures false positive rate
- **Calibration:** DSPy can learn to be stricter about SUPPORTS classification
- **Consistency:** Optimized prompts are more reliable across different inputs

---

## Success Criteria

### Primary Goals (Quantitative)

#### Goal 1: Claim Extraction Quality

**Baseline (current):** 40% of claims are low-quality
**Target:** <15% low-quality claims
**Minimum acceptable:** <20% low-quality claims

**How measured:**

- Manual labeling of 100 claims from test set
- Automated metrics:
  - % claims with pronouns (he/she/they/his/her/their)
  - % claims with vague words (very/really/amazing/terrible)
  - % claims with opinion markers (think/believe/feel)
  - % claims missing entity names (check against NER)

#### Goal 2: Entailment Validation Accuracy

**Baseline (current):** 30% false positive rate (RELATED → SUPPORTS)
**Target:** <10% false positive rate
**Minimum acceptable:** <15% false positive rate

**How measured:**

- Manual labeling of 100 claim-quote pairs from test set
- Confusion matrix:

  ```
                  Predicted SUPPORTS | Predicted RELATED
  Actual SUPPORTS      TP                   FN
  Actual RELATED       FP (minimize!)       TN
  ```

- False positive rate = FP / (FP + TN)

### Deliverables

#### Phase 1 Deliverables (Week 2)

- [ ] 30-50 labeled claim quality examples
- [ ] 50-100 labeled entailment examples
- [ ] Baseline metrics measured and documented
- [ ] DSPy signatures defined for both tasks
- [ ] Metrics implemented and tested

#### Phase 2 Deliverables (Week 4)

- [ ] Optimized claim extraction module (JSON file)
- [ ] Optimized entailment validation module (JSON file)
- [ ] Before/after comparison report
- [ ] Integration code connecting DSPy to existing pipeline

#### Phase 3 Deliverables (Week 6)

- [ ] Production deployment of optimized modules
- [ ] A/B test results (old vs new prompts)
- [ ] Documentation for maintaining and re-optimizing
- [ ] Expanded evaluation dataset (200+ examples)

---

## Phase-by-Phase Backlog

### PHASE 0: Environment Setup & Validation

**Goal:** Ensure all infrastructure is working before building on it
**Time estimate:** 1-2 days
**Priority:** P0 (blocker)

#### TASK 0.1: Validate Ollama Setup

- [ ] Test Ollama API connectivity (localhost:11434)
- [ ] Confirm Qwen 2.5 7B model is available
- [ ] Test generation with simple prompt
- [ ] Test embedding generation with nomic-embed-text
- [ ] Document API response format

**Acceptance criteria:**

```python
# This should work:
import requests
response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'qwen2.5:7b-instruct',
    'prompt': 'What is 2+2?',
    'stream': False
})
assert response.status_code == 200
print(response.json()['response'])  # Should print "4" or similar
```

#### TASK 0.2: Install and Configure DSPy

- [ ] Install DSPy: `uv add dspy`
- [ ] Test DSPy with Ollama backend
- [ ] Verify DSPy can call Ollama models
- [ ] Test basic signature execution

**Acceptance criteria:**

```python
import dspy

# Configure DSPy to use Ollama
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Test simple signature
class BasicQA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa = dspy.Predict(BasicQA)
result = qa(question="What is 2+2?")
print(result.answer)  # Should print "4" or similar
```

#### TASK 0.3: Verify Reranker Service

- [ ] Confirm reranker Docker container is running (localhost:8080)
- [ ] Test reranker API with sample claim-quote pair
- [ ] Measure reranker latency (should be <200ms per batch)

**Note:** Reranker is optional for DSPy but used in the existing pipeline.

#### TASK 0.4: Database Connection Test

- [ ] Verify PostgreSQL connection
- [ ] Test pgvector extension is installed
- [ ] Query existing claims/quotes tables
- [ ] Understand current data schema

---

### PHASE 1: Dataset Creation

**Goal:** Build ground truth evaluation datasets
**Time estimate:** 1-2 weeks
**Priority:** P0 (blocker for optimization)

This is the **critical path** - without labeled data, DSPy cannot optimize.

#### TASK 1.1: Select Representative Transcript Samples

- [ ] Choose 10-15 diverse podcast episodes from database
- [ ] Variety in topics (crypto, tech, finance, general interest)
- [ ] Range of transcript lengths (short, medium, long)
- [ ] Include edge cases (heavy jargon, informal speech, multiple speakers)

**SQL query to help select:**

```sql
SELECT id, name, length(transcript) as transcript_length,
       (SELECT COUNT(*) FROM claims WHERE episode_id = podcast_episodes.id) as existing_claims_count
FROM podcast_episodes
WHERE transcript IS NOT NULL
ORDER BY RANDOM()
LIMIT 20;
```

Pick 10-15 that cover different scenarios.

#### TASK 1.2: Create Claim Quality Dataset

**Goal:** 30-50 labeled examples of claim extraction quality
**Time estimate:** 4-6 hours of manual labeling

**Process:**

1. Run existing claim extraction on selected episodes
2. For each claim, label as:
   - **GOOD:** Factual, specific, self-contained, no pronouns, verifiable
   - **BAD:** Has pronouns, vague, opinion, missing context, not verifiable
   - **EDGE:** Borderline cases (document why)

3. Save in format:

```python
# claims_dataset.json
{
  "examples": [
    {
      "transcript_chunk": "...",
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quality": "good",
      "issues": [],
      "notes": "Specific, factual, self-contained"
    },
    {
      "transcript_chunk": "...",
      "claim": "He said it was amazing",
      "quality": "bad",
      "issues": ["pronoun", "vague", "missing_context"],
      "notes": "Who is 'he'? What is 'it'? 'Amazing' is opinion"
    }
  ]
}
```

**Labeling guidelines document:**
Create `evaluation/CLAIM_LABELING_GUIDELINES.md`:

```markdown
# Claim Quality Labeling Guidelines

## GOOD Claims
✅ "Elon Musk announced Tesla's Cybertruck in November 2019"
   - Named entity (Elon Musk, Tesla, Cybertruck)
   - Specific fact (announced, November 2019)
   - Self-contained (no pronouns)
   - Verifiable

✅ "The iPhone 15 Pro features a titanium frame and USB-C port"
   - Product features, specific model
   - Factual, verifiable

## BAD Claims
❌ "He said the product was revolutionary"
   - Pronoun "he" - who?
   - "Revolutionary" is subjective opinion

❌ "The company announced something big"
   - "The company" - which company?
   - "Something big" - vague

❌ "It was very impressive"
   - All pronouns and opinion words
   - Not factual or specific

## EDGE Cases
⚠️ "Bitcoin's price volatility makes it risky for investors"
   - Partially factual (volatility) but "risky" is subjective
   - Label: BAD (opinion)

⚠️ "The CEO stated the company exceeded revenue targets"
   - Has "the CEO" and "the company" but might be clear from context
   - Check if names appear earlier in transcript
   - If yes → GOOD, if no → BAD
```

**Tool to build:**

```python
# evaluation/label_claims.py
# Simple CLI tool to speed up labeling
import json

def label_claim_interactive(transcript_chunk, claim):
    print(f"\n{'='*60}")
    print(f"Transcript: {transcript_chunk[:200]}...")
    print(f"Claim: {claim}")
    print(f"{'='*60}")

    quality = input("Quality (good/bad/edge): ").lower()

    issues = []
    if quality in ['bad', 'edge']:
        print("Issues (comma-separated):")
        print("  - pronoun, vague, opinion, missing_context, not_verifiable")
        issues = input("Issues: ").split(',')
        issues = [i.strip() for i in issues]

    notes = input("Notes: ")

    return {
        "transcript_chunk": transcript_chunk,
        "claim": claim,
        "quality": quality,
        "issues": issues,
        "notes": notes
    }

# Usage:
# Run existing claim extraction, then interactively label each claim
```

#### TASK 1.3: Create Entailment Dataset

**Goal:** 50-100 labeled examples of claim-quote entailment
**Time estimate:** 6-10 hours of manual labeling

**Process:**

1. For each claim from labeled dataset, get top 10 quotes from semantic search
2. For each claim-quote pair, label as:
   - **SUPPORTS:** Quote directly asserts the claim or provides clear evidence
   - **RELATED:** Quote is topically related but doesn't validate the claim
   - **NEUTRAL:** Quote is unrelated or provides no evidence
   - **CONTRADICTS:** Quote contradicts or undermines the claim

3. Save in format:

```python
# entailment_dataset.json
{
  "examples": [
    {
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quote": "Bitcoin hit its all-time high of $69,000 in November 2021",
      "relationship": "SUPPORTS",
      "reasoning": "Quote directly states the claim with exact figures and date",
      "confidence": 1.0
    },
    {
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quote": "Cryptocurrency markets were very volatile in 2021",
      "relationship": "RELATED",
      "reasoning": "Topically related but doesn't mention Bitcoin's specific price",
      "confidence": 0.9
    },
    {
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quote": "Bitcoin was trading around $45,000 in early 2021",
      "relationship": "RELATED",
      "reasoning": "Related to Bitcoin price but different time period",
      "confidence": 0.8
    }
  ]
}
```

**Labeling guidelines document:**
Create `evaluation/ENTAILMENT_LABELING_GUIDELINES.md`:

```markdown
# Entailment Labeling Guidelines

## SUPPORTS - Quote provides direct evidence for the claim
The quote must:
- Directly assert the claim, OR
- Provide specific evidence that validates the claim
- Use similar or synonymous language

Examples:
✅ Claim: "Bitcoin reached $69,000 in November 2021"
   Quote: "Bitcoin hit its all-time high of $69,000 in November"
   → SUPPORTS (exact figures, same event)

✅ Claim: "Tesla delivered 1 million cars in 2022"
   Quote: "Tesla's 2022 deliveries crossed the million mark"
   → SUPPORTS (synonymous language, same fact)

## RELATED - Quote is topically connected but doesn't validate
The quote:
- Discusses same topic/entity
- But doesn't provide evidence for the specific claim
- May provide context or background

Examples:
⚠️ Claim: "Bitcoin reached $69,000 in November 2021"
   Quote: "Cryptocurrency markets were volatile in 2021"
   → RELATED (topical connection, no specific validation)

⚠️ Claim: "Tesla delivered 1 million cars in 2022"
   Quote: "Tesla's production capacity has grown significantly"
   → RELATED (same company, related topic, but no specific numbers)

## NEUTRAL - Quote is unrelated
No topical connection or relevance.

## CONTRADICTS - Quote undermines or opposes the claim
The quote provides evidence against the claim.

## Common Pitfalls (Label as RELATED, not SUPPORTS)

❌ WRONG: Marking "Cryptocurrency volatility" as SUPPORTS for "Bitcoin price"
   → This is RELATED only

❌ WRONG: Marking general statement as SUPPORTS for specific claim
   Claim: "Company X revenue was $10B"
   Quote: "Company X has strong financials"
   → This is RELATED (general) not SUPPORTS (specific)

❌ WRONG: Marking temporal proximity as SUPPORTS
   Claim: "Event happened on June 1"
   Quote: "Event was discussed in early June"
   → This is RELATED (timeframe) not SUPPORTS (exact date)
```

**Tool to build:**

```python
# evaluation/label_entailment.py
def label_entailment_interactive(claim, quote):
    print(f"\n{'='*60}")
    print(f"Claim: {claim}")
    print(f"Quote: {quote}")
    print(f"{'='*60}")

    relationship = input("Relationship (supports/related/neutral/contradicts): ").lower()
    reasoning = input("Reasoning (why?): ")
    confidence = float(input("Confidence (0.0-1.0): "))

    return {
        "claim": claim,
        "quote": quote,
        "relationship": relationship.upper(),
        "reasoning": reasoning,
        "confidence": confidence
    }
```

#### TASK 1.4: Split Dataset (Train/Val/Test)

- [ ] Split claim quality dataset: 70% train, 15% val, 15% test
- [ ] Split entailment dataset: 70% train, 15% val, 15% test
- [ ] Ensure no overlap (same claim shouldn't be in train and test)
- [ ] Document split methodology

**Code:**

```python
import random
import json

def split_dataset(examples, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train/val/test."""
    random.seed(42)  # Reproducibility
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        'train': examples[:train_end],
        'val': examples[train_end:val_end],
        'test': examples[val_end:]
    }

# Usage
with open('claims_dataset.json') as f:
    claims_data = json.load(f)

split = split_dataset(claims_data['examples'])
print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")

# Save splits
for split_name, split_data in split.items():
    with open(f'claims_dataset_{split_name}.json', 'w') as f:
        json.dump(split_data, f, indent=2)
```

#### TASK 1.5: Dataset Statistics & Validation

- [ ] Count examples per split
- [ ] Analyze label distribution (good/bad ratio, SUPPORTS/RELATED ratio)
- [ ] Check for data quality issues (missing fields, malformed JSON)
- [ ] Document dataset statistics

**Desired statistics:**

```
Claim Quality Dataset:
- Total examples: 50
- Train: 35 (70%), Val: 8 (15%), Test: 7 (15%)
- Label distribution:
  - GOOD: 30 (60%)
  - BAD: 20 (40%)
- Issue distribution:
  - pronoun: 12
  - vague: 8
  - opinion: 6
  - missing_context: 5

Entailment Dataset:
- Total examples: 100
- Train: 70, Val: 15, Test: 15
- Label distribution:
  - SUPPORTS: 40 (40%)
  - RELATED: 45 (45%)
  - NEUTRAL: 10 (10%)
  - CONTRADICTS: 5 (5%)
```

---

### PHASE 2: Baseline Measurement

**Goal:** Measure current performance objectively
**Time estimate:** 3-5 days
**Priority:** P0 (needed before optimization)

#### TASK 2.1: Convert Existing Prompts to DSPy Signatures

- [ ] Define ClaimExtraction signature
- [ ] Define EntailmentValidation signature
- [ ] Test that signatures produce same output as old prompts
- [ ] Document signature design choices

**Code template:**

```python
# dspy_signatures.py
import dspy
from typing import List

class ClaimExtraction(dspy.Signature):
    """Extract factual, verifiable claims from podcast transcript text.

    Quality criteria for claims:
    - Factual and verifiable (not opinions or speculation)
    - Self-contained (includes all necessary context, no pronouns)
    - Specific (names entities, gives numbers/dates when relevant)
    - Concise (1-2 sentences maximum)

    Avoid:
    - Pronouns (he, she, they, it) without clear antecedents
    - Vague language (very, really, amazing, terrible)
    - Opinion markers (I think, I believe, seems like)
    - Claims requiring external context to understand
    """

    transcript_chunk: str = dspy.InputField(
        desc="Segment of podcast transcript to extract claims from"
    )
    claims: List[str] = dspy.OutputField(
        desc="List of factual claims extracted from the transcript, each self-contained and verifiable"
    )


class EntailmentValidation(dspy.Signature):
    """Determine whether a quote provides evidence that supports a claim.

    Classification guide:
    - SUPPORTS: Quote directly asserts the claim OR provides clear, specific evidence that validates it.
      The quote must contain the key factual details of the claim (numbers, names, dates, etc.)

    - RELATED: Quote is topically connected to the claim but does NOT provide evidence for it.
      May discuss same entity/topic but lacks the specific details needed to validate the claim.

    - NEUTRAL: Quote is unrelated to the claim, provides no relevant information.

    - CONTRADICTS: Quote provides evidence that undermines or opposes the claim.

    Be strict: Only use SUPPORTS when the quote genuinely validates the claim's specific assertions.
    If the quote is merely on the same topic but doesn't prove the claim, use RELATED.
    """

    claim: str = dspy.InputField(desc="The factual claim to validate")
    quote: str = dspy.InputField(desc="The quote text from the podcast transcript")

    relationship: str = dspy.OutputField(
        desc="Classification: SUPPORTS, RELATED, NEUTRAL, or CONTRADICTS"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation (1-2 sentences) for the classification"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0"
    )
```

**Testing:**

```python
# Test that DSPy signatures work
import dspy

lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Test claim extraction
extractor = dspy.ChainOfThought(ClaimExtraction)
result = extractor(transcript_chunk="Bitcoin hit $69,000 in November 2021...")
print("Extracted claims:", result.claims)

# Test entailment
validator = dspy.ChainOfThought(EntailmentValidation)
result = validator(
    claim="Bitcoin reached $69,000 in November 2021",
    quote="Bitcoin hit its all-time high of $69,000 in November"
)
print(f"Relationship: {result.relationship}")
print(f"Reasoning: {result.reasoning}")
```

#### TASK 2.2: Implement Evaluation Metrics

- [ ] Implement claim_quality_metric
- [ ] Implement entailment_accuracy_metric
- [ ] Test metrics on sample data
- [ ] Validate metric edge cases

**Code template:**

```python
# metrics.py
import dspy
from typing import List

def claim_quality_metric(example, pred, trace=None):
    """
    Evaluate claim quality based on presence of common quality issues.

    Returns a score from 0.0 to 1.0:
    - 1.0 = perfect claim (no issues)
    - 0.0 = all claims have issues

    Checks for:
    - Pronouns (he, she, they, it, his, her, their)
    - Vague words (very, really, something, someone, stuff, thing)
    - Opinion markers (think, believe, feel, seems, appears)
    """
    predicted_claims = pred.claims

    if not predicted_claims:
        return 0.0

    # Define quality issue patterns
    PRONOUNS = ['he', 'she', 'they', 'it', 'him', 'her', 'them', 'his', 'hers', 'their', 'its']
    VAGUE_WORDS = ['very', 'really', 'something', 'someone', 'stuff', 'thing', 'some']
    OPINION_WORDS = ['think', 'believe', 'feel', 'seems', 'appears', 'probably', 'maybe']

    issues_found = 0
    total_claims = len(predicted_claims)

    for claim in predicted_claims:
        claim_lower = claim.lower()
        words = claim_lower.split()

        # Check for pronouns (as standalone words)
        if any(pronoun in words for pronoun in PRONOUNS):
            issues_found += 1
            continue

        # Check for vague words
        if any(vague in words for vague in VAGUE_WORDS):
            issues_found += 1
            continue

        # Check for opinion markers
        if any(opinion in claim_lower for opinion in OPINION_WORDS):
            issues_found += 1
            continue

    # Calculate quality score
    quality_score = 1.0 - (issues_found / total_claims)
    return quality_score


def entailment_accuracy_metric(example, pred, trace=None):
    """
    Measure accuracy of entailment classification.

    Returns:
    - 1.0 if prediction matches ground truth
    - 0.0 if prediction is wrong
    - -2.0 if RELATED was incorrectly predicted as SUPPORTS (extra penalty)

    The extra penalty for false positives helps DSPy learn to be stricter.
    """
    gold_relationship = example.relationship.upper().strip()
    pred_relationship = pred.relationship.upper().strip()

    # Exact match
    if gold_relationship == pred_relationship:
        return 1.0

    # Extra penalty for the specific error we want to minimize
    # (RELATED misclassified as SUPPORTS)
    if gold_relationship == "RELATED" and pred_relationship == "SUPPORTS":
        return -2.0

    # Regular penalty for other misclassifications
    return 0.0


def entailment_false_positive_rate(predictions):
    """
    Calculate false positive rate for SUPPORTS classification.

    False Positive Rate = FP / (FP + TN)
    where FP = RELATED predicted as SUPPORTS
          TN = RELATED correctly predicted as RELATED

    Returns value between 0.0 and 1.0 (lower is better).
    """
    false_positives = 0
    true_negatives = 0

    for example, pred, score in predictions:
        gold = example.relationship.upper()
        predicted = pred.relationship.upper()

        if gold == "RELATED":
            if predicted == "SUPPORTS":
                false_positives += 1
            elif predicted == "RELATED":
                true_negatives += 1

    total_negatives = false_positives + true_negatives
    if total_negatives == 0:
        return 0.0

    return false_positives / total_negatives
```

#### TASK 2.3: Load Evaluation Datasets

- [ ] Write data loader for claim quality dataset
- [ ] Write data loader for entailment dataset
- [ ] Convert to DSPy Example format
- [ ] Validate data loading works correctly

**Code template:**

```python
# data_loader.py
import json
import dspy
from typing import List

def load_claim_quality_dataset(split: str = 'train') -> List[dspy.Example]:
    """
    Load claim quality dataset and convert to DSPy Examples.

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        List of dspy.Example objects with inputs marked
    """
    with open(f'evaluation/claims_dataset_{split}.json') as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = dspy.Example(
            transcript_chunk=item['transcript_chunk'],
            claims=[item['claim']],  # Wrap in list for consistency
            quality=item['quality'],
            issues=item.get('issues', []),
            notes=item.get('notes', '')
        ).with_inputs('transcript_chunk')

        examples.append(example)

    return examples


def load_entailment_dataset(split: str = 'train') -> List[dspy.Example]:
    """
    Load entailment dataset and convert to DSPy Examples.

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        List of dspy.Example objects with inputs marked
    """
    with open(f'evaluation/entailment_dataset_{split}.json') as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = dspy.Example(
            claim=item['claim'],
            quote=item['quote'],
            relationship=item['relationship'],
            reasoning=item['reasoning'],
            confidence=item['confidence']
        ).with_inputs('claim', 'quote')

        examples.append(example)

    return examples


# Usage
trainset_claims = load_claim_quality_dataset('train')
valset_claims = load_claim_quality_dataset('val')
testset_claims = load_claim_quality_dataset('test')

print(f"Loaded {len(trainset_claims)} claim training examples")
print(f"Loaded {len(valset_claims)} claim validation examples")
print(f"Loaded {len(testset_claims)} claim test examples")
```

#### TASK 2.4: Run Baseline Evaluation - Claim Extraction

- [ ] Create evaluation script for claim extraction
- [ ] Run baseline (unoptimized) on validation set
- [ ] Run baseline on test set (document but don't optimize on it)
- [ ] Generate evaluation report with metrics

**Code template:**

```python
# baseline_evaluation_claims.py
import dspy
from dspy.evaluate import Evaluate
from data_loader import load_claim_quality_dataset
from metrics import claim_quality_metric
from dspy_signatures import ClaimExtraction

# Configure DSPy
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Load datasets
trainset = load_claim_quality_dataset('train')
valset = load_claim_quality_dataset('val')
testset = load_claim_quality_dataset('test')

# Create baseline program (unoptimized)
baseline_extractor = dspy.ChainOfThought(ClaimExtraction)

# Set up evaluator
evaluator = Evaluate(
    devset=valset,
    metric=claim_quality_metric,
    num_threads=4,
    display_progress=True,
    display_table=10
)

# Run evaluation
print("="*60)
print("BASELINE EVALUATION - Claim Extraction")
print("="*60)

baseline_score = evaluator(baseline_extractor)

print(f"\nBaseline Score on Validation Set: {baseline_score}%")
print(f"Target Score: >85% (to achieve <15% low-quality claims)")
print(f"Gap to Target: {85 - baseline_score}%")

# Save results
import json
results = {
    'baseline_score': baseline_score,
    'target_score': 85,
    'validation_set_size': len(valset),
    'date': '2025-10-25',
    'model': 'qwen2.5:7b-instruct',
    'metric': 'claim_quality_metric'
}

with open('results/baseline_claims.json', 'w') as f:
    json.dump(results, f, indent=2)
```

#### TASK 2.5: Run Baseline Evaluation - Entailment

- [ ] Create evaluation script for entailment validation
- [ ] Run baseline on validation set
- [ ] Calculate false positive rate
- [ ] Generate evaluation report

**Code template:**

```python
# baseline_evaluation_entailment.py
import dspy
from dspy.evaluate import Evaluate
from data_loader import load_entailment_dataset
from metrics import entailment_accuracy_metric, entailment_false_positive_rate
from dspy_signatures import EntailmentValidation

# Configure DSPy
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Load datasets
trainset = load_entailment_dataset('train')
valset = load_entailment_dataset('val')
testset = load_entailment_dataset('test')

# Create baseline program
baseline_validator = dspy.ChainOfThought(EntailmentValidation)

# Set up evaluator
evaluator = Evaluate(
    devset=valset,
    metric=entailment_accuracy_metric,
    num_threads=4,
    display_progress=True,
    display_table=10
)

# Run evaluation
print("="*60)
print("BASELINE EVALUATION - Entailment Validation")
print("="*60)

result = evaluator(baseline_validator)

# Calculate false positive rate
fp_rate = entailment_false_positive_rate(result.results)

print(f"\nBaseline Accuracy: {result.score}%")
print(f"False Positive Rate (RELATED→SUPPORTS): {fp_rate*100:.1f}%")
print(f"Target FP Rate: <10%")
print(f"Gap to Target: {fp_rate*100 - 10:.1f}%")

# Save results
import json
results = {
    'baseline_accuracy': result.score,
    'false_positive_rate': fp_rate,
    'target_fp_rate': 0.10,
    'validation_set_size': len(valset),
    'date': '2025-10-25',
    'model': 'qwen2.5:7b-instruct',
    'metric': 'entailment_accuracy_metric'
}

with open('results/baseline_entailment.json', 'w') as f:
    json.dump(results, f, indent=2)
```

#### TASK 2.6: Document Baseline Results

- [ ] Create baseline report document
- [ ] Include example predictions (good and bad)
- [ ] Analyze failure modes
- [ ] Set optimization targets based on baseline

**Report template:**

```markdown
# Baseline Evaluation Report

Date: 2025-10-25
Model: qwen2.5:7b-instruct (Ollama)

## Claim Extraction

### Metrics
- **Validation Score:** 62.5%
- **Target Score:** >85%
- **Gap:** 22.5%

### Analysis
- Total validation examples: 8
- Examples with issues: 3 (37.5%)

### Common Failure Modes
1. **Pronouns without context** (2 examples)
   - "He announced the new product" → should be "Elon Musk announced Tesla's new product"

2. **Vague language** (1 example)
   - "It was very successful" → should be specific about what succeeded and how

### Example Predictions

Good:
- Input: "Bitcoin hit $69,000 in November..."
- Output: "Bitcoin reached $69,000 in November 2021"
- Issues: None ✅

Bad:
- Input: "Elon said Tesla is amazing..."
- Output: "He said the company was amazing"
- Issues: Pronoun ("he"), vague ("amazing"), missing context ❌

## Entailment Validation

### Metrics
- **Validation Accuracy:** 73.3%
- **False Positive Rate:** 28.6% (2/7 RELATED cases)
- **Target FP Rate:** <10%
- **Gap:** 18.6%

### Confusion Matrix
```

                  Pred SUPPORTS | Pred RELATED
Actual SUPPORTS        5              1
Actual RELATED         2              5

```

### Common Failure Modes
1. **Topical similarity mistaken for support** (2 examples)
   - Claim: "Bitcoin reached $69,000"
   - Quote: "Cryptocurrency markets were volatile"
   - Predicted: SUPPORTS ❌
   - Should be: RELATED

### Next Steps
1. Optimize with DSPy MIPROv2
2. Target: Reduce FP rate from 28.6% → <10%
3. Add few-shot examples of borderline cases
```

---

### PHASE 3: DSPy Optimization

**Goal:** Use DSPy teleprompters to improve prompts
**Time estimate:** 1-2 weeks
**Priority:** P1

#### TASK 3.1: Optimize Claim Extraction with MIPROv2

- [ ] Configure MIPROv2 optimizer for claim extraction
- [ ] Run optimization (may take 4-8 hours compute time)
- [ ] Evaluate optimized version on validation set
- [ ] Save optimized module

**Code template:**

```python
# optimize_claims.py
import dspy
from dspy.teleprompt import MIPROv2
from data_loader import load_claim_quality_dataset
from metrics import claim_quality_metric
from dspy_signatures import ClaimExtraction

# Configure DSPy
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Load data
trainset = load_claim_quality_dataset('train')
valset = load_claim_quality_dataset('val')

# Create student program
student = dspy.ChainOfThought(ClaimExtraction)

# Configure optimizer
optimizer = MIPROv2(
    metric=claim_quality_metric,
    num_candidates=10,        # Try 10 different prompt variations
    init_temperature=1.0,     # Creativity in prompt generation
    verbose=True              # Show optimization progress
)

print("="*60)
print("OPTIMIZING CLAIM EXTRACTION WITH MIPROv2")
print("="*60)
print(f"Training examples: {len(trainset)}")
print(f"Validation examples: {len(valset)}")
print("This may take 4-8 hours...")
print("="*60)

# Run optimization
optimized_extractor = optimizer.compile(
    student=student,
    trainset=trainset,
    valset=valset,
    num_trials=30,            # Number of optimization iterations
    max_bootstrapped_demos=3  # Use up to 3 few-shot examples
)

# Evaluate optimized version
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=valset,
    metric=claim_quality_metric,
    num_threads=4,
    display_progress=True,
    display_table=10
)

optimized_score = evaluator(optimized_extractor)

print(f"\nOptimized Score: {optimized_score}%")
print(f"Baseline Score: 62.5%")  # From baseline evaluation
print(f"Improvement: {optimized_score - 62.5}%")

# Save optimized module
optimized_extractor.save('models/optimized_claim_extractor.json')
print("\nSaved optimized module to models/optimized_claim_extractor.json")

# Save results
import json
results = {
    'optimized_score': optimized_score,
    'baseline_score': 62.5,
    'improvement': optimized_score - 62.5,
    'validation_set_size': len(valset),
    'optimizer': 'MIPROv2',
    'num_trials': 30,
    'max_bootstrapped_demos': 3,
    'date': '2025-10-25'
}

with open('results/optimized_claims.json', 'w') as f:
    json.dump(results, f, indent=2)
```

**Expected output:**

```
Iteration 1/30: Score 65.2%
Iteration 2/30: Score 68.7%
Iteration 3/30: Score 71.3%
...
Iteration 30/30: Score 87.5%

Best performing prompt found at iteration 24: 87.5%
```

#### TASK 3.2: Optimize Entailment Validation with MIPROv2

- [ ] Configure MIPROv2 optimizer for entailment
- [ ] Run optimization
- [ ] Evaluate on validation set
- [ ] Calculate false positive rate improvement
- [ ] Save optimized module

**Code template:**

```python
# optimize_entailment.py
import dspy
from dspy.teleprompt import MIPROv2
from data_loader import load_entailment_dataset
from metrics import entailment_accuracy_metric, entailment_false_positive_rate
from dspy_signatures import EntailmentValidation

# Configure DSPy
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Load data
trainset = load_entailment_dataset('train')
valset = load_entailment_dataset('val')

# Create student program
student = dspy.ChainOfThought(EntailmentValidation)

# Configure optimizer
optimizer = MIPROv2(
    metric=entailment_accuracy_metric,
    num_candidates=10,
    init_temperature=1.0,
    verbose=True
)

print("="*60)
print("OPTIMIZING ENTAILMENT VALIDATION WITH MIPROv2")
print("="*60)
print(f"Training examples: {len(trainset)}")
print(f"Validation examples: {len(valset)}")
print("Focus: Reducing false positives (RELATED → SUPPORTS)")
print("This may take 4-8 hours...")
print("="*60)

# Run optimization
optimized_validator = optimizer.compile(
    student=student,
    trainset=trainset,
    valset=valset,
    num_trials=30,
    max_bootstrapped_demos=4  # More examples for nuanced task
)

# Evaluate
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=valset,
    metric=entailment_accuracy_metric,
    num_threads=4,
    display_progress=True
)

result = evaluator(optimized_validator)
fp_rate = entailment_false_positive_rate(result.results)

print(f"\nOptimized Accuracy: {result.score}%")
print(f"Baseline Accuracy: 73.3%")
print(f"Improvement: {result.score - 73.3}%")
print()
print(f"Optimized FP Rate: {fp_rate*100:.1f}%")
print(f"Baseline FP Rate: 28.6%")
print(f"Improvement: {28.6 - fp_rate*100:.1f}%")

# Save
optimized_validator.save('models/optimized_entailment_validator.json')
print("\nSaved to models/optimized_entailment_validator.json")
```

#### TASK 3.3: Inspect Optimized Prompts

- [ ] Load optimized modules
- [ ] Inspect what DSPy changed (instructions, examples)
- [ ] Document optimizations DSPy discovered
- [ ] Understand why changes improved performance

**Code template:**

```python
# inspect_optimizations.py
import dspy
from dspy_signatures import ClaimExtraction, EntailmentValidation

# Load optimized modules
optimized_extractor = dspy.ChainOfThought(ClaimExtraction)
optimized_extractor.load('models/optimized_claim_extractor.json')

optimized_validator = dspy.ChainOfThought(EntailmentValidation)
optimized_validator.load('models/optimized_entailment_validator.json')

# Inspect claim extractor
print("="*60)
print("OPTIMIZED CLAIM EXTRACTOR")
print("="*60)
print("\nOptimized prompt:")
print(optimized_extractor.dump_state())
print("\nFew-shot examples included:")
for i, demo in enumerate(optimized_extractor.demos):
    print(f"\nExample {i+1}:")
    print(f"  Input: {demo.transcript_chunk[:100]}...")
    print(f"  Output: {demo.claims}")

# Inspect entailment validator
print("\n" + "="*60)
print("OPTIMIZED ENTAILMENT VALIDATOR")
print("="*60)
print("\nOptimized prompt:")
print(optimized_validator.dump_state())
```

#### TASK 3.4: Test on Hold-out Test Set

- [ ] Run optimized claim extractor on test set (never seen before)
- [ ] Run optimized entailment validator on test set
- [ ] Compare test performance to validation performance
- [ ] Ensure no overfitting (test ≈ validation scores)

**Code template:**

```python
# test_set_evaluation.py
import dspy
from dspy.evaluate import Evaluate
from data_loader import load_claim_quality_dataset, load_entailment_dataset
from metrics import claim_quality_metric, entailment_accuracy_metric
from dspy_signatures import ClaimExtraction, EntailmentValidation

# Configure
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Load optimized modules
optimized_extractor = dspy.ChainOfThought(ClaimExtraction)
optimized_extractor.load('models/optimized_claim_extractor.json')

optimized_validator = dspy.ChainOfThought(EntailmentValidation)
optimized_validator.load('models/optimized_entailment_validator.json')

# Load test sets (NEVER USED DURING OPTIMIZATION)
testset_claims = load_claim_quality_dataset('test')
testset_entailment = load_entailment_dataset('test')

print("="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)
print("⚠️  This is the first time seeing test data")
print("="*60)

# Evaluate claims
evaluator_claims = Evaluate(
    devset=testset_claims,
    metric=claim_quality_metric,
    num_threads=4,
    display_progress=True
)

test_score_claims = evaluator_claims(optimized_extractor)

print(f"\nClaim Extraction Test Score: {test_score_claims}%")
print(f"Claim Extraction Validation Score: 87.5%")  # From optimization
print(f"Difference: {abs(test_score_claims - 87.5):.1f}%")

# Evaluate entailment
evaluator_entailment = Evaluate(
    devset=testset_entailment,
    metric=entailment_accuracy_metric,
    num_threads=4,
    display_progress=True
)

result_entailment = evaluator_entailment(optimized_validator)

print(f"\nEntailment Validation Test Accuracy: {result_entailment.score}%")
print(f"Entailment Validation Val Accuracy: 91.2%")  # From optimization
print(f"Difference: {abs(result_entailment.score - 91.2):.1f}%")

# Check for overfitting
if abs(test_score_claims - 87.5) > 5:
    print("\n⚠️  WARNING: Claim extraction may be overfitting")
else:
    print("\n✅ Claim extraction generalizes well")

if abs(result_entailment.score - 91.2) > 5:
    print("⚠️  WARNING: Entailment validation may be overfitting")
else:
    print("✅ Entailment validation generalizes well")
```

---

### PHASE 4: Integration & Deployment

**Goal:** Integrate optimized DSPy modules into existing pipeline
**Time estimate:** 1-2 weeks
**Priority:** P1

#### TASK 4.1: Create DSPy-to-Python Integration Layer

- [ ] Write Python wrapper for optimized claim extractor
- [ ] Write Python wrapper for optimized entailment validator
- [ ] Handle error cases (timeouts, malformed responses)
- [ ] Match existing API interface for easy drop-in replacement

**Code template:**

```python
# dspy_integration.py
import dspy
from typing import List, Dict
from dspy_signatures import ClaimExtraction, EntailmentValidation

class DSPyClaimExtractor:
    """Wrapper around optimized DSPy claim extraction module."""

    def __init__(self, model_path: str = 'models/optimized_claim_extractor.json'):
        """Initialize with saved optimized module."""
        lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
        dspy.configure(lm=lm)

        self.extractor = dspy.ChainOfThought(ClaimExtraction)
        self.extractor.load(model_path)

    def extract_claims(self, transcript_chunk: str) -> List[str]:
        """
        Extract claims from transcript chunk.

        Args:
            transcript_chunk: Text segment from podcast transcript

        Returns:
            List of factual claims extracted from the text
        """
        try:
            result = self.extractor(transcript_chunk=transcript_chunk)
            return result.claims if result.claims else []
        except Exception as e:
            print(f"Error extracting claims: {e}")
            return []


class DSPyEntailmentValidator:
    """Wrapper around optimized DSPy entailment validation module."""

    def __init__(self, model_path: str = 'models/optimized_entailment_validator.json'):
        """Initialize with saved optimized module."""
        lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
        dspy.configure(lm=lm)

        self.validator = dspy.ChainOfThought(EntailmentValidation)
        self.validator.load(model_path)

    def validate_entailment(self, claim: str, quote: str) -> Dict[str, any]:
        """
        Validate whether quote supports claim.

        Args:
            claim: The factual claim to validate
            quote: The quote text to check against claim

        Returns:
            Dict with keys: relationship, reasoning, confidence
        """
        try:
            result = self.validator(claim=claim, quote=quote)
            return {
                'relationship': result.relationship,
                'reasoning': result.reasoning,
                'confidence': float(result.confidence)
            }
        except Exception as e:
            print(f"Error validating entailment: {e}")
            return {
                'relationship': 'NEUTRAL',
                'reasoning': f'Error during validation: {e}',
                'confidence': 0.0
            }


# Usage example
if __name__ == '__main__':
    # Initialize
    extractor = DSPyClaimExtractor()
    validator = DSPyEntailmentValidator()

    # Test claim extraction
    transcript = "Bitcoin hit $69,000 in November 2021, setting a new all-time high."
    claims = extractor.extract_claims(transcript)
    print(f"Extracted claims: {claims}")

    # Test entailment validation
    for claim in claims:
        result = validator.validate_entailment(
            claim=claim,
            quote=transcript
        )
        print(f"Claim: {claim}")
        print(f"Relationship: {result['relationship']}")
        print(f"Reasoning: {result['reasoning']}")
```

#### TASK 4.2: Update Existing Pipeline to Use DSPy Modules

- [ ] Identify where to inject DSPy modules (replace OllamaClient calls)
- [ ] Update claim extraction to use DSPyClaimExtractor
- [ ] Update entailment validation to use DSPyEntailmentValidator
- [ ] Maintain backward compatibility (flag to switch between old/new)

**Integration points in old architecture:**

1. **Claim Extraction:** `src/etl/podcasts/claims/ollama-client.ts:extractClaimsWithQuotes()`
   - Replace with call to Python DSPy wrapper
   - Use subprocess or HTTP API to communicate with Python

2. **Entailment Validation:** Currently done with hardcoded prompt
   - Add new validation step using DSPyEntailmentValidator
   - Filter quotes based on relationship (keep only SUPPORTS)

**Option A: Python subprocess approach**

```python
# dspy_service.py
# Run this as a simple HTTP service
from flask import Flask, request, jsonify
from dspy_integration import DSPyClaimExtractor, DSPyEntailmentValidator

app = Flask(__name__)

extractor = DSPyClaimExtractor()
validator = DSPyEntailmentValidator()

@app.route('/extract_claims', methods=['POST'])
def extract_claims():
    data = request.json
    claims = extractor.extract_claims(data['transcript_chunk'])
    return jsonify({'claims': claims})

@app.route('/validate_entailment', methods=['POST'])
def validate_entailment():
    data = request.json
    result = validator.validate_entailment(
        claim=data['claim'],
        quote=data['quote']
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
```

Then from TypeScript:

```typescript
// Call Python DSPy service
async function extractClaimsWithDSPy(transcriptChunk: string): Promise<string[]> {
  const response = await fetch('http://localhost:5000/extract_claims', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript_chunk: transcriptChunk })
  });
  const data = await response.json();
  return data.claims;
}
```

**Option B: Full Python rewrite (more work but cleaner)**

- Rewrite entire pipeline in Python
- Use DSPy modules directly without HTTP layer
- Cleaner architecture, better performance

#### TASK 4.3: A/B Testing Framework

- [ ] Implement side-by-side comparison of old vs new prompts
- [ ] Process same episodes with both systems
- [ ] Collect metrics for comparison
- [ ] Statistical significance testing

**Code template:**

```python
# ab_test.py
import json
from typing import List, Dict
from old_system import OldClaimExtractor  # Hypothetical old system
from dspy_integration import DSPyClaimExtractor

def ab_test_claim_extraction(episode_ids: List[int]) -> Dict:
    """
    Run A/B test comparing old and new claim extraction.

    Args:
        episode_ids: List of episode IDs to test on

    Returns:
        Comparison metrics
    """
    old_extractor = OldClaimExtractor()
    new_extractor = DSPyClaimExtractor()

    results = {
        'episode_count': len(episode_ids),
        'old_claims': [],
        'new_claims': [],
        'metrics': {}
    }

    for episode_id in episode_ids:
        # Load episode transcript
        transcript = load_transcript(episode_id)

        # Extract with old system
        old_claims = old_extractor.extract(transcript)

        # Extract with new system
        new_claims = new_extractor.extract_claims(transcript)

        results['old_claims'].extend(old_claims)
        results['new_claims'].extend(new_claims)

    # Calculate metrics
    results['metrics'] = {
        'old_total_claims': len(results['old_claims']),
        'new_total_claims': len(results['new_claims']),
        'old_quality_score': calculate_quality(results['old_claims']),
        'new_quality_score': calculate_quality(results['new_claims']),
    }

    return results

# Run test
test_results = ab_test_claim_extraction(episode_ids=[1, 2, 3, 4, 5])
print(json.dumps(test_results['metrics'], indent=2))
```

#### TASK 4.4: Production Deployment

- [ ] Deploy Python DSPy service to production environment
- [ ] Set up monitoring and logging
- [ ] Configure fallback to old system if DSPy service fails
- [ ] Gradual rollout (10% → 50% → 100% of traffic)

#### TASK 4.5: Documentation

- [ ] Document how to use optimized modules
- [ ] Document how to re-optimize with new data
- [ ] Create runbook for production issues
- [ ] Training materials for team

---

## Code Templates & Examples

### Complete End-to-End Example

```python
# complete_example.py
"""
Complete DSPy optimization example for claim extraction.
This shows the full workflow from data loading to optimization.
"""

import dspy
from typing import List
import json

# 1. Configure DSPy with Ollama
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# 2. Define signature
class ClaimExtraction(dspy.Signature):
    """Extract factual claims from transcript text."""
    transcript_chunk: str = dspy.InputField()
    claims: List[str] = dspy.OutputField()

# 3. Load evaluation dataset
def load_dataset(path: str) -> List[dspy.Example]:
    with open(path) as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = dspy.Example(
            transcript_chunk=item['transcript_chunk'],
            claims=item['claims']
        ).with_inputs('transcript_chunk')
        examples.append(example)

    return examples

trainset = load_dataset('claims_train.json')
valset = load_dataset('claims_val.json')

# 4. Define metric
def claim_quality_metric(example, pred, trace=None):
    """Simple metric: no pronouns allowed."""
    pronouns = ['he', 'she', 'they', 'it']

    for claim in pred.claims:
        if any(p in claim.lower().split() for p in pronouns):
            return 0.0
    return 1.0

# 5. Create baseline
baseline = dspy.ChainOfThought(ClaimExtraction)

# 6. Evaluate baseline
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=valset, metric=claim_quality_metric)
baseline_score = evaluator(baseline)
print(f"Baseline: {baseline_score}%")

# 7. Optimize with MIPROv2
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(metric=claim_quality_metric, num_candidates=5)
optimized = optimizer.compile(
    student=baseline,
    trainset=trainset,
    valset=valset,
    num_trials=10
)

# 8. Evaluate optimized
optimized_score = evaluator(optimized)
print(f"Optimized: {optimized_score}%")
print(f"Improvement: {optimized_score - baseline_score}%")

# 9. Save
optimized.save('optimized_claim_extractor.json')
```

### Alternative Optimizer: BootstrapFewShot

If MIPROv2 is too slow or complex, start with simpler BootstrapFewShot:

```python
from dspy.teleprompt import BootstrapFewShot

# Simpler optimizer - just finds good few-shot examples
optimizer = BootstrapFewShot(
    metric=claim_quality_metric,
    max_bootstrapped_demos=4  # Use 4 few-shot examples
)

optimized_simple = optimizer.compile(
    student=baseline,
    trainset=trainset
)
```

---

## Metrics Definitions

### Claim Quality Metrics

#### 1. Pronoun Detection (Simple)

```python
def no_pronouns_metric(example, pred, trace=None):
    """Returns 1.0 if no pronouns, 0.0 otherwise."""
    pronouns = ['he', 'she', 'they', 'it', 'him', 'her', 'them']
    for claim in pred.claims:
        if any(p in claim.lower().split() for p in pronouns):
            return 0.0
    return 1.0
```

#### 2. Composite Quality Score (Advanced)

```python
def composite_quality_metric(example, pred, trace=None):
    """
    Comprehensive quality score combining multiple factors.
    Returns score from 0.0 to 1.0.
    """
    if not pred.claims:
        return 0.0

    scores = []
    for claim in pred.claims:
        score = 1.0

        # Penalty for pronouns
        pronouns = ['he', 'she', 'they', 'it', 'him', 'her', 'them', 'his', 'hers', 'their', 'its']
        if any(p in claim.lower().split() for p in pronouns):
            score -= 0.4

        # Penalty for vague words
        vague = ['very', 'really', 'something', 'someone', 'thing', 'stuff']
        if any(v in claim.lower().split() for v in vague):
            score -= 0.2

        # Penalty for opinion markers
        opinion = ['think', 'believe', 'feel', 'seems', 'appears']
        if any(o in claim.lower() for o in opinion):
            score -= 0.3

        # Bonus for named entities (simple heuristic: capitalized words)
        words = claim.split()
        capitalized = [w for w in words if w[0].isupper() and w.lower() not in ['the', 'a', 'an']]
        if len(capitalized) >= 2:
            score += 0.1

        scores.append(max(0.0, min(1.0, score)))

    return sum(scores) / len(scores)
```

#### 3. LLM-as-Judge (Expensive but Accurate)

```python
class AssessClaimQuality(dspy.Signature):
    """Judge claim quality on multiple dimensions."""
    claim: str = dspy.InputField()
    has_pronouns: bool = dspy.OutputField(desc="Does claim contain pronouns without clear antecedents?")
    is_specific: bool = dspy.OutputField(desc="Is claim specific rather than vague?")
    is_factual: bool = dspy.OutputField(desc="Is claim factual rather than opinion?")
    is_self_contained: bool = dspy.OutputField(desc="Can claim be understood without additional context?")

def llm_judge_metric(example, pred, trace=None):
    """Use LLM to judge claim quality."""
    judge = dspy.ChainOfThought(AssessClaimQuality)

    total_score = 0.0
    for claim in pred.claims:
        assessment = judge(claim=claim)

        # Calculate score (1.0 = perfect)
        score = 0.0
        if not assessment.has_pronouns:
            score += 0.25
        if assessment.is_specific:
            score += 0.25
        if assessment.is_factual:
            score += 0.25
        if assessment.is_self_contained:
            score += 0.25

        total_score += score

    return total_score / len(pred.claims) if pred.claims else 0.0
```

### Entailment Metrics

#### 1. Exact Accuracy

```python
def entailment_exact_accuracy(example, pred, trace=None):
    """Simple exact match metric."""
    gold = example.relationship.upper().strip()
    predicted = pred.relationship.upper().strip()
    return 1.0 if gold == predicted else 0.0
```

#### 2. Weighted Accuracy with False Positive Penalty

```python
def entailment_weighted_metric(example, pred, trace=None):
    """
    Weighted metric that penalizes false positives more heavily.

    Scoring:
    - Correct: +1.0
    - Wrong: 0.0
    - RELATED→SUPPORTS (false positive): -2.0 (extra penalty)
    """
    gold = example.relationship.upper()
    predicted = pred.relationship.upper()

    if gold == predicted:
        return 1.0

    # Extra penalty for the specific error we want to avoid
    if gold == "RELATED" and predicted == "SUPPORTS":
        return -2.0

    return 0.0
```

#### 3. F1 Score for SUPPORTS Class

```python
def entailment_f1_supports(predictions):
    """
    Calculate F1 score specifically for SUPPORTS classification.

    Args:
        predictions: List of (example, pred, score) tuples from evaluator

    Returns:
        F1 score between 0.0 and 1.0
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for example, pred, _ in predictions:
        gold = example.relationship.upper()
        predicted = pred.relationship.upper()

        if gold == "SUPPORTS" and predicted == "SUPPORTS":
            true_positives += 1
        elif gold != "SUPPORTS" and predicted == "SUPPORTS":
            false_positives += 1
        elif gold == "SUPPORTS" and predicted != "SUPPORTS":
            false_negatives += 1

    if true_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

---

## Key Decisions & Trade-offs

### Decision 1: Which Optimizer to Use?

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| **BootstrapFewShot** | Fast (1-2 hours), simple, good starting point | Only optimizes few-shot examples, not instructions | Initial experiments, small datasets |
| **MIPROv2** | Optimizes both instructions and examples, best results | Slow (4-8 hours), uses many LLM calls | Production optimization, when quality matters most |
| **COPRO** | Optimizes signature fields (descriptions) | Limited to field-level changes | When signature descriptions need tuning |
| **GRPO** | Fine-tunes model weights | Requires GPU training, complex setup | Advanced use cases only |

**Recommendation:** Start with BootstrapFewShot for quick validation, then use MIPROv2 for production.

### Decision 2: How Much Labeled Data?

| Amount | Quality Level | Time to Create | Suitable For |
|--------|---------------|----------------|--------------|
| 20-30 examples | Minimum viable | 2-3 hours | Initial experiments, proof of concept |
| 50-100 examples | Good | 1 day | Production optimization |
| 200-500 examples | Excellent | 2-3 days | Robust production system |
| 1000+ examples | Research-grade | 1-2 weeks | Continuous improvement |

**Recommendation:** Start with 50 examples (1 day), expand to 200 as time permits.

### Decision 3: Train on All Data or Hold Out Test Set?

**Option A: Use all data for optimization**

- Pros: More training data = better optimization
- Cons: Can't measure generalization

**Option B: Hold out test set (recommended)**

- Pros: Validates that optimization generalizes
- Cons: Less training data

**Recommendation:** Use 70/15/15 split (train/val/test). Never optimize on test set.

### Decision 4: Single Task or Multi-Task Optimization?

**Option A: Optimize tasks separately** (recommended for start)

- Optimize claim extraction first
- Then optimize entailment validation
- Easier to debug, clearer metrics

**Option B: Joint optimization**

- Optimize entire pipeline end-to-end
- More complex but potentially better
- Requires more sophisticated metrics

**Recommendation:** Start with separate optimization, consider joint later.

### Decision 5: Python or TypeScript Implementation?

**Option A: Python-only (recommended)**

- DSPy is Python-native
- Easier integration
- Better ML ecosystem

**Option B: TypeScript with Python service**

- Keep existing TypeScript codebase
- Call Python DSPy via HTTP
- More operational complexity

**Recommendation:** Consider full Python rewrite for cleaner architecture.

---

## Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: Optimizing Without Ground Truth

**What:** Running optimization without labeled evaluation data
**Why bad:** DSPy has nothing to optimize for, results are meaningless
**Fix:** Always create evaluation dataset first (Phase 1)

### ❌ Anti-Pattern 2: Using Test Set for Optimization

**What:** Including test examples in training or validation
**Why bad:** Overfitting, false sense of performance
**Fix:** Strict train/val/test split, never look at test during development

### ❌ Anti-Pattern 3: Ignoring Baseline Measurement

**What:** Starting optimization without measuring current performance
**Why bad:** Can't tell if optimization helped
**Fix:** Always run baseline evaluation (Phase 2) before optimizing

### ❌ Anti-Pattern 4: Over-Complex Metrics

**What:** Creating metrics that are too sophisticated or hard to interpret
**Why bad:** Hard to debug, unclear what's being optimized
**Fix:** Start simple (e.g., "no pronouns"), add complexity gradually

### ❌ Anti-Pattern 5: Trusting First Optimization Run

**What:** Deploying optimized module without validation
**Why bad:** May not generalize, may have bugs
**Fix:** Always validate on held-out test set (Task 3.4)

### ❌ Anti-Pattern 6: Neglecting Edge Cases

**What:** Only labeling "obvious" examples in evaluation dataset
**Why bad:** System fails on edge cases
**Fix:** Deliberately include borderline cases in dataset

### ❌ Anti-Pattern 7: Not Inspecting Optimizations

**What:** Treating DSPy as black box, not understanding what changed
**Why bad:** Can't learn from optimizations, hard to debug
**Fix:** Always inspect optimized prompts (Task 3.3)

### ❌ Anti-Pattern 8: Assuming More Data = Better

**What:** Creating 1000s of low-quality labeled examples
**Why bad:** Garbage in, garbage out
**Fix:** 50 high-quality examples > 500 low-quality examples

---

## Resource Requirements

### Compute Resources

#### Development Environment

- **CPU:** Any modern CPU (8+ cores recommended)
- **RAM:** 16GB minimum, 32GB recommended
- **GPU:** Not required (Ollama can run on CPU), but helpful for speed
- **Disk:** 50GB for models and data

#### Ollama Resource Usage

- **qwen2.5:7b-instruct:** ~4GB RAM (Q4 quantization)
- **nomic-embed-text:** ~500MB RAM
- **Concurrent requests:** Can handle 3-5 parallel requests

#### DSPy Optimization Resource Usage

- **MIPROv2:** 1000-5000 LLM calls per optimization run
  - At ~2 seconds per call: 30 minutes - 3 hours
  - With batching: Can be faster
- **BootstrapFewShot:** 50-200 LLM calls
  - Much faster, good for iteration

### Time Estimates

| Phase | Tasks | Time Estimate | Dependencies |
|-------|-------|---------------|--------------|
| Phase 0: Setup | Environment configuration | 1-2 days | None |
| Phase 1: Dataset | Create 50 claim examples | 4-6 hours | Episodes selected |
| Phase 1: Dataset | Create 100 entailment examples | 6-10 hours | Claims extracted |
| Phase 1: Dataset | Split and validate | 2-3 hours | Labeling complete |
| Phase 2: Baseline | Implement signatures and metrics | 1-2 days | Phase 0 complete |
| Phase 2: Baseline | Run baseline evaluation | 2-4 hours | Datasets ready |
| Phase 3: Optimize | MIPROv2 optimization (claims) | 4-8 hours | Baseline measured |
| Phase 3: Optimize | MIPROv2 optimization (entailment) | 4-8 hours | Baseline measured |
| Phase 3: Optimize | Test set validation | 2-4 hours | Optimization complete |
| Phase 4: Deploy | Integration code | 2-3 days | Optimization validated |
| Phase 4: Deploy | Production deployment | 1-2 days | Integration tested |

**Total time estimate:**

- **Minimum viable (Phases 0-2):** 1-2 weeks
- **Optimized and tested (Phases 0-3):** 3-4 weeks
- **Production ready (Phases 0-4):** 4-6 weeks

### Budget Considerations

#### LLM API Costs

- **Local Ollama:** Free (using your own compute)
- **Alternative (OpenAI GPT-4):** Would cost $50-200 per optimization run
  - DSPy works with any LLM provider
  - Local Ollama is recommended for cost savings

#### Human Time

- **Labeling:** ~10-20 hours to create quality datasets
- **Development:** ~40-60 hours for implementation and testing
- **Total:** ~50-80 hours of developer time

---

## References & Documentation

### DSPy Official Resources

- **Main GitHub:** <https://github.com/stanfordnlp/dspy>
- **Documentation:** <https://dspy-docs.vercel.app/>
- **Paper:** "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"

### Key DSPy Concepts Documentation

- **Signatures:** <https://dspy-docs.vercel.app/docs/building-blocks/signatures>
- **Modules:** <https://dspy-docs.vercel.app/docs/building-blocks/modules>
- **Optimizers:** <https://dspy-docs.vercel.app/docs/building-blocks/optimizers>
- **Evaluation:** <https://dspy-docs.vercel.app/docs/building-blocks/evaluate>

### Project-Specific Documentation

- **Old Architecture:** `specs/OLD_ARCHITECTURE.md`
- **Project Guidelines:** `CLAUDE.md`
- **This Backlog:** `BACKLOG.md`

### Related Technologies

- **Ollama:** <https://ollama.ai/>
- **Qwen 2.5:** <https://github.com/QwenLM/Qwen2.5>
- **nomic-embed-text:** <https://github.com/nomic-ai/contrastors>

---

## Appendix: Quick Reference Commands

### Environment Setup

```bash
# Install DSPy
uv add dspy

# Verify Ollama
curl http://localhost:11434/api/tags

# Test Ollama generation
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b-instruct",
  "prompt": "What is 2+2?",
  "stream": false
}'
```

### Running Optimizations

```bash
# Run claim extraction optimization
python optimize_claims.py

# Run entailment optimization
python optimize_entailment.py

# Run baseline evaluation
python baseline_evaluation_claims.py
python baseline_evaluation_entailment.py

# Run test set validation
python test_set_evaluation.py
```

### Inspecting Results

```bash
# View optimization results
cat results/optimized_claims.json
cat results/optimized_entailment.json

# Inspect optimized prompts
python inspect_optimizations.py

# Compare baseline vs optimized
python compare_results.py
```

---

## Changelog

### 2025-10-25 - Initial Creation

- Created comprehensive backlog for DSPy optimization initiative
- Defined 4 phases with detailed tasks
- Documented current state and target goals
- Included code templates and examples
- Documented decision points and trade-offs

---

**END OF BACKLOG**
