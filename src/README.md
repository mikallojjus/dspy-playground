# DSPy Metrics

This directory contains evaluation metrics for DSPy optimization experiments.

## Available Metrics

### `claim_quality_metric(example, pred, trace=None)`

**Purpose:** Evaluate the quality of extracted claims from podcast transcripts.

**Returns:** Float between 0.0 and 1.0
- 1.0 = All claims are high quality (no issues detected)
- Lower scores = Higher proportion of claims with quality issues

**What it detects:**
1. **Pronouns** (28% of bad claims in manual review)
   - he, she, they, it, him, her, them, his, hers, their, its
2. **Vague words** (36% of bad claims)
   - recently, soon, thing, stuff, someone, something, very, really
3. **Opinion indicators** (14% of bad claims)
   - think, believe, feel, seems, appears, looks like, might, could, probably
4. **Advertisement language** (14% of bad claims)
   - offers, provides, delivers, high-quality basics, for a fraction
5. **Missing context** (57% of bad claims)
   - disapproval is, approval is, the mantra, there was a mantra

**Performance:**
- Accuracy: 85.7% (agrees with manual review 18/21 times)
- Precision: 100% (never flags good claims as bad)
- Recall: 78.6% (catches ~79% of bad claims)

**Usage:**
```python
from src.metrics import claim_quality_metric
import dspy

# In DSPy evaluation
evaluator = dspy.evaluate.Evaluate(
    devset=examples,
    metric=claim_quality_metric,
    display_progress=True
)

score = evaluator(model)
```

### `strict_claim_quality_metric(example, pred, trace=None)`

**Purpose:** More conservative version that detects additional issues.

**Additional checks:**
- Questions (claims ending with '?')
- Too short claims (<5 words)
- Too long claims (>40 words)
- Generic context indicators ("the ban", "the bill", "this policy")

**Performance:**
- Accuracy: 76.2%
- Precision: 100%
- Recall: 64.3%

**When to use:**
- Use `claim_quality_metric` for most cases (better accuracy)
- Use `strict_claim_quality_metric` when you want to be extra conservative

## Development History

Based on patterns identified in Experiment 2.1 manual review of 21 claims:
- 14 bad claims (67%)
- 7 good claims (33%)

Key issues found in bad claims:
- Missing context: 57%
- Pronouns: 28%
- Vague language: 36%
- Opinions: 14%
- Advertisements: 14%

## Testing

Run tests:
```bash
python exp_2_2_test_metric.py
```

This will:
1. Test against all 21 manually reviewed examples
2. Show detailed results for each claim
3. Calculate accuracy, precision, and recall
4. Test on sample crafted examples
