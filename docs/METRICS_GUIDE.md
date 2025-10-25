# Claim Quality Metrics - Comprehensive Guide

## The Problem with Pattern Matching

Your current metric (`src/metrics.py`) uses hardcoded patterns:

```python
PRONOUNS = ['he', 'she', 'they', ...]
AD_WORDS = ['get it today', 'order now', ...]
CONTEXT_INDICATORS = ['the ban', 'the bill', ...]
```

### Issues

1. **Can't understand context:**
   - ❌ Flags "Trump said he would build a ballroom" (GOOD claim, has referent)
   - ✅ Flags "He said he would build a ballroom" (BAD claim, no referent)
   - Pattern matching sees "he" in both → treats them the same

2. **Doesn't scale:**
   - Can't enumerate all ad phrases
   - Can't cover all vague indicators
   - Requires constant maintenance for thousands of episodes

3. **False positives/negatives:**
   - Misses semantic issues
   - Flags valid claims with pronouns that have referents

## Better Approaches

### Option 1: LLM-as-Judge ⭐ (Recommended)

**File:** `src/metrics_llm_judge.py`

**How it works:**
- Asks an LLM to judge each claim
- LLM understands semantic meaning
- Can distinguish "Trump said he..." (good) from "He said..." (bad)

**Pros:**
- ✅ Understands context and semantics
- ✅ Generalizes to new patterns
- ✅ No hardcoded lists to maintain
- ✅ Handles nuanced cases correctly

**Cons:**
- ❌ Slower (1 LLM call per claim = ~100-200ms)
- ❌ Costs inference budget (with API) or compute (with local model)

**When to use:**
- When accuracy is critical
- When you have diverse content (thousands of episodes)
- When you can afford the latency

**Example:**
```python
from src.metrics_llm_judge import llm_judge_metric

score = llm_judge_metric(None, pred)
```

---

### Option 2: Few-Shot LLM Judge

**File:** `src/metrics_fewshot_judge.py`

**How it works:**
- Shows the LLM examples from your manual review
- LLM learns YOUR specific quality criteria
- Then judges new claims based on those examples

**Pros:**
- ✅ All benefits of LLM judge
- ✅ Adapts to YOUR domain and criteria
- ✅ More consistent than zero-shot

**Cons:**
- ❌ Same speed/cost as LLM judge
- ❌ Requires good manual review examples

**When to use:**
- When you have clear examples of good/bad (you do!)
- When domain-specific understanding matters (podcast claims)
- When consistency with your manual review is critical

**Example:**
```python
from src.metrics_fewshot_judge import fewshot_judge_metric

score = fewshot_judge_metric(None, pred)
```

---

### Option 3: Hybrid (Fast + Smart) ⚡

**File:** `src/metrics_hybrid.py`

**How it works:**
1. Quick pattern checks catch obvious bad claims (no LLM call)
   - Advertisements, questions, very short claims
   - Pronouns at START of claim
2. Claims with pronouns in middle/end → LLM judge (semantic check)
3. Claims with no issues → mark as good (no LLM call)

**Pros:**
- ✅ 2-3x faster than pure LLM judge
- ✅ Still accurate for most cases
- ✅ Best of both worlds

**Cons:**
- ❌ More complex code
- ❌ Still needs some LLM calls

**When to use:**
- When speed AND accuracy both matter
- When you want to optimize for throughput
- Production use with large-scale evaluation

**Example:**
```python
from src.metrics_hybrid import hybrid_metric

score = hybrid_metric(None, pred)
```

---

### Option 4: Keep Pattern Matching (Not Recommended)

**File:** `src/metrics.py`

Only use if:
- You have very limited LLM budget
- Your claims are extremely simple
- You can enumerate all patterns (you can't for thousands of episodes)

---

## Comparison: Run This

```bash
uv run python compare_metrics.py
```

This will test all 4 approaches on your 34 manual review examples and show:
- Accuracy vs your manual judgments
- Speed (ms per claim)
- Which claims each metric gets wrong

**Expected results:**
- **Pattern matching:** ~85% accuracy, ~1ms per claim
- **LLM judge:** ~92-95% accuracy, ~150ms per claim
- **Few-shot judge:** ~95-98% accuracy, ~150ms per claim
- **Hybrid:** ~93-96% accuracy, ~50ms per claim

---

## Recommendation Matrix

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| **Optimizing prompts with DSPy** | Few-shot LLM Judge | Need accuracy during optimization, speed less critical |
| **Production evaluation** | Hybrid | Balance speed and accuracy for large scale |
| **Initial experimentation** | LLM Judge | Simplest to implement, very accurate |
| **Very tight budget** | Pattern matching (improved) | Fast, but limited accuracy |
| **Your case (thousands of episodes)** | **Few-shot LLM Judge for optimization**, **Hybrid for production** | Best of both worlds |

---

## Implementation Plan

### Phase 1: Test Metrics (Now)

```bash
# Compare all metrics
uv run python compare_metrics.py

# Review results, pick the best one
```

### Phase 2: Update Optimization (Next)

Update `exp_3_1b_optimize_with_positive_only.py` to use the better metric:

```python
# Change this line:
from src.metrics import claim_quality_metric

# To this:
from src.metrics_fewshot_judge import fewshot_judge_metric as claim_quality_metric

# Or for hybrid:
from src.metrics_hybrid import hybrid_metric as claim_quality_metric
```

### Phase 3: Re-run Optimization

```bash
uv run python exp_3_1b_optimize_with_positive_only.py
```

---

## Advanced: Pronoun Detection Done Right

If you want to improve pattern matching without full LLM:

```python
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def has_unresolved_pronoun(claim: str) -> bool:
    """Check if claim has pronouns without referents."""
    doc = nlp(claim)

    for token in doc:
        if token.pos_ == "PRON" and token.dep_ == "nsubj":
            # Found subject pronoun (he, she, they)

            # Look for referent in same sentence
            has_referent = any(
                t.pos_ == "PROPN"  # Proper noun (name)
                for t in doc
                if t.i < token.i  # Appears before pronoun
            )

            if not has_referent:
                return True  # Unresolved pronoun

    return False
```

But honestly, LLM-as-judge is simpler and more robust.

---

## Key Takeaway

**You're processing thousands of episodes with diverse content. Pattern matching can't possibly cover all cases. Use LLM-as-judge or hybrid approach.**

The small extra cost per claim (~100ms) is worth the dramatic improvement in accuracy and maintainability.

---

## Next Steps

1. **Run the comparison:** `uv run python compare_metrics.py`
2. **Pick the best metric** based on accuracy/speed tradeoff
3. **Update optimization script** to use new metric
4. **Re-run optimization** and see if results improve
5. **Document learnings** in your journal

---

## Questions?

**Q: Won't LLM judge be too slow for thousands of episodes?**
A: It's ~100-200ms per claim. If you have 10 claims per episode × 1000 episodes = 10,000 claims = ~20 minutes total. That's acceptable for batch evaluation. Use hybrid for even better speed.

**Q: What if the LLM judge makes mistakes?**
A: It will be more consistent than pattern matching. You can also use few-shot examples to teach it your criteria. And you can cache results for identical claims.

**Q: Should I use this for production real-time?**
A: For real-time, use hybrid (50ms per claim is acceptable). For batch/offline, use LLM judge.

**Q: Can I combine metrics?**
A: Yes! Average their scores or use weighted combination. E.g., `final_score = 0.7 * llm_score + 0.3 * pattern_score`
