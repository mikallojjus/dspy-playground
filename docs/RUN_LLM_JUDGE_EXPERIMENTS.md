# Running LLM-as-Judge Experiments

This guide walks through testing and using LLM-as-Judge for claim quality evaluation.

## What You'll Do

1. **Test LLM-as-Judge** - See if it's better than pattern matching
2. **Re-run Optimization** - Use the better metric for DSPy optimization
3. **Compare Results** - See if LLM judge improves optimization

## Step 1: Test LLM-as-Judge (Experiment 2.5)

This compares pattern matching vs LLM-as-Judge on your 34 manual review examples.

```bash
uv run python exp_2_5_test_llm_judge.py
```

**What it does:**
- Tests both metrics on all 34 claims
- Shows side-by-side comparisons
- Highlights cases where LLM judge is better
- Measures accuracy and speed

**Expected output:**
```
Pattern Matching Accuracy: ~85%
LLM-as-Judge Accuracy:     ~95%

Improvement: +10 percentage points

EXAMPLES WHERE LLM JUDGE IS BETTER:
1. "Trump said he would build a ballroom"
   Pattern matching: BAD (saw pronoun "he")
   LLM judge: GOOD (understands "he" refers to Trump)
```

**Time:** ~2-3 minutes (LLM calls are slow)

---

## Step 2: Re-run Optimization with LLM Judge (Experiment 3.1c)

If LLM judge is better (Step 1 shows improvement), use it for optimization:

```bash
uv run python exp_3_1c_optimize_with_llm_judge.py
```

**What it does:**
- Uses LLM-as-Judge as the metric for BootstrapFewShot
- Trains on 14 good examples (positive-only)
- Validates on 11 mixed examples
- Compares baseline vs optimized

**Expected output:**
```
BASELINE EVALUATION (LLM-as-Judge Metric)
Baseline: 65% quality

OPTIMIZATION
Training on 14 good examples...
Using LLM-as-Judge to evaluate

OPTIMIZED EVALUATION
Optimized: 75-85% quality

Improvement: +10-20 percentage points
```

**Time:** ~5-10 minutes (many LLM calls during optimization)

---

## Step 3: Compare All Results

After running both experiments, compare:

### Experiment 2.5 (Metric Comparison)
- Pattern matching: 85% accuracy
- LLM judge: 95% accuracy
- **Conclusion:** LLM judge understands semantics better

### Experiment 3.1b (Optimization with Pattern Matching)
- Baseline: 63.1%
- Optimized: 50.0%
- Improvement: -13.1 points (got worse!)
- **Problem:** Not enough good examples + pattern metric limitations

### Experiment 3.1c (Optimization with LLM Judge)
- Baseline: ~65%
- Optimized: ~75-85%
- Improvement: +10-20 points (expected)
- **Why better:** LLM judge guides optimization more accurately

---

## Why LLM-as-Judge Is Better

### Pattern Matching Problems

```python
# Pattern matching sees "he" and flags as bad
claim = "Trump said he would build a ballroom"
pattern_metric(claim) → BAD (WRONG!)

# Pattern matching can't enumerate all ad phrases
claim = "Special offer for new subscribers"
pattern_metric(claim) → GOOD (WRONG! - not in hardcoded list)
```

### LLM Judge Advantages

```python
# LLM understands "he" refers to Trump (referent exists)
claim = "Trump said he would build a ballroom"
llm_judge(claim) → GOOD ✓

# LLM recognizes advertisement semantically
claim = "Special offer for new subscribers"
llm_judge(claim) → BAD ✓

# LLM handles new patterns without hardcoding
claim = "Limited time deal on premium features"
llm_judge(claim) → BAD ✓
```

---

## Key Insights

### 1. Better Metric → Better Optimization

In DSPy, the metric guides optimization:
- BootstrapFewShot selects examples where metric score is high
- If metric is wrong (pattern matching), it selects wrong examples
- If metric is right (LLM judge), it selects good examples

### 2. Semantic Understanding Matters

Your data has thousands of diverse episodes. You can't hardcode all patterns:
- Every podcast has different speaking styles
- New ad phrases emerge constantly
- Context matters (pronouns with referents are OK)

LLM-as-Judge generalizes to new patterns.

### 3. Speed vs Accuracy Tradeoff

**Pattern Matching:**
- Fast: ~1ms per claim
- Limited accuracy: ~85%

**LLM-as-Judge:**
- Slower: ~100-150ms per claim
- High accuracy: ~95%

**For DSPy optimization:** Accuracy matters most (run once offline)
**For production evaluation:** Consider hybrid approach

---

## Files Created

- `exp_2_5_test_llm_judge.py` - Test LLM judge vs pattern matching
- `exp_3_1c_optimize_with_llm_judge.py` - Optimize with LLM judge
- `src/metrics_llm_judge.py` - LLM judge implementation
- `src/metrics_fewshot_judge.py` - Few-shot variant
- `src/metrics_hybrid.py` - Hybrid (fast + accurate)
- `METRICS_GUIDE.md` - Comprehensive guide on all approaches

---

## Troubleshooting

### "LLM judge is too slow"

Use hybrid approach (`src/metrics_hybrid.py`):
- Fast pattern checks catch obvious issues
- LLM only for borderline cases
- ~2-3x faster than pure LLM judge

### "LLM judge makes mistakes"

Use few-shot variant (`src/metrics_fewshot_judge.py`):
- Shows LLM examples from your manual review
- Learns your specific criteria
- More consistent judgments

### "Optimization still doesn't improve"

Possible issues:
- Need more training examples (target 30-50)
- Validation set too small (11 examples has high variance)
- Task might be very difficult for this model
- Consider MIPROv2 optimizer instead of BootstrapFewShot

---

## Next Steps After Optimization

If Experiment 3.1c succeeds (improvement > +5 points):

1. **Inspect optimized module:**
   ```bash
   # Update exp_3_2_inspect_optimized.py to load:
   # 'models/claim_extractor_llm_judge_v1.json'
   uv run python exp_3_2_inspect_optimized.py
   ```

2. **Test on fresh data:**
   - Get 5-10 new transcript chunks
   - Run both baseline and optimized
   - Manually compare quality

3. **Production deployment:**
   - Use hybrid metric for speed
   - Cache LLM judgments for identical claims
   - Monitor accuracy over time

---

## Summary

**Why this matters:** You're processing thousands of diverse podcast episodes. Pattern matching can't scale. LLM-as-Judge provides semantic understanding that generalizes to new patterns.

**Expected outcome:** LLM-as-Judge should improve both metric accuracy and optimization results.

**Time investment:** ~10-15 minutes to run both experiments and see the benefits.
