# DSPy Experimentation Guide

**Last Updated:** 2025-10-25
**Status:** Active Experimentation
**Approach:** Iterative learning, pause for data when needed

---

## Philosophy

This is an **experimental playground**, not a production project. The goal is to:

- **Learn DSPy** while exploring its capabilities
- **Iterate quickly** with small experiments (30min - 3 hours each)
- **Pause for data** when needed - create just enough examples to answer the current question
- **Build intuition** about what works before scaling up
- **Fail fast** - it's OK if experiments don't work, learning is the goal

**Not aiming for:**
- Production-grade software (yet)
- Perfect evaluation datasets upfront
- Multi-week phases
- Predetermined success metrics

**Complementary doc:** See [BACKLOG.md](BACKLOG.md) for comprehensive reference material, code templates, and metrics.

---

## How to Use This Guide

1. **Work through experiments roughly in order** (but feel free to jump around)
2. **Each experiment is time-boxed** - if it's taking way longer, stop and reassess
3. **Pause points** indicate decision moments - don't blindly continue
4. **Keep a journal** - write notes on what you learn (create `JOURNAL.md`)
5. **Reference BACKLOG.md** when you need detailed code examples or explanations

---

## Current Status

**Experiments Completed:** 0
**Current Arc:** Arc 1 - DSPy Basics
**Next Experiment:** 1.1 - Hello World

---

## Arc 1: DSPy Basics

**Goal:** Can I use DSPy at all? What does it feel like?
**Why this matters:** Validate the toolchain works before doing real work

### Experiment 1.1: Hello World

**Goal:** Get DSPy calling Ollama successfully
**Time:** 30 minutes
**Prerequisites:** Ollama running with qwen2.5:7b-instruct

**Steps:**

1. Install DSPy: `uv add dspy`
2. Create `test_dspy_hello.py`:

```python
import dspy

# Configure DSPy to use Ollama
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Define simple signature
class BasicQA(dspy.Signature):
    """Answer questions concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Use it
qa = dspy.Predict(BasicQA)
result = qa(question="What is 2+2?")
print(f"Answer: {result.answer}")
```

3. Run it: `python test_dspy_hello.py`

**Expected output:** Should print "Answer: 4" (or similar)

**What you'll learn:**
- Does DSPy work with Ollama?
- What does the basic API feel like?
- How fast is inference?

**Pause point:**
- ✅ If it works → Continue to 1.2
- ❌ If it fails → Debug Ollama connection, check model is pulled

**Artifact:** Working `test_dspy_hello.py` script

---

### Experiment 1.2: Claim Extraction Signature

**Goal:** Define a ClaimExtraction signature for our actual task
**Time:** 1 hour
**Prerequisites:** Experiment 1.1 completed

**Steps:**

1. Get one transcript chunk from the database (or use the example from CLAUDE.md)
2. Create `test_claim_extraction.py`:

```python
import dspy
from typing import List

lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

class ClaimExtraction(dspy.Signature):
    """Extract factual, verifiable claims from podcast transcript text.

    Claims should be:
    - Factual (not opinions)
    - Self-contained (no pronouns like he/she/they)
    - Specific (include names, numbers, dates)
    """
    transcript_chunk: str = dspy.InputField()
    claims: List[str] = dspy.OutputField()

# Test with real transcript
transcript = """
2 (16s):
Welcome to Bankless where today we explore a defense of the Ethereum roadmap.
This is Ryan Sean Adams. I'm here with David Hoffman and we are here to help
you become more bankless. The Ethereum roadmap has been called into question
recently, I think Bankless has aired some of these dissents...
"""

extractor = dspy.Predict(ClaimExtraction)
result = extractor(transcript_chunk=transcript)

print("Extracted claims:")
for i, claim in enumerate(result.claims, 1):
    print(f"{i}. {claim}")
```

3. Run it and examine the output

**What you'll learn:**
- Does the signature work for our task?
- What kind of claims does it extract?
- Are they good quality or do they have pronouns/vague language?
- Does the docstring actually influence the output?

**Questions to answer:**
- How many claims does it extract from this chunk?
- Do the claims have pronouns? (he/she/they)
- Are they specific or vague?
- Would you consider them "good" claims?

**Pause point:**
- Write down 2-3 observations about the output quality
- If output is JSON parsing errors → adjust signature
- If output is completely wrong → revisit task definition

**Artifact:** `test_claim_extraction.py` + notes on output quality

---

### Experiment 1.3: Predict vs ChainOfThought

**Goal:** Does ChainOfThought help claim extraction?
**Time:** 30 minutes
**Prerequisites:** Experiment 1.2 completed

**Steps:**

1. Modify `test_claim_extraction.py` to compare:

```python
# Test both approaches
extractor_simple = dspy.Predict(ClaimExtraction)
extractor_cot = dspy.ChainOfThought(ClaimExtraction)

print("=== SIMPLE PREDICT ===")
result1 = extractor_simple(transcript_chunk=transcript)
for claim in result1.claims:
    print(f"  - {claim}")

print("\n=== CHAIN OF THOUGHT ===")
result2 = extractor_cot(transcript_chunk=transcript)
print(f"Reasoning: {result2.rationale}")
for claim in result2.claims:
    print(f"  - {claim}")
```

2. Compare outputs side-by-side

**What you'll learn:**
- Does CoT produce different claims?
- Does the reasoning make sense?
- Is CoT slower? (time both approaches)
- Is CoT worth the extra latency?

**Pause point:**
- If CoT is clearly better → use it going forward
- If no difference → use Predict (simpler and faster)
- Save your conclusion for later experiments

**Artifact:** Notes on Predict vs CoT comparison

---

## Arc 2: Understanding Claim Quality

**Goal:** What makes a claim good vs bad? Can I measure it?
**Why this matters:** Need to define quality before we can optimize for it

### Experiment 2.1: Generate and Review Claims

**Goal:** Build intuition for claim quality
**Time:** 1-2 hours
**Prerequisites:** Arc 1 completed

**Steps:**

1. Select 5 diverse transcript chunks from your database:
   - Different podcasts if possible
   - Different topics (crypto, tech, general)
   - Mix of technical and non-technical content

```python
# Example: Get random chunks from database
import psycopg2

conn = psycopg2.connect("your_connection_string")
cursor = conn.cursor()

cursor.execute("""
    SELECT id, name, SUBSTRING(transcript, 1, 1000) as chunk
    FROM podcast_episodes
    WHERE transcript IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 5
""")

chunks = cursor.fetchall()
```

2. Run claim extraction on each chunk
3. Create `evaluation/claims_manual_review.json` and label each claim:

```json
{
  "examples": [
    {
      "episode_id": 123,
      "transcript_chunk": "...",
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quality": "good",
      "issues": [],
      "notes": "Specific, factual, self-contained"
    },
    {
      "episode_id": 123,
      "transcript_chunk": "...",
      "claim": "He said it was amazing",
      "quality": "bad",
      "issues": ["pronoun", "vague", "missing_context"],
      "notes": "Who is 'he'? What is 'it'? 'Amazing' is opinion"
    }
  ]
}
```

4. For each claim, mark:
   - `quality`: "good", "bad", or "edge"
   - `issues`: List of problems (pronoun, vague, opinion, missing_context)
   - `notes`: Why you made this judgment

**What you'll learn:**
- What patterns emerge in bad claims?
- Is "good vs bad" clear-cut or fuzzy?
- What percentage are bad? (rough estimate)

**Questions to answer in your journal:**
- What's the most common quality issue? (pronouns? vagueness? opinions?)
- Are there edge cases where you're unsure?
- Do you need clearer guidelines?

**Pause point:**
- If patterns are clear (e.g., "pronouns are the main issue") → Continue to 2.2
- If everything is fuzzy and unclear → Get 5 more examples or revisit task definition
- If almost all claims are good (>90%) → Task might be too easy or you're being too lenient

**Artifact:** `evaluation/claims_manual_review.json` with 15-25 labeled claims

---

### Experiment 2.2: Write a Simple Metric

**Goal:** Turn quality patterns into code
**Time:** 30 minutes
**Prerequisites:** Experiment 2.1 completed

**Steps:**

1. Based on patterns from 2.1, write a metric
2. Create `src/metrics.py`:

```python
def claim_quality_metric(example, pred, trace=None):
    """
    Evaluate claim quality based on common issues found in manual review.

    Returns 1.0 if no issues, 0.0 if claim has issues.

    Customize this based on what you found in Experiment 2.1!
    """
    predicted_claims = pred.claims

    if not predicted_claims:
        return 0.0

    # TODO: Customize these based on YOUR patterns from 2.1
    PRONOUNS = ['he', 'she', 'they', 'it', 'him', 'her', 'them']
    VAGUE_WORDS = ['very', 'really', 'something', 'someone']
    OPINION_WORDS = ['think', 'believe', 'feel', 'seems', 'amazing', 'terrible']

    issues_found = 0
    for claim in predicted_claims:
        claim_lower = claim.lower()
        words = claim_lower.split()

        # Check for issues you care about
        if any(pronoun in words for pronoun in PRONOUNS):
            issues_found += 1
            continue

        if any(vague in words for vague in VAGUE_WORDS):
            issues_found += 1
            continue

        if any(opinion in claim_lower for opinion in OPINION_WORDS):
            issues_found += 1
            continue

    quality_score = 1.0 - (issues_found / len(predicted_claims))
    return quality_score
```

**What you'll learn:**
- Can I capture quality rules programmatically?
- What's hard to detect automatically?

**Pause point:**
- Metric is ready for testing → Continue to 2.3

**Artifact:** `src/metrics.py`

---

### Experiment 2.3: Test the Metric

**Goal:** Does the metric match my judgment?
**Time:** 30 minutes
**Prerequisites:** Experiments 2.1 and 2.2 completed

**Steps:**

1. Create `test_metric.py`:

```python
import dspy
import json
from src.metrics import claim_quality_metric
from typing import List

# Load your manual review
with open('evaluation/claims_manual_review.json') as f:
    data = json.load(f)

# Test metric on each example
print("Testing metric on manual review examples:\n")

for example in data['examples']:
    # Create a mock prediction
    pred = type('obj', (object,), {
        'claims': [example['claim']]
    })

    score = claim_quality_metric(None, pred)

    # Compare to your manual judgment
    manual_quality = example['quality']

    match = "✅" if (score == 1.0 and manual_quality == "good") or \
                    (score < 1.0 and manual_quality == "bad") else "❌"

    print(f"{match} Claim: {example['claim'][:60]}...")
    print(f"   Metric score: {score:.2f} | Manual: {manual_quality}")
    print(f"   Issues: {example['issues']}\n")

# Calculate agreement
# TODO: Add calculation of how often metric agrees with manual labels
```

2. Run it and review results

**What you'll learn:**
- Does metric correlate with my judgment?
- What's the agreement rate?
- What does the metric miss?

**Questions to answer:**
- What percentage of examples does the metric agree with you on?
- Are there false positives? (metric says bad, but you said good)
- Are there false negatives? (metric says good, but you said bad)

**Pause point:**
- If agreement is >80% → Metric is good enough, continue
- If agreement is 60-80% → Refine metric, add more patterns
- If agreement is <60% → Rethink metric completely

**Artifact:** Test results showing metric vs manual agreement

---

### Experiment 2.4: Measure Baseline

**Goal:** How bad is current claim extraction?
**Time:** 30 minutes
**Prerequisites:** Experiments 2.1-2.3 completed

**Steps:**

1. Create `measure_baseline.py`:

```python
import dspy
from dspy.evaluate import Evaluate
from src.metrics import claim_quality_metric

# Load your manual review as DSPy examples
import json
with open('evaluation/claims_manual_review.json') as f:
    data = json.load(f)

examples = []
for item in data['examples']:
    example = dspy.Example(
        transcript_chunk=item['transcript_chunk'],
        claims=[item['claim']]
    ).with_inputs('transcript_chunk')
    examples.append(example)

# Configure DSPy
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Create baseline extractor (unoptimized)
from test_claim_extraction import ClaimExtraction  # Use your signature
baseline = dspy.ChainOfThought(ClaimExtraction)

# Evaluate
evaluator = Evaluate(
    devset=examples,
    metric=claim_quality_metric,
    display_progress=True
)

score = evaluator(baseline)

print(f"\n{'='*60}")
print(f"BASELINE QUALITY SCORE: {score:.1f}%")
print(f"{'='*60}")
print(f"This means {100-score:.1f}% of claims have quality issues")
```

2. Run it and note the score

**What you'll learn:**
- What's the baseline quality score?
- Is optimization worth attempting?

**Pause point - CRITICAL DECISION:**

- **If baseline is >85%:** Current prompts are already good! Maybe optimization isn't needed?
  - Consider: Is the task too easy? Is the metric too lenient?

- **If baseline is 60-85%:** Good candidate for optimization → Continue to Arc 3
  - This is the sweet spot

- **If baseline is <60%:** Task might be too hard for this model
  - Consider: Simpler task? Better base model? Rethink approach?

**Write in your journal:**
- Baseline score
- Your interpretation
- Decision: Continue to Arc 3 or pivot?

**Artifact:** `results/baseline_score.txt` with the score

---

## Arc 3: First Optimization Attempt

**Goal:** Can DSPy improve the prompt? By how much?
**Why this matters:** Validate that optimization actually works before investing more time

**Prerequisites:** Arc 2 completed with baseline score 60-85%

### Experiment 3.1: BootstrapFewShot Optimization

**Goal:** Try the simplest DSPy optimizer
**Time:** 1-2 hours (includes compute time)
**Prerequisites:** Baseline measured

**Steps:**

1. Split your data (from Experiment 2.1):

```python
import random
import json

with open('evaluation/claims_manual_review.json') as f:
    data = json.load(f)

# Shuffle
examples = data['examples']
random.seed(42)
random.shuffle(examples)

# Split: 70% train, 30% val (adjust based on your data size)
split_point = int(len(examples) * 0.7)
train = examples[:split_point]
val = examples[split_point:]

print(f"Train: {len(train)} examples")
print(f"Val: {len(val)} examples")

# Save splits
with open('evaluation/claims_train.json', 'w') as f:
    json.dump(train, f, indent=2)
with open('evaluation/claims_val.json', 'w') as f:
    json.dump(val, f, indent=2)
```

2. Create `optimize_bootstrap.py`:

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json

# Configure
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Load data
from src.data_loader import load_claim_dataset  # You'll need to create this
trainset = load_claim_dataset('evaluation/claims_train.json')
valset = load_claim_dataset('evaluation/claims_val.json')

# Import your signature and metric
from test_claim_extraction import ClaimExtraction
from src.metrics import claim_quality_metric

# Create baseline
baseline = dspy.ChainOfThought(ClaimExtraction)

# Optimize with BootstrapFewShot (simplest optimizer)
optimizer = BootstrapFewShot(
    metric=claim_quality_metric,
    max_bootstrapped_demos=3  # Use 3 few-shot examples
)

print("Optimizing with BootstrapFewShot...")
print(f"Training on {len(trainset)} examples")

optimized = optimizer.compile(
    student=baseline,
    trainset=trainset
)

# Evaluate both
evaluator = Evaluate(devset=valset, metric=claim_quality_metric)

baseline_score = evaluator(baseline)
optimized_score = evaluator(optimized)

print(f"\n{'='*60}")
print(f"BASELINE:   {baseline_score:.1f}%")
print(f"OPTIMIZED:  {optimized_score:.1f}%")
print(f"IMPROVEMENT: {optimized_score - baseline_score:+.1f}%")
print(f"{'='*60}")

# Save the optimized module
optimized.save('models/claim_extractor_bootstrap_v1.json')
print("\nSaved optimized module to models/claim_extractor_bootstrap_v1.json")
```

3. Run it and wait for results (may take 30-60 minutes)

**What you'll learn:**
- Did optimization improve the score?
- By how much?
- How long did it take?

**Pause point - CRITICAL DECISION:**

- **If improvement >10%:** Significant! → Continue to 3.2 to understand why
- **If improvement 5-10%:** Modest but positive → Continue to 3.2
- **If improvement <5%:** Barely any change → Debug
  - Is the metric too simple?
  - Is there not enough training data?
  - Is the task already maxed out?
- **If score got worse:** Something's wrong → Debug
  - Check metric is working correctly
  - Check data splits are reasonable

**Write in your journal:**
- Baseline vs optimized scores
- Improvement delta
- Time it took
- Your interpretation

**Artifact:** `models/claim_extractor_bootstrap_v1.json` + results

**Reference:** See BACKLOG.md lines 1921-1938 for BootstrapFewShot details

---

### Experiment 3.2: Inspect What Changed

**Goal:** Understand what DSPy actually did
**Time:** 30 minutes
**Prerequisites:** Experiment 3.1 completed with >5% improvement

**Steps:**

1. Create `inspect_optimized.py`:

```python
import dspy
from test_claim_extraction import ClaimExtraction

# Load optimized module
optimized = dspy.ChainOfThought(ClaimExtraction)
optimized.load('models/claim_extractor_bootstrap_v1.json')

# Inspect the module
print("="*60)
print("OPTIMIZED MODULE INSPECTION")
print("="*60)

# Show the few-shot examples it selected
if hasattr(optimized, 'demos') and optimized.demos:
    print(f"\nDSPy selected {len(optimized.demos)} few-shot examples:")
    for i, demo in enumerate(optimized.demos, 1):
        print(f"\n--- Example {i} ---")
        print(f"Input (first 100 chars): {demo.transcript_chunk[:100]}...")
        print(f"Output claims:")
        for claim in demo.claims:
            print(f"  - {claim}")
else:
    print("\nNo few-shot examples found (using zero-shot)")

# Try it on a new example
print("\n" + "="*60)
print("TESTING ON NEW EXAMPLE")
print("="*60)

test_transcript = """
Your test transcript here...
"""

result = optimized(transcript_chunk=test_transcript)
print("\nExtracted claims:")
for claim in result.claims:
    print(f"  - {claim}")
```

2. Run it and examine the output

**What you'll learn:**
- Which examples did DSPy select as few-shot demonstrations?
- Why those examples? (look for patterns)
- Do the selected examples represent "good" claims?

**Questions to answer:**
- What do the selected few-shot examples have in common?
- Are they diverse or similar?
- Do they demonstrate the quality patterns you care about?

**Artifact:** Notes on what DSPy changed

---

### Experiment 3.3: Test on Fresh Examples

**Goal:** Does improvement generalize to unseen data?
**Time:** 1 hour
**Prerequisites:** Experiment 3.1 completed

**Steps:**

1. Get 5 NEW transcript chunks (not in your training/val set)
2. Run baseline and optimized on them
3. Manually compare the quality

```python
import dspy
from test_claim_extraction import ClaimExtraction

# Configure
lm = dspy.LM('ollama/qwen2.5:7b-instruct', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# Load both modules
baseline = dspy.ChainOfThought(ClaimExtraction)
optimized = dspy.ChainOfThought(ClaimExtraction)
optimized.load('models/claim_extractor_bootstrap_v1.json')

# Test on fresh examples
fresh_chunks = [
    # Get 5 new chunks from database
]

print("Testing on fresh unseen examples:\n")

for i, chunk in enumerate(fresh_chunks, 1):
    print(f"{'='*60}")
    print(f"CHUNK {i}")
    print(f"{'='*60}")

    print("\n--- BASELINE ---")
    result_baseline = baseline(transcript_chunk=chunk)
    for claim in result_baseline.claims:
        print(f"  - {claim}")

    print("\n--- OPTIMIZED ---")
    result_optimized = optimized(transcript_chunk=chunk)
    for claim in result_optimized.claims:
        print(f"  - {claim}")

    print("\n")

# Manually review: which is better?
```

**What you'll learn:**
- Does improvement hold on completely fresh data?
- Are the differences noticeable?
- Is optimized consistently better or hit-or-miss?

**Pause point - DECISION:**

- **If optimized is clearly better on fresh data:** Success! You can trust this optimization
  - Decision: Move to Arc 4 to expand? Or try entailment next (Arc 5)?

- **If optimized is only slightly better or inconsistent:** The improvement might not be robust
  - Decision: More training data? Better metric? Different approach?

- **If optimized is worse on fresh data:** Overfitting!
  - The model memorized the validation set
  - Decision: Simpler optimizer? More diverse data?

**Write in your journal:**
- Subjective comparison of baseline vs optimized
- Is the improvement real and generalizable?
- Next steps

**Artifact:** Fresh evaluation results + decision on next steps

---

## Arc 4: Expand and Iterate (Optional)

**Goal:** Make optimization more robust
**When to do this:** Only if Arc 3 showed clear improvement and you want to scale up

**Prerequisites:**
- Arc 3 completed successfully
- Clear evidence of >10% improvement
- Decided this is worth investing more time in

### Experiment 4.1: Create More Diverse Examples

**Goal:** Expand dataset with edge cases and failure modes
**Time:** 2-3 hours
**Prerequisites:** Arc 3 completed

**Focus areas:**
- Edge cases you discovered in Arc 3
- Failure modes from fresh testing (Experiment 3.3)
- Diverse podcast styles, topics, speaking patterns

**Target:** 30-50 total labeled examples (you have ~15-20 from Arc 2)

**Steps:**

1. Review failures and edge cases from previous experiments
2. Deliberately sample chunks that exhibit those patterns
3. Label them using the same process as Experiment 2.1
4. Update `evaluation/claims_manual_review.json`

**What you'll learn:**
- What edge cases exist?
- How diverse can the task be?

**Pause point:**
- When you have 30-50 examples → Continue to 4.2

**Artifact:** Expanded dataset with 30-50 examples

---

### Experiment 4.2: Re-optimize with More Data

**Goal:** Does more data lead to better optimization?
**Time:** 1-2 hours
**Prerequisites:** Experiment 4.1 completed

**Steps:**

1. Re-split data (70/30 with larger dataset)
2. Run BootstrapFewShot again
3. Compare results to previous optimization

**What you'll learn:**
- Does more data help?
- Is there a point of diminishing returns?

**Artifact:** New optimized module + comparison results

---

### Experiment 4.3: Try MIPROv2 (Advanced)

**Goal:** Can a more sophisticated optimizer do better?
**Time:** 4-8 hours (compute time)
**Prerequisites:**
- Experiment 4.2 completed
- You have 30+ examples
- BootstrapFewShot has plateaued

**Why MIPROv2:**
- Optimizes both instructions AND few-shot examples
- More sophisticated than BootstrapFewShot
- But slower and more complex

**Steps:**

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=claim_quality_metric,
    num_candidates=10,
    verbose=True
)

optimized_mipro = optimizer.compile(
    student=baseline,
    trainset=trainset,
    valset=valset,
    num_trials=20,  # Start with 20, can increase later
    max_bootstrapped_demos=3
)

# Compare BootstrapFewShot vs MIPROv2
```

**What you'll learn:**
- Is MIPROv2 significantly better?
- Is the extra compute time worth it?
- What did it optimize differently?

**Pause point:**
- If MIPROv2 is clearly better → Use it going forward
- If MIPROv2 is only marginally better → Stick with BootstrapFewShot (faster iteration)

**Artifact:** MIPROv2 optimized module + comparison

**Reference:** See BACKLOG.md lines 2118-2125 for optimizer comparison

---

## Arc 5: Entailment Task (Second Problem)

**Goal:** Apply learnings to the entailment validation task
**When to do this:** After you're comfortable with claim extraction (Arc 3 or 4 complete)

**Note:** This follows a similar progression to Arcs 1-3 but for entailment

### Experiment 5.1: Entailment Signature

**Goal:** Define EntailmentValidation signature
**Time:** 1 hour

**Steps:**

```python
import dspy

class EntailmentValidation(dspy.Signature):
    """Determine whether a quote provides evidence for a claim.

    SUPPORTS: Quote directly asserts the claim or provides clear evidence
    RELATED: Quote is topically related but doesn't validate the claim
    NEUTRAL: Quote is unrelated
    CONTRADICTS: Quote undermines the claim

    Be strict: only use SUPPORTS when quote truly validates the claim.
    """
    claim: str = dspy.InputField()
    quote: str = dspy.InputField()
    relationship: str = dspy.OutputField(desc="SUPPORTS, RELATED, NEUTRAL, or CONTRADICTS")
    reasoning: str = dspy.OutputField(desc="Brief explanation")
```

**What you'll learn:**
- Does the signature work for entailment?
- What do outputs look like?

**Artifact:** Working entailment signature

---

### Experiment 5.2: Label Entailment Examples

**Goal:** Create evaluation dataset for entailment
**Time:** 2-3 hours

**Steps:**

1. For 10 claims from your claim dataset
2. Get top 5 quotes from semantic search (or manually select quotes)
3. For each claim-quote pair, label the relationship

```json
{
  "examples": [
    {
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quote": "Bitcoin hit its all-time high of $69,000 in November",
      "relationship": "SUPPORTS",
      "reasoning": "Quote directly states the claim"
    },
    {
      "claim": "Bitcoin reached $69,000 in November 2021",
      "quote": "Cryptocurrency markets were volatile in 2021",
      "relationship": "RELATED",
      "reasoning": "Topically related but doesn't mention specific price"
    }
  ]
}
```

**Focus on:**
- SUPPORTS vs RELATED boundary (this is the hard part!)
- Include clear SUPPORTS examples
- Include borderline cases that might be mislabeled

**Target:** 20-30 labeled claim-quote pairs to start

**Artifact:** `evaluation/entailment_manual_review.json`

---

### Experiment 5.3: Entailment Metric

**Goal:** Write a metric for entailment accuracy
**Time:** 30 minutes

**Steps:**

```python
def entailment_accuracy_metric(example, pred, trace=None):
    """
    Measure entailment accuracy with extra penalty for false positives.

    We especially want to avoid RELATED being misclassified as SUPPORTS.
    """
    gold = example.relationship.upper()
    predicted = pred.relationship.upper()

    # Exact match
    if gold == predicted:
        return 1.0

    # Extra penalty for false positive (RELATED → SUPPORTS)
    if gold == "RELATED" and predicted == "SUPPORTS":
        return -2.0  # Strong penalty to guide optimization

    return 0.0
```

**Artifact:** Entailment metric in `src/metrics.py`

---

### Experiment 5.4: Baseline + Optimize Entailment

**Goal:** Measure baseline and optimize (similar to Arc 2-3 for claims)
**Time:** 2-3 hours

**Steps:**

1. Measure baseline entailment accuracy
2. Run BootstrapFewShot optimization
3. Compare results

**What you'll learn:**
- Is entailment harder or easier than claim extraction?
- Does optimization help?
- What's the false positive rate?

**Artifact:** Optimized entailment module

**Reference:** See BACKLOG.md lines 1386-1455 for entailment optimization code

---

## Arc 6: Integration Exploration (Way Later)

**Goal:** How would we use these optimized modules in production?
**When to do this:** Only after Arcs 3-5 are done and optimization clearly works

**Note:** This is deferred because you're in experimentation mode, not production mode

### Experiment 6.1: Extract Optimized Prompt as String

**Goal:** Can we get the optimized prompt out of DSPy for use elsewhere?
**Time:** 1 hour

**Why this matters:**
- Maybe you don't want a Python service in production
- Could extract the optimized prompt and hardcode it in TypeScript
- Simpler deployment, lower latency

**Steps:**

```python
# Inspect what the optimized prompt actually is
optimized = dspy.ChainOfThought(ClaimExtraction)
optimized.load('models/claim_extractor_bootstrap_v1.json')

# Get the prompt
state = optimized.dump_state()
print(state)

# Extract the few-shot examples
for demo in optimized.demos:
    print(f"Example input: {demo.transcript_chunk}")
    print(f"Example output: {demo.claims}")
```

**What you'll learn:**
- Can you manually replicate what DSPy does?
- Is it simple enough to hardcode?

**Decision:**
- If simple → Extract and use in TypeScript
- If complex → Need Python service or full rewrite

**Reference:** See BACKLOG.md lines 2168-2183 for integration decision matrix

---

## Reference Material

### When to Consult BACKLOG.md

- **Detailed code examples:** BACKLOG.md has full working scripts
- **Metrics patterns:** Lines 1941-2110 show simple → advanced → LLM-judge progression
- **DSPy concepts explained:** Lines 182-272 explain signatures, modules, optimizers
- **Anti-patterns:** Lines 2186-2235 list common mistakes
- **Optimizer comparison:** Lines 2118-2125 compare BootstrapFewShot vs MIPROv2

### Journal Template

Create `JOURNAL.md` and track your learnings:

```markdown
# DSPy Experimentation Journal

## 2025-10-25

### Experiment 1.1: Hello World
- Status: ✅ Completed
- Time: 20 minutes
- Learnings:
  - DSPy works with Ollama
  - Inference is ~2 seconds per call
  - API is intuitive
- Next: Move to 1.2

### Experiment 2.1: Generate and Review Claims
- Status: ✅ Completed
- Time: 1.5 hours
- Learnings:
  - Pronouns are the #1 issue (60% of bad claims)
  - Vagueness is second (30%)
  - Opinions are rare (10%)
- Decisions:
  - Focus metric on pronouns and vagueness
  - Less worried about opinions for now
- Next: Write metric (2.2)
```

### Key DSPy Commands Reference

```bash
# Install
uv add dspy

# Basic usage
python test_dspy_hello.py

# Run optimization
python optimize_bootstrap.py

# Inspect optimized module
python inspect_optimized.py
```

---

## Decision Trees

### After Baseline Measurement (Experiment 2.4)

```
Baseline score?
│
├─ >85% → Is task too easy? Metric too lenient?
│          → Review metric, try harder examples
│
├─ 60-85% → Perfect! Continue to Arc 3 (optimization)
│
└─ <60% → Task too hard for model?
           → Simplify task or use better base model
```

### After First Optimization (Experiment 3.1)

```
Improvement?
│
├─ >10% → Significant! Continue to 3.2, inspect what changed
│
├─ 5-10% → Modest. Continue but watch for overfitting in 3.3
│
├─ <5% → Barely any change
│         → Check: metric too simple? need more data?
│
└─ Negative → Something wrong
              → Debug: metric, data splits, task definition
```

### After Fresh Testing (Experiment 3.3)

```
Fresh data performance?
│
├─ Consistently better → Success! Move to Arc 4 or Arc 5
│
├─ Slightly better → Improvement might not be robust
│                    → More data? Better metric?
│
└─ Worse → Overfitting to validation set
           → Simpler optimizer? More diverse data?
```

---

## FAQ

**Q: Can I skip experiments?**
A: Yes! But Arc 1 and Arc 2 are foundational. Arc 3+ can be flexible.

**Q: What if an experiment fails?**
A: That's fine! Write notes in your journal, pivot to something else.

**Q: How many examples do I really need?**
A: Start with 5-10, then 15-20, then 30-50. Don't create 100 upfront.

**Q: Should I optimize claim extraction or entailment first?**
A: Claim extraction (Arc 2-3) is simpler. Get familiar with DSPy there, then tackle entailment (Arc 5).

**Q: When do I know I'm "done" with experimentation?**
A: When you can answer: "Does DSPy improve prompts meaningfully?" If yes and you want to use it → Arc 6. If no → that's OK, you learned something.

**Q: What if Ollama is too slow?**
A: Try reducing `num_trials` in optimizers or use a smaller model. Or use OpenAI API if you have credits.

---

## Next Steps

**Start here:** Experiment 1.1 (Hello World)

**Expected timeline:**
- Arc 1: 2-3 hours
- Arc 2: 3-5 hours
- Arc 3: 3-5 hours
- **Total to first meaningful result:** ~10-15 hours of focused work

**Remember:** This is exploration. Take notes, learn, iterate. Production comes later (if at all).

---

**Good luck! 🚀**
