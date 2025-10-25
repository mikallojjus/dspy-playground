"""
Experiment 2.4: Measure Baseline

Goal: Establish baseline performance for claim extraction using the metric from Experiment 2.2

This experiment:
1. Loads manual review data as DSPy examples
2. Creates an unoptimized baseline claim extractor
3. Evaluates it using DSPy's Evaluate framework
4. Records the baseline score for comparison with future optimizations

Expected baseline: ~40% low-quality claims (60% quality score)
Target after optimization: <15% low-quality claims (>85% quality score)
"""

import dspy
from dspy.evaluate import Evaluate
import json
import sys
from typing import List
from ...metrics import claim_quality_metric

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# Define the baseline claim extraction signature
class ClaimExtraction(dspy.Signature):
    """Extract factual, verifiable claims from podcast transcript text.

    Claims should be:
    - Factual (not opinions)
    - Self-contained (no pronouns like he/she/they)
    - Specific (include names, numbers, dates)
    - Concise (5-40 words)
    """
    transcript_chunk: str = dspy.InputField(desc="The podcast transcript text to analyze")
    claims: List[str] = dspy.OutputField(desc="List of factual claims extracted from the transcript")


def load_manual_review_as_dspy_examples():
    """Load manual review data and convert to DSPy examples."""

    with open('evaluation/claims_manual_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data['examples']:
        # Create DSPy example with transcript as input and claims as expected output
        example = dspy.Example(
            transcript_chunk=item['transcript_chunk'],
            claims=[item['claim']],  # The claim from manual review
            quality=item['quality']
        ).with_inputs('transcript_chunk')
        examples.append(example)

    return examples


def main():
    print("=" * 80)
    print("Experiment 2.4: Measure Baseline")
    print("=" * 80)
    print("\nGoal: Establish baseline claim extraction quality using DSPy Evaluate")
    print()

    # Load data
    print("Loading manual review data...")
    examples = load_manual_review_as_dspy_examples()
    print(f"✅ Loaded {len(examples)} examples from manual review")

    good_count = sum(1 for ex in examples if ex.quality == 'good')
    bad_count = sum(1 for ex in examples if ex.quality == 'bad')
    print(f"   Good claims: {good_count}")
    print(f"   Bad claims: {bad_count}")
    print()

    # Configure DSPy with Ollama
    print("Configuring DSPy with Ollama (qwen2.5:7b-instruct-q4_0)...")
    lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
    dspy.configure(lm=lm)
    print("✅ DSPy configured")
    print()

    # Create baseline extractor (unoptimized)
    print("Creating baseline extractor (ChainOfThought, unoptimized)...")
    baseline = dspy.ChainOfThought(ClaimExtraction)
    print("✅ Baseline extractor created")
    print()

    # Evaluate baseline
    print("=" * 80)
    print("BASELINE EVALUATION")
    print("=" * 80)
    print("\nEvaluating baseline extractor on all examples...")
    print("(This will take a few minutes as each example requires an LLM call)")
    print()

    evaluator = Evaluate(
        devset=examples,
        metric=claim_quality_metric,
        display_progress=True,
        display_table=0  # Don't display full table
    )

    result = evaluator(baseline)

    # Extract score from EvaluationResult
    # The result contains the average metric score
    score = result.score if hasattr(result, 'score') else float(result)

    # Display results
    print()
    print("=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print()
    print(f"Quality Score: {score:.1f}%")
    print(f"Claims with Issues: {(100 - score):.1f}%")
    print()
    print("Interpretation:")
    print(f"  - {score:.1f}% of extracted claims pass quality checks")
    print(f"  - {(100 - score):.1f}% of claims have quality issues")
    print(f"    (pronouns, vague language, opinions, ads, missing context)")
    print()
    print("Comparison to Goals:")
    print(f"  - Current baseline: {(100 - score):.1f}% low-quality claims")
    print(f"  - Expected baseline: ~40% low-quality claims")
    print(f"  - Target after optimization: <15% low-quality claims")
    print()

    issues_pct = 100 - score
    if issues_pct > 85:
        print("❌ CRITICAL: Baseline is very poor (>85% issues)")
        print("   → Task might be too hard for this model")
        print("   → Consider: Simpler task? Better base model?")
    elif issues_pct < 15:
        print("⚠️  WARNING: Baseline is already very good (<15% issues)")
        print("   → Task might be too easy or metric too lenient")
        print("   → Consider: Harder examples? Stricter metric?")
    elif 40 <= issues_pct <= 85:
        print("✅ EXCELLENT: Baseline is in the sweet spot (40-85% issues)")
        print("   → Perfect candidate for optimization")
        print("   → Continue to Arc 3 (Optimization)")
    else:
        print("✅ GOOD: Baseline shows room for improvement")
        print("   → Continue to Arc 3 (Optimization)")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Record this baseline score in your journal")
    print("2. Proceed to Arc 3: First Optimization Attempt")
    print("3. Use BootstrapFewShot to optimize the prompt")
    print("4. Compare optimized score against this baseline")
    print()

    # Save baseline score for reference
    with open('results/baseline_score.txt', 'w') as f:
        f.write(f"Baseline Quality Score: {score:.1f}%\n")
        f.write(f"Claims with Issues: {(100 - score):.1f}%\n")
        f.write(f"Date: 2025-10-25\n")
        f.write(f"Model: qwen2.5:7b-instruct-q4_0\n")
        f.write(f"Examples: {len(examples)}\n")

    print("✅ Baseline score saved to results/baseline_score.txt")
    print()


if __name__ == "__main__":
    main()
