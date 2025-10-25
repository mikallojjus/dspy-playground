"""
Experiment 3.1b: BootstrapFewShot with Positive Examples Only

Goal: Optimize using ONLY good claim examples

Key difference from 3.1:
- Train on 20 good examples (positive-only)
- Validate on mixed good/bad examples
- This teaches "extract claims like THESE" rather than confusing with bad examples

Expected improvement: Should now see positive gains
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
from typing import List
from ...metrics import claim_quality_metric
import time
import random

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# Define the claim extraction signature
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


def load_dataset(filepath):
    """Load dataset and convert to DSPy examples."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data['examples']:
        example = dspy.Example(
            transcript_chunk=item['transcript_chunk'],
            claims=[item['claim']],
            quality=item['quality']
        ).with_inputs('transcript_chunk')
        examples.append(example)

    return examples


def main():
    print("=" * 80)
    print("Experiment 3.1b: BootstrapFewShot with Positive Examples Only")
    print("=" * 80)
    print("\nKey Change: Training on GOOD examples only")
    print()

    # Load data
    print("Loading datasets...")

    # Load positive-only for training
    positive_trainset = load_dataset('evaluation/claims_positive_only.json')

    # Shuffle and split positive examples into train/val
    random.seed(42)
    random.shuffle(positive_trainset)

    split_point = int(len(positive_trainset) * 0.7)
    trainset = positive_trainset[:split_point]  # 70% for training
    positive_valset = positive_trainset[split_point:]  # 30% held out

    # Also load the full validation set (mixed good/bad) for final evaluation
    full_valset = load_dataset('evaluation/claims_val.json')

    print(f"Positive training set: {len(trainset)} good examples")
    print(f"Positive validation set: {len(positive_valset)} good examples")
    print(f"Full validation set: {len(full_valset)} mixed examples")

    # Count good/bad in full valset
    good_val = sum(1 for ex in full_valset if ex.quality == 'good')
    bad_val = sum(1 for ex in full_valset if ex.quality == 'bad')
    print(f"  (Full val: {good_val} good, {bad_val} bad)")
    print()

    # Configure DSPy
    print("Configuring DSPy with Ollama...")
    lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
    dspy.configure(lm=lm)
    print("Done")
    print()

    # Create baseline
    print("Creating baseline extractor...")
    baseline = dspy.ChainOfThought(ClaimExtraction)
    print("Done")
    print()

    # Evaluate baseline on full validation set
    print("=" * 80)
    print("BASELINE EVALUATION (Full Validation Set)")
    print("=" * 80)

    evaluator = Evaluate(
        devset=full_valset,
        metric=claim_quality_metric,
        display_progress=True,
        display_table=0
    )

    baseline_result = evaluator(baseline)
    baseline_score = baseline_result.score if hasattr(baseline_result, 'score') else float(baseline_result)

    print(f"\nBaseline Quality Score: {baseline_score:.1f}%")
    print(f"Claims with Issues: {(100 - baseline_score):.1f}%")
    print()

    # Optimize with BootstrapFewShot using positive examples only
    print("=" * 80)
    print("OPTIMIZATION (Positive Examples Only)")
    print("=" * 80)
    print("\nOptimizing with BootstrapFewShot...")
    print(f"Training on {len(trainset)} GOOD examples only")
    print()
    print("This approach:")
    print("  1. Shows the model only high-quality claim examples")
    print("  2. Teaches 'extract claims like THESE'")
    print("  3. Avoids confusing the model with bad examples")
    print()

    optimizer = BootstrapFewShot(
        metric=claim_quality_metric,
        max_bootstrapped_demos=4  # Use 4 few-shot examples (we have more data now)
    )

    start_time = time.time()

    optimized = optimizer.compile(
        student=baseline,
        trainset=trainset
    )

    optimization_time = time.time() - start_time

    print(f"\nOptimization completed in {optimization_time:.1f} seconds")
    print()

    # Evaluate optimized on full validation set
    print("=" * 80)
    print("OPTIMIZED EVALUATION (Full Validation Set)")
    print("=" * 80)

    optimized_result = evaluator(optimized)
    optimized_score = optimized_result.score if hasattr(optimized_result, 'score') else float(optimized_result)

    print(f"\nOptimized Quality Score: {optimized_score:.1f}%")
    print(f"Claims with Issues: {(100 - optimized_score):.1f}%")
    print()

    # Compare results
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"Dataset: {len(full_valset)} mixed examples ({good_val} good, {bad_val} bad)")
    print()
    print(f"Baseline:   {baseline_score:.1f}% quality ({(100 - baseline_score):.1f}% issues)")
    print(f"Optimized:  {optimized_score:.1f}% quality ({(100 - optimized_score):.1f}% issues)")
    print()

    improvement = optimized_score - baseline_score
    print(f"Improvement: {improvement:+.1f} percentage points")
    print()

    # Interpretation
    optimized_issues = 100 - optimized_score
    target_met = optimized_issues < 15

    if improvement > 10:
        print("SUCCESS: SIGNIFICANT IMPROVEMENT (>10 percentage points)")
        print("   The optimization clearly helped!")
    elif improvement >= 5:
        print("SUCCESS: MODEST IMPROVEMENT (5-10 percentage points)")
        print("   Optimization helped")
    elif improvement >= 0:
        print("MINIMAL IMPROVEMENT (<5 percentage points)")
        print("   Optimization barely helped")
    else:
        print("NEGATIVE RESULT - Score got worse")
        print("   This is unexpected with positive-only training")

    print()
    if target_met:
        print("TARGET ACHIEVED: <15% low-quality claims")
    else:
        print(f"Target not yet met: {optimized_issues:.1f}% issues (need <15%)")

    print()
    print(f"Training approach: Positive examples only")
    print(f"Training set size: {len(trainset)} good examples")
    print(f"Time: {optimization_time:.1f} seconds")
    print()

    # Save optimized module
    print("=" * 80)
    print("SAVING OPTIMIZED MODULE")
    print("=" * 80)
    print()

    import os
    os.makedirs('models', exist_ok=True)

    model_path = 'models/claim_extractor_positive_v1.json'
    optimized.save(model_path)

    print(f"Optimized module saved to {model_path}")
    print()

    # Save results summary
    results = {
        "experiment": "3.1b BootstrapFewShot with Positive Examples",
        "date": "2025-10-25",
        "model": "qwen2.5:7b-instruct-q4_0",
        "train_size": len(trainset),
        "train_composition": "positive_only",
        "val_size": len(full_valset),
        "val_composition": f"{good_val} good, {bad_val} bad",
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": improvement,
        "target_met": target_met,
        "optimization_time_seconds": optimization_time,
        "max_bootstrapped_demos": 4
    }

    with open('results/experiment_3_1b_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to results/experiment_3_1b_results.json")
    print()

    # Next steps
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()

    if improvement > 0:
        print("1. Experiment 3.2: Inspect what DSPy changed")
        print("   Modify exp_3_2_inspect_optimized.py to load:")
        print("   'models/claim_extractor_positive_v1.json'")
        print()
        print("2. Compare actual claim outputs side-by-side")
        print()
        print("3. If satisfied, consider this optimization successful")
        print("   and move to testing on fresh data")
    else:
        print("Optimization didn't improve results.")
        print("Consider:")
        print("  - More diverse training examples")
        print("  - Different optimizer (MIPROv2)")
        print("  - Manual prompt engineering instead")


if __name__ == "__main__":
    main()
