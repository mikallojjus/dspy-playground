"""
Experiment 3.1: BootstrapFewShot Optimization

Goal: Use DSPy's simplest optimizer (BootstrapFewShot) to improve claim extraction

This experiment:
1. Loads train/val splits
2. Creates baseline (unoptimized) extractor
3. Optimizes using BootstrapFewShot
4. Evaluates both on validation set
5. Saves the optimized module

Expected improvement: Reduce low-quality claims from ~34% to <15%
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
from typing import List
from ...metrics import claim_quality_metric
import time

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# Define the claim extraction signature (same as baseline)
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
    print("Experiment 3.1: BootstrapFewShot Optimization")
    print("=" * 80)
    print("\nGoal: Optimize claim extraction using DSPy's BootstrapFewShot")
    print()

    # Load data
    print("Loading train/val datasets...")
    trainset = load_dataset('evaluation/claims_train.json')
    valset = load_dataset('evaluation/claims_val.json')

    print(f"Training set: {len(trainset)} examples")
    print(f"Validation set: {len(valset)} examples")
    print()

    # Configure DSPy
    print("Configuring DSPy with Ollama (qwen2.5:7b-instruct-q4_0)...")
    lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
    dspy.configure(lm=lm)
    print("Done")
    print()

    # Create baseline
    print("Creating baseline extractor...")
    baseline = dspy.ChainOfThought(ClaimExtraction)
    print("Done")
    print()

    # Evaluate baseline on validation set
    print("=" * 80)
    print("BASELINE EVALUATION")
    print("=" * 80)
    print("\nEvaluating baseline on validation set...")

    evaluator = Evaluate(
        devset=valset,
        metric=claim_quality_metric,
        display_progress=True,
        display_table=0
    )

    baseline_result = evaluator(baseline)
    baseline_score = baseline_result.score if hasattr(baseline_result, 'score') else float(baseline_result)

    print(f"\nBaseline Quality Score: {baseline_score:.1f}%")
    print(f"Claims with Issues: {(100 - baseline_score):.1f}%")
    print()

    # Optimize with BootstrapFewShot
    print("=" * 80)
    print("OPTIMIZATION")
    print("=" * 80)
    print("\nOptimizing with BootstrapFewShot...")
    print("This will:")
    print("  1. Generate successful examples from training set")
    print("  2. Select best few-shot demonstrations")
    print("  3. Create optimized prompt with those examples")
    print()
    print("This may take several minutes...")
    print()

    optimizer = BootstrapFewShot(
        metric=claim_quality_metric,
        max_bootstrapped_demos=3  # Use 3 few-shot examples
    )

    start_time = time.time()

    optimized = optimizer.compile(
        student=baseline,
        trainset=trainset
    )

    optimization_time = time.time() - start_time

    print(f"\nOptimization completed in {optimization_time:.1f} seconds")
    print()

    # Evaluate optimized on validation set
    print("=" * 80)
    print("OPTIMIZED EVALUATION")
    print("=" * 80)
    print("\nEvaluating optimized extractor on validation set...")

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
    print(f"Baseline:   {baseline_score:.1f}% quality ({(100 - baseline_score):.1f}% issues)")
    print(f"Optimized:  {optimized_score:.1f}% quality ({(100 - optimized_score):.1f}% issues)")
    print()

    improvement = optimized_score - baseline_score
    print(f"Improvement: {improvement:+.1f} percentage points")
    print()

    # Interpretation
    optimized_issues = 100 - optimized_score
    if improvement > 10:
        print("SIGNIFICANT IMPROVEMENT (>10 percentage points)")
        print("   The optimization clearly helped!")
        if optimized_issues < 15:
            print("   TARGET ACHIEVED: <15% low-quality claims")
        else:
            print(f"   Still above target (need <15%, currently {optimized_issues:.1f}%)")
    elif improvement >= 5:
        print("MODEST IMPROVEMENT (5-10 percentage points)")
        print("   Optimization helped, but gains are moderate")
        if optimized_issues < 15:
            print("   TARGET ACHIEVED: <15% low-quality claims")
        else:
            print(f"   Still above target (need <15%, currently {optimized_issues:.1f}%)")
    elif improvement >= 0:
        print("MINIMAL IMPROVEMENT (<5 percentage points)")
        print("   Optimization barely helped")
        print("   Consider: Better metric? More training data? Different approach?")
    else:
        print("NEGATIVE RESULT - Score got worse!")
        print("   Something might be wrong:")
        print("   - Check metric is working correctly")
        print("   - Check data quality")
        print("   - Try different random seed?")

    print()
    print(f"Time: {optimization_time:.1f} seconds")
    print()

    # Save optimized module
    print("=" * 80)
    print("SAVING OPTIMIZED MODULE")
    print("=" * 80)
    print()

    # Create models directory if needed
    import os
    os.makedirs('models', exist_ok=True)

    model_path = 'models/claim_extractor_bootstrap_v1.json'
    optimized.save(model_path)

    print(f"Optimized module saved to {model_path}")
    print()

    # Save results summary
    results = {
        "experiment": "3.1 BootstrapFewShot Optimization",
        "date": "2025-10-25",
        "model": "qwen2.5:7b-instruct-q4_0",
        "train_size": len(trainset),
        "val_size": len(valset),
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": improvement,
        "optimization_time_seconds": optimization_time,
        "max_bootstrapped_demos": 3
    }

    with open('results/experiment_3_1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to results/experiment_3_1_results.json")
    print()

    # Next steps
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Experiment 3.2: Inspect what DSPy changed")
    print("   Run: python exp_3_2_inspect_optimized.py")
    print()
    print("2. Experiment 3.3: Test on fresh examples")
    print("   Get new transcript chunks and compare baseline vs optimized")
    print()


if __name__ == "__main__":
    main()
