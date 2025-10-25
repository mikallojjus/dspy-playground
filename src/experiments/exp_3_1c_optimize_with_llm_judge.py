"""
Experiment 3.1c: BootstrapFewShot with LLM-as-Judge Metric

Goal: Optimize using LLM-as-Judge instead of pattern matching

Key differences from 3.1b:
- Uses LLM-as-Judge metric (semantic understanding)
- Should get more accurate optimization
- Slower but more reliable

Expected outcome: Better optimization results due to more accurate metric guiding the process
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
from typing import List
import time
import random

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# Claim extraction signature
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


# LLM-as-Judge metric
class ClaimQualityJudge(dspy.Signature):
    """Evaluate if a claim is high quality and self-contained.

    A HIGH-QUALITY claim must be:
    1. Self-contained - understandable without external context
    2. Specific - includes names, not just pronouns without referents
    3. Factual - not opinion or speculation
    4. Clear - no vague language that makes it unverifiable

    Be strict: claims must be understandable on their own.
    """
    claim: str = dspy.InputField(desc="The claim to evaluate")
    is_high_quality: bool = dspy.OutputField(desc="True if high quality, False otherwise")
    reason: str = dspy.OutputField(desc="Brief explanation")


def llm_judge_metric(example, pred, trace=None):
    """Evaluate claim quality using an LLM judge."""
    predicted_claims = pred.claims if hasattr(pred, 'claims') else []

    if not predicted_claims:
        return 0.0

    judge = dspy.ChainOfThought(ClaimQualityJudge)
    high_quality_count = 0

    for claim in predicted_claims:
        result = judge(claim=claim)
        if result.is_high_quality:
            high_quality_count += 1

    return high_quality_count / len(predicted_claims)


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
    print("Experiment 3.1c: BootstrapFewShot with LLM-as-Judge")
    print("=" * 80)
    print()
    print("Key Change: Using LLM-as-Judge metric instead of pattern matching")
    print("Expected: More accurate optimization due to better metric")
    print()

    # Load data
    print("Loading datasets...")
    positive_trainset = load_dataset('evaluation/claims_positive_only.json')

    # Split positive examples
    random.seed(42)
    random.shuffle(positive_trainset)

    split_point = int(len(positive_trainset) * 0.7)
    trainset = positive_trainset[:split_point]

    # Load full validation set
    full_valset = load_dataset('evaluation/claims_val.json')

    print(f"Positive training set: {len(trainset)} good examples")
    print(f"Full validation set: {len(full_valset)} mixed examples")

    good_val = sum(1 for ex in full_valset if ex.quality == 'good')
    bad_val = sum(1 for ex in full_valset if ex.quality == 'bad')
    print(f"  (Val: {good_val} good, {bad_val} bad)")
    print()

    # Configure DSPy
    print("Configuring DSPy with Ollama...")
    lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
    dspy.configure(lm=lm)
    print("Done")
    print()

    # Create baseline
    baseline = dspy.ChainOfThought(ClaimExtraction)

    # Evaluate baseline with LLM judge
    print("=" * 80)
    print("BASELINE EVALUATION (LLM-as-Judge Metric)")
    print("=" * 80)
    print()
    print("Note: This will be slower due to LLM judge calls")
    print()

    evaluator = Evaluate(
        devset=full_valset,
        metric=llm_judge_metric,
        display_progress=True,
        display_table=0
    )

    print("Evaluating baseline...")
    baseline_result = evaluator(baseline)
    baseline_score = baseline_result.score if hasattr(baseline_result, 'score') else float(baseline_result)

    print(f"\nBaseline Quality Score (LLM Judge): {baseline_score:.1f}%")
    print(f"Claims with Issues: {(100 - baseline_score):.1f}%")
    print()

    # Optimize
    print("=" * 80)
    print("OPTIMIZATION (LLM-as-Judge Metric)")
    print("=" * 80)
    print()
    print(f"Training on {len(trainset)} good examples...")
    print("Using LLM-as-Judge to evaluate training examples")
    print()
    print("This will be slower but more accurate!")
    print()

    optimizer = BootstrapFewShot(
        metric=llm_judge_metric,
        max_bootstrapped_demos=4
    )

    start_time = time.time()
    print("Optimizing... (this may take several minutes)")
    print()

    optimized = optimizer.compile(
        student=baseline,
        trainset=trainset
    )

    optimization_time = time.time() - start_time
    print(f"\nOptimization completed in {optimization_time:.1f} seconds")
    print()

    # Evaluate optimized
    print("=" * 80)
    print("OPTIMIZED EVALUATION (LLM-as-Judge Metric)")
    print("=" * 80)
    print()

    print("Evaluating optimized model...")
    optimized_result = evaluator(optimized)
    optimized_score = optimized_result.score if hasattr(optimized_result, 'score') else float(optimized_result)

    print(f"\nOptimized Quality Score (LLM Judge): {optimized_score:.1f}%")
    print(f"Claims with Issues: {(100 - optimized_score):.1f}%")
    print()

    # Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"Metric: LLM-as-Judge (semantic understanding)")
    print(f"Dataset: {len(full_valset)} mixed examples ({good_val} good, {bad_val} bad)")
    print()
    print(f"Baseline:   {baseline_score:.1f}% quality")
    print(f"Optimized:  {optimized_score:.1f}% quality")
    print()

    improvement = optimized_score - baseline_score
    print(f"Improvement: {improvement:+.1f} percentage points")
    print()

    optimized_issues = 100 - optimized_score
    target_met = optimized_issues < 15

    if improvement > 10:
        print("EXCELLENT: Significant improvement!")
    elif improvement >= 5:
        print("SUCCESS: Good improvement")
    elif improvement > 0:
        print("MODEST: Small improvement")
    else:
        print("NO IMPROVEMENT: Optimization didn't help")

    print()
    if target_met:
        print("TARGET ACHIEVED: <15% low-quality claims")
    else:
        print(f"Target not met: {optimized_issues:.1f}% issues (need <15%)")

    print()
    print(f"Time: {optimization_time:.1f} seconds")
    print()

    # Save
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()

    import os
    os.makedirs('models', exist_ok=True)

    model_path = 'models/claim_extractor_llm_judge_v1.json'
    optimized.save(model_path)
    print(f"Model saved: {model_path}")

    results = {
        "experiment": "3.1c BootstrapFewShot with LLM-as-Judge",
        "date": "2025-10-25",
        "model": "qwen2.5:7b-instruct-q4_0",
        "metric": "llm_judge",
        "train_size": len(trainset),
        "val_size": len(full_valset),
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": improvement,
        "target_met": target_met,
        "optimization_time_seconds": optimization_time
    }

    with open('results/experiment_3_1c_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved: results/experiment_3_1c_results.json")
    print()

    # Next steps
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()

    if improvement > 0:
        print("1. Compare this to pattern matching optimization (exp 3.1b)")
        print("2. Inspect what DSPy learned (exp_3_2_inspect_optimized.py)")
        print("3. Test on fresh examples")
        print()
        print("Expected: LLM judge should give more reliable optimization")
    else:
        print("Optimization didn't improve results.")
        print("Consider:")
        print("  - More training examples")
        print("  - Different optimizer (MIPROv2)")
        print("  - Manual prompt engineering")


if __name__ == "__main__":
    main()
