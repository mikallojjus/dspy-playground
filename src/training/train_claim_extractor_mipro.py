"""
Train/retrain the claim extraction model using MIPROv2 optimizer.

MIPROv2 jointly optimizes:
1. Instruction phrasing (finding better ways to describe the task)
2. Few-shot example selection (choosing best demonstrations)

This is more sophisticated than BootstrapFewShot and can achieve higher quality,
but takes longer to run (1-3 hours with your 389 training examples).

Usage:
    python -m src.training.train_claim_extractor_mipro

Optional arguments:
    --train-path PATH     Path to training dataset (default: evaluation/claims_train.json)
    --val-path PATH       Path to validation dataset (default: evaluation/claims_val.json)
    --output PATH         Path to save model (default: models/claim_extractor_mipro_v1.json)
    --max-demos INT       Max bootstrapped demos (default: 4)
    --num-trials INT      Bayesian optimization iterations (default: 30)
    --num-candidates INT  Instruction variants to try (default: 10)
"""

import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
import json
import sys
import argparse
from typing import List
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.metrics_llm_judge import llm_judge_metric
from src.config.settings import settings


class ClaimExtraction(dspy.Signature):
    """
    Extract factual, verifiable claims from podcast transcript text.

    Claims should be:
    - Factual (not opinions)
    - Self-contained (no pronouns like he/she/they without clear referents)
    - Specific (include names, numbers, dates)
    - Concise (5-40 words)
    """

    transcript_chunk: str = dspy.InputField(desc="The podcast transcript text to analyze")
    claims: List[str] = dspy.OutputField(desc="List of factual claims extracted from the transcript")


def load_dataset(filepath):
    """Load claims dataset and convert to DSPy examples."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data['examples']:
        # For positive-only training: only include good claims
        # For validation: include both good and bad
        if item['quality'] == 'good':
            example = dspy.Example(
                transcript_chunk=item['transcript_chunk'],
                claims=[item['claim']]  # Wrap in list
            ).with_inputs('transcript_chunk')
            examples.append(example)
        else:
            # For validation set with bad examples, expected claims is empty list
            example = dspy.Example(
                transcript_chunk=item['transcript_chunk'],
                claims=[]  # Bad claims should produce empty output
            ).with_inputs('transcript_chunk')
            examples.append(example)

    return examples


def print_metrics(dataset, dataset_name):
    """Print distribution statistics for a dataset."""
    print(f"{dataset_name}: {len(dataset)} examples")


def main():
    parser = argparse.ArgumentParser(description='Train claim extraction model with MIPROv2')
    parser.add_argument('--train-path', default='evaluation/claims_train.json',
                        help='Path to training dataset (mixed good/bad examples)')
    parser.add_argument('--val-path', default='evaluation/claims_val.json',
                        help='Path to validation dataset (mixed good/bad)')
    parser.add_argument('--output', default='models/claim_extractor_mipro_v1.json',
                        help='Path to save trained model')
    parser.add_argument('--max-demos', type=int, default=4,
                        help='Maximum bootstrapped demos')
    parser.add_argument('--num-trials', type=int, default=30,
                        help='Number of Bayesian optimization iterations (higher = better but slower)')
    parser.add_argument('--num-candidates', type=int, default=10,
                        help='Number of instruction variants to try per iteration')

    args = parser.parse_args()

    start_time = datetime.now()

    print("=" * 80)
    print("Claim Extraction Model Training - MIPROv2")
    print("=" * 80)
    print()
    print("MIPROv2 optimizes both instructions AND few-shot examples.")
    print("Expected runtime: 1-3 hours for 389 training examples")
    print("Started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print()

    # Configure DSPy
    print(f"Configuring DSPy with Ollama at {settings.ollama_url}")
    lm = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url
    )
    dspy.configure(lm=lm)

    # Configure prompt model for MIPROv2 instruction proposal
    prompt_model = None
    if settings.mipro_prompt_model:
        print(f"Using {settings.mipro_prompt_model} for instruction proposal")
        prompt_model = dspy.LM(settings.mipro_prompt_model, api_key=settings.anthropic_api_key)
    else:
        print("WARNING: No prompt_model configured. MIPROv2 may fail with Qwen3 4B.")
        print("Add MIPRO_PROMPT_MODEL to .env (e.g., 'anthropic/claude-3-5-haiku-20241022')")
    print()

    # Load datasets
    print("Loading datasets...")
    trainset = load_dataset(args.train_path)
    valset = load_dataset(args.val_path)
    print()

    print_metrics(trainset, "Training set (mixed)")
    print_metrics(valset, "Validation set (mixed)")
    print()

    # Create baseline model
    print("Creating baseline model (zero-shot)...")
    baseline = dspy.ChainOfThought(ClaimExtraction)
    print()

    # Evaluate baseline
    print("Evaluating baseline on validation set...")
    evaluator = Evaluate(
        devset=valset,
        metric=llm_judge_metric,
        num_threads=1,
        display_progress=True
    )

    baseline_score = evaluator(baseline)
    baseline_score_value = float(baseline_score)
    print(f"Baseline quality score: {baseline_score_value:.3f}")
    print()

    # Optimize with MIPROv2
    print("=" * 80)
    print("Starting MIPROv2 Optimization")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Metric: LLM-as-judge (claim quality evaluation)")
    print(f"  Max bootstrapped demos: {args.max_demos}")
    print(f"  Max labeled demos: {args.max_demos}")
    print(f"  Bayesian optimization trials: {args.num_trials}")
    print(f"  Instruction candidates per trial: {args.num_candidates}")
    print(f"  Total LLM calls (est): {args.num_trials * args.num_candidates * 2} - {args.num_trials * args.num_candidates * 5}")
    print(f"  Training on: {len(trainset)} examples")
    print()
    print("Progress tracking:")
    print("  MIPROv2 will try different instruction phrasings")
    print("  Each trial tests num_candidates variants")
    print("  Best performer becomes the next baseline")
    print("  This is Bayesian optimization - it gets smarter over time")
    print()

    optimizer = MIPROv2(
        metric=llm_judge_metric,
        num_candidates=args.num_candidates,
        init_temperature=1.0,
        auto=None,  # Disable auto mode to use manual num_trials/num_candidates
        verbose=True,  # Show progress
        prompt_model=prompt_model  # Use Claude for instruction proposal
    )

    print("Starting optimization (this will take 1-3 hours)...")
    optimization_start = datetime.now()

    optimized = optimizer.compile(
        baseline,
        trainset=trainset,
        num_trials=args.num_trials,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos
    )

    optimization_end = datetime.now()
    optimization_duration = optimization_end - optimization_start

    print()
    print("Optimization complete!")
    print(f"Time taken: {optimization_duration}")
    print()

    # Evaluate optimized model
    print("Evaluating optimized model on validation set...")
    optimized_score = evaluator(optimized)
    optimized_score_value = float(optimized_score)
    print(f"Optimized quality score: {optimized_score_value:.3f}")
    print(f"Improvement: {optimized_score_value - baseline_score_value:+.3f}")
    print()

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    optimized.save(str(output_path))
    print(f"✓ Model saved to {args.output}")
    print()

    # Print few-shot examples count
    if hasattr(optimized, 'demos') and optimized.demos:
        print(f"Model has {len(optimized.demos)} few-shot examples")
    print()

    # Print example predictions
    print("=" * 80)
    print("Example Predictions (first 3 validation examples)")
    print("=" * 80)
    print()

    for i, example in enumerate(valset[:3], 1):
        pred = optimized(transcript_chunk=example.transcript_chunk)

        print(f"Example {i}:")
        print(f"  Transcript: {example.transcript_chunk[:100]}...")
        print(f"  Ground truth claims: {example.claims}")
        print(f"  Predicted claims: {pred.claims}")
        print()

    # Print summary
    print("=" * 80)
    print("Training Summary")
    print("=" * 80)
    print()
    print(f"Started at:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:  {datetime.now() - start_time}")
    print()
    print(f"Baseline quality score:  {baseline_score_value:.3f}")
    print(f"Optimized quality score: {optimized_score_value:.3f}")
    print(f"Improvement:             {optimized_score_value - baseline_score_value:+.3f} ({(optimized_score_value - baseline_score_value) / baseline_score_value * 100:+.1f}%)")
    print()
    print(f"Model saved to: {args.output}")
    print()

    # Check if we met goals
    issues_pct = (1 - optimized_score_value) * 100
    if issues_pct < 15:
        print(f"✓ Goal achieved: Low-quality claims = {issues_pct:.1f}% (target: <15%)")
    else:
        print(f"⚠ Goal not met: Low-quality claims = {issues_pct:.1f}% (target: <15%)")

    if optimized_score_value > baseline_score_value + 0.05:
        print(f"✓ Significant improvement over BootstrapFewShot!")
    elif optimized_score_value > baseline_score_value:
        print(f"✓ Modest improvement over baseline")
    else:
        print(f"⚠ No improvement - BootstrapFewShot may be sufficient")

    print()

    # Save detailed results
    results = {
        "optimizer": "MIPROv2",
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "duration_seconds": (datetime.now() - start_time).total_seconds(),
        "optimization_duration_seconds": optimization_duration.total_seconds(),
        "config": {
            "num_trials": args.num_trials,
            "num_candidates": args.num_candidates,
            "max_demos": args.max_demos,
            "model": settings.ollama_model,
            "train_examples": len(trainset),
            "val_examples": len(valset),
        },
        "scores": {
            "baseline": float(baseline_score_value),
            "optimized": float(optimized_score_value),
            "improvement": float(optimized_score_value - baseline_score_value),
            "improvement_pct": float((optimized_score_value - baseline_score_value) / baseline_score_value * 100),
        }
    }

    results_path = output_path.parent / f"{output_path.stem}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to {results_path}")
    print()


if __name__ == '__main__':
    main()
