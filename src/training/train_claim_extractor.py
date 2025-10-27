"""
Train/retrain the claim extraction model.

This script:
1. Loads claim train/val datasets (positive-only for training)
2. Uses BootstrapFewShot optimizer with LLM-as-judge metric
3. Saves optimized model to models/claim_extractor_llm_judge_v1.json
4. Evaluates on full validation set (good + bad examples)

Usage:
    python -m src.training.train_claim_extractor

Optional arguments:
    --train-path PATH     Path to training dataset (default: evaluation/claims_positive_only.json)
    --val-path PATH       Path to validation dataset (default: evaluation/claims_val.json)
    --output PATH         Path to save model (default: models/claim_extractor_llm_judge_v1.json)
    --max-demos INT       Max bootstrapped demos (default: from settings)
    --max-rounds INT      Max optimization rounds (default: from settings)
    --max-errors INT      Max errors tolerated (default: from settings)
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
import argparse
from typing import List
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore

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


def load_positive_only_dataset(filepath):
    """Load only positive (good quality) examples for training."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data['examples']:
        if item['quality'] == 'good':
            example = dspy.Example(
                transcript_chunk=item['transcript_chunk'],
                claims=[item['claim']]
            ).with_inputs('transcript_chunk')
            examples.append(example)

    return examples


def print_metrics(dataset, dataset_name):
    """Print distribution statistics for a dataset."""
    print(f"{dataset_name}: {len(dataset)} examples")


def main():
    parser = argparse.ArgumentParser(description='Train claim extraction model')
    parser.add_argument('--train-path', default='evaluation/claims_positive_only.json',
                        help='Path to training dataset (positive examples only)')
    parser.add_argument('--val-path', default='evaluation/claims_val.json',
                        help='Path to validation dataset (mixed good/bad)')
    parser.add_argument('--output', default='models/claim_extractor_llm_judge_v1.json',
                        help='Path to save trained model')
    parser.add_argument('--max-demos', type=int, default=None,
                        help='Maximum bootstrapped demos (default: from settings)')
    parser.add_argument('--max-rounds', type=int, default=None,
                        help='Maximum optimization rounds (default: from settings)')
    parser.add_argument('--max-errors', type=int, default=None,
                        help='Maximum errors tolerated (default: from settings)')

    args = parser.parse_args()

    print("=" * 80)
    print("Claim Extraction Model Training")
    print("=" * 80)
    print()

    # Configure DSPy
    print(f"Configuring DSPy with Ollama at {settings.ollama_url}")
    lm = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url
    )
    dspy.configure(lm=lm)
    print()

    # Load datasets
    print("Loading datasets...")
    trainset = load_positive_only_dataset(args.train_path)
    valset = load_dataset(args.val_path)
    print()

    print_metrics(trainset, "Training set (positive only)")
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
    print(f"Baseline quality score: {baseline_score:.3f}")
    print()

    # Optimize with BootstrapFewShot
    print("=" * 80)
    print("Starting BootstrapFewShot Optimization")
    print("=" * 80)
    print()

    max_demos = args.max_demos or settings.dspy_max_bootstrapped_demos
    max_rounds = args.max_rounds or settings.dspy_max_rounds
    max_labeled_demos = max_demos  # Use same value
    max_errors = args.max_errors or settings.dspy_max_errors

    print(f"Configuration:")
    print(f"  Metric: LLM-as-judge (claim quality evaluation)")
    print(f"  Max bootstrapped demos: {max_demos}")
    print(f"  Max labeled demos: {max_labeled_demos}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Max errors: {max_errors}")
    print(f"  Training on: Positive examples only ({len(trainset)} good claims)")
    print()

    optimizer = BootstrapFewShot(
        metric=llm_judge_metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        max_errors=max_errors
    )

    optimized = optimizer.compile(
        baseline,
        trainset=trainset
    )

    print()
    print("Optimization complete!")
    print()

    # Evaluate optimized model
    print("Evaluating optimized model on validation set...")
    optimized_score = evaluator(optimized)  # type: ignore
    print(f"Optimized quality score: {optimized_score:.3f}")
    print(f"Improvement: {optimized_score - baseline_score:+.3f}")  # type: ignore
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
    print(f"Baseline quality score: {baseline_score:.3f}")
    print(f"Optimized quality score: {optimized_score:.3f}")
    print(f"Improvement: {optimized_score - baseline_score:+.3f}")  # type: ignore
    print()
    print(f"Model saved to: {args.output}")
    print()

    # Check if we met goals
    if optimized_score > 0.85:
        print("✓ Goal achieved: Quality score > 0.85")
    else:
        print(f"⚠ Goal not met: Quality score {optimized_score:.3f} (target: >0.85)")

    print()


if __name__ == '__main__':
    main()
