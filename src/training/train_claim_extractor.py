"""
Train/retrain the claim extraction model.

This script:
1. Loads claim train/val datasets (positive-only for training)
2. Uses BootstrapFewShot optimizer with LLM-as-judge metric
3. Saves optimized model to models/claim_extractor_TIMESTAMP/ folder
4. Evaluates on full validation set (good + bad examples)

Each training run creates a timestamped folder containing:
  - model.json (DSPy model)
  - results.json (training metrics)

Usage:
    python -m src.training.train_claim_extractor

Optional arguments:
    --train-path PATH     Path to training dataset (default: evaluation/claims_train.json)
    --val-path PATH       Path to validation dataset (default: evaluation/claims_val.json)
    --output PATH         Custom path to save model (default: auto-generated folder)
    --max-demos INT       Max bootstrapped demos (default: 4)
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
import argparse
import random
import time
from datetime import datetime
from typing import List
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.metrics_llm_judge import llm_judge_metric
from src.config.settings import settings
from src.training.training_utils import (
    generate_model_filename,
    save_training_results,
    format_metric_comparison,
    format_duration,
)


class ClaimExtraction(dspy.Signature):
    """
    Extract factual, verifiable claims from podcast transcript text.

    Claims should be:
    - Factual (not opinions)
    - Self-contained (no pronouns like he/she/they without clear referents)
    - Specific (include names, numbers, dates when relevant)
    - Concise (5-40 words)

    OUTPUT FORMAT REQUIREMENT:
    Return claims as a valid JSON array using ONLY double quotes.
    Example: ["claim one", "claim two", "claim three"]
    Do NOT use single quotes or mix quote styles - use double quotes only.
    """

    transcript_chunk: str = dspy.InputField(
        desc="The podcast transcript text to analyze",
        prefix="Transcript:",
    )
    claims: List[str] = dspy.OutputField(
        desc='List of factual claims as JSON array with double quotes only: ["claim1", "claim2"]',
        prefix="Claims (extract the actual facts, not meta-commentary about what the speaker discusses):",
    )


def load_dataset(filepath):
    """Load claims dataset and convert to DSPy examples."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data["examples"]:
        # For positive-only training: only include good claims
        # For validation: include both good and bad
        if item["quality"] == "good":
            example = dspy.Example(
                transcript_chunk=item["transcript_chunk"],
                claims=item["claims"],  # List of all claims from this chunk
            ).with_inputs("transcript_chunk")
            examples.append(example)
        else:
            # For validation set with bad examples, expected claims is empty list
            example = dspy.Example(
                transcript_chunk=item["transcript_chunk"],
                claims=[],  # Bad claims should produce empty output
            ).with_inputs("transcript_chunk")
            examples.append(example)

    return examples


def print_metrics(dataset, dataset_name):
    """Print distribution statistics for a dataset."""
    print(f"{dataset_name}: {len(dataset)} examples")


def main():
    parser = argparse.ArgumentParser(description="Train claim extraction model")
    parser.add_argument(
        "--train-path",
        default="evaluation/claims_train.json",
        help="Path to training dataset (mixed good/bad examples)",
    )
    parser.add_argument(
        "--val-path",
        default="evaluation/claims_val.json",
        help="Path to validation dataset (mixed good/bad)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save trained model (default: auto-generated timestamp-based filename)",
    )
    parser.add_argument(
        "--max-demos", type=int, default=4, help="Maximum bootstrapped demos"
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Delete existing model file before training",
    )

    args = parser.parse_args()

    # Generate timestamp-based filenames or use provided output path
    if args.output is None:
        model_path, results_path = generate_model_filename("claim_extractor")
        print("=" * 80)
        print("Claim Extraction Model Training")
        print("=" * 80)
        print()
        print(f"Training run folder: {model_path.parent}")
        print(f"  Model: {model_path.name}")
        print(f"  Results: {results_path.name}")
        print()
    else:
        model_path = Path(args.output)
        results_path = model_path.parent / f"{model_path.stem}_results.json"
        print("=" * 80)
        print("Claim Extraction Model Training")
        print("=" * 80)
        print()
        print(f"Using custom output path:")
        print(f"  Model: {model_path}")
        print(f"  Results: {results_path}")
        print()

    # Delete existing model file if fresh start requested
    if args.fresh_start:
        if model_path.exists():
            print(f"Deleting existing model file: {model_path}")
            model_path.unlink()
            print("✓ Model file deleted for fresh start")
            print()

    # NOTE: Monkey-patch NOT needed for Track A (format="json") or Track B (ChatAdapter)
    # The patch was only needed for JSONAdapter without format constraints
    # If Track A works, malformed JSON never occurs (prevented at source)
    print("Track A: Testing format='json' WITHOUT monkey-patching (clean test)")
    print()

    # Configure DSPy
    random_seed = random.randint(1, 1000000)
    print(f"Configuring DSPy with Ollama at {settings.ollama_url}")
    print(f"Using random seed: {random_seed}")

    # 🎯 TRAINING CONFIGURATION: Use format="json" for compatibility
    # NOTE: Cannot use JSON schema during training because DSPy uses multiple signatures:
    #   - ClaimExtraction (reasoning, claims)
    #   - ClaimQualityJudge (is_high_quality, reason)
    # Global schema would conflict with LLM judge metric
    #
    # STRATEGY: Clean training with format="json", strict schema at inference
    lm = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        cache=False,  # Disable caching to force fresh training
        temperature=0.3,  # Non-deterministic generation
        seed=random_seed,  # Random seed for each training run
        format="json",  # Valid JSON mode (compatible with all signatures)
        num_ctx=32768,  # 32K context to prevent truncation with large few-shot examples
    )
    dspy.configure(lm=lm)
    print("✓ DSPy configured with:")
    print(f"  - JSON Mode: ENABLED (format='json' for multi-signature compatibility)")
    print(f"  - Caching: DISABLED")
    print(f"  - Temperature: 0.3 (balanced quality and variation)")
    print(f"  - Seed: {random_seed} (randomized to break cache)")
    print(f"  NOTE: Strict schema will be enforced at INFERENCE time")
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
        devset=valset, metric=llm_judge_metric, num_threads=1, display_progress=True
    )

    baseline_score = evaluator(baseline)
    baseline_score_value = float(baseline_score)
    print(f"Baseline quality score: {baseline_score_value:.3f}")
    print()

    # Optimize with BootstrapFewShot
    print("=" * 80)
    print("Starting BootstrapFewShot Optimization")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Metric: LLM-as-judge (claim quality evaluation)")
    print(f"  Max bootstrapped demos: {args.max_demos}")
    print(f"  Max labeled demos: {args.max_demos}")
    print(f"  Training on: Mixed examples ({len(trainset)} examples)")
    print()

    # Track training time
    training_start_time = time.time()
    training_start_timestamp = datetime.now()

    optimizer = BootstrapFewShot(
        metric=llm_judge_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos,
    )

    optimized = optimizer.compile(baseline, trainset=trainset)

    training_duration = time.time() - training_start_time

    print()
    print(f"Optimization complete! (took {format_duration(training_duration)})")
    print()

    # Evaluate optimized model
    print("Evaluating optimized model on validation set...")
    optimized_score = evaluator(optimized)
    optimized_score_value = float(optimized_score)
    print(f"Optimized quality score: {optimized_score_value:.3f}")
    print(f"Improvement: {optimized_score_value - baseline_score_value:+.3f}")
    print()

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(model_path))
    print(f"✓ Model saved to {model_path}")

    # Get few-shot examples count (check multiple possible locations)
    few_shot_count = 0
    if hasattr(optimized, "demos") and optimized.demos:
        few_shot_count = len(optimized.demos)
    elif hasattr(optimized, "predictor") and hasattr(optimized.predictor, "demos") and optimized.predictor.demos:
        few_shot_count = len(optimized.predictor.demos)

    if few_shot_count > 0:
        print(f"  Model has {few_shot_count} few-shot examples")

    # Save comprehensive results
    improvement = optimized_score_value - baseline_score_value
    targets_met = optimized_score_value > 0.85

    results = {
        "model_path": str(model_path),
        "model_name": model_path.parent.name,  # Use folder name, not "model.json"
        "timestamp": training_start_timestamp.isoformat(),
        "model_type": "claim_extractor",
        "optimizer": "BootstrapFewShot",
        "config": {
            "max_demos": args.max_demos,
            "train_path": args.train_path,
            "val_path": args.val_path,
            "train_size": len(trainset),
            "val_size": len(valset),
            "temperature": 0.3,
            "cache": False,
        },
        "baseline": {
            "score": baseline_score_value,
            "quality_score": baseline_score_value,
        },
        "optimized": {
            "score": optimized_score_value,
            "quality_score": optimized_score_value,
        },
        "improvement": {
            "score": improvement,
            "quality_score": improvement,
        },
        "training_time_seconds": training_duration,
        "few_shot_demos": few_shot_count,
        "targets_met": targets_met,
        "target_score": 0.85,
    }

    save_training_results(results_path, results)
    print(f"✓ Results saved to {results_path}")
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
    print(f"Quality Score: {format_metric_comparison(baseline_score_value, optimized_score_value)}")
    print(f"Training Time: {format_duration(training_duration)}")
    print(f"Few-shot Demos: {few_shot_count}")
    print()
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    print()

    # Check if we met goals
    if targets_met:
        print("✓ Goal achieved: Quality score > 0.85")
    else:
        print(
            f"⚠ Goal not met: Quality score {optimized_score_value:.3f} (target: >0.85)"
        )

    print()
    print("To compare all trained models, run:")
    print("  uv run python -m src.cli.compare_models --model-type claim_extractor")
    print()


if __name__ == "__main__":
    main()
