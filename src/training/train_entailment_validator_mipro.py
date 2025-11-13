"""
Train/retrain the entailment validation model using MIPROv2 optimizer.

MIPROv2 is particularly valuable for entailment because:
1. The SUPPORTS vs RELATED boundary is subtle and requires precise instructions
2. Instruction optimization can clarify "directly asserts" vs "topically related"
3. This is your highest-value optimization target

Expected improvement: 8-15% reduction in false positives

Usage:
    python -m src.training.train_entailment_validator_mipro

Optional arguments:
    --train-path PATH     Path to training dataset (default: evaluation/entailment_train.json)
    --val-path PATH       Path to validation dataset (default: evaluation/entailment_val.json)
    --output PATH         Path to save model (default: models/entailment_validator_mipro_v1.json)
    --max-demos INT       Max bootstrapped demos (default: 5)
    --num-trials INT      Bayesian optimization iterations (default: 20)
    --num-candidates INT  Instruction variants to try (default: 8)
"""

import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
import json
import sys
import argparse
import time
from typing import Literal
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.metrics.entailment_metrics import entailment_llm_judge_metric, calculate_entailment_metrics
from src.config.settings import settings
from src.training.training_utils import (
    generate_model_filename,
    save_training_results,
    format_metric_comparison,
    format_duration,
)


class EntailmentValidation(dspy.Signature):
    """
    Validate whether a quote supports a claim.

    Relationship types:
    - SUPPORTS: Quote directly asserts the claim or provides clear evidence
    - RELATED: Quote is topically related but doesn't validate the claim
    - NEUTRAL: Quote is unrelated or provides no evidence
    - CONTRADICTS: Quote contradicts or undermines the claim
    """

    claim: str = dspy.InputField(desc="The claim to validate")
    quote: str = dspy.InputField(desc="The quote to check for support")
    relationship: Literal["SUPPORTS", "RELATED", "NEUTRAL", "CONTRADICTS"] = dspy.OutputField(
        desc="The relationship between quote and claim"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the relationship")
    confidence: float = dspy.OutputField(desc="Confidence score (0.0-1.0)")


def load_dataset(filepath):
    """Load entailment dataset and convert to DSPy examples."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data['examples']:
        example = dspy.Example(
            claim=item['claim'],
            quote=item['quote'],
            relationship=item['relationship'],
            reasoning=item['reasoning'],
            confidence=item['confidence']
        ).with_inputs('claim', 'quote')
        examples.append(example)

    return examples


def print_metrics(dataset, dataset_name):
    """Print distribution statistics for a dataset."""
    supports = sum(1 for ex in dataset if ex.relationship == 'SUPPORTS')
    related = sum(1 for ex in dataset if ex.relationship == 'RELATED')
    neutral = sum(1 for ex in dataset if ex.relationship == 'NEUTRAL')
    contradicts = sum(1 for ex in dataset if ex.relationship == 'CONTRADICTS')

    print(f"{dataset_name}: {len(dataset)} examples")
    print(f"  SUPPORTS={supports}, RELATED={related}, NEUTRAL={neutral}, CONTRADICTS={contradicts}")


def main():
    parser = argparse.ArgumentParser(description='Train entailment validation model with MIPROv2')
    parser.add_argument('--train-path', default='evaluation/entailment_train.json',
                        help='Path to training dataset')
    parser.add_argument('--val-path', default='evaluation/entailment_val.json',
                        help='Path to validation dataset')
    parser.add_argument('--output', default=None,
                        help='Path to save trained model (default: auto-generated timestamp-based folder)')
    parser.add_argument('--max-demos', type=int, default=5,
                        help='Maximum bootstrapped demos (using 5 for binary-ish task)')
    parser.add_argument('--num-trials', type=int, default=20,
                        help='Number of Bayesian optimization iterations (lower for smaller dataset)')
    parser.add_argument('--num-candidates', type=int, default=8,
                        help='Number of instruction variants to try per iteration')

    args = parser.parse_args()

    # Generate timestamp-based filenames or use provided output path
    if args.output is None:
        model_path, results_path = generate_model_filename("entailment_validator_mipro")
        print("=" * 80)
        print("Entailment Validation Model Training - MIPROv2")
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
        print("Entailment Validation Model Training - MIPROv2")
        print("=" * 80)
        print()
        print(f"Using custom output path:")
        print(f"  Model: {model_path}")
        print(f"  Results: {results_path}")
        print()

    training_start_time = time.time()
    training_start_timestamp = datetime.now()

    print("MIPROv2 is ideal for entailment's subtle SUPPORTS vs RELATED boundary.")
    print("Expected runtime: 30-90 minutes for 31 training examples")
    print("Started at:", training_start_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    print()

    # Configure DSPy
    print(f"Configuring DSPy with Ollama at {settings.ollama_url}")
    lm = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        num_ctx=32768
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

    print_metrics(trainset, "Training set")
    print_metrics(valset, "Validation set")
    print()

    # Create baseline model
    print("Creating baseline model (zero-shot)...")
    baseline = dspy.ChainOfThought(EntailmentValidation)
    print()

    # Evaluate baseline
    print("Evaluating baseline on validation set...")
    evaluator = Evaluate(
        devset=valset,
        metric=entailment_llm_judge_metric,
        num_threads=1,
        display_progress=True
    )

    baseline_score = evaluator(baseline)
    baseline_score_value = float(baseline_score)
    print(f"Baseline score: {baseline_score_value:.3f}")
    print()

    # Calculate detailed baseline metrics
    print("Calculating detailed baseline metrics...")
    baseline_metrics = calculate_entailment_metrics(
        valset,
        lambda ex: baseline(claim=ex.claim, quote=ex.quote)
    )

    print(f"Baseline Metrics:")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.1%}")
    print(f"  False Positive Rate: {baseline_metrics['false_positive_rate']:.1%}")
    print()

    # Optimize with MIPROv2
    print("=" * 80)
    print("Starting MIPROv2 Optimization")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Metric: LLM-as-judge (heavy penalty for false positives)")
    print(f"  Max bootstrapped demos: {args.max_demos}")
    print(f"  Max labeled demos: {args.max_demos}")
    print(f"  Bayesian optimization trials: {args.num_trials}")
    print(f"  Instruction candidates per trial: {args.num_candidates}")
    print(f"  Total LLM calls (est): {args.num_trials * args.num_candidates * 2} - {args.num_trials * args.num_candidates * 5}")
    print(f"  Training on: {len(trainset)} examples")
    print()
    print("Progress tracking:")
    print("  MIPROv2 will find better ways to phrase SUPPORTS vs RELATED distinction")
    print("  Goal: Reduce false positives (RELATED misclassified as SUPPORTS)")
    print()

    optimizer = MIPROv2(
        metric=entailment_llm_judge_metric,
        num_candidates=args.num_candidates,
        init_temperature=1.0,
        auto=None,  # Disable auto mode to use manual num_trials/num_candidates
        verbose=True,  # Show progress
        prompt_model=prompt_model  # Use Claude for instruction proposal
    )

    print("Starting optimization (this will take 30-90 minutes)...")

    optimized = optimizer.compile(
        baseline,
        trainset=trainset,
        num_trials=args.num_trials,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos
    )

    training_duration = time.time() - training_start_time

    print()
    print(f"Optimization complete! (took {format_duration(training_duration)})")
    print()

    # Evaluate optimized model
    print("Evaluating optimized model on validation set...")
    optimized_score = evaluator(optimized)
    optimized_score_value = float(optimized_score)
    print(f"Optimized score: {optimized_score_value:.3f}")
    print(f"Improvement: {optimized_score_value - baseline_score_value:+.3f}")
    print()

    # Calculate detailed optimized metrics
    print("Calculating detailed optimized metrics...")
    optimized_metrics = calculate_entailment_metrics(
        valset,
        lambda ex: optimized(claim=ex.claim, quote=ex.quote)
    )

    print(f"Optimized Metrics:")
    print(f"  Accuracy: {optimized_metrics['accuracy']:.1%} ({optimized_metrics['accuracy'] - baseline_metrics['accuracy']:+.1%})")
    print(f"  False Positive Rate: {optimized_metrics['false_positive_rate']:.1%} ({optimized_metrics['false_positive_rate'] - baseline_metrics['false_positive_rate']:+.1%})")
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
    print()

    # Print example predictions
    print("=" * 80)
    print("Example Predictions (first 3 validation examples)")
    print("=" * 80)
    print()

    for i, example in enumerate(valset[:3], 1):
        pred = optimized(claim=example.claim, quote=example.quote)

        print(f"Example {i}:")
        print(f"  Claim: {example.claim}")
        print(f"  Quote: {example.quote[:100]}...")
        print(f"  Ground truth: {example.relationship}")
        print(f"  Predicted: {pred.relationship}")
        print(f"  Reasoning: {pred.reasoning}")
        print(f"  Match: {'✓' if pred.relationship == example.relationship else '✗'}")
        print()

    # Save comprehensive results
    accuracy_improvement = optimized_metrics['accuracy'] - baseline_metrics['accuracy']
    fp_reduction = baseline_metrics['false_positive_rate'] - optimized_metrics['false_positive_rate']
    targets_met = (optimized_metrics['false_positive_rate'] < 0.10 and
                   optimized_metrics['accuracy'] > 0.90)

    results = {
        "model_path": str(model_path),
        "model_name": model_path.parent.name,  # Use folder name, not "model.json"
        "timestamp": training_start_timestamp.isoformat(),
        "model_type": "entailment_validator",
        "optimizer": "MIPROv2",
        "config": {
            "max_demos": args.max_demos,
            "num_trials": args.num_trials,
            "num_candidates": args.num_candidates,
            "train_path": args.train_path,
            "val_path": args.val_path,
            "train_size": len(trainset),
            "val_size": len(valset),
        },
        "baseline": {
            "score": baseline_score_value,
            "accuracy": baseline_metrics['accuracy'],
            "precision": baseline_metrics['precision'],
            "recall": baseline_metrics['recall'],
            "false_positive_rate": baseline_metrics['false_positive_rate'],
        },
        "optimized": {
            "score": optimized_score_value,
            "accuracy": optimized_metrics['accuracy'],
            "precision": optimized_metrics['precision'],
            "recall": optimized_metrics['recall'],
            "false_positive_rate": optimized_metrics['false_positive_rate'],
        },
        "improvement": {
            "score": optimized_score_value - baseline_score_value,
            "accuracy": accuracy_improvement,
            "false_positive_rate_reduction": fp_reduction,
        },
        "training_time_seconds": training_duration,
        "few_shot_demos": few_shot_count,
        "targets_met": targets_met,
        "target_accuracy": 0.90,
        "target_fp_rate": 0.10,
    }

    save_training_results(results_path, results)
    print(f"✓ Results saved to {results_path}")
    print()

    # Print summary
    print("=" * 80)
    print("Training Summary")
    print("=" * 80)
    print()
    print(f"Accuracy: {baseline_metrics['accuracy']:.1%} → {optimized_metrics['accuracy']:.1%} ({accuracy_improvement:+.1%})")
    print(f"False Positive Rate: {baseline_metrics['false_positive_rate']:.1%} → {optimized_metrics['false_positive_rate']:.1%} ({-fp_reduction:+.1%})")
    print(f"Training Time: {format_duration(training_duration)}")
    print(f"Few-shot Demos: {few_shot_count}")
    print()
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    print()

    # Check if we met goals
    if optimized_metrics['false_positive_rate'] < 0.10:
        print("✓ Goal achieved: False positive rate < 10%")
    else:
        print(f"⚠ Goal not met: False positive rate {optimized_metrics['false_positive_rate']:.1%} (target: <10%)")

    if optimized_metrics['accuracy'] > 0.90:
        print("✓ Goal achieved: Accuracy > 90%")
    else:
        print(f"⚠ Goal not met: Accuracy {optimized_metrics['accuracy']:.1%} (target: >90%)")

    print()
    print("To compare all trained models, run:")
    print("  uv run python -m src.cli.compare_models --model-type entailment_validator")
    print()


if __name__ == '__main__':
    main()
