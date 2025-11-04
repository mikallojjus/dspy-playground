"""
Train/retrain the entailment validation model.

This script:
1. Loads entailment train/val datasets
2. Uses BootstrapFewShot optimizer with LLM-as-judge metric
3. Saves optimized model to models/entailment_validator_v1.json
4. Evaluates on validation set and prints metrics

Usage:
    python -m src.training.train_entailment_validator

Optional arguments:
    --train-path PATH     Path to training dataset (default: evaluation/entailment_train.json)
    --val-path PATH       Path to validation dataset (default: evaluation/entailment_val.json)
    --output PATH         Path to save model (default: models/entailment_validator_v1.json)
    --max-demos INT       Max bootstrapped demos (default: 4)
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
import argparse
from typing import Literal
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.metrics.entailment_metrics import entailment_llm_judge_metric, calculate_entailment_metrics
from src.config.settings import settings


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
    parser = argparse.ArgumentParser(description='Train entailment validation model')
    parser.add_argument('--train-path', default='evaluation/entailment_train.json',
                        help='Path to training dataset')
    parser.add_argument('--val-path', default='evaluation/entailment_val.json',
                        help='Path to validation dataset')
    parser.add_argument('--output', default='models/entailment_validator_v1.json',
                        help='Path to save trained model')
    parser.add_argument('--max-demos', type=int, default=4,
                        help='Maximum bootstrapped demos')

    args = parser.parse_args()

    print("=" * 80)
    print("Entailment Validation Model Training")
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

    # Optimize with BootstrapFewShot
    print("=" * 80)
    print("Starting BootstrapFewShot Optimization")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Metric: LLM-as-judge (heavy penalty for false positives)")
    print(f"  Max bootstrapped demos: {args.max_demos}")
    print(f"  Max labeled demos: {args.max_demos}")
    print()

    optimizer = BootstrapFewShot(
        metric=entailment_llm_judge_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos
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
        pred = optimized(claim=example.claim, quote=example.quote)

        print(f"Example {i}:")
        print(f"  Claim: {example.claim}")
        print(f"  Quote: {example.quote[:100]}...")
        print(f"  Ground truth: {example.relationship}")
        print(f"  Predicted: {pred.relationship}")
        print(f"  Reasoning: {pred.reasoning}")
        print(f"  Match: {'✓' if pred.relationship == example.relationship else '✗'}")
        print()

    # Print summary
    print("=" * 80)
    print("Training Summary")
    print("=" * 80)
    print()
    print(f"Baseline accuracy: {baseline_metrics['accuracy']:.1%}")
    print(f"Optimized accuracy: {optimized_metrics['accuracy']:.1%}")
    print(f"Improvement: {optimized_metrics['accuracy'] - baseline_metrics['accuracy']:+.1%}")
    print()
    print(f"Baseline false positive rate: {baseline_metrics['false_positive_rate']:.1%}")
    print(f"Optimized false positive rate: {optimized_metrics['false_positive_rate']:.1%}")
    print(f"Reduction: {baseline_metrics['false_positive_rate'] - optimized_metrics['false_positive_rate']:.1%}")
    print()
    print(f"Model saved to: {args.output}")
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


if __name__ == '__main__':
    main()
