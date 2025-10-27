"""
Train/retrain the quote finder model.

This script:
1. Loads quote finding train/val datasets
2. Uses BootstrapFewShot optimizer with composite metric (verification + entailment + recall)
3. Saves optimized model to models/quote_finder_v1.json
4. Evaluates on validation set and prints metrics

Usage:
    python -m src.training.train_quote_finder

Optional arguments:
    --train-path PATH     Path to training dataset (default: evaluation/quote_finding_train.json)
    --val-path PATH       Path to validation dataset (default: evaluation/quote_finding_val.json)
    --output PATH         Path to save model (default: models/quote_finder_v1.json)
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
from typing import List, Dict
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore

from src.search.llm_quote_finder import QuoteFinder
from src.metrics.quote_finder_metrics import QuoteFinderMetric
from src.dspy_models.entailment_validator import EntailmentValidatorModel
from src.config.settings import settings


def load_dataset(filepath: str) -> List[dspy.Example]:
    """
    Load quote finding dataset and convert to DSPy examples.

    Expected JSON format:
    {
        "examples": [
            {
                "claim": "Bitcoin reached $69,000 in November 2021",
                "transcript_chunks": "...",
                "gold_quotes": ["Bitcoin hit $69k in November 2021"]
            },
            ...
        ]
    }
    """
    filepath_obj = Path(filepath)

    if not filepath_obj.exists():
        raise FileNotFoundError(f"Training data not found: {filepath}")

    with open(filepath_obj, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both wrapped {"examples": [...]} and flat [...] formats
    if isinstance(data, dict) and 'examples' in data:
        items = data['examples']
    else:
        items = data

    examples = []
    for item in items:
        example = dspy.Example(
            claim=item['claim'],
            transcript_chunks=item['transcript_chunks'],
            gold_quotes=item['gold_quotes']
        ).with_inputs('claim', 'transcript_chunks')
        examples.append(example)

    return examples


def print_dataset_stats(dataset: List[dspy.Example], dataset_name: str):
    """Print statistics about the dataset."""
    total_quotes = sum(len(ex.gold_quotes) for ex in dataset)
    avg_quotes = total_quotes / len(dataset) if dataset else 0

    print(f"{dataset_name}: {len(dataset)} examples")
    print(f"  Total gold quotes: {total_quotes}")
    print(f"  Avg quotes/claim: {avg_quotes:.1f}")


def evaluate_model(finder: QuoteFinder, dataset: List[dspy.Example], metric: QuoteFinderMetric) -> Dict:
    """
    Evaluate model on dataset and return detailed metrics.

    Returns:
        Dict with mean_score, individual scores, and component metrics
    """
    scores = []
    verification_rates = []
    entailment_rates = []
    recalls = []

    for example in dataset:
        # Generate prediction
        try:
            prediction = finder(
                claim=example.claim,
                transcript_chunks=example.transcript_chunks
            )
        except Exception as e:
            print(f"Warning: Prediction failed for claim '{example.claim[:50]}...': {e}")
            prediction = dspy.Prediction(quotes=[])

        # Calculate score
        score = metric(example, prediction)
        scores.append(score)

    mean_score = sum(scores) / len(scores) if scores else 0.0

    return {
        'mean_score': mean_score,
        'scores': scores,
        'min_score': min(scores) if scores else 0.0,
        'max_score': max(scores) if scores else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='Train quote finder model')
    parser.add_argument('--train-path', default='evaluation/quote_finding_train.json',
                        help='Path to training dataset')
    parser.add_argument('--val-path', default='evaluation/quote_finding_val.json',
                        help='Path to validation dataset (optional)')
    parser.add_argument('--output', default='models/quote_finder_v1.json',
                        help='Path to save trained model')
    parser.add_argument('--max-demos', type=int, default=None,
                        help='Maximum bootstrapped demos (default: from settings)')
    parser.add_argument('--max-rounds', type=int, default=None,
                        help='Maximum optimization rounds (default: from settings)')
    parser.add_argument('--max-errors', type=int, default=None,
                        help='Maximum errors tolerated (default: from settings)')

    args = parser.parse_args()

    print("=" * 80)
    print("Quote Finder Model Training")
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

    # Load training dataset
    print("Loading training dataset...")
    try:
        trainset = load_dataset(args.train_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print()
        print("To create sample training data, run:")
        print("  python examples/generate_training_data.py")
        print()
        print("Or manually create a JSON file with the format:")
        print('  [{"claim": "...", "transcript_chunks": "...", "gold_quotes": ["..."]}]')
        return 1

    print_dataset_stats(trainset, "Training set")
    print()

    # Load validation dataset (optional)
    valset = None
    if Path(args.val_path).exists():
        print("Loading validation dataset...")
        valset = load_dataset(args.val_path)
        print_dataset_stats(valset, "Validation set")
        print()
    else:
        print(f"Validation dataset not found at {args.val_path} (skipping)")
        print()

    # Create baseline model
    print("Creating baseline model (zero-shot)...")
    baseline = QuoteFinder()
    print()

    # Create evaluation metric
    print("Initializing evaluation metric...")
    entailment_validator = EntailmentValidatorModel()
    metric = QuoteFinderMetric(entailment_validator=entailment_validator)
    print(f"  Metric weights: verification=40%, entailment=40%, recall=20%")
    print()

    # Evaluate baseline (on trainset if no valset)
    eval_dataset = valset if valset else trainset
    eval_name = "validation" if valset else "training"

    print(f"Evaluating baseline on {eval_name} set...")
    baseline_metrics = evaluate_model(baseline, eval_dataset, metric)
    print(f"Baseline score: {baseline_metrics['mean_score']:.3f}")
    print()

    # Optimize with BootstrapFewShot
    print("=" * 80)
    print("Starting BootstrapFewShot Optimization")
    print("=" * 80)
    print()

    max_demos = args.max_demos or settings.dspy_max_bootstrapped_demos
    max_rounds = args.max_rounds or settings.dspy_max_rounds
    max_labeled_demos = max_demos  # Use same value
    max_errors = settings.dspy_max_errors

    print(f"Configuration:")
    print(f"  Metric: Composite (verification + entailment + recall)")
    print(f"  Max bootstrapped demos: {max_demos}")
    print(f"  Max labeled demos: {max_labeled_demos}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Max errors: {max_errors}")
    print()
    print("This may take 10-30 minutes depending on data size...")
    print()

    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        max_errors=max_errors
    )

    try:
        optimized = optimizer.compile(
            baseline,
            trainset=trainset
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        print("Falling back to baseline model")
        optimized = baseline

    print()
    print("Optimization complete!")
    print()

    # Evaluate optimized model
    print(f"Evaluating optimized model on {eval_name} set...")
    optimized_metrics = evaluate_model(optimized, eval_dataset, metric)
    print(f"Optimized score: {optimized_metrics['mean_score']:.3f}")
    print(f"Improvement: {optimized_metrics['mean_score'] - baseline_metrics['mean_score']:+.3f}")
    print()

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    optimized.find_quotes.save(str(output_path))
    print(f"Model saved to {args.output}")
    print()

    # Print few-shot examples count
    if hasattr(optimized.find_quotes, 'demos') and optimized.find_quotes.demos:
        demos = getattr(optimized.find_quotes, 'demos', [])
        print(f"Model has {len(demos)} few-shot examples")
    else:
        print("Model has no few-shot examples (zero-shot)")
    print()

    # Print example predictions
    if eval_dataset:
        print("=" * 80)
        print("Example Predictions (first 3 examples)")
        print("=" * 80)
        print()

        for i, example in enumerate(eval_dataset[:3], 1):
            try:
                pred = optimized(claim=example.claim, transcript_chunks=example.transcript_chunks)

                print(f"Example {i}:")
                print(f"  Claim: {example.claim}")
                print(f"  Gold quotes: {len(example.gold_quotes)}")
                print(f"  Predicted quotes: {len(pred.quotes) if hasattr(pred, 'quotes') and isinstance(pred.quotes, list) else 0}")

                if hasattr(pred, 'quotes') and isinstance(pred.quotes, list):
                    for j, quote_data in enumerate(pred.quotes[:2], 1):  # Show first 2
                        if isinstance(quote_data, dict) and 'text' in quote_data:
                            print(f"    Quote {j}: {quote_data['text'][:80]}...")
                print()
            except Exception as e:
                print(f"Example {i}: Error generating prediction: {e}")
                print()

    # Print summary
    print("=" * 80)
    print("Training Summary")
    print("=" * 80)
    print()
    print(f"Dataset: {len(trainset)} training examples")
    if valset:
        print(f"         {len(valset)} validation examples")
    print()
    print(f"Baseline score: {baseline_metrics['mean_score']:.3f}")
    print(f"Optimized score: {optimized_metrics['mean_score']:.3f}")
    print(f"Improvement: {optimized_metrics['mean_score'] - baseline_metrics['mean_score']:+.3f}")
    print()
    print(f"Model saved to: {args.output}")
    print()

    # Check if we met goals
    if optimized_metrics['mean_score'] >= 0.80:
        print("Goal achieved: Score >= 0.80 (Excellent)")
    elif optimized_metrics['mean_score'] >= 0.60:
        print("Goal partially met: Score >= 0.60 (Good)")
    else:
        print(f"Warning: Score {optimized_metrics['mean_score']:.3f} < 0.60 (needs improvement)")

    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
