"""
Train/retrain the ad classification model.

This script:
1. Loads ad classification train/val datasets
2. Uses BootstrapFewShot optimizer with LLM-as-judge metric
3. Saves optimized model to models/ad_classifier_TIMESTAMP/ folder
4. Evaluates on validation set

Each training run creates a timestamped folder containing:
  - model.json (DSPy model)
  - results.json (training metrics)

Usage:
    python -m src.training.train_ad_classifier

Optional arguments:
    --train-path PATH     Path to training dataset (default: evaluation/ad_train.json)
    --val-path PATH       Path to validation dataset (default: evaluation/ad_val.json)
    --output PATH         Custom path to save model (default: auto-generated folder)
    --max-demos INT       Max bootstrapped demos (default: 4)
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.metrics.ad_metrics import ad_classification_llm_judge_metric
from src.config.settings import settings
from src.infrastructure.logger import get_logger
from src.training.training_utils import (
    generate_model_filename,
    save_training_results,
    format_metric_comparison,
    format_duration,
)

logger = get_logger(__name__)


class AdClassification(dspy.Signature):
    """
    Determine if a claim is advertisement/promotional content.

    Advertisement claims include:
    - Product or service promotions
    - Discount codes or special offers
    - Sponsor mentions or endorsements
    - Calls to action for commercial products
    - Affiliate links or referral codes

    Content claims include:
    - Factual statements about topics discussed
    - Guest opinions or expert insights
    - Historical facts or data points
    - Industry news or analysis
    - Technical explanations or tutorials

    Examples of ADVERTISEMENT claims:
    - "Use code BANKLESS for 20% off Athletic Greens"
    - "Athletic Greens contains 75 vitamins and minerals"
    - "Visit athleticgreens.com/bankless for a special offer"
    - "Today's episode is sponsored by Ledger"

    Examples of CONTENT claims:
    - "Ethereum's merge reduced energy consumption by 99%"
    - "Bitcoin reached $69,000 in November 2021"
    - "Layer 2 solutions improve transaction throughput"
    - "Mike Neuder thinks the Ethereum roadmap is on track"

    Output format instructions:
    - For is_advertisement: respond with exactly "True" or "False" (case-sensitive)
    - For confidence: respond with a decimal number between 0.0 and 1.0
    """

    claim_text: str = dspy.InputField(desc="The claim to classify")
    is_advertisement: str = dspy.OutputField(desc="'True' if claim is promotional content, 'False' otherwise")
    confidence: str = dspy.OutputField(desc="Classification confidence as string decimal (0.0-1.0)")


class AdClassifier(dspy.Module):
    """Wrapper module that handles string-to-typed-value parsing."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(AdClassification)

    def forward(self, claim_text):
        # Get raw prediction with string outputs
        raw_pred = self.predictor(claim_text=claim_text)

        # Log raw outputs for debugging (only if values are None)
        if raw_pred.is_advertisement is None or raw_pred.confidence is None:
            logger.debug(
                f"Raw prediction has None values: "
                f"is_advertisement={raw_pred.is_advertisement}, "
                f"confidence={raw_pred.confidence}"
            )

        # Parse string outputs to proper types
        is_advertisement = self._parse_bool(raw_pred.is_advertisement)
        confidence = self._parse_float(raw_pred.confidence)

        # Return new prediction with parsed values
        return dspy.Prediction(
            is_advertisement=is_advertisement,
            confidence=confidence,
            reasoning=raw_pred.reasoning if hasattr(raw_pred, 'reasoning') else None
        )

    def _parse_bool(self, value):
        """Parse string to bool, with fallback."""
        if value is None:
            logger.warning("is_advertisement is None, defaulting to False")
            return False

        value_str = str(value).strip().lower()
        if value_str in ['true', '1', 'yes']:
            return True
        elif value_str in ['false', '0', 'no']:
            return False
        else:
            logger.warning(f"Could not parse bool from '{value}', defaulting to False")
            return False

    def _parse_float(self, value):
        """Parse string to float, with fallback."""
        if value is None:
            logger.warning("confidence is None, defaulting to 0.5")
            return 0.5

        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse float from '{value}', defaulting to 0.5")
            return 0.5


def load_dataset(filepath):
    """Load ad classification dataset and convert to DSPy examples."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []
    for item in data['examples']:
        example = dspy.Example(
            claim_text=item['claim_text'],
            is_advertisement=item['is_advertisement']
        ).with_inputs('claim_text')
        examples.append(example)

    return examples


def print_metrics(dataset, dataset_name):
    """Print distribution statistics for a dataset."""
    ad_count = sum(1 for ex in dataset if ex.is_advertisement)
    content_count = len(dataset) - ad_count

    print(f"{dataset_name}: {len(dataset)} examples")
    print(f"  - Advertisements: {ad_count} ({ad_count/len(dataset)*100:.1f}%)")
    print(f"  - Content: {content_count} ({content_count/len(dataset)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Train ad classification model')
    parser.add_argument('--train-path', default='evaluation/ad_train.json',
                        help='Path to training dataset')
    parser.add_argument('--val-path', default='evaluation/ad_val.json',
                        help='Path to validation dataset')
    parser.add_argument('--output', default=None,
                        help='Path to save trained model (default: auto-generated timestamp-based filename)')
    parser.add_argument('--max-demos', type=int, default=4,
                        help='Maximum bootstrapped demos')

    args = parser.parse_args()

    # Generate timestamp-based filenames or use provided output path
    if args.output is None:
        model_path, results_path = generate_model_filename("ad_classifier")
        print("=" * 80)
        print("Ad Classification Model Training")
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
        print("Ad Classification Model Training")
        print("=" * 80)
        print()
        print(f"Using custom output path:")
        print(f"  Model: {model_path}")
        print(f"  Results: {results_path}")
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
    baseline = AdClassifier()
    print()

    # Evaluate baseline
    print("Evaluating baseline on validation set...")
    evaluator = Evaluate(
        devset=valset,
        metric=ad_classification_llm_judge_metric,
        num_threads=1,
        display_progress=True
    )

    baseline_score = evaluator(baseline)
    baseline_score_value = float(baseline_score)
    print(f"Baseline accuracy score: {baseline_score_value:.3f}")
    print()

    # Optimize with BootstrapFewShot
    print("=" * 80)
    print("Starting BootstrapFewShot Optimization")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Metric: LLM-as-judge (ad classification evaluation)")
    print(f"  Max bootstrapped demos: {args.max_demos}")
    print(f"  Max labeled demos: {args.max_demos}")
    print(f"  Training on: {len(trainset)} examples")
    print()

    # Track training time
    training_start_time = time.time()
    training_start_timestamp = datetime.now()

    optimizer = BootstrapFewShot(
        metric=ad_classification_llm_judge_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_demos
    )

    optimized = optimizer.compile(
        baseline,
        trainset=trainset
    )

    training_duration = time.time() - training_start_time

    print()
    print(f"Optimization complete! (took {format_duration(training_duration)})")
    print()

    # Evaluate optimized model
    print("Evaluating optimized model on validation set...")
    optimized_score = evaluator(optimized)
    optimized_score_value = float(optimized_score)
    print(f"Optimized accuracy score: {optimized_score_value:.3f}")
    print(f"Improvement: {optimized_score_value - baseline_score_value:+.3f}")
    print()

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(model_path))
    print(f"✓ Model saved to {model_path}")

    # Get few-shot examples count
    few_shot_count = len(optimized.demos) if hasattr(optimized, 'demos') and optimized.demos else 0
    if few_shot_count > 0:
        print(f"  Model has {few_shot_count} few-shot examples")

    # Save comprehensive results
    improvement = optimized_score_value - baseline_score_value
    targets_met = optimized_score_value > 0.90

    results = {
        "model_path": str(model_path),
        "model_name": model_path.name,
        "timestamp": training_start_timestamp.isoformat(),
        "model_type": "ad_classifier",
        "optimizer": "BootstrapFewShot",
        "config": {
            "max_demos": args.max_demos,
            "train_path": args.train_path,
            "val_path": args.val_path,
            "train_size": len(trainset),
            "val_size": len(valset),
        },
        "baseline": {
            "score": baseline_score_value,
            "accuracy": baseline_score_value,
        },
        "optimized": {
            "score": optimized_score_value,
            "accuracy": optimized_score_value,
        },
        "improvement": {
            "score": improvement,
            "accuracy": improvement,
        },
        "training_time_seconds": training_duration,
        "few_shot_demos": few_shot_count,
        "targets_met": targets_met,
        "target_accuracy": 0.90,
    }

    save_training_results(results_path, results)
    print(f"✓ Results saved to {results_path}")
    print()

    # Print example predictions
    print("=" * 80)
    print("Example Predictions (first 5 validation examples)")
    print("=" * 80)
    print()

    for i, example in enumerate(valset[:5], 1):
        pred = optimized(claim_text=example.claim_text)

        actual_label = "AD" if example.is_advertisement else "CONTENT"
        predicted_label = "AD" if pred.is_advertisement else "CONTENT"
        match_icon = "✓" if example.is_advertisement == pred.is_advertisement else "✗"

        print(f"Example {i}: {match_icon}")
        print(f"  Claim: {example.claim_text[:80]}...")
        print(f"  Ground truth: {actual_label}")
        print(f"  Predicted: {predicted_label} (confidence: {pred.confidence:.2f})")
        print()

    # Print summary
    print("=" * 80)
    print("Training Summary")
    print("=" * 80)
    print()
    print(f"Accuracy: {format_metric_comparison(baseline_score_value, optimized_score_value)}")
    print(f"Training Time: {format_duration(training_duration)}")
    print(f"Few-shot Demos: {few_shot_count}")
    print()
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    print()

    # Check if we met goals
    if targets_met:
        print("✓ Goal achieved: Accuracy score > 0.90")
    else:
        print(f"⚠ Goal not met: Accuracy score {optimized_score_value:.3f} (target: >0.90)")

    print()
    print("To compare all trained models, run:")
    print("  uv run python -m src.cli.compare_models --model-type ad_classifier")
    print()


if __name__ == '__main__':
    main()
