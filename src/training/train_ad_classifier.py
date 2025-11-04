"""
Train/retrain the ad classification model.

This script:
1. Loads ad classification train/val datasets
2. Uses BootstrapFewShot optimizer with LLM-as-judge metric
3. Saves optimized model to models/ad_classifier_v1.json
4. Evaluates on validation set

Usage:
    python -m src.training.train_ad_classifier

Optional arguments:
    --train-path PATH     Path to training dataset (default: evaluation/ad_train.json)
    --val-path PATH       Path to validation dataset (default: evaluation/ad_val.json)
    --output PATH         Path to save model (default: models/ad_classifier_v1.json)
    --max-demos INT       Max bootstrapped demos (default: 4)
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
import argparse
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.metrics.ad_metrics import ad_classification_llm_judge_metric
from src.config.settings import settings


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
    """

    claim_text: str = dspy.InputField(desc="The claim to classify")
    is_advertisement: bool = dspy.OutputField(desc="True if claim is promotional content")
    confidence: float = dspy.OutputField(desc="Classification confidence (0.0-1.0)")


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
    parser.add_argument('--output', default='models/ad_classifier_v1.json',
                        help='Path to save trained model')
    parser.add_argument('--max-demos', type=int, default=4,
                        help='Maximum bootstrapped demos')

    args = parser.parse_args()

    print("=" * 80)
    print("Ad Classification Model Training")
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
    baseline = dspy.ChainOfThought(AdClassification)
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

    optimizer = BootstrapFewShot(
        metric=ad_classification_llm_judge_metric,
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
    print(f"Optimized accuracy score: {optimized_score_value:.3f}")
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
    print(f"Baseline accuracy score: {baseline_score_value:.3f}")
    print(f"Optimized accuracy score: {optimized_score_value:.3f}")
    print(f"Improvement: {optimized_score_value - baseline_score_value:+.3f}")
    print()
    print(f"Model saved to: {args.output}")
    print()

    # Check if we met goals
    if optimized_score_value > 0.90:
        print("✓ Goal achieved: Accuracy score > 0.90")
    else:
        print(f"⚠ Goal not met: Accuracy score {optimized_score_value:.3f} (target: >0.90)")

    print()


if __name__ == '__main__':
    main()
