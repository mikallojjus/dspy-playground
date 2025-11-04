"""
Compare BootstrapFewShot vs MIPROv2 optimization results.

This script loads both optimized models and evaluates them side-by-side
to help you decide which optimizer is better for your use case.

Usage:
    # Compare claim extraction optimizers
    python -m src.evaluation.compare_optimizers --task claims

    # Compare entailment validation optimizers
    python -m src.evaluation.compare_optimizers --task entailment

Optional arguments:
    --task TEXT              Task to compare: claims or entailment (required)
    --val-path PATH          Path to validation dataset (auto-detected if not provided)
    --bootstrap-model PATH   Path to BootstrapFewShot model (auto-detected if not provided)
    --mipro-model PATH       Path to MIPROv2 model (auto-detected if not provided)
"""

import dspy
from dspy.evaluate import Evaluate
import json
import sys
import argparse
from pathlib import Path
from typing import List, Literal

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.config.settings import settings


class ClaimExtraction(dspy.Signature):
    """Extract factual, verifiable claims from podcast transcript text."""
    transcript_chunk: str = dspy.InputField(desc="The podcast transcript text to analyze")
    claims: List[str] = dspy.OutputField(desc="List of factual claims extracted from the transcript")


class EntailmentValidation(dspy.Signature):
    """Validate whether a quote supports a claim."""
    claim: str = dspy.InputField(desc="The claim to validate")
    quote: str = dspy.InputField(desc="The quote to check for support")
    relationship: Literal["SUPPORTS", "RELATED", "NEUTRAL", "CONTRADICTS"] = dspy.OutputField(
        desc="The relationship between quote and claim"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the relationship")
    confidence: float = dspy.OutputField(desc="Confidence score (0.0-1.0)")


def load_claims_dataset(filepath):
    """Load claims dataset and convert to DSPy examples."""
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
        else:
            example = dspy.Example(
                transcript_chunk=item['transcript_chunk'],
                claims=[]
            ).with_inputs('transcript_chunk')
            examples.append(example)

    return examples


def load_entailment_dataset(filepath):
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


def compare_claims(valset, bootstrap_model, mipro_model, metric):
    """Compare claim extraction models."""
    evaluator = Evaluate(
        devset=valset,
        metric=metric,
        num_threads=1,
        display_progress=True
    )

    print("Evaluating BootstrapFewShot model...")
    bootstrap_score = evaluator(bootstrap_model)

    print()
    print("Evaluating MIPROv2 model...")
    mipro_score = evaluator(mipro_model)

    return float(bootstrap_score), float(mipro_score)


def compare_entailment(valset, bootstrap_model, mipro_model, metric, calculate_metrics_func):
    """Compare entailment validation models."""
    evaluator = Evaluate(
        devset=valset,
        metric=metric,
        num_threads=1,
        display_progress=True
    )

    print("Evaluating BootstrapFewShot model...")
    bootstrap_score = evaluator(bootstrap_model)
    bootstrap_metrics = calculate_metrics_func(
        valset,
        lambda ex: bootstrap_model(claim=ex.claim, quote=ex.quote)
    )

    print()
    print("Evaluating MIPROv2 model...")
    mipro_score = evaluator(mipro_model)
    mipro_metrics = calculate_metrics_func(
        valset,
        lambda ex: mipro_model(claim=ex.claim, quote=ex.quote)
    )

    return float(bootstrap_score), float(mipro_score), bootstrap_metrics, mipro_metrics


def main():
    parser = argparse.ArgumentParser(description='Compare BootstrapFewShot vs MIPROv2')
    parser.add_argument('--task', required=True, choices=['claims', 'entailment'],
                        help='Task to compare: claims or entailment')
    parser.add_argument('--val-path', default=None,
                        help='Path to validation dataset (auto-detected if not provided)')
    parser.add_argument('--bootstrap-model', default=None,
                        help='Path to BootstrapFewShot model (auto-detected if not provided)')
    parser.add_argument('--mipro-model', default=None,
                        help='Path to MIPROv2 model (auto-detected if not provided)')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Optimizer Comparison: BootstrapFewShot vs MIPROv2")
    print(f"Task: {args.task.upper()}")
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

    # Auto-detect paths if not provided
    if args.task == 'claims':
        val_path = args.val_path or 'evaluation/claims_val.json'
        bootstrap_model_path = args.bootstrap_model or 'models/claim_extractor_llm_judge_v1.json'
        mipro_model_path = args.mipro_model or 'models/claim_extractor_mipro_v1.json'
        signature = ClaimExtraction

        from src.metrics_llm_judge import llm_judge_metric as metric

    else:  # entailment
        val_path = args.val_path or 'evaluation/entailment_val.json'
        bootstrap_model_path = args.bootstrap_model or 'models/entailment_validator_v1.json'
        mipro_model_path = args.mipro_model or 'models/entailment_validator_mipro_v1.json'
        signature = EntailmentValidation

        from src.metrics.entailment_metrics import entailment_llm_judge_metric as metric
        from src.metrics.entailment_metrics import calculate_entailment_metrics

    # Check if models exist
    if not Path(bootstrap_model_path).exists():
        print(f"❌ BootstrapFewShot model not found: {bootstrap_model_path}")
        print(f"   Run: python -m src.training.train_{args.task.rstrip('s')}_extractor" if args.task == 'claims'
              else f"   Run: python -m src.training.train_entailment_validator")
        sys.exit(1)

    if not Path(mipro_model_path).exists():
        print(f"❌ MIPROv2 model not found: {mipro_model_path}")
        print(f"   Run: python -m src.training.train_{args.task.rstrip('s')}_extractor_mipro" if args.task == 'claims'
              else f"   Run: python -m src.training.train_entailment_validator_mipro")
        sys.exit(1)

    # Load validation dataset
    print(f"Loading validation dataset: {val_path}")
    if args.task == 'claims':
        valset = load_claims_dataset(val_path)
    else:
        valset = load_entailment_dataset(val_path)
    print(f"Loaded {len(valset)} validation examples")
    print()

    # Load models
    print("Loading BootstrapFewShot model...")
    bootstrap_model = dspy.ChainOfThought(signature)
    bootstrap_model.load(bootstrap_model_path)

    if hasattr(bootstrap_model, 'demos') and bootstrap_model.demos:
        print(f"  - Has {len(bootstrap_model.demos)} few-shot examples")

    print()
    print("Loading MIPROv2 model...")
    mipro_model = dspy.ChainOfThought(signature)
    mipro_model.load(mipro_model_path)

    if hasattr(mipro_model, 'demos') and mipro_model.demos:
        print(f"  - Has {len(mipro_model.demos)} few-shot examples")

    print()
    print("=" * 80)
    print("Running Evaluations")
    print("=" * 80)
    print()

    # Compare models
    if args.task == 'claims':
        bootstrap_score, mipro_score = compare_claims(
            valset, bootstrap_model, mipro_model, metric
        )

        # Print results
        print()
        print("=" * 80)
        print("RESULTS: Claim Extraction")
        print("=" * 80)
        print()
        print(f"BootstrapFewShot quality score: {bootstrap_score:.3f} ({(1-bootstrap_score)*100:.1f}% issues)")
        print(f"MIPROv2 quality score:          {mipro_score:.3f} ({(1-mipro_score)*100:.1f}% issues)")
        print()
        print(f"Absolute improvement: {mipro_score - bootstrap_score:+.3f}")
        print(f"Relative improvement: {(mipro_score - bootstrap_score) / bootstrap_score * 100:+.1f}%")
        print()

        # Recommendation
        if mipro_score > bootstrap_score + 0.05:
            print("✓ RECOMMENDATION: Use MIPROv2 (significant improvement)")
        elif mipro_score > bootstrap_score + 0.02:
            print("⚠ RECOMMENDATION: MIPROv2 shows modest improvement")
            print("  Consider: Is +{:.1f}% worth the extra training time?".format(
                (mipro_score - bootstrap_score) * 100
            ))
        elif mipro_score > bootstrap_score:
            print("⚠ RECOMMENDATION: Marginal improvement")
            print("  Stick with BootstrapFewShot (faster iteration)")
        else:
            print("❌ RECOMMENDATION: No improvement")
            print("  BootstrapFewShot is sufficient for this task")

    else:  # entailment
        bootstrap_score, mipro_score, bootstrap_metrics, mipro_metrics = compare_entailment(
            valset, bootstrap_model, mipro_model, metric, calculate_entailment_metrics
        )

        # Print results
        print()
        print("=" * 80)
        print("RESULTS: Entailment Validation")
        print("=" * 80)
        print()
        print(f"BootstrapFewShot:")
        print(f"  Overall score:         {bootstrap_score:.3f}")
        print(f"  Accuracy:              {bootstrap_metrics['accuracy']:.1%}")
        print(f"  False Positive Rate:   {bootstrap_metrics['false_positive_rate']:.1%}")
        print()
        print(f"MIPROv2:")
        print(f"  Overall score:         {mipro_score:.3f}")
        print(f"  Accuracy:              {mipro_metrics['accuracy']:.1%}")
        print(f"  False Positive Rate:   {mipro_metrics['false_positive_rate']:.1%}")
        print()
        print(f"Improvements:")
        print(f"  Overall score:         {mipro_score - bootstrap_score:+.3f}")
        print(f"  Accuracy:              {mipro_metrics['accuracy'] - bootstrap_metrics['accuracy']:+.1%}")
        print(f"  False Positive Rate:   {mipro_metrics['false_positive_rate'] - bootstrap_metrics['false_positive_rate']:+.1%}")
        print()

        # Recommendation
        fp_improvement = bootstrap_metrics['false_positive_rate'] - mipro_metrics['false_positive_rate']

        if mipro_metrics['false_positive_rate'] < 0.10 and bootstrap_metrics['false_positive_rate'] >= 0.10:
            print("✓ RECOMMENDATION: Use MIPROv2 (achieved <10% false positive goal)")
        elif fp_improvement > 0.10:
            print("✓ RECOMMENDATION: Use MIPROv2 (significant FP reduction)")
        elif fp_improvement > 0.05:
            print("⚠ RECOMMENDATION: MIPROv2 shows improvement")
            print(f"  FP rate reduced by {fp_improvement:.1%}")
        elif mipro_score > bootstrap_score:
            print("⚠ RECOMMENDATION: Marginal improvement")
            print("  Consider if worth the extra training time")
        else:
            print("❌ RECOMMENDATION: No improvement")
            print("  BootstrapFewShot is sufficient")

    print()


if __name__ == '__main__':
    main()
