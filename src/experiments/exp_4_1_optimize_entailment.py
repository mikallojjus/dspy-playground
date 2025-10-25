"""
Experiment 4.1: Optimize Entailment Validation with BootstrapFewShot

Goal: Optimize entailment validator to reduce false positive rate (<10%)

This experiment:
1. Loads entailment train/val datasets
2. Uses LLM-as-judge metric (heavy penalty for false positives)
3. Runs BootstrapFewShot optimization
4. Saves optimized model to models/entailment_validator_v1.json
5. Evaluates on validation set

Expected outcome: <10% false positive rate, >90% accuracy
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
import json
import sys
from typing import Literal
import time

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Import the metric
from src.metrics.entailment_metrics import entailment_llm_judge_metric, calculate_entailment_metrics
from src.config.settings import settings


# Entailment validation signature (already defined in entailment_validator.py)
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


def main():
    print("=" * 80)
    print("Experiment 4.1: Optimize Entailment Validation")
    print("=" * 80)
    print()
    print("Goal: <10% false positive rate, >90% accuracy")
    print("Method: BootstrapFewShot with LLM-as-judge metric")
    print()

    # Load datasets
    print("Loading datasets...")
    trainset = load_dataset('evaluation/entailment_train.json')
    valset = load_dataset('evaluation/entailment_val.json')

    print(f"Training set: {len(trainset)} examples")
    print(f"Validation set: {len(valset)} examples")
    print()

    # Count distribution
    for dataset_name, dataset in [('Train', trainset), ('Val', valset)]:
        supports = sum(1 for ex in dataset if ex.relationship == 'SUPPORTS')
        related = sum(1 for ex in dataset if ex.relationship == 'RELATED')
        neutral = sum(1 for ex in dataset if ex.relationship == 'NEUTRAL')
        contradicts = sum(1 for ex in dataset if ex.relationship == 'CONTRADICTS')
        print(
            f"{dataset_name}: SUPPORTS={supports}, RELATED={related}, "
            f"NEUTRAL={neutral}, CONTRADICTS={contradicts}"
        )
    print()

    # Configure DSPy
    print("Configuring DSPy with Ollama...")
    lm = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url
    )
    dspy.configure(lm=lm)
    print("Done")
    print()

    # Create baseline model
    baseline = dspy.ChainOfThought(EntailmentValidation)

    # Evaluate baseline
    print("=" * 80)
    print("BASELINE EVALUATION")
    print("=" * 80)
    print()
    print("Evaluating baseline (zero-shot) model on validation set...")
    print("This will take a few minutes due to LLM judge calls...")
    print()

    evaluator = Evaluate(
        devset=valset,
        metric=entailment_llm_judge_metric,
        display_progress=True,
        display_table=0
    )

    baseline_start = time.time()
    baseline_result = evaluator(baseline)
    baseline_time = time.time() - baseline_start

    # Extract score from result object (it's already a percentage 0-100)
    baseline_score = float(baseline_result) / 100.0 if float(baseline_result) > 1.0 else float(baseline_result)

    print()
    print(f"Baseline score: {baseline_score:.1%}")
    print(f"Time: {baseline_time:.1f}s")
    print()

    # Collect predictions for detailed metrics
    print("Collecting baseline predictions for detailed metrics...")
    baseline_predictions = []
    baseline_ground_truth = []

    for example in valset:
        pred = baseline(claim=example.claim, quote=example.quote)
        baseline_predictions.append(pred.relationship)
        baseline_ground_truth.append(example.relationship)

    baseline_metrics = calculate_entailment_metrics(baseline_predictions, baseline_ground_truth)

    print("Baseline detailed metrics:")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.1%}")
    print(f"  Precision (SUPPORTS): {baseline_metrics['precision_supports']:.1%}")
    print(f"  Recall (SUPPORTS): {baseline_metrics['recall_supports']:.1%}")
    print(f"  FALSE POSITIVE RATE: {baseline_metrics['false_positive_rate']:.1%}")
    print()

    if baseline_metrics['false_positive_rate'] < 0.10:
        print("✅ Baseline already meets target (<10% FP rate)!")
        print("   But let's optimize anyway to improve overall accuracy.")
    else:
        print(f"❌ Baseline FP rate: {baseline_metrics['false_positive_rate']:.1%} (target: <10%)")
        print("   Optimization needed!")
    print()

    # Optimize with BootstrapFewShot
    print("=" * 80)
    print("OPTIMIZATION")
    print("=" * 80)
    print()
    print("Running BootstrapFewShot optimization...")
    print("This will take 10-20 minutes depending on dataset size...")
    print()

    optimizer = BootstrapFewShot(
        metric=entailment_llm_judge_metric,
        max_bootstrapped_demos=4,  # 4-6 few-shot examples
        max_labeled_demos=4,
        max_rounds=1
    )

    optimize_start = time.time()
    optimized_model = optimizer.compile(
        student=baseline,
        trainset=trainset
    )
    optimize_time = time.time() - optimize_start

    print()
    print(f"✅ Optimization complete! ({optimize_time:.1f}s)")
    print()

    # Evaluate optimized model
    print("=" * 80)
    print("OPTIMIZED MODEL EVALUATION")
    print("=" * 80)
    print()
    print("Evaluating optimized model on validation set...")
    print()

    eval_start = time.time()
    optimized_result = evaluator(optimized_model)
    eval_time = time.time() - eval_start

    # Extract score from result object (it's already a percentage 0-100)
    optimized_score = float(optimized_result) / 100.0 if float(optimized_result) > 1.0 else float(optimized_result)

    print()
    print(f"Optimized score: {optimized_score:.1%}")
    print(f"Time: {eval_time:.1f}s")
    print()

    # Collect predictions for detailed metrics
    print("Collecting optimized predictions for detailed metrics...")
    optimized_predictions = []
    optimized_ground_truth = []

    for example in valset:
        pred = optimized_model(claim=example.claim, quote=example.quote)
        optimized_predictions.append(pred.relationship)
        optimized_ground_truth.append(example.relationship)

    optimized_metrics = calculate_entailment_metrics(optimized_predictions, optimized_ground_truth)

    print("Optimized detailed metrics:")
    print(f"  Accuracy: {optimized_metrics['accuracy']:.1%}")
    print(f"  Precision (SUPPORTS): {optimized_metrics['precision_supports']:.1%}")
    print(f"  Recall (SUPPORTS): {optimized_metrics['recall_supports']:.1%}")
    print(f"  FALSE POSITIVE RATE: {optimized_metrics['false_positive_rate']:.1%}")
    print()

    # Compare results
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()

    improvement = optimized_score - baseline_score
    fp_improvement = baseline_metrics['false_positive_rate'] - optimized_metrics['false_positive_rate']

    print(f"Overall Score:")
    print(f"  Baseline:  {baseline_score:.1%}")
    print(f"  Optimized: {optimized_score:.1%}")
    print(f"  Change:    {improvement:+.1%}")
    print()

    print(f"False Positive Rate:")
    print(f"  Baseline:  {baseline_metrics['false_positive_rate']:.1%}")
    print(f"  Optimized: {optimized_metrics['false_positive_rate']:.1%}")
    print(f"  Change:    {-fp_improvement:+.1%} (lower is better)")
    print()

    print(f"Accuracy:")
    print(f"  Baseline:  {baseline_metrics['accuracy']:.1%}")
    print(f"  Optimized: {optimized_metrics['accuracy']:.1%}")
    print(f"  Change:    {optimized_metrics['accuracy'] - baseline_metrics['accuracy']:+.1%}")
    print()

    # Check if targets met
    targets_met = []
    targets_failed = []

    if optimized_metrics['false_positive_rate'] < 0.10:
        targets_met.append("✅ False positive rate <10%")
    else:
        targets_failed.append(
            f"❌ False positive rate {optimized_metrics['false_positive_rate']:.1%} "
            f"(target: <10%)"
        )

    if optimized_metrics['accuracy'] >= 0.90:
        targets_met.append("✅ Accuracy ≥90%")
    else:
        targets_failed.append(
            f"⚠️  Accuracy {optimized_metrics['accuracy']:.1%} "
            f"(target: ≥90%)"
        )

    if targets_met:
        print("Targets Met:")
        for target in targets_met:
            print(f"  {target}")
        print()

    if targets_failed:
        print("Targets Not Met:")
        for target in targets_failed:
            print(f"  {target}")
        print()

    # Save optimized model
    model_path = 'models/entailment_validator_v1.json'
    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    print()
    print(f"Saving optimized model to {model_path}...")

    optimized_model.save(model_path)

    print("✅ Model saved!")
    print()

    # Check few-shot examples
    if hasattr(optimized_model, 'demos') and optimized_model.demos:
        print(f"Model includes {len(optimized_model.demos)} few-shot examples")
        print()
        print("Example demonstrations:")
        for i, demo in enumerate(optimized_model.demos[:3], 1):
            print(f"{i}. Claim: {demo.claim[:60]}...")
            print(f"   Quote: {demo.quote[:60]}...")
            print(f"   Relationship: {demo.relationship}")
            print()

    # Save results
    results = {
        "experiment": "exp_4_1_optimize_entailment",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "baseline": {
            "score": baseline_score,
            "accuracy": baseline_metrics['accuracy'],
            "precision_supports": baseline_metrics['precision_supports'],
            "recall_supports": baseline_metrics['recall_supports'],
            "false_positive_rate": baseline_metrics['false_positive_rate'],
            "time_seconds": baseline_time
        },
        "optimized": {
            "score": optimized_score,
            "accuracy": optimized_metrics['accuracy'],
            "precision_supports": optimized_metrics['precision_supports'],
            "recall_supports": optimized_metrics['recall_supports'],
            "false_positive_rate": optimized_metrics['false_positive_rate'],
            "time_seconds": eval_time
        },
        "improvement": {
            "score": improvement,
            "accuracy": optimized_metrics['accuracy'] - baseline_metrics['accuracy'],
            "false_positive_rate": -(optimized_metrics['false_positive_rate'] - baseline_metrics['false_positive_rate'])
        },
        "targets_met": len(targets_failed) == 0,
        "optimization_time_seconds": optimize_time,
        "few_shot_demos": len(optimized_model.demos) if hasattr(optimized_model, 'demos') else 0
    }

    results_path = 'results/exp_4_1_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    print()

    # Recommendation
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()

    if len(targets_failed) == 0:
        print("SUCCESS! All targets met.")
        print()
        print("Next steps:")
        print("1. Update EntailmentValidatorModel to load optimized model")
        print("2. Integrate entailment validation into ExtractionPipeline")
        print("3. Test end-to-end with real episodes")
    else:
        print("Targets not fully met. Consider:")
        print("1. Adding more training examples (especially false positives)")
        print("2. Adjusting the metric penalty for false positives")
        print("3. Trying different optimization parameters")
        print()
        print("Current model may still be usable if improvement is significant.")

    print()


if __name__ == "__main__":
    main()
