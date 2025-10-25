"""
Entailment validation metrics for DSPy optimization.

Provides LLM-as-judge metric for evaluating entailment validation quality.
Focus: Reducing false positives (RELATED misclassified as SUPPORTS).

Usage:
    from src.metrics.entailment_metrics import entailment_llm_judge_metric

    score = entailment_llm_judge_metric(example, prediction)
    # Returns 1.0 for correct, 0.0 for wrong, -2.0 for false positive
"""

import dspy
from typing import Literal

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class EntailmentQualityJudge(dspy.Signature):
    """
    Evaluate if an entailment relationship classification is correct.

    This judge evaluates whether a predicted relationship between a claim and quote
    is accurate. It's particularly strict about false positives (RELATED → SUPPORTS).

    Relationship definitions:
    - SUPPORTS: Quote directly asserts the claim or provides clear evidence that validates it
      * Quote must contain specific information that confirms the claim
      * Not just topically related - must actually validate

    - RELATED: Quote is topically related but doesn't validate or provide evidence
      * Mentions related concepts or entities
      * Provides context but not validation
      * Discusses the same topic but doesn't confirm the claim

    - NEUTRAL: Quote is unrelated or provides no evidence
      * Different topic entirely
      * No connection to the claim

    - CONTRADICTS: Quote contradicts or undermines the claim
      * States opposite information
      * Provides evidence against the claim

    Examples of correct SUPPORTS:
    - Claim: "Bitcoin reached $69,000 in November 2021"
      Quote: "Bitcoin hit its all-time high of $69,000 in November 2021"
      → SUPPORTS (directly confirms price and date)

    Examples of false positives (RELATED misclassified as SUPPORTS):
    - Claim: "Bitcoin reached $69,000 in November 2021"
      Quote: "Cryptocurrency markets were volatile in 2021"
      → RELATED, not SUPPORTS (doesn't confirm Bitcoin's specific price)

    - Claim: "Tesla acquired $1.5 billion worth of Bitcoin"
      Quote: "Many companies added Bitcoin to their balance sheets"
      → RELATED, not SUPPORTS (doesn't validate Tesla's specific purchase)

    Be strict: Only classify as SUPPORTS if the quote genuinely validates the claim.
    """

    claim: str = dspy.InputField(desc="The claim being validated")
    quote: str = dspy.InputField(desc="The quote being checked for support")
    predicted_relationship: Literal["SUPPORTS", "RELATED", "NEUTRAL", "CONTRADICTS"] = dspy.InputField(
        desc="The predicted relationship by the model"
    )
    actual_relationship: Literal["SUPPORTS", "RELATED", "NEUTRAL", "CONTRADICTS"] = dspy.InputField(
        desc="The ground truth relationship from manual labeling"
    )
    is_correct: bool = dspy.OutputField(
        desc="True if predicted relationship matches actual relationship"
    )
    is_false_positive: bool = dspy.OutputField(
        desc="True if predicted SUPPORTS but actual is RELATED (false positive)"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why the classification is correct or incorrect"
    )


def entailment_llm_judge_metric(example, pred, trace=None) -> float:
    """
    Evaluate entailment validation using an LLM judge.

    Scoring:
    - Correct classification: +1.0
    - Incorrect classification: 0.0
    - False positive (RELATED → SUPPORTS): -2.0 (heavy penalty)

    This metric is designed to guide DSPy optimization toward models that:
    1. Correctly classify all relationships
    2. Are especially strict about SUPPORTS vs RELATED distinction
    3. Avoid false positives at all costs

    Args:
        example: DSPy example with ground truth (claim, quote, relationship)
        pred: Model prediction with predicted relationship
        trace: Optional DSPy trace (unused)

    Returns:
        Score: 1.0 (correct), 0.0 (wrong), or -2.0 (false positive)

    Example:
        ```python
        example = dspy.Example(
            claim="Bitcoin reached $69,000",
            quote="Bitcoin hit $69k in November",
            relationship="SUPPORTS"
        ).with_inputs('claim', 'quote')

        pred = dspy.Example(relationship="SUPPORTS")
        score = entailment_llm_judge_metric(example, pred)
        # Returns 1.0 (correct)

        pred_wrong = dspy.Example(relationship="RELATED")
        score = entailment_llm_judge_metric(example, pred_wrong)
        # Returns 0.0 (wrong)

        # False positive scenario
        example_related = dspy.Example(
            claim="Bitcoin reached $69,000",
            quote="Crypto was volatile in 2021",
            relationship="RELATED"
        ).with_inputs('claim', 'quote')

        pred_false_positive = dspy.Example(relationship="SUPPORTS")
        score = entailment_llm_judge_metric(example_related, pred_false_positive)
        # Returns -2.0 (false positive - heavy penalty)
        ```
    """
    # Extract values
    actual_relationship = example.relationship
    predicted_relationship = pred.relationship if hasattr(pred, 'relationship') else None

    if not predicted_relationship:
        logger.warning("Prediction missing 'relationship' field")
        return 0.0

    # Create judge
    judge = dspy.ChainOfThought(EntailmentQualityJudge)

    try:
        result = judge(
            claim=example.claim,
            quote=example.quote,
            predicted_relationship=predicted_relationship,
            actual_relationship=actual_relationship
        )

        is_correct = result.is_correct
        is_false_positive = result.is_false_positive

        # Score calculation
        if is_false_positive:
            # Heavy penalty for false positives (RELATED → SUPPORTS)
            score = -2.0
            logger.debug(
                f"FALSE POSITIVE detected: predicted SUPPORTS but actual {actual_relationship}. "
                f"Score: -2.0. Reasoning: {result.reasoning}"
            )
        elif is_correct:
            # Correct classification
            score = 1.0
            logger.debug(
                f"Correct classification: {predicted_relationship}. Score: 1.0"
            )
        else:
            # Incorrect but not false positive
            score = 0.0
            logger.debug(
                f"Incorrect: predicted {predicted_relationship}, actual {actual_relationship}. "
                f"Score: 0.0. Reasoning: {result.reasoning}"
            )

        return score

    except Exception as e:
        logger.error(f"Error in entailment judge: {e}", exc_info=True)
        # Default to simple check
        return 1.0 if predicted_relationship == actual_relationship else 0.0


def calculate_entailment_metrics(predictions, ground_truth):
    """
    Calculate comprehensive entailment validation metrics.

    Metrics:
    - Accuracy: Overall classification accuracy
    - Precision (SUPPORTS): Of all predicted SUPPORTS, how many were correct?
    - Recall (SUPPORTS): Of all actual SUPPORTS, how many were predicted?
    - False Positive Rate: Of all actual RELATED, how many were predicted SUPPORTS?

    Args:
        predictions: List of predicted relationships
        ground_truth: List of actual relationships

    Returns:
        Dict with accuracy, precision, recall, false_positive_rate

    Example:
        ```python
        predictions = ["SUPPORTS", "RELATED", "SUPPORTS", "SUPPORTS"]
        ground_truth = ["SUPPORTS", "RELATED", "RELATED", "SUPPORTS"]

        metrics = calculate_entailment_metrics(predictions, ground_truth)
        # {
        #     "accuracy": 0.75,  # 3/4 correct
        #     "precision_supports": 0.67,  # 2/3 predicted SUPPORTS were correct
        #     "recall_supports": 1.0,  # 2/2 actual SUPPORTS were found
        #     "false_positive_rate": 0.5  # 1/2 RELATED were misclassified as SUPPORTS
        # }
        ```
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground_truth must have same length")

    n = len(predictions)

    # Overall accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / n if n > 0 else 0.0

    # SUPPORTS precision and recall
    predicted_supports = [i for i, p in enumerate(predictions) if p == "SUPPORTS"]
    actual_supports = [i for i, g in enumerate(ground_truth) if g == "SUPPORTS"]

    true_positives = len([i for i in predicted_supports if i in actual_supports])

    precision_supports = (
        true_positives / len(predicted_supports)
        if predicted_supports else 0.0
    )

    recall_supports = (
        true_positives / len(actual_supports)
        if actual_supports else 0.0
    )

    # False positive rate (RELATED → SUPPORTS)
    actual_related = [i for i, g in enumerate(ground_truth) if g == "RELATED"]
    false_positives = len([
        i for i in actual_related
        if predictions[i] == "SUPPORTS"
    ])

    false_positive_rate = (
        false_positives / len(actual_related)
        if actual_related else 0.0
    )

    logger.info(
        f"Entailment metrics: accuracy={accuracy:.1%}, "
        f"precision={precision_supports:.1%}, recall={recall_supports:.1%}, "
        f"FP_rate={false_positive_rate:.1%}"
    )

    return {
        "accuracy": accuracy,
        "precision_supports": precision_supports,
        "recall_supports": recall_supports,
        "false_positive_rate": false_positive_rate,
        "total_examples": n,
        "true_positives": true_positives,
        "false_positives": false_positives,
    }
