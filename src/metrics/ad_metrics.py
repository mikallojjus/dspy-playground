"""
Ad classification metrics for DSPy optimization.

Provides LLM-as-judge metric for evaluating ad classification quality.
Focus: Accurately distinguishing advertisement claims from content claims.

Usage:
    from src.metrics.ad_metrics import ad_classification_llm_judge_metric

    score = ad_classification_llm_judge_metric(example, prediction)
    # Returns 1.0 for correct, 0.0 for incorrect
"""

import dspy

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class AdClassificationJudge(dspy.Signature):
    """
    Evaluate if an ad classification prediction is correct.

    This judge evaluates whether a claim was correctly classified as
    advertisement or content.

    Advertisement claims include:
    - Product or service promotions (explicit sales pitches)
    - Discount codes or special offers
    - Sponsor mentions with calls to action
    - Affiliate links or referral codes
    - Promotional endorsements with commercial intent

    Content claims include:
    - Factual statements about topics discussed in the podcast
    - Guest opinions or expert insights
    - Historical facts or data points
    - Industry news or analysis
    - Technical explanations

    Examples of ADVERTISEMENT claims:
    - "Use code BANKLESS for 20% off Athletic Greens"
      → Clear promotional code and discount offer
    - "Visit athleticgreens.com/bankless to get your first order"
      → Call to action with referral link
    - "Today's episode is sponsored by Ledger"
      → Explicit sponsor mention
    - "Athletic Greens contains 75 vitamins and minerals"
      → Product feature claim from sponsor segment

    Examples of CONTENT claims:
    - "Ethereum's merge reduced energy consumption by 99%"
      → Technical fact about Ethereum
    - "Bitcoin reached $69,000 in November 2021"
      → Historical market data
    - "Mike Neuder thinks the Ethereum roadmap is on track"
      → Guest opinion about technical topic
    - "Layer 2 solutions improve transaction throughput"
      → Educational content

    Edge cases:
    - "Athletic Greens contains 75 vitamins" could be ad or content
      → Consider context: if discussing health products generally = CONTENT
      → If part of sponsor read with promo codes = ADVERTISEMENT
    - For this judge, be conservative: mark as AD only if clearly promotional

    Be balanced: Correctly identify both ads and content.
    """

    claim_text: str = dspy.InputField(desc="The claim being classified")
    predicted_is_ad: bool = dspy.InputField(
        desc="The predicted classification (True = advertisement)"
    )
    actual_is_ad: bool = dspy.InputField(
        desc="The ground truth classification (True = advertisement)"
    )
    is_correct: bool = dspy.OutputField(
        desc="True if predicted classification matches actual classification"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why the classification is correct or incorrect"
    )


def ad_classification_llm_judge_metric(example, pred, trace=None) -> float:
    """
    Evaluate ad classification using an LLM judge.

    This metric evaluates whether the predicted ad classification matches
    the ground truth, using semantic understanding rather than keyword matching.

    Scoring:
    - Correct classification: 1.0
    - Incorrect classification: 0.0

    Args:
        example: DSPy example with ground truth (claim_text, is_advertisement)
        pred: Model prediction with predicted is_advertisement
        trace: Optional DSPy trace (unused)

    Returns:
        Score: 1.0 (correct) or 0.0 (incorrect)

    Example:
        ```python
        # Correctly classified as ad
        example = dspy.Example(
            claim_text="Use code BANKLESS for 20% off",
            is_advertisement=True
        ).with_inputs('claim_text')

        pred = dspy.Example(is_advertisement=True)
        score = ad_classification_llm_judge_metric(example, pred)
        # Returns 1.0 (correct)

        # Incorrectly classified as content
        pred_wrong = dspy.Example(is_advertisement=False)
        score = ad_classification_llm_judge_metric(example, pred_wrong)
        # Returns 0.0 (incorrect)

        # Correctly classified as content
        example_content = dspy.Example(
            claim_text="Ethereum's merge reduced energy consumption by 99%",
            is_advertisement=False
        ).with_inputs('claim_text')

        pred_correct = dspy.Example(is_advertisement=False)
        score = ad_classification_llm_judge_metric(example_content, pred_correct)
        # Returns 1.0 (correct)
        ```
    """
    # Extract values
    actual_is_ad = example.is_advertisement
    predicted_is_ad = pred.is_advertisement if hasattr(pred, 'is_advertisement') else None

    if predicted_is_ad is None:
        logger.warning("Prediction missing 'is_advertisement' field")
        return 0.0

    # Create judge
    judge = dspy.ChainOfThought(AdClassificationJudge)

    try:
        result = judge(
            claim_text=example.claim_text,
            predicted_is_ad=predicted_is_ad,
            actual_is_ad=actual_is_ad
        )

        is_correct = result.is_correct

        if is_correct:
            score = 1.0
            logger.debug(
                f"Correct classification: predicted_ad={predicted_is_ad}, "
                f"actual_ad={actual_is_ad}. Score: 1.0"
            )
        else:
            score = 0.0
            classification_type = "False positive" if predicted_is_ad and not actual_is_ad else "False negative"
            logger.debug(
                f"{classification_type}: predicted_ad={predicted_is_ad}, "
                f"actual_ad={actual_is_ad}. Score: 0.0. "
                f"Reasoning: {result.reasoning}"
            )

        return score

    except Exception as e:
        logger.error(f"Error in ad classification judge: {e}", exc_info=True)
        # Fallback to simple comparison
        return 1.0 if predicted_is_ad == actual_is_ad else 0.0


def calculate_ad_classification_metrics(predictions, ground_truth):
    """
    Calculate comprehensive ad classification metrics.

    Metrics:
    - Accuracy: Overall classification accuracy
    - Precision (AD): Of all predicted ADs, how many were correct?
    - Recall (AD): Of all actual ADs, how many were predicted?
    - False Positive Rate: Of all actual CONTENT, how many were predicted AD?
    - False Negative Rate: Of all actual AD, how many were predicted CONTENT?

    Args:
        predictions: List of predicted is_advertisement (bool)
        ground_truth: List of actual is_advertisement (bool)

    Returns:
        Dict with accuracy, precision, recall, false_positive_rate, false_negative_rate

    Example:
        ```python
        predictions = [True, False, True, False]
        ground_truth = [True, False, False, False]

        metrics = calculate_ad_classification_metrics(predictions, ground_truth)
        # {
        #     "accuracy": 0.75,  # 3/4 correct
        #     "precision_ad": 0.5,  # 1/2 predicted ADs were correct
        #     "recall_ad": 1.0,  # 1/1 actual ADs were found
        #     "false_positive_rate": 0.33,  # 1/3 CONTENT misclassified as AD
        #     "false_negative_rate": 0.0  # 0/1 AD misclassified as CONTENT
        # }
        ```
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground_truth must have same length")

    n = len(predictions)
    if n == 0:
        return {
            "accuracy": 0.0,
            "precision_ad": 0.0,
            "recall_ad": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "total_examples": 0,
        }

    # Overall accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / n

    # Calculate confusion matrix components
    true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    false_positives = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    false_negatives = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    true_negatives = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)

    # Precision and recall for AD class
    predicted_ad_count = true_positives + false_positives
    actual_ad_count = true_positives + false_negatives
    actual_content_count = true_negatives + false_positives

    precision_ad = true_positives / predicted_ad_count if predicted_ad_count > 0 else 0.0
    recall_ad = true_positives / actual_ad_count if actual_ad_count > 0 else 0.0

    # False positive/negative rates
    false_positive_rate = false_positives / actual_content_count if actual_content_count > 0 else 0.0
    false_negative_rate = false_negatives / actual_ad_count if actual_ad_count > 0 else 0.0

    logger.info(
        f"Ad classification metrics: accuracy={accuracy:.1%}, "
        f"precision={precision_ad:.1%}, recall={recall_ad:.1%}, "
        f"FP_rate={false_positive_rate:.1%}, FN_rate={false_negative_rate:.1%}"
    )

    return {
        "accuracy": accuracy,
        "precision_ad": precision_ad,
        "recall_ad": recall_ad,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "total_examples": n,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }
