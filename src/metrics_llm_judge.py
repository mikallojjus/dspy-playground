"""
LLM-as-Judge metric for claim quality evaluation.

Instead of hardcoded pattern matching, use an LLM to evaluate semantic quality.
This can understand context, pronouns with referents, and generalize to new patterns.
"""

import dspy
from typing import Optional


class ClaimQualityJudge(dspy.Signature):
    """Evaluate if a claim is high quality and self-contained.

    A HIGH-QUALITY claim must be:
    1. Self-contained - understandable without external context
    2. Specific - includes names, not just pronouns without referents
    3. Factual - not opinion or speculation
    4. Clear - no vague language that makes it unverifiable

    Examples of GOOD claims:
    - "Trump said he would build a ballroom" (Trump is named, 'he' refers to Trump)
    - "Bitcoin reached $69,000 in November 2021" (specific, verifiable)
    - "USAID has a $40 billion budget" (clear, factual)

    Examples of BAD claims:
    - "He said he would build a ballroom" (Who is 'he'? Not self-contained)
    - "The new bill will help people" (Which bill? Which people? Vague)
    - "His approval rating is 44.9%" (Whose approval? Missing context)
    - "Get your subscription today for $9.99" (Advertisement, not a claim)

    Be strict: claims must be understandable on their own.
    """

    claim: str = dspy.InputField(desc="The claim to evaluate")
    is_high_quality: bool = dspy.OutputField(desc="True if high quality, False otherwise")
    reason: str = dspy.OutputField(desc="Brief explanation of the judgment")


def llm_judge_metric(example, pred, trace=None):
    """
    Evaluate claim quality using an LLM judge.

    This approach:
    - Understands semantic meaning (pronouns with referents are OK)
    - Generalizes to new patterns (no hardcoded lists)
    - Can handle nuanced cases

    Tradeoffs:
    - Slower than pattern matching (1 LLM call per claim)
    - Uses LLM inference budget
    - Need to tune the judge's prompt

    Returns:
        float: 1.0 if all claims are high quality, 0.0 to 1.0 based on proportion
    """
    predicted_claims = pred.claims if hasattr(pred, 'claims') else []

    if not predicted_claims:
        return 0.0

    # Create judge
    judge = dspy.ChainOfThought(ClaimQualityJudge)

    high_quality_count = 0

    for claim in predicted_claims:
        result = judge(claim=claim)

        if result.is_high_quality:
            high_quality_count += 1

    quality_score = high_quality_count / len(predicted_claims)
    return quality_score


def llm_judge_metric_with_cache(example, pred, trace=None, cache: Optional[dict] = None):
    """
    LLM judge with simple caching to avoid re-judging the same claims.

    Pass a shared dict as cache across evaluations:
        cache = {}
        score = llm_judge_metric_with_cache(example, pred, cache=cache)
    """
    if cache is None:
        cache = {}

    predicted_claims = pred.claims if hasattr(pred, 'claims') else []

    if not predicted_claims:
        return 0.0

    judge = dspy.ChainOfThought(ClaimQualityJudge)

    high_quality_count = 0

    for claim in predicted_claims:
        # Check cache first
        if claim in cache:
            is_high_quality = cache[claim]
        else:
            result = judge(claim=claim)
            is_high_quality = result.is_high_quality
            cache[claim] = is_high_quality

        if is_high_quality:
            high_quality_count += 1

    quality_score = high_quality_count / len(predicted_claims)
    return quality_score


# Lightweight wrapper for backward compatibility
def claim_quality_metric_llm(example, pred, trace=None):
    """Alias for llm_judge_metric."""
    return llm_judge_metric(example, pred, trace)
