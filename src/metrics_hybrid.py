"""
Hybrid metric: Fast pattern matching + LLM judge for borderline cases.

Strategy:
1. Quick pattern checks catch obvious bad claims (fast, no LLM call)
2. LLM judge handles borderline cases (slower, more accurate)

This gives you speed + accuracy.
"""

import dspy
import re


class ClaimQualityJudge(dspy.Signature):
    """Evaluate if a claim is high quality and self-contained."""
    claim: str = dspy.InputField(desc="The claim to evaluate")
    is_high_quality: bool = dspy.OutputField(desc="True if high quality")
    reason: str = dspy.OutputField(desc="Brief explanation")


def has_obvious_issues(claim: str) -> tuple[bool, str]:
    """
    Fast pattern matching for OBVIOUS quality issues.

    Returns:
        (has_issue, reason) - True if claim has obvious problems
    """
    claim_lower = claim.lower()

    # 1. Check for advertisement patterns (very specific phrases)
    if any(phrase in claim_lower for phrase in [
        'get it today', 'order now', 'limited time offer',
        'subscribe now', 'discount code', 'act now'
    ]):
        return True, "advertisement"

    # 2. Check for question marks (claims shouldn't be questions)
    if '?' in claim:
        return True, "question"

    # 3. Check for very short claims (< 5 words, likely incomplete)
    if len(claim.split()) < 5:
        return True, "too_short"

    # 4. Check for standalone pronouns at START (obvious missing referent)
    # "He said..." or "She mentioned..." are clearly bad
    claim_stripped = claim.strip()
    if re.match(r'^(He|She|They|It|His|Her|Their)\s', claim_stripped):
        return True, "pronoun_at_start"

    # 5. Check for "the X" without specification (very vague)
    if claim_lower.startswith('the new ') or claim_lower.startswith('the ban '):
        return True, "vague_reference"

    return False, ""


def needs_llm_judgment(claim: str) -> bool:
    """
    Decide if claim needs LLM judgment.

    If it passes all quick checks, it might still be bad
    (e.g., "Trump said he would..." needs semantic understanding)
    """
    # If it has obvious issues, we don't need LLM
    has_issue, _ = has_obvious_issues(claim)
    if has_issue:
        return False

    # Check for pronouns in the middle/end (these need semantic analysis)
    claim_lower = claim.lower()
    pronouns = ['he', 'she', 'they', 'him', 'her', 'his', 'their']

    for pronoun in pronouns:
        # Look for pronoun as whole word (not part of "the", "other", etc.)
        if re.search(rf'\b{pronoun}\b', claim_lower):
            return True  # Needs LLM to check if referent exists

    # No obvious issues and no pronouns → probably good
    return False


def hybrid_metric(example, pred, trace=None):
    """
    Hybrid evaluation: Fast checks + LLM for borderline cases.

    Flow:
    1. Quick pattern check catches obvious bad claims → mark as bad (no LLM call)
    2. Claims with pronouns but no obvious issues → LLM judge (needs context understanding)
    3. Claims that pass all checks → mark as good (no LLM call)

    This is ~2-3x faster than pure LLM judge because most claims don't need LLM.

    Returns:
        float: Quality score from 0.0 to 1.0
    """
    predicted_claims = pred.claims if hasattr(pred, 'claims') else []

    if not predicted_claims:
        return 0.0

    # Lazy-init judge (only create if needed)
    judge = None

    high_quality_count = 0
    llm_calls_made = 0

    for claim in predicted_claims:
        # Step 1: Fast pattern check
        has_issue, reason = has_obvious_issues(claim)

        if has_issue:
            # Obviously bad, no need for LLM
            continue

        # Step 2: Check if needs LLM judgment
        if needs_llm_judgment(claim):
            # Create judge on first use
            if judge is None:
                judge = dspy.ChainOfThought(ClaimQualityJudge)

            # Use LLM for borderline case
            result = judge(claim=claim)
            llm_calls_made += 1

            if result.is_high_quality:
                high_quality_count += 1
        else:
            # Passed all checks and no borderline issues
            high_quality_count += 1

    quality_score = high_quality_count / len(predicted_claims)

    # Debug info
    if trace:
        trace['llm_calls_made'] = llm_calls_made
        trace['total_claims'] = len(predicted_claims)
        trace['llm_call_rate'] = llm_calls_made / len(predicted_claims) if predicted_claims else 0

    return quality_score
