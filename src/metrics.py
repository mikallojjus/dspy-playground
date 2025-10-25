"""
DSPy metrics for evaluating claim extraction quality.

Based on patterns identified in Experiment 2.1 manual review:
- Primary issue (57%): Missing context / unclear references
- Secondary issue (28%): Pronouns (he/she/they/his/her/them)
- Third issue (36%): Vague language
- Other issues: Opinions (14%), Advertisements (14%)
"""


def claim_quality_metric(example, pred, trace=None):
    """
    Evaluate claim quality based on common issues found in manual review.

    Returns a score between 0.0 and 1.0:
    - 1.0 if no issues detected (all claims are high quality)
    - Lower scores indicate proportion of claims with quality issues

    Quality issues detected:
    1. Pronouns - claims containing he/she/they/his/her/them etc.
    2. Vague words - recently, soon, thing, stuff, someone, something
    3. Opinion words - think, believe, feel, seems, appears, looks like
    4. Advertisement indicators - offers, provides, delivers (product context)

    Args:
        example: The DSPy example (not used in this simple metric)
        pred: The prediction object containing 'claims' field
        trace: Optional trace information (not used)

    Returns:
        float: Quality score from 0.0 to 1.0
    """
    predicted_claims = pred.claims if hasattr(pred, 'claims') else []

    if not predicted_claims:
        return 0.0

    # Pronouns are the most common issue (28% of bad claims in manual review)
    # Using word boundaries to avoid false positives (e.g., "the" contains "he")
    PRONOUNS = [
        'he', 'she', 'they', 'it',
        'him', 'her', 'them', 'his', 'hers', 'their', 'its',
        "he's", "she's", "they're", "it's"
    ]

    # Vague words that indicate missing context (36% of bad claims)
    VAGUE_WORDS = [
        'recently', 'soon', 'thing', 'stuff',
        'someone', 'something', 'somehow',
        'very', 'really', 'quite'
    ]

    # Opinion indicators (14% of bad claims)
    OPINION_WORDS = [
        'think', 'believe', 'feel', 'seems', 'appears',
        'looks like', 'might', 'could', 'probably', 'maybe',
        'in my opinion', 'i believe'
    ]

    # Advertisement indicators (14% of bad claims)
    AD_WORDS = [
        'get it today', 'order now', 'limited time',
        'sign up', 'subscribe', 'discount code',
        'offers', 'provides topical', 'provides oral',
        'high-quality basics', 'for a fraction',
        'delivers directly',
        'get your', 'today for only', 'only $'
    ]

    # Missing context indicators - vague references without specifics
    MISSING_CONTEXT = [
        'disapproval is', 'approval is', 'approval has',
        'the mantra', 'there was a mantra',
        'the new policy', 'the new bill', 'the policy will', 'the bill will'
    ]

    issues_found = 0

    for claim in predicted_claims:
        claim_lower = claim.lower()
        words = claim_lower.split()

        # Check for pronouns with word boundaries to avoid partial matches
        # e.g., "the" should not match "he"
        has_pronoun = False
        for pronoun in PRONOUNS:
            # Check if pronoun appears as a complete word
            if f" {pronoun} " in f" {claim_lower} " or \
               claim_lower.startswith(f"{pronoun} ") or \
               claim_lower.endswith(f" {pronoun}"):
                has_pronoun = True
                break

        if has_pronoun:
            issues_found += 1
            continue

        # Check for vague words
        if any(vague in claim_lower for vague in VAGUE_WORDS):
            issues_found += 1
            continue

        # Check for opinion indicators
        if any(opinion in claim_lower for opinion in OPINION_WORDS):
            issues_found += 1
            continue

        # Check for advertisement language
        if any(ad_word in claim_lower for ad_word in AD_WORDS):
            issues_found += 1
            continue

        # Check for missing context indicators
        if any(context in claim_lower for context in MISSING_CONTEXT):
            issues_found += 1
            continue

    # Calculate quality score
    # Score = 1.0 - (proportion of claims with issues)
    quality_score = 1.0 - (issues_found / len(predicted_claims))

    return quality_score


def strict_claim_quality_metric(example, pred, trace=None):
    """
    Stricter version of claim quality metric.

    This version is more conservative and detects additional issues:
    - Questions (claims ending with '?')
    - Too short claims (<5 words)
    - Too long claims (>40 words)
    - Missing context indicators (words like "the ban", "the bill", "this policy")

    Use this when you want to be more rigorous in filtering claims.

    Returns:
        float: Quality score from 0.0 to 1.0
    """
    predicted_claims = pred.claims if hasattr(pred, 'claims') else []

    if not predicted_claims:
        return 0.0

    # All checks from the basic metric
    PRONOUNS = [
        'he', 'she', 'they', 'it',
        'him', 'her', 'them', 'his', 'hers', 'their', 'its',
        "he's", "she's", "they're", "it's"
    ]

    VAGUE_WORDS = [
        'recently', 'soon', 'thing', 'stuff',
        'someone', 'something', 'somehow',
        'very', 'really', 'quite'
    ]

    OPINION_WORDS = [
        'think', 'believe', 'feel', 'seems', 'appears',
        'looks like', 'might', 'could', 'probably', 'maybe'
    ]

    AD_WORDS = [
        'get it today', 'order now', 'limited time',
        'sign up', 'subscribe', 'discount code'
    ]

    # Additional strict checks
    CONTEXT_INDICATORS = [
        'the ban', 'the bill', 'this policy', 'the new',
        'the plan', 'the proposal'
    ]

    issues_found = 0

    for claim in predicted_claims:
        claim_lower = claim.lower()
        word_count = len(claim.split())

        # All basic checks
        has_pronoun = False
        for pronoun in PRONOUNS:
            if f" {pronoun} " in f" {claim_lower} " or \
               claim_lower.startswith(f"{pronoun} ") or \
               claim_lower.endswith(f" {pronoun}"):
                has_pronoun = True
                break

        if has_pronoun:
            issues_found += 1
            continue

        if any(vague in claim_lower for vague in VAGUE_WORDS):
            issues_found += 1
            continue

        if any(opinion in claim_lower for opinion in OPINION_WORDS):
            issues_found += 1
            continue

        if any(ad_word in claim_lower for ad_word in AD_WORDS):
            issues_found += 1
            continue

        # Additional strict checks
        if "?" in claim:
            issues_found += 1
            continue

        if word_count < 5:
            issues_found += 1
            continue

        if word_count > 40:
            issues_found += 1
            continue

        if any(context in claim_lower for context in CONTEXT_INDICATORS):
            issues_found += 1
            continue

    quality_score = 1.0 - (issues_found / len(predicted_claims))

    return quality_score
