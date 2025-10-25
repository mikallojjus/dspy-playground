"""
Experiment 2.1: Generate and Review Claims

Goal: Implement a two-stage pipeline where:
1. Generate claims from transcript (generation stage)
2. Review and filter claims for quality (critique stage)

This pattern should help reduce low-quality claims by having a second LLM pass
that critiques and filters the generated claims.

Expected improvement: 40% low-quality → <15% low-quality
"""

import dspy
from typing import List
from pydantic import BaseModel
import time


# Configure DSPy to use Ollama with Qwen 2.5 7B
lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
dspy.configure(lm=lm)


# Stage 1: Generate claims
class ClaimGeneration(dspy.Signature):
    """Extract all potential factual claims from podcast transcript text.

    Be comprehensive - extract all statements that could be claims.
    They will be reviewed in a second stage.
    """

    transcript: str = dspy.InputField(desc="The podcast transcript text to analyze")
    claims: List[str] = dspy.OutputField(desc="List of potential claims extracted from the transcript")


# Stage 2: Review a single claim
class ClaimReview(dspy.Signature):
    """Review a claim for quality based on strict criteria.

    A HIGH-QUALITY claim must be:
    - Factual and verifiable (not opinion, speculation, or questions)
    - Self-contained with no pronouns (must use full names, not he/she/they)
    - Concise and specific (5-40 words)
    - Based only on information explicitly stated

    LOW-QUALITY claims include:
    - Opinions, beliefs, or speculation
    - Contains pronouns (he, she, they, him, her, his, their)
    - Vague or requires external context
    - Questions or hypotheticals
    - Too short (<5 words) or too long (>40 words)
    """

    claim: str = dspy.InputField(desc="The claim to review")
    transcript_context: str = dspy.InputField(desc="Original transcript for context")
    is_high_quality: bool = dspy.OutputField(desc="True if claim meets all quality criteria, False otherwise")
    issues: str = dspy.OutputField(desc="List of specific issues found, or 'None' if high quality")


# Generate-and-Review Pipeline
class GenerateAndReviewClaims(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ClaimGeneration)
        self.review = dspy.ChainOfThought(ClaimReview)

    def forward(self, transcript: str):
        # Stage 1: Generate all potential claims
        generation_result = self.generate(transcript=transcript)
        generated_claims = generation_result.claims

        # Stage 2: Review each claim
        reviewed_claims = []
        rejected_claims = []

        for claim in generated_claims:
            review_result = self.review(
                claim=claim,
                transcript_context=transcript[:1000]  # Provide context but limit length
            )

            if review_result.is_high_quality:
                reviewed_claims.append({
                    "claim": claim,
                    "issues": review_result.issues
                })
            else:
                rejected_claims.append({
                    "claim": claim,
                    "issues": review_result.issues
                })

        return dspy.Prediction(
            generated_claims=generated_claims,
            reviewed_claims=[c["claim"] for c in reviewed_claims],
            rejected_claims=rejected_claims,
            generation_reasoning=generation_result.reasoning if hasattr(generation_result, 'reasoning') else None
        )


# Baseline extractor (no review stage)
class BaselineExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ClaimGeneration)

    def forward(self, transcript: str):
        result = self.extract(transcript=transcript)
        return result


def load_sample_transcript():
    """Load the sample transcript from data directory."""
    with open("data/sample_transcript.txt", "r", encoding="utf-8") as f:
        return f.read()


def analyze_claim_quality(claims: List[str]) -> dict:
    """Analyze claims for quality issues."""
    issues = {
        "total": len(claims),
        "with_pronouns": [],
        "opinion_words": [],
        "vague_words": [],
        "too_short": [],
        "too_long": [],
        "questions": []
    }

    pronouns = ["he", "she", "they", "him", "her", "them", "his", "hers", "their", "he's", "she's", "they're"]
    opinion_words = ["think", "believe", "feel", "opinion", "seems", "appears", "maybe", "probably", "might", "could"]
    vague_words = ["thing", "stuff", "someone", "something", "somehow"]

    for i, claim in enumerate(claims, 1):
        claim_lower = claim.lower()

        # Check for pronouns
        if any(f" {p} " in f" {claim_lower} " or f" {p}'" in f" {claim_lower} " or claim_lower.startswith(f"{p} ") for p in pronouns):
            issues["with_pronouns"].append((i, claim))

        # Check for opinion words
        if any(word in claim_lower for word in opinion_words):
            issues["opinion_words"].append((i, claim))

        # Check for vague words
        if any(word in claim_lower for word in vague_words):
            issues["vague_words"].append((i, claim))

        # Check if it's a question
        if "?" in claim:
            issues["questions"].append((i, claim))

        # Check length
        word_count = len(claim.split())
        if word_count < 5:
            issues["too_short"].append((i, claim))
        elif word_count > 40:
            issues["too_long"].append((i, claim))

    return issues


def print_quality_analysis(issues: dict, title: str):
    """Print quality analysis results."""
    print(f"\n{title}")
    print("-" * 80)
    print(f"Total claims: {issues['total']}")
    print(f"Claims with pronouns: {len(issues['with_pronouns'])}")
    print(f"Claims with opinion words: {len(issues['opinion_words'])}")
    print(f"Claims with vague words: {len(issues['vague_words'])}")
    print(f"Questions: {len(issues['questions'])}")
    print(f"Too short (<5 words): {len(issues['too_short'])}")
    print(f"Too long (>40 words): {len(issues['too_long'])}")

    # Calculate quality issue rate
    quality_issues = (
        len(issues['with_pronouns']) +
        len(issues['opinion_words']) +
        len(issues['vague_words']) +
        len(issues['questions']) +
        len(issues['too_short'])
    )

    if issues['total'] > 0:
        quality_rate = (quality_issues / issues['total']) * 100
        print(f"\n{'🔴' if quality_rate >= 15 else '🟢'} Quality issue rate: {quality_rate:.1f}%")
        print(f"   Target: <15% | Baseline: 40%")

    print()


def print_claims_with_issues(claims: List[str], issues: dict, title: str, max_display: int = 5):
    """Print claims and highlight issues."""
    print(f"\n{title}")
    print("=" * 80)

    if not claims:
        print("(No claims)")
        return

    # Create a map of claim index to issues
    issue_map = {}
    for i in range(len(claims)):
        claim_issues = []

        # Check each issue type
        for issue_type, issue_list in issues.items():
            if issue_type == "total":
                continue
            for idx, claim in issue_list:
                if idx == i + 1:
                    claim_issues.append(issue_type)

        if claim_issues:
            issue_map[i] = claim_issues

    # Display claims
    for i, claim in enumerate(claims[:max_display], 1):
        issue_tags = ""
        if i - 1 in issue_map:
            tags = issue_map[i - 1]
            issue_tags = f" ⚠️  [{', '.join(tags)}]"

        print(f"{i}. {claim}{issue_tags}")

    if len(claims) > max_display:
        print(f"\n... and {len(claims) - max_display} more")
    print()


def main():
    print("=" * 80)
    print("Experiment 2.1: Generate and Review Claims")
    print("=" * 80)
    print("\nThis experiment tests a two-stage pipeline:")
    print("  Stage 1: Generate claims (comprehensive)")
    print("  Stage 2: Review and filter for quality")
    print()

    # Load sample transcript
    transcript = load_sample_transcript()
    print(f"Loaded transcript: {len(transcript)} characters")
    print(f"Transcript preview: {transcript[:200]}...")
    print()

    # ===== BASELINE: Single-stage extraction =====
    print("\n" + "=" * 80)
    print("BASELINE: Single-Stage Extraction (No Review)")
    print("=" * 80)

    baseline_extractor = BaselineExtractor()
    start = time.time()
    baseline_result = baseline_extractor(transcript=transcript)
    baseline_time = time.time() - start

    print(f"\nExtraction time: {baseline_time:.2f}s")
    print(f"Claims extracted: {len(baseline_result.claims)}")

    baseline_issues = analyze_claim_quality(baseline_result.claims)
    print_claims_with_issues(baseline_result.claims, baseline_issues, "Baseline Claims")
    print_quality_analysis(baseline_issues, "Baseline Quality Analysis")

    # ===== EXPERIMENTAL: Two-stage generate and review =====
    print("\n" + "=" * 80)
    print("EXPERIMENTAL: Two-Stage Generate and Review")
    print("=" * 80)

    review_extractor = GenerateAndReviewClaims()
    start = time.time()
    review_result = review_extractor(transcript=transcript)
    review_time = time.time() - start

    print(f"\nTotal time: {review_time:.2f}s")
    print(f"Generated claims: {len(review_result.generated_claims)}")
    print(f"Accepted after review: {len(review_result.reviewed_claims)}")
    print(f"Rejected: {len(review_result.rejected_claims)}")
    print(f"Acceptance rate: {len(review_result.reviewed_claims) / len(review_result.generated_claims) * 100:.1f}%")

    # Show rejected claims with reasons
    if review_result.rejected_claims:
        print(f"\nSample Rejected Claims:")
        print("-" * 80)
        for i, rejected in enumerate(review_result.rejected_claims[:3], 1):
            print(f"{i}. {rejected['claim']}")
            print(f"   Issues: {rejected['issues']}")
            print()

    review_issues = analyze_claim_quality(review_result.reviewed_claims)
    print_claims_with_issues(review_result.reviewed_claims, review_issues, "Accepted Claims (After Review)")
    print_quality_analysis(review_issues, "After-Review Quality Analysis")

    # ===== COMPARISON =====
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    baseline_quality_issues = (
        len(baseline_issues['with_pronouns']) +
        len(baseline_issues['opinion_words']) +
        len(baseline_issues['vague_words']) +
        len(baseline_issues['questions']) +
        len(baseline_issues['too_short'])
    )

    review_quality_issues = (
        len(review_issues['with_pronouns']) +
        len(review_issues['opinion_words']) +
        len(review_issues['vague_words']) +
        len(review_issues['questions']) +
        len(review_issues['too_short'])
    )

    baseline_rate = (baseline_quality_issues / baseline_issues['total'] * 100) if baseline_issues['total'] > 0 else 0
    review_rate = (review_quality_issues / review_issues['total'] * 100) if review_issues['total'] > 0 else 0

    print(f"\nBaseline (Single-Stage):")
    print(f"  Claims: {baseline_issues['total']}")
    print(f"  Quality issues: {baseline_quality_issues} ({baseline_rate:.1f}%)")
    print(f"  Time: {baseline_time:.2f}s")

    print(f"\nGenerate-and-Review (Two-Stage):")
    print(f"  Claims: {review_issues['total']}")
    print(f"  Quality issues: {review_quality_issues} ({review_rate:.1f}%)")
    print(f"  Time: {review_time:.2f}s")
    print(f"  Filtered out: {len(review_result.rejected_claims)} claims")

    improvement = baseline_rate - review_rate
    print(f"\n{'✅' if review_rate < 15 else '⚠️'} Improvement: {improvement:.1f} percentage points")
    print(f"   {'✅ Target achieved!' if review_rate < 15 else '❌ Target not met (need <15%)'}")
    print(f"   Time overhead: {review_time - baseline_time:.2f}s ({((review_time / baseline_time - 1) * 100):.1f}% slower)")
    print()


if __name__ == "__main__":
    main()
