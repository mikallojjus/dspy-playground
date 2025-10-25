"""
Experiment 1.3: Predict vs ChainOfThought Comparison

Goal: Compare dspy.Predict and dspy.ChainOfThought for claim extraction to understand:
- Which produces higher quality claims
- Impact on reasoning transparency
- Performance differences
"""

import dspy
from typing import List
import time


# Configure DSPy to use Ollama with Qwen 2.5 7B
lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
dspy.configure(lm=lm)


# Define the claim extraction signature
class ClaimExtraction(dspy.Signature):
    """Extract factual, verifiable claims from podcast transcript text.

    A claim must be:
    - Factual and verifiable (not opinion or speculation)
    - Self-contained with no pronouns (use full names)
    - Concise and specific (1-2 sentences)
    - Based only on information explicitly stated in the transcript

    Exclude:
    - Opinions, questions, or speculative statements
    - Claims with pronouns (he, she, they) or vague references
    - Inferred information not explicitly stated
    """

    transcript: str = dspy.InputField(desc="The podcast transcript text to analyze")
    claims: List[str] = dspy.OutputField(desc="List of factual claims extracted from the transcript")


# Predictor using simple Predict
class SimpleClaimExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(ClaimExtraction)

    def forward(self, transcript: str):
        result = self.extract(transcript=transcript)
        return result


# Predictor using ChainOfThought
class CoTClaimExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ClaimExtraction)

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
        "vague": [],
        "too_short": [],
        "too_long": []
    }

    pronouns = ["he", "she", "they", "him", "her", "them", "his", "hers", "their"]
    opinion_words = ["think", "believe", "feel", "opinion", "seems", "appears", "maybe", "probably"]

    for i, claim in enumerate(claims, 1):
        claim_lower = claim.lower()

        # Check for pronouns
        if any(f" {p} " in f" {claim_lower} " or f" {p}'" in f" {claim_lower} " for p in pronouns):
            issues["with_pronouns"].append((i, claim))

        # Check for opinion words
        if any(word in claim_lower for word in opinion_words):
            issues["opinion_words"].append((i, claim))

        # Check length
        if len(claim.split()) < 5:
            issues["too_short"].append((i, claim))
        elif len(claim.split()) > 40:
            issues["too_long"].append((i, claim))

    return issues


def print_claims(claims: List[str], title: str):
    """Print claims with formatting."""
    print(f"\n{title}")
    print("=" * 80)
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim}")
    print()


def print_quality_analysis(issues: dict, title: str):
    """Print quality analysis results."""
    print(f"\n{title}")
    print("-" * 80)
    print(f"Total claims: {issues['total']}")
    print(f"Claims with pronouns: {len(issues['with_pronouns'])}")
    print(f"Claims with opinion words: {len(issues['opinion_words'])}")
    print(f"Claims too short (<5 words): {len(issues['too_short'])}")
    print(f"Claims too long (>40 words): {len(issues['too_long'])}")

    quality_issues = len(issues['with_pronouns']) + len(issues['opinion_words']) + len(issues['too_short'])
    if issues['total'] > 0:
        quality_rate = (quality_issues / issues['total']) * 100
        print(f"\nQuality issue rate: {quality_rate:.1f}%")
        print(f"(Target: <15%, Baseline: 40%)")

    # Show problematic claims
    if issues['with_pronouns']:
        print(f"\nClaims with pronouns:")
        for idx, claim in issues['with_pronouns'][:3]:  # Show first 3
            print(f"  {idx}. {claim[:100]}...")

    if issues['opinion_words']:
        print(f"\nClaims with opinion words:")
        for idx, claim in issues['opinion_words'][:3]:  # Show first 3
            print(f"  {idx}. {claim[:100]}...")
    print()


def main():
    print("=" * 80)
    print("Experiment 1.3: Predict vs ChainOfThought Comparison")
    print("=" * 80)
    print()

    # Load sample transcript
    transcript = load_sample_transcript()
    print(f"Loaded transcript: {len(transcript)} characters")
    print()

    # Create both extractors
    simple_extractor = SimpleClaimExtractor()
    cot_extractor = CoTClaimExtractor()

    # Test 1: Simple Predict
    print("\n" + "=" * 80)
    print("TEST 1: Using dspy.Predict (Simple)")
    print("=" * 80)
    start = time.time()
    simple_result = simple_extractor(transcript=transcript)
    simple_time = time.time() - start

    print(f"Extraction time: {simple_time:.2f}s")
    print_claims(simple_result.claims, "Claims from Predict")
    simple_issues = analyze_claim_quality(simple_result.claims)
    print_quality_analysis(simple_issues, "Quality Analysis - Predict")

    # Test 2: ChainOfThought
    print("\n" + "=" * 80)
    print("TEST 2: Using dspy.ChainOfThought (with Reasoning)")
    print("=" * 80)
    start = time.time()
    cot_result = cot_extractor(transcript=transcript)
    cot_time = time.time() - start

    print(f"Extraction time: {cot_time:.2f}s")

    # Show reasoning if available
    if hasattr(cot_result, 'reasoning'):
        print(f"\nReasoning:")
        print("-" * 80)
        print(cot_result.reasoning[:500])  # Show first 500 chars
        if len(cot_result.reasoning) > 500:
            print(f"... (truncated, total length: {len(cot_result.reasoning)} chars)")
        print()

    print_claims(cot_result.claims, "Claims from ChainOfThought")
    cot_issues = analyze_claim_quality(cot_result.claims)
    print_quality_analysis(cot_issues, "Quality Analysis - ChainOfThought")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nPredict:")
    print(f"  - Claims extracted: {simple_issues['total']}")
    print(f"  - Time: {simple_time:.2f}s")
    print(f"  - Quality issues: {len(simple_issues['with_pronouns']) + len(simple_issues['opinion_words']) + len(simple_issues['too_short'])}")

    print(f"\nChainOfThought:")
    print(f"  - Claims extracted: {cot_issues['total']}")
    print(f"  - Time: {cot_time:.2f}s")
    print(f"  - Quality issues: {len(cot_issues['with_pronouns']) + len(cot_issues['opinion_words']) + len(cot_issues['too_short'])}")
    print(f"  - Provides reasoning: Yes")

    print(f"\nTime difference: {cot_time - simple_time:.2f}s ({((cot_time / simple_time - 1) * 100):.1f}% slower)")
    print()


if __name__ == "__main__":
    main()
