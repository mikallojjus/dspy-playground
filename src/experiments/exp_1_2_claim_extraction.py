"""
Experiment 1.2: Claim Extraction Signature

Goal: Use DSPy to define and optimize a claim extraction signature that:
- Reduces low-quality claims from 40% baseline to <15%
- Extracts factual, verifiable claims
- Avoids opinions, pronouns, and vague statements
"""

import dspy
from typing import List
from pydantic import BaseModel, Field


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


# Create a predictor module
class ClaimExtractor(dspy.Module):
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


def main():
    print("=" * 80)
    print("Experiment 1.2: Claim Extraction with DSPy")
    print("=" * 80)
    print()

    # Load sample transcript
    transcript = load_sample_transcript()
    print(f"Loaded transcript: {len(transcript)} characters")
    print()

    # Create extractor
    extractor = ClaimExtractor()

    # Extract claims
    print("Extracting claims...")
    print("-" * 80)
    result = extractor(transcript=transcript)

    # Display results
    print(f"\nExtracted {len(result.claims)} claims:")
    print()
    for i, claim in enumerate(result.claims, 1):
        print(f"{i}. {claim}")
        print()

    # Display reasoning (from ChainOfThought)
    if hasattr(result, 'reasoning'):
        print("-" * 80)
        print("Reasoning:")
        print(result.reasoning)


if __name__ == "__main__":
    main()
