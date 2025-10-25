"""
Experiment 3.2: Inspect What Changed

Goal: Understand what DSPy actually did during optimization

This experiment:
1. Loads the optimized module
2. Inspects the few-shot examples DSPy selected
3. Tests it on a new example
4. Compares behavior to baseline
"""

import dspy
import sys
from typing import List

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# Same signature as optimization
class ClaimExtraction(dspy.Signature):
    """Extract factual, verifiable claims from podcast transcript text.

    Claims should be:
    - Factual (not opinions)
    - Self-contained (no pronouns like he/she/they)
    - Specific (include names, numbers, dates)
    - Concise (5-40 words)
    """
    transcript_chunk: str = dspy.InputField(desc="The podcast transcript text to analyze")
    claims: List[str] = dspy.OutputField(desc="List of factual claims extracted from the transcript")


def main():
    print("=" * 80)
    print("Experiment 3.2: Inspect Optimized Module")
    print("=" * 80)
    print()

    # Configure DSPy
    print("Configuring DSPy with Ollama...")
    lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
    dspy.configure(lm=lm)
    print("Done")
    print()

    # Load optimized module
    print("Loading optimized module...")
    optimized = dspy.ChainOfThought(ClaimExtraction)

    # Try to load the successful LLM judge optimized model first
    model_paths = [
        'models/claim_extractor_llm_judge_v1.json',
        'models/claim_extractor_positive_v1.json',
        'models/claim_extractor_bootstrap_v1.json'
    ]

    loaded = False
    for model_path in model_paths:
        try:
            optimized.load(model_path)
            print(f"Loaded: {model_path}")
            loaded = True
            break
        except Exception:
            continue

    if not loaded:
        print(f"ERROR: Could not load any optimized module")
        print("\nTried:")
        for p in model_paths:
            print(f"  - {p}")
        print("\nMake sure you've run one of the optimization experiments first!")
        return
    print()

    # Inspect the module
    print("=" * 80)
    print("MODULE INSPECTION")
    print("=" * 80)
    print()

    # Check for few-shot examples
    if hasattr(optimized, 'demos') and optimized.demos:
        print(f"DSPy selected {len(optimized.demos)} few-shot examples:")
        print()

        for i, demo in enumerate(optimized.demos, 1):
            print(f"--- Example {i} ---")
            print(f"Input (transcript):")
            # Show first 150 chars of transcript
            transcript_preview = demo.transcript_chunk[:150]
            if len(demo.transcript_chunk) > 150:
                transcript_preview += "..."
            print(f"  {transcript_preview}")
            print()
            print(f"Output (claims):")
            for claim in demo.claims:
                print(f"  - {claim}")
            print()

        print("=" * 80)
        print("ANALYSIS OF SELECTED EXAMPLES")
        print("=" * 80)
        print()
        print("Questions to consider:")
        print("  - What do these examples have in common?")
        print("  - Are they diverse or similar?")
        print("  - Do they demonstrate good claim quality?")
        print("  - Why might DSPy have selected these specific examples?")
        print()

    else:
        print("No few-shot examples found in the optimized module.")
        print("The module is using zero-shot prompting.")
        print()

    # Test on a new example
    print("=" * 80)
    print("TESTING ON NEW EXAMPLE")
    print("=" * 80)
    print()

    # Load sample transcript
    try:
        with open("data/sample_transcript.txt", "r", encoding="utf-8") as f:
            test_transcript = f.read()

        # Take first 500 chars as test
        test_chunk = test_transcript[:500]
    except:
        # Fallback test example
        test_chunk = """
        2 (16s):
        Welcome to Bankless where today we explore a defense of the Ethereum roadmap.
        This is Ryan Sean Adams. I'm here with David Hoffman and we are here to help
        you become more bankless. The Ethereum roadmap has been called into question
        recently, I think Bankless has aired some of these dissents, and I'll say maybe
        two things about that before we begin.
        """

    print("Test transcript:")
    print(test_chunk[:200] + "..." if len(test_chunk) > 200 else test_chunk)
    print()

    # Create baseline for comparison
    print("Extracting claims with BASELINE (no optimization):")
    baseline = dspy.ChainOfThought(ClaimExtraction)
    baseline_result = baseline(transcript_chunk=test_chunk)

    print()
    for i, claim in enumerate(baseline_result.claims, 1):
        print(f"  {i}. {claim}")
    print()

    print("-" * 80)
    print()

    print("Extracting claims with OPTIMIZED (with few-shot examples):")
    optimized_result = optimized(transcript_chunk=test_chunk)

    print()
    for i, claim in enumerate(optimized_result.claims, 1):
        print(f"  {i}. {claim}")
    print()

    # Manual comparison
    print("=" * 80)
    print("MANUAL COMPARISON")
    print("=" * 80)
    print()
    print("Compare the two outputs above:")
    print()
    print("Questions to consider:")
    print("  - Are optimized claims more specific?")
    print("  - Do optimized claims avoid pronouns better?")
    print("  - Are optimized claims more factual (less opinion)?")
    print("  - Do you notice any quality difference?")
    print()
    print("This subjective comparison helps validate the metric scores.")
    print()


if __name__ == "__main__":
    main()
