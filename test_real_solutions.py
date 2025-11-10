"""
Test REAL root cause solutions (no kwargs hacks).

1. ChatAdapter - Official DSPy adapter that doesn't use JSON
2. Direct Predict - Skip ChainOfThought reasoning
3. Simplified signature - Clearer instructions
"""

import sys
import dspy
from typing import List
from src.config.settings import settings

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

test_transcript = """
Trump's tariff policies have caused significant economic disruption.
The Fed's response has been to maintain interest rates.
Biden's administration disagrees with this approach.
"""

print("=" * 80)
print("Testing REAL Root Cause Solutions")
print("=" * 80)
print()

# Configure base LM (no format tricks)
lm = dspy.LM(
    f"ollama/{settings.ollama_model}",
    api_base=settings.ollama_url,
    temperature=0.3,
)


# Solution 1: ChatAdapter (uses field markers, not JSON)
print("Solution 1: ChatAdapter (officially supported)")
print("-" * 80)

try:
    from dspy.adapters.chat_adapter import ChatAdapter

    dspy.configure(lm=lm, adapter=ChatAdapter())

    class ClaimExtraction(dspy.Signature):
        """Extract factual, verifiable claims from podcast transcript text."""
        transcript_chunk: str = dspy.InputField()
        claims: List[str] = dspy.OutputField()

    # Still use ChainOfThought
    predictor = dspy.ChainOfThought(ClaimExtraction)
    result = predictor(transcript_chunk=test_transcript)

    print(f"✓ ChatAdapter + ChainOfThought completed")
    print(f"  Claims: {len(result.claims) if isinstance(result.claims, list) else 'ERROR'}")
    if isinstance(result.claims, list):
        for i, claim in enumerate(result.claims[:3], 1):
            print(f"    {i}. {claim[:60]}...")
        splits = [c for c in result.claims if c.startswith("s ")]
        print(f"  Splits: {len(splits)}")
    print()

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# Solution 2: Direct Predict (skip ChainOfThought reasoning)
print("Solution 2: Direct Predict (no reasoning, just extraction)")
print("-" * 80)

try:
    from dspy.adapters.json_adapter import JSONAdapter

    dspy.configure(lm=lm, adapter=JSONAdapter())  # Back to JSON adapter

    # Use Predict instead of ChainOfThought
    predictor = dspy.Predict(ClaimExtraction)
    result = predictor(transcript_chunk=test_transcript)

    print(f"✓ Predict (no CoT) completed")
    print(f"  Claims: {len(result.claims) if isinstance(result.claims, list) else 'ERROR'}")
    if isinstance(result.claims, list):
        for i, claim in enumerate(result.claims[:3], 1):
            print(f"    {i}. {claim[:60]}...")
        splits = [c for c in result.claims if c.startswith("s ")]
        print(f"  Splits: {len(splits)}")
    print()

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# Solution 3: Simplified signature instructions
print("Solution 3: Ultra-simple signature (minimal instructions)")
print("-" * 80)

try:
    dspy.configure(lm=lm, adapter=JSONAdapter())

    class SimpleClaimExtraction(dspy.Signature):
        """Extract claims. Return JSON array only."""
        transcript_chunk: str = dspy.InputField()
        claims: List[str] = dspy.OutputField(desc="JSON array of claims")

    predictor = dspy.ChainOfThought(SimpleClaimExtraction)
    result = predictor(transcript_chunk=test_transcript)

    print(f"✓ Simplified signature completed")
    print(f"  Claims: {len(result.claims) if isinstance(result.claims, list) else 'ERROR'}")
    if isinstance(result.claims, list):
        for i, claim in enumerate(result.claims[:3], 1):
            print(f"    {i}. {claim[:60]}...")
        splits = [c for c in result.claims if c.startswith("s ")]
        print(f"  Splits: {len(splits)}")
    print()

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# Solution 4: Our enhanced parser (band-aid but works)
print("Solution 4: Enhanced parser (band-aid)")
print("-" * 80)

try:
    from src.training.dspy_json_patch import apply_json_patch

    apply_json_patch()  # Apply our nuclear patch
    dspy.configure(lm=lm, adapter=JSONAdapter())

    predictor = dspy.ChainOfThought(ClaimExtraction)
    result = predictor(transcript_chunk=test_transcript)

    print(f"✓ Enhanced parser completed")
    print(f"  Claims: {len(result.claims) if isinstance(result.claims, list) else 'ERROR'}")
    if isinstance(result.claims, list):
        for i, claim in enumerate(result.claims[:3], 1):
            print(f"    {i}. {claim[:60]}...")
        splits = [c for c in result.claims if c.startswith("s ")]
        print(f"  Splits: {len(splits)}")
    print()

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()


print("=" * 80)
print("Recommendation:")
print("=" * 80)
print()
print("1️⃣ ChatAdapter - Most robust, officially supported")
print("2️⃣ Enhanced parser - Works but is a band-aid")
print("3️⃣ Simplified signature - May help but not guaranteed")
print()
