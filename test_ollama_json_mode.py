"""
Test Ollama JSON mode with DSPy to eliminate malformed JSON.

This tests whether we can force Ollama to generate ONLY valid JSON,
eliminating the need for complex parsing hacks.
"""

import sys
import dspy
from typing import List
from src.config.settings import settings

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 80)
print("Testing Ollama JSON Mode with DSPy")
print("=" * 80)
print()

# Test 1: Standard DSPy (current approach - generates malformed JSON)
print("Test 1: Standard DSPy LM (baseline - may generate malformed JSON)")
print("-" * 80)

try:
    lm_standard = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
    )

    dspy.configure(lm=lm_standard)

    class ClaimExtraction(dspy.Signature):
        """Extract claims from transcript."""
        transcript_chunk: str = dspy.InputField()
        claims: List[str] = dspy.OutputField()

    predictor = dspy.ChainOfThought(ClaimExtraction)

    test_transcript = """
    Trump's tariff policies have caused significant economic disruption.
    The Fed's response has been to maintain interest rates.
    Biden's administration disagrees with this approach.
    """

    result = predictor(transcript_chunk=test_transcript)

    print(f"✓ Standard LM completed")
    print(f"  Claims returned: {len(result.claims) if isinstance(result.claims, list) else 'ERROR'}")
    if isinstance(result.claims, list):
        for i, claim in enumerate(result.claims[:3], 1):
            print(f"    {i}. {claim[:80]}...")
    else:
        print(f"  ❌ ERROR: Claims is not a list: {type(result.claims)}")
    print()

except Exception as e:
    print(f"❌ Standard LM failed: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 2: Try passing format="json" directly
print("Test 2: DSPy LM with format='json' parameter")
print("-" * 80)

try:
    lm_json_mode = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
        format="json",  # Force JSON mode
    )

    dspy.configure(lm=lm_json_mode)

    predictor = dspy.ChainOfThought(ClaimExtraction)
    result = predictor(transcript_chunk=test_transcript)

    print(f"✓ JSON mode LM completed")
    print(f"  Claims returned: {len(result.claims) if isinstance(result.claims, list) else 'ERROR'}")
    if isinstance(result.claims, list):
        for i, claim in enumerate(result.claims[:3], 1):
            print(f"    {i}. {claim[:80]}...")

        # Check for splits
        split_items = [c for c in result.claims if isinstance(c, str) and c.startswith("s ")]
        if split_items:
            print(f"  ⚠️  WARNING: Found {len(split_items)} split items: {split_items}")
        else:
            print(f"  ✓ No splits detected!")
    else:
        print(f"  ❌ ERROR: Claims is not a list: {type(result.claims)}")
    print()

except Exception as e:
    print(f"❌ JSON mode LM failed: {e}")
    print(f"   This might mean:")
    print(f"   1. Ollama doesn't support format parameter yet")
    print(f"   2. DSPy/LiteLLM doesn't pass it through correctly")
    print(f"   3. The model doesn't support JSON mode")
    import traceback
    traceback.print_exc()
    print()

# Test 3: Try as model_kwargs (alternative approach)
print("Test 3: DSPy LM with format in model_kwargs (alternative)")
print("-" * 80)

try:
    lm_model_kwargs = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
        model_kwargs={"format": "json"},  # Try passing as model_kwargs
    )

    dspy.configure(lm=lm_model_kwargs)

    predictor = dspy.ChainOfThought(ClaimExtraction)
    result = predictor(transcript_chunk=test_transcript)

    print(f"✓ model_kwargs approach completed")
    print(f"  Claims returned: {len(result.claims) if isinstance(result.claims, list) else 'ERROR'}")
    if isinstance(result.claims, list):
        for i, claim in enumerate(result.claims[:3], 1):
            print(f"    {i}. {claim[:80]}...")

        # Check for splits
        split_items = [c for c in result.claims if isinstance(c, str) and c.startswith("s ")]
        if split_items:
            print(f"  ⚠️  WARNING: Found {len(split_items)} split items: {split_items}")
        else:
            print(f"  ✓ No splits detected!")
    else:
        print(f"  ❌ ERROR: Claims is not a list: {type(result.claims)}")
    print()

except Exception as e:
    print(f"❌ model_kwargs approach failed: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test 4: ChatAdapter (alternative approach - doesn't use JSON at all)
print("Test 4: ChatAdapter (no JSON parsing needed)")
print("-" * 80)

try:
    from dspy.adapters.chat_adapter import ChatAdapter

    lm_chat = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
    )

    dspy.configure(lm=lm_chat, adapter=ChatAdapter())

    predictor = dspy.ChainOfThought(ClaimExtraction)
    result = predictor(transcript_chunk=test_transcript)

    print(f"✓ ChatAdapter completed")
    print(f"  Claims returned: {len(result.claims) if isinstance(result.claims, list) else 'ERROR'}")
    if isinstance(result.claims, list):
        for i, claim in enumerate(result.claims[:3], 1):
            print(f"    {i}. {claim[:80]}...")

        # Check for splits
        split_items = [c for c in result.claims if isinstance(c, str) and c.startswith("s ")]
        if split_items:
            print(f"  ⚠️  WARNING: Found {len(split_items)} split items: {split_items}")
        else:
            print(f"  ✓ No splits detected!")
    else:
        print(f"  ❌ ERROR: Claims is not a list: {type(result.claims)}")
    print()

except Exception as e:
    print(f"❌ ChatAdapter failed: {e}")
    import traceback
    traceback.print_exc()
    print()

print("=" * 80)
print("Summary: Which approach should we use?")
print("=" * 80)
print()
print("✅ If Test 2 or 3 works → Use Ollama JSON mode (BEST solution)")
print("✅ If Test 4 works → Use ChatAdapter (GOOD solution)")
print("⚠️  If all fail → Keep enhanced parser (ACCEPTABLE band-aid)")
print()
