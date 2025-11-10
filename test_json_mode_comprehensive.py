"""
Comprehensive validation test for Ollama JSON mode with DSPy.

Tests:
1. BootstrapFewShot training (where the bug originally appeared)
2. Long, complex transcripts with many possessives
3. Model save/load behavior
4. Multiple iterations to catch intermittent issues
5. Actual Ollama API call inspection
"""

import sys
import dspy
import json
import tempfile
import logging
from pathlib import Path
from typing import List
from dspy.teleprompt import BootstrapFewShot
from src.config.settings import settings

# Fix encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Enable debug logging to see actual API calls
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

print("=" * 80)
print("COMPREHENSIVE JSON MODE VALIDATION TEST")
print("=" * 80)
print()

# Long, complex transcript with MANY possessives (the stress test)
STRESS_TRANSCRIPT = """
Trump's tariff policies have caused significant economic disruption in Biden's
administration. The Fed's response has been to maintain interest rates despite
concerns about Netanyahu's government's actions in the Middle East.

Powell's Federal Reserve maintains that the economy's resilience depends on
workers' ability to adapt. The market's reaction to Trump's announcements shows
investors' nervousness about China's retaliation. Biden's team disagrees with
the Fed's approach, believing the economy's fundamentals are stronger than
Powell's projections suggest.

The IMF's latest report contradicts Trump's claims about the dollar's strength.
Europe's economy shows signs of recovery, with Germany's manufacturing sector
outperforming analysts' expectations. France's President Macron's policy changes
have boosted consumer confidence according to the ECB's recent survey.

Tech companies' earnings reports reveal Silicon Valley's concerns about AI's
impact on workers' jobs. Microsoft's CEO's statements about OpenAI's partnership
drew criticism from Google's leadership team. Apple's new product lineup reflects
Tim Cook's vision for the company's future in AR/VR markets.
"""

# Expected claims (for validation)
EXPECTED_CLAIMS_COUNT_MIN = 8
EXPECTED_CLAIMS_COUNT_MAX = 20


class ClaimExtraction(dspy.Signature):
    """Extract factual, verifiable claims from podcast transcript text."""
    transcript_chunk: str = dspy.InputField()
    claims: List[str] = dspy.OutputField()


def check_for_splits(claims: List[str], context: str) -> int:
    """Check for split claims (items starting with 's ')."""
    if not isinstance(claims, list):
        print(f"  ⚠️  {context}: Claims is not a list: {type(claims)}")
        return -1

    split_items = [c for c in claims if isinstance(c, str) and c.startswith("s ")]

    if split_items:
        print(f"  ❌ {context}: Found {len(split_items)} SPLITS:")
        for split in split_items[:3]:
            print(f"     - '{split[:60]}...'")
        return len(split_items)
    else:
        print(f"  ✅ {context}: No splits detected")
        return 0


def check_for_malformed_patterns(claims: List[str], context: str) -> bool:
    """Check for patterns indicating malformed JSON was parsed."""
    issues = []

    # Check for empty strings
    empty = [i for i, c in enumerate(claims) if not c.strip()]
    if empty:
        issues.append(f"Empty claims at indices: {empty}")

    # Check for very short fragments (< 10 chars, likely splits)
    fragments = [(i, c) for i, c in enumerate(claims) if len(c.strip()) < 10]
    if fragments:
        issues.append(f"Suspiciously short claims: {fragments[:3]}")

    # Check for claims ending with comma or open quote
    malformed = [c for c in claims if c.rstrip().endswith((',', '"', "'", '['))]
    if malformed:
        issues.append(f"Malformed claim endings: {[c[:40] for c in malformed[:3]]}")

    if issues:
        print(f"  ⚠️  {context}: Quality issues detected:")
        for issue in issues:
            print(f"     - {issue}")
        return False
    else:
        print(f"  ✅ {context}: Claims quality looks good")
        return True


# =============================================================================
# TEST 1: Simple Inference with JSON Mode
# =============================================================================
print("TEST 1: Simple Inference (baseline)")
print("-" * 80)

try:
    lm_json = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
        format="json",  # Enable JSON mode
    )
    dspy.configure(lm=lm_json)

    predictor = dspy.ChainOfThought(ClaimExtraction)
    result = predictor(transcript_chunk=STRESS_TRANSCRIPT)

    print(f"✓ Completed")
    print(f"  Claims count: {len(result.claims)}")
    splits = check_for_splits(result.claims, "Test 1")
    quality = check_for_malformed_patterns(result.claims, "Test 1")
    print()

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# =============================================================================
# TEST 2: BootstrapFewShot Training (THE CRITICAL TEST)
# =============================================================================
print("TEST 2: BootstrapFewShot Training (where original bug appeared)")
print("-" * 80)
print("This is the REAL test - training is where malformed JSON appeared!")
print()

try:
    # Create small training dataset
    trainset = [
        dspy.Example(
            transcript_chunk=STRESS_TRANSCRIPT,
            claims=[
                "Trump's tariff policies have caused significant economic disruption.",
                "The Fed's response has been to maintain interest rates.",
                "Biden's administration disagrees with the Fed's approach.",
            ]
        ).with_inputs("transcript_chunk"),
        dspy.Example(
            transcript_chunk="Apple's CEO Tim Cook announced the company's new AI strategy. Microsoft's partnership with OpenAI continues to evolve.",
            claims=[
                "Apple's CEO Tim Cook announced the company's new AI strategy.",
                "Microsoft's partnership with OpenAI continues to evolve.",
            ]
        ).with_inputs("transcript_chunk"),
    ]

    # Simple metric (just check if it's a list)
    def simple_metric(example, pred, trace=None):
        return isinstance(pred.claims, list) and len(pred.claims) > 0

    # Configure with JSON mode
    lm_json = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
        cache=False,
        format="json",  # Enable JSON mode
    )
    dspy.configure(lm=lm_json)

    print("Starting BootstrapFewShot optimization...")
    print("(This will make multiple LLM calls - watch for malformed JSON in logs)")
    print()

    baseline = dspy.ChainOfThought(ClaimExtraction)
    optimizer = BootstrapFewShot(
        metric=simple_metric,
        max_bootstrapped_demos=2,  # Small for quick test
        max_labeled_demos=2,
    )

    optimized = optimizer.compile(baseline, trainset=trainset)

    print()
    print("✓ BootstrapFewShot completed")
    print(f"  Demos collected: {len(optimized.demos) if hasattr(optimized, 'demos') else 0}")

    # Test the optimized model
    result = optimized(transcript_chunk=STRESS_TRANSCRIPT)
    print(f"  Claims count: {len(result.claims)}")
    splits = check_for_splits(result.claims, "Test 2 (optimized)")
    quality = check_for_malformed_patterns(result.claims, "Test 2 (optimized)")
    print()

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# =============================================================================
# TEST 3: Model Save/Load (Does format="json" persist?)
# =============================================================================
print("TEST 3: Model Save/Load Behavior")
print("-" * 80)
print("Testing if format='json' needs to be set again after load...")
print()

try:
    # Train a simple model
    lm_json = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
        format="json",
    )
    dspy.configure(lm=lm_json)

    model1 = dspy.ChainOfThought(ClaimExtraction)

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    model1.save(temp_path)
    print(f"✓ Model saved to {temp_path}")

    # Load WITHOUT setting format="json"
    print("  Loading model WITHOUT format='json'...")
    lm_no_format = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
        # NO format="json" here!
    )
    dspy.configure(lm=lm_no_format)

    model2 = dspy.ChainOfThought(ClaimExtraction)
    model2.load(temp_path)

    result = model2(transcript_chunk=STRESS_TRANSCRIPT)
    print(f"  Claims count: {len(result.claims)}")
    splits = check_for_splits(result.claims, "Test 3 (no format)")
    quality = check_for_malformed_patterns(result.claims, "Test 3 (no format)")

    # Load WITH format="json"
    print()
    print("  Loading model WITH format='json'...")
    lm_with_format = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
        format="json",
    )
    dspy.configure(lm=lm_with_format)

    model3 = dspy.ChainOfThought(ClaimExtraction)
    model3.load(temp_path)

    result = model3(transcript_chunk=STRESS_TRANSCRIPT)
    print(f"  Claims count: {len(result.claims)}")
    splits = check_for_splits(result.claims, "Test 3 (with format)")
    quality = check_for_malformed_patterns(result.claims, "Test 3 (with format)")

    # Cleanup
    Path(temp_path).unlink()
    print()

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# =============================================================================
# TEST 4: Stress Test - Multiple Iterations
# =============================================================================
print("TEST 4: Stress Test - 20 iterations")
print("-" * 80)
print("Testing for intermittent failures...")
print()

try:
    lm_json = dspy.LM(
        f"ollama/{settings.ollama_model}",
        api_base=settings.ollama_url,
        temperature=0.3,
        format="json",
    )
    dspy.configure(lm=lm_json)

    predictor = dspy.ChainOfThought(ClaimExtraction)

    total_splits = 0
    total_quality_issues = 0

    for i in range(20):
        result = predictor(transcript_chunk=STRESS_TRANSCRIPT)

        if not isinstance(result.claims, list):
            print(f"  Iteration {i+1}: ❌ Not a list!")
            continue

        splits = check_for_splits(result.claims, f"Iteration {i+1}")
        quality = check_for_malformed_patterns(result.claims, f"Iteration {i+1}")

        if splits > 0:
            total_splits += splits
        if not quality:
            total_quality_issues += 1

    print()
    print(f"Summary after 20 iterations:")
    print(f"  Total split claims detected: {total_splits}")
    print(f"  Iterations with quality issues: {total_quality_issues}/20")

    if total_splits == 0 and total_quality_issues == 0:
        print(f"  ✅ ALL ITERATIONS PASSED!")
    else:
        print(f"  ❌ ISSUES DETECTED - format='json' not reliable!")
    print()

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    print()


# =============================================================================
# FINAL VERDICT
# =============================================================================
print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()
print("Key findings:")
print("1. Does format='json' work in simple inference? [See Test 1]")
print("2. Does it work during BootstrapFewShot training? [See Test 2] ← CRITICAL")
print("3. Does it require re-setting after model load? [See Test 3]")
print("4. Is it reliable across iterations? [See Test 4]")
print()
print("If ALL tests pass → format='json' is production-ready")
print("If Test 2 fails → format='json' doesn't help during training (original bug)")
print("If Test 3 shows difference → Must set format='json' in both train & inference")
print("If Test 4 shows issues → Intermittent failures, not reliable")
print()
