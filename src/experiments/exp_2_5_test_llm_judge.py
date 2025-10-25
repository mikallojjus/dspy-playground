"""
Experiment 2.5: Test LLM-as-Judge Metric

Goal: Demonstrate that LLM-as-Judge understands semantic quality better than pattern matching

This experiment:
1. Tests both metrics on manual review data
2. Shows side-by-side comparisons
3. Highlights cases where LLM-as-Judge is superior
4. Measures accuracy and speed

Expected outcome: LLM judge should achieve ~95% accuracy vs ~85% for pattern matching
"""

import dspy
import json
import sys
import time
from typing import List

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


# Import both metrics
from ...metrics import claim_quality_metric as pattern_metric


# Define LLM judge inline (same as src/metrics_llm_judge.py)
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
    """Evaluate claim quality using an LLM judge."""
    predicted_claims = pred.claims if hasattr(pred, 'claims') else []

    if not predicted_claims:
        return 0.0

    judge = dspy.ChainOfThought(ClaimQualityJudge)

    high_quality_count = 0

    for claim in predicted_claims:
        result = judge(claim=claim)

        if result.is_high_quality:
            high_quality_count += 1

    quality_score = high_quality_count / len(predicted_claims)
    return quality_score


def test_single_claim(claim, manual_quality, pattern_metric, llm_judge):
    """Test a single claim with both metrics."""
    pred = type('obj', (object,), {
        'claims': [claim]
    })

    # Pattern matching
    pattern_score = pattern_metric(None, pred)
    pattern_says_good = (pattern_score == 1.0)

    # LLM judge
    result = llm_judge(claim=claim)
    llm_says_good = result.is_high_quality
    llm_reason = result.reason

    # Compare to manual
    manual_says_good = (manual_quality == 'good')

    pattern_correct = (pattern_says_good == manual_says_good)
    llm_correct = (llm_says_good == manual_says_good)

    return {
        'claim': claim,
        'manual': manual_quality,
        'pattern_says': 'good' if pattern_says_good else 'bad',
        'llm_says': 'good' if llm_says_good else 'bad',
        'llm_reason': llm_reason,
        'pattern_correct': pattern_correct,
        'llm_correct': llm_correct,
        'both_wrong': not pattern_correct and not llm_correct,
        'llm_better': llm_correct and not pattern_correct,
        'pattern_better': pattern_correct and not llm_correct
    }


def main():
    print("=" * 80)
    print("Experiment 2.5: Test LLM-as-Judge Metric")
    print("=" * 80)
    print()
    print("This experiment compares pattern matching vs LLM-as-Judge")
    print("on your manual review data to show which is more accurate.")
    print()

    # Load manual review data
    print("Loading manual review data...")
    with open('evaluation/claims_manual_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = data['examples']
    print(f"Loaded {len(examples)} manually reviewed claims")
    print(f"  Good: {sum(1 for ex in examples if ex['quality'] == 'good')}")
    print(f"  Bad: {sum(1 for ex in examples if ex['quality'] == 'bad')}")
    print()

    # Configure DSPy
    print("Configuring DSPy with Ollama...")
    lm = dspy.LM("ollama/qwen2.5:7b-instruct-q4_0", api_base="http://localhost:11434")
    dspy.configure(lm=lm)
    print("Done")
    print()

    # Create LLM judge
    llm_judge = dspy.ChainOfThought(ClaimQualityJudge)

    # Test each claim
    print("=" * 80)
    print("TESTING BOTH METRICS")
    print("=" * 80)
    print()
    print("This will take a few minutes (LLM calls are slow)...")
    print()

    results = []
    start_time = time.time()

    for i, example in enumerate(examples, 1):
        print(f"Testing {i}/{len(examples)}...", end='\r')

        result = test_single_claim(
            example['claim'],
            example['quality'],
            pattern_metric,
            llm_judge
        )
        results.append(result)

    total_time = time.time() - start_time

    print()
    print(f"Completed in {total_time:.1f} seconds")
    print()

    # Calculate accuracies
    pattern_correct = sum(1 for r in results if r['pattern_correct'])
    llm_correct = sum(1 for r in results if r['llm_correct'])

    pattern_accuracy = (pattern_correct / len(results) * 100)
    llm_accuracy = (llm_correct / len(results) * 100)

    # Find interesting cases
    llm_better_cases = [r for r in results if r['llm_better']]
    pattern_better_cases = [r for r in results if r['pattern_better']]
    both_wrong_cases = [r for r in results if r['both_wrong']]

    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Pattern Matching Accuracy: {pattern_accuracy:.1f}% ({pattern_correct}/{len(results)})")
    print(f"LLM-as-Judge Accuracy:     {llm_accuracy:.1f}% ({llm_correct}/{len(results)})")
    print()

    improvement = llm_accuracy - pattern_accuracy
    print(f"Improvement: {improvement:+.1f} percentage points")
    print()

    if improvement > 5:
        print("SUCCESS: LLM-as-Judge is significantly better!")
    elif improvement > 0:
        print("MODEST: LLM-as-Judge is slightly better")
    elif improvement == 0:
        print("EQUAL: Both metrics perform the same")
    else:
        print("SURPRISING: Pattern matching is better (unexpected)")

    print()
    print(f"Cases where LLM judge is better: {len(llm_better_cases)}")
    print(f"Cases where pattern matching is better: {len(pattern_better_cases)}")
    print(f"Cases where both are wrong: {len(both_wrong_cases)}")
    print()

    # Show examples where LLM is better
    if llm_better_cases:
        print("=" * 80)
        print("EXAMPLES WHERE LLM JUDGE IS BETTER")
        print("=" * 80)
        print()
        print("These cases show why LLM-as-Judge is superior:")
        print()

        for i, case in enumerate(llm_better_cases[:5], 1):  # Show up to 5
            print(f"{i}. Claim: {case['claim'][:70]}...")
            print(f"   Manual review: {case['manual'].upper()}")
            print(f"   Pattern matching: {case['pattern_says'].upper()} (WRONG)")
            print(f"   LLM judge: {case['llm_says'].upper()} (CORRECT)")
            print(f"   LLM reasoning: {case['llm_reason']}")
            print()

        if len(llm_better_cases) > 5:
            print(f"... and {len(llm_better_cases) - 5} more cases")
            print()

    # Show examples where pattern matching is better (if any)
    if pattern_better_cases:
        print("=" * 80)
        print("EXAMPLES WHERE PATTERN MATCHING IS BETTER")
        print("=" * 80)
        print()

        for i, case in enumerate(pattern_better_cases[:3], 1):
            print(f"{i}. Claim: {case['claim'][:70]}...")
            print(f"   Manual review: {case['manual'].upper()}")
            print(f"   Pattern matching: {case['pattern_says'].upper()} (CORRECT)")
            print(f"   LLM judge: {case['llm_says'].upper()} (WRONG)")
            print(f"   LLM reasoning: {case['llm_reason']}")
            print()

    # Show cases where both are wrong
    if both_wrong_cases:
        print("=" * 80)
        print("HARD CASES (Both Metrics Wrong)")
        print("=" * 80)
        print()
        print("These cases are genuinely difficult:")
        print()

        for i, case in enumerate(both_wrong_cases[:3], 1):
            print(f"{i}. Claim: {case['claim'][:70]}...")
            print(f"   Manual review: {case['manual'].upper()}")
            print(f"   Pattern matching: {case['pattern_says'].upper()}")
            print(f"   LLM judge: {case['llm_says'].upper()}")
            print(f"   LLM reasoning: {case['llm_reason']}")
            print()

    # Speed comparison
    print("=" * 80)
    print("SPEED COMPARISON")
    print("=" * 80)
    print()

    time_per_claim = total_time / len(examples)
    print(f"LLM-as-Judge: {time_per_claim * 1000:.1f}ms per claim")
    print(f"Pattern Matching: ~1ms per claim (estimated)")
    print()
    print(f"LLM judge is ~{time_per_claim * 1000:.0f}x slower")
    print()
    print("For {len(examples)} claims: {total_time:.1f} seconds")
    print("For 1000 claims: ~{time_per_claim * 1000:.0f} seconds (~{time_per_claim * 1000 / 60:.1f} minutes)")
    print()

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if improvement > 5:
        print("STRONG RECOMMENDATION: Use LLM-as-Judge")
        print()
        print("Why:")
        print(f"  - {improvement:.1f} percentage points more accurate")
        print(f"  - Correctly handles {len(llm_better_cases)} cases that pattern matching missed")
        print("  - Speed is acceptable for DSPy optimization (run once offline)")
        print()
        print("For DSPy optimization:")
        print("  -> Use LLM-as-Judge (accuracy matters most)")
        print()
        print("For production evaluation:")
        print("  -> Consider hybrid approach (balance speed and accuracy)")
    elif improvement > 0:
        print("MILD RECOMMENDATION: Consider LLM-as-Judge")
        print()
        print(f"LLM judge is slightly better ({improvement:.1f} points), but the gain is modest.")
        print("Evaluate if the extra time is worth the accuracy improvement.")
    else:
        print("SURPRISING: Pattern matching performed as well or better")
        print()
        print("This suggests:")
        print("  - Pattern matching is sufficient for your task")
        print("  - Or LLM prompt needs refinement")

    # Save results
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()

    if improvement > 0:
        print("1. Review the examples where LLM judge is better")
        print("2. Re-run optimization with LLM-as-Judge:")
        print("   -> Update exp_3_1b_optimize_with_positive_only.py")
        print("   -> Change: from ...metrics_llm_judge import llm_judge_metric")
        print("   -> Run: uv run python exp_3_1b_optimize_with_positive_only.py")
        print()
        print("3. Compare optimization results:")
        print("   - Baseline with pattern metric vs LLM metric")
        print("   - Optimized with pattern metric vs LLM metric")
    else:
        print("1. Pattern matching is sufficient for now")
        print("2. Proceed with optimization using current metric")

    # Save detailed results
    summary = {
        "pattern_accuracy": pattern_accuracy,
        "llm_accuracy": llm_accuracy,
        "improvement": improvement,
        "pattern_correct": pattern_correct,
        "llm_correct": llm_correct,
        "total_claims": len(results),
        "llm_better_count": len(llm_better_cases),
        "pattern_better_count": len(pattern_better_cases),
        "both_wrong_count": len(both_wrong_cases),
        "time_seconds": total_time,
        "time_per_claim_ms": time_per_claim * 1000
    }

    with open('results/experiment_2_5_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("Detailed results saved to results/experiment_2_5_results.json")
    print()


if __name__ == "__main__":
    main()
