"""
Experiment 2.2: Test the Simple Metric

Goal: Verify that the claim_quality_metric correctly identifies quality issues
      and agrees with manual judgments from Experiment 2.1

This test loads the manual review data and checks how well the automated
metric agrees with human judgment.
"""

import json
import sys
from ...metrics import claim_quality_metric, strict_claim_quality_metric

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def test_metric_on_manual_review():
    """Test the metric against manual review examples."""

    # Load manual review data
    with open('evaluation/claims_manual_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 80)
    print("Experiment 2.2: Testing Claim Quality Metric")
    print("=" * 80)
    print(f"\nTesting on {len(data['examples'])} manually reviewed examples")
    print()

    # Test both metrics
    for metric_name, metric_func in [
        ("Standard Metric", claim_quality_metric),
        ("Strict Metric", strict_claim_quality_metric)
    ]:
        print("\n" + "=" * 80)
        print(f"Testing: {metric_name}")
        print("=" * 80)

        correct_good = 0  # Metric says good (score=1.0), manual says good
        correct_bad = 0   # Metric says bad (score<1.0), manual says bad
        false_positive = 0  # Metric says bad, manual says good
        false_negative = 0  # Metric says good, manual says bad

        total_good = 0
        total_bad = 0

        print("\nDetailed results:")
        print("-" * 80)

        for i, example in enumerate(data['examples'], 1):
            # Create a mock prediction object
            pred = type('obj', (object,), {
                'claims': [example['claim']]
            })

            # Get metric score
            score = metric_func(None, pred)

            # Compare to manual judgment
            manual_quality = example['quality']
            metric_says_good = (score == 1.0)
            manual_says_good = (manual_quality == 'good')

            # Update counters
            if manual_says_good:
                total_good += 1
                if metric_says_good:
                    correct_good += 1
                    match = "✅"
                else:
                    false_positive += 1
                    match = "❌ FALSE POSITIVE"
            else:  # manual says bad
                total_bad += 1
                if not metric_says_good:
                    correct_bad += 1
                    match = "✅"
                else:
                    false_negative += 1
                    match = "❌ FALSE NEGATIVE"

            # Print result
            claim_preview = example['claim'][:60] + "..." if len(example['claim']) > 60 else example['claim']
            print(f"\n{i:2d}. {match}")
            print(f"    Claim: {claim_preview}")
            print(f"    Metric: {score:.2f} (says {'GOOD' if metric_says_good else 'BAD'})")
            print(f"    Manual: {manual_quality.upper()}")
            if example['issues']:
                print(f"    Issues: {', '.join(example['issues'])}")

        # Calculate metrics
        total = len(data['examples'])
        accuracy = (correct_good + correct_bad) / total * 100 if total > 0 else 0
        precision = correct_bad / (correct_bad + false_positive) * 100 if (correct_bad + false_positive) > 0 else 0
        recall = correct_bad / total_bad * 100 if total_bad > 0 else 0

        print("\n" + "=" * 80)
        print(f"RESULTS FOR {metric_name}")
        print("=" * 80)
        print(f"\nTotal examples: {total}")
        print(f"  Good claims (manual): {total_good}")
        print(f"  Bad claims (manual): {total_bad}")
        print()
        print(f"Metric Performance:")
        print(f"  Correct (Good → Good): {correct_good}/{total_good}")
        print(f"  Correct (Bad → Bad):   {correct_bad}/{total_bad}")
        print(f"  False Positives:       {false_positive} (flagged good claims as bad)")
        print(f"  False Negatives:       {false_negative} (missed bad claims)")
        print()
        print(f"Overall Metrics:")
        print(f"  Accuracy:  {accuracy:.1f}% (how often metric matches manual judgment)")
        print(f"  Precision: {precision:.1f}% (when metric says bad, how often is it right)")
        print(f"  Recall:    {recall:.1f}% (what % of bad claims does metric catch)")
        print()

        # Interpretation
        if accuracy >= 80:
            print("✅ EXCELLENT - Metric agrees with manual review >80% of the time")
        elif accuracy >= 60:
            print("⚠️  GOOD - Metric agrees >60% but could be refined")
        else:
            print("❌ NEEDS WORK - Metric agreement <60%, rethink approach")
        print()


def test_metric_on_sample_claims():
    """Test the metric on some crafted example claims."""

    print("\n" + "=" * 80)
    print("Testing on Sample Claims")
    print("=" * 80)

    test_cases = [
        {
            "claim": "Bitcoin reached $69,000 in November 2021",
            "expected": "good",
            "reason": "Specific, factual, self-contained"
        },
        {
            "claim": "He said it was amazing",
            "expected": "bad",
            "reason": "Contains pronoun 'he'"
        },
        {
            "claim": "The new policy will help people",
            "expected": "bad",
            "reason": "Vague - which policy? which people?"
        },
        {
            "claim": "President Biden signed the Infrastructure Investment and Jobs Act in November 2021",
            "expected": "good",
            "reason": "Specific president, specific bill, specific timeframe"
        },
        {
            "claim": "I think the economy is doing well",
            "expected": "bad",
            "reason": "Opinion - contains 'think'"
        },
        {
            "claim": "Get your subscription today for only $9.99",
            "expected": "bad",
            "reason": "Advertisement"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        pred = type('obj', (object,), {
            'claims': [test_case['claim']]
        })

        score = claim_quality_metric(None, pred)
        metric_says_good = (score == 1.0)
        expected_good = (test_case['expected'] == 'good')

        match = "✅" if metric_says_good == expected_good else "❌"

        print(f"\n{i}. {match} {test_case['claim']}")
        print(f"   Expected: {test_case['expected'].upper()}")
        print(f"   Metric: {score:.2f} ({'GOOD' if metric_says_good else 'BAD'})")
        print(f"   Reason: {test_case['reason']}")

    print()


if __name__ == "__main__":
    test_metric_on_manual_review()
    test_metric_on_sample_claims()
