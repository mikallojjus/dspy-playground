"""
Test the enhanced safe_json_loads() parser with real malformed examples from training.
"""

import sys
import json

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Apply the patch
from src.training.dspy_json_patch import apply_json_patch, safe_json_loads

print("=" * 80)
print("Testing Enhanced Parser with Real Malformed JSON Examples")
print("=" * 80)
print()

# Apply patch
apply_json_patch()
print()

# Real malformed examples from training logs
test_cases = [
    {
        "name": "Double-nested array with leading brackets",
        "input": '[[ "Ruby is a luxury language", "The main cost component is human capacity"]]',
        "expected_count": 2,
    },
    {
        "name": "Leading comma before array",
        "input": ', [\n"Active Record is a pattern",\n"Ruby on Rails uses Active Record"\n]',
        "expected_count": 2,
    },
    {
        "name": "Leading text with markdown",
        "input": ', based on the discussion:\n- The London production includes different actors\n- Brad Pitt\'s film production company acquired rights',
        "expected_count": 2,
    },
    {
        "name": "Leading closing paren",
        "input": ')\n[\'Cory Booker\'s speech was aimed at drawing attention\',\n\'The senator prepared extensively\']',
        "expected_count": 2,
    },
    {
        "name": "Leading closing bracket",
        "input": ']\n[\'Talk radio provides a platform\',\n\'People who call in are often brilliant\']',
        "expected_count": 2,
    },
    {
        "name": "Double-nested with leading comma",
        "input": ',[[ "Gemini has a complex process", "The development process includes multiple stages"]]',
        "expected_count": 2,
    },
    {
        "name": "Valid double-quote JSON",
        "input": '["Trump\'s policy", "Biden\'s response", "Netanyahu\'s position"]',
        "expected_count": 3,
    },
]

all_passed = True

for i, test_case in enumerate(test_cases, 1):
    print(f"Test {i}: {test_case['name']}")
    print(f"  Input (first 100 chars): {test_case['input'][:100]}...")

    try:
        result = safe_json_loads(test_case['input'], context=f"Test{i}")

        print(f"  Result type: {type(result)}")
        print(f"  Count: {len(result)} (expected: {test_case['expected_count']})")

        if isinstance(result, list):
            print(f"  Items:")
            for item in result[:3]:  # Show first 3
                print(f"    - {item[:80] if len(item) > 80 else item}")

        # Check count
        if len(result) != test_case['expected_count']:
            print(f"  ❌ FAILED: Wrong count! Expected {test_case['expected_count']}, got {len(result)}")
            all_passed = False
        else:
            # Check for split items (items starting with "s ")
            if isinstance(result, list):
                split_items = [item for item in result if isinstance(item, str) and item.startswith("s ")]
                if split_items:
                    print(f"  ❌ FAILED: Found split items: {split_items}")
                    all_passed = False
                else:
                    print(f"  ✓ PASSED")
            else:
                print(f"  ✓ PASSED")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print()

print("=" * 80)
if all_passed:
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("The enhanced parser successfully handles all malformed JSON patterns!")
else:
    print("❌❌❌ SOME TESTS FAILED ❌❌❌")
    print("The parser needs further refinement.")
print("=" * 80)
