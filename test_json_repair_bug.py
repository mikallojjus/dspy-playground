"""
Test script to reproduce the json_repair list splitting bug with apostrophes.

This script tests whether json_repair.loads() incorrectly splits list items
containing apostrophes (possessives) into separate items.
"""
import json
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import json_repair
    print(f"json_repair version: {json_repair.__version__}")
except AttributeError:
    print("json_repair version: unknown (no __version__ attribute)")

print("=" * 80)
print("Testing json_repair.loads() with apostrophes in list items")
print("=" * 80)
print()

test_cases = [
    {
        "name": "Double quotes with apostrophes (valid JSON)",
        "input": '["Trump\'s tariffs", "Biden\'s response", "The Fed\'s independence"]',
        "expected_count": 3,
    },
    {
        "name": "Single quotes (Python-style)",
        "input": "['Trump\\'s tariffs', 'Biden\\'s response']",
        "expected_count": 2,
    },
    {
        "name": "Mixed quotes",
        "input": '["Trump\'s tariffs", \'Biden\'s response\']',
        "expected_count": 2,
    },
    {
        "name": "Kelley Blue Book case",
        "input": '["Kelley Blue Book\'s kbb.com ranks Toyota"]',
        "expected_count": 1,
    },
    {
        "name": "Example from model file",
        "input": '["The Fed president acknowledges the potential for Trump", "s tariff policies to cause a recession"]',
        "expected_count": 2,
    },
]

all_passed = True

for i, test_case in enumerate(test_cases, 1):
    print(f"Test {i}: {test_case['name']}")
    print(f"  Input: {test_case['input']}")

    try:
        # Test with standard json.loads()
        try:
            result_json = json.loads(test_case['input'])
            print(f"  ✓ json.loads(): {len(result_json)} items - {result_json}")
        except json.JSONDecodeError as e:
            print(f"  ✗ json.loads() failed: {e}")
            result_json = None

        # Test with json_repair.loads()
        result_repair = json_repair.loads(test_case['input'])
        print(f"  → json_repair.loads(): {len(result_repair)} items - {result_repair}")

        # Check for split items (items starting with "s ")
        if isinstance(result_repair, list):
            split_items = [item for item in result_repair if isinstance(item, str) and item.startswith("s ")]
            if split_items:
                print(f"  ❌ SPLIT DETECTED: Found {len(split_items)} items starting with 's '")
                print(f"     Split items: {split_items}")
                all_passed = False

            if len(result_repair) != test_case['expected_count']:
                print(f"  ❌ WRONG COUNT: Expected {test_case['expected_count']}, got {len(result_repair)}")
                all_passed = False
            elif not split_items:
                print(f"  ✓ PASSED: Correct count, no splits detected")
        else:
            print(f"  ❌ FAILED: Result is not a list: {type(result_repair)}")
            all_passed = False

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print()

print("=" * 80)
if all_passed:
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
else:
    print("❌❌❌ SOME TESTS FAILED ❌❌❌")
    print()
    print("This confirms json_repair has a bug with apostrophes in list items.")
    print("The monkey patch approach may not work if json_repair corrupts data")
    print("BEFORE parse_value() is called in JSONAdapter.parse() line 154.")
print("=" * 80)
