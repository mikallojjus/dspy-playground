"""
Test script to verify the fix_split_claims function works correctly.

This tests the fix for the json_repair bug that splits claims at apostrophes.
"""

import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

from src.dspy_models.claim_extractor import fix_split_claims


def test_fix_split_claims():
    """Test various cases of split claims."""

    print("Testing fix_split_claims function\n")
    print("=" * 80)

    # Test case 1: Split claim with possessive
    print("\n1. Split possessive claim:")
    input1 = ["Trump", "s tariff policy could create economic issues"]
    output1 = fix_split_claims(input1)
    print(f"   Input:  {input1}")
    print(f"   Output: {output1}")
    assert len(output1) == 1
    assert output1[0] == "Trump's tariff policy could create economic issues"
    print("   ✓ PASSED")

    # Test case 2: Multiple split claims
    print("\n2. Multiple split claims:")
    input2 = [
        "Netanyahu",
        "s family faced persecution",
        "Israel",
        "s withdrawal from Lebanon led to conflict"
    ]
    output2 = fix_split_claims(input2)
    print(f"   Input:  {input2}")
    print(f"   Output: {output2}")
    assert len(output2) == 2
    assert output2[0] == "Netanyahu's family faced persecution"
    assert output2[1] == "Israel's withdrawal from Lebanon led to conflict"
    print("   ✓ PASSED")

    # Test case 3: Normal claims (no splits)
    print("\n3. Normal claims without splits:")
    input3 = [
        "Bitcoin reached $69,000 in November 2021",
        "The pandemic affected global economies"
    ]
    output3 = fix_split_claims(input3)
    print(f"   Input:  {input3}")
    print(f"   Output: {output3}")
    assert output3 == input3
    print("   ✓ PASSED")

    # Test case 4: Claims with proper apostrophes (not split)
    print("\n4. Claims with proper apostrophes:")
    input4 = ["Trump's policy was controversial"]
    output4 = fix_split_claims(input4)
    print(f"   Input:  {input4}")
    print(f"   Output: {output4}")
    assert output4 == input4
    print("   ✓ PASSED")

    # Test case 5: Mixed - some split, some not
    print("\n5. Mixed claims:")
    input5 = [
        "Bitcoin reached new highs",
        "Assad",
        "s family rule ended after civil war",
        "The economy recovered"
    ]
    output5 = fix_split_claims(input5)
    print(f"   Input:  {input5}")
    print(f"   Output: {output5}")
    assert len(output5) == 3
    assert output5[1] == "Assad's family rule ended after civil war"
    print("   ✓ PASSED")

    # Test case 6: Empty list
    print("\n6. Empty list:")
    input6 = []
    output6 = fix_split_claims(input6)
    print(f"   Input:  {input6}")
    print(f"   Output: {output6}")
    assert output6 == []
    print("   ✓ PASSED")

    # Test case 7: Single claim not split
    print("\n7. Single claim:")
    input7 = ["The Federal Reserve maintained interest rates"]
    output7 = fix_split_claims(input7)
    print(f"   Input:  {input7}")
    print(f"   Output: {output7}")
    assert output7 == input7
    print("   ✓ PASSED")

    # Test case 8: Your actual database examples
    print("\n8. Real-world database examples:")
    input8 = [
        "leading them to believe they could maintain the status quo",
        "Netanyahu",
        "s family faced persecution due to their Jewish roots.'",
        "The episode discusses foreign policy"
    ]
    output8 = fix_split_claims(input8)
    print(f"   Input:  {input8}")
    print(f"   Output: {output8}")
    assert len(output8) == 3
    assert "Netanyahu's family faced persecution" in output8[1]
    print("   ✓ PASSED")

    print("\n" + "=" * 80)
    print("All tests PASSED! ✓")
    print("\nThe fix_split_claims function is working correctly.")


if __name__ == "__main__":
    test_fix_split_claims()
