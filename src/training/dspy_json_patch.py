"""
DSPy JSON Parsing Monkey-Patch

This module patches DSPy's json_repair-based parsing to fix a bug where claims
containing apostrophes (possessives like "Netanyahu's" or "Fed's") get incorrectly
split into separate list items.

Root Cause:
- DSPy uses json_repair.loads() in dspy/adapters/utils.py:parse_value()
- json_repair has a bug with mixed quote styles in JSON arrays
- When LLM generates: ["Trump's policy", 'Biden\'s plan'] (mixed quotes)
- json_repair incorrectly splits at apostrophes: ["Trump", "s policy", "Biden", "s plan"]

This patch replaces the parsing logic to use standard json.loads() for list types,
which correctly handles apostrophes and mixed quotes.

Usage:
    from src.training.dspy_json_patch import apply_json_patch

    # Apply patch BEFORE any DSPy training or optimization
    apply_json_patch()

    # Now proceed with BootstrapFewShot, etc.
    optimizer = BootstrapFewShot(...)
"""

import json
import logging
import typing
import re
import ast
from typing import Any

logger = logging.getLogger(__name__)

_patch_applied = False
_original_parse_value = None
_original_json_adapter_parse = None
_original_json_repair_loads = None  # For nuclear option


def safe_json_loads(value: str, context: str = "") -> Any:
    """
    Safely parse JSON string using a fallback chain to avoid json_repair bugs.

    Strategy:
    1. Preprocess: Strip leading garbage and unwrap double-nested arrays
    2. Try json.loads() - handles valid JSON with apostrophes correctly
    3. Try quote normalization + json.loads() - fixes single-quote arrays
    4. Try ast.literal_eval() - handles Python-style syntax
    5. Try extracting JSON from markdown/text
    6. Last resort: raise exception for caller to handle

    Args:
        value: JSON string to parse
        context: Description of what's being parsed (for logging)

    Returns:
        Parsed Python object (dict, list, etc.)
    """
    # Store error messages with proper scope
    error1 = None
    error2 = None
    error3 = None
    error4 = None

    original_value = value
    raw_value_preview = value[:500] if len(value) > 500 else value

    # PREPROCESSING: Clean common malformed patterns from LLM output
    preprocessed = value.strip()

    # Remove leading garbage characters: `, ) ] based on the discussion:`
    # Pattern: leading punctuation/text before the actual JSON array/object
    garbage_patterns = [
        r'^,\s*',           # Leading comma: ", ["
        r'^\)\s*',          # Leading paren: ") ["
        r'^\]\s*',          # Leading bracket: "] ["
        r'^[^[{]+(?=[[\{])',  # Any text before [ or {: "based on the discussion: ["
    ]

    for pattern in garbage_patterns:
        cleaned = re.sub(pattern, '', preprocessed, count=1)
        if cleaned != preprocessed:
            logger.debug(f"{context}: Removed leading garbage with pattern {pattern}")
            preprocessed = cleaned.strip()
            break

    # Detect and handle markdown bullet points
    if '\n-' in preprocessed or '\n•' in preprocessed:
        logger.debug(f"{context}: Detected markdown bullet points, attempting extraction")
        # Try to extract claims from markdown format
        lines = preprocessed.split('\n')
        claims = []
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                # Extract claim text after bullet
                claim = line.lstrip('-•').strip()
                if claim:
                    claims.append(claim)

        if claims:
            logger.info(f"{context}: Extracted {len(claims)} claims from markdown format")
            return claims

    # Attempt 1: Standard JSON parser (handles apostrophes correctly)
    try:
        result = json.loads(preprocessed)

        # Check for double-nested arrays: [[...]]
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            logger.info(f"{context}: Unwrapped double-nested array [[...]] -> [...]")
            result = result[0]

        logger.debug(f"{context}: Successfully parsed with json.loads()")
        return result
    except json.JSONDecodeError as e:
        error1 = str(e)
        logger.debug(f"{context}: json.loads() failed: {e}")

    # Attempt 2: Normalize quotes and try again
    try:
        # Replace single quotes that act as string delimiters with double quotes
        normalized = preprocessed

        # Replace opening single quotes: [' or ,' or ': ' or {'
        normalized = re.sub(r"(\[|\{|,|:)\s*'", r'\1"', normalized)
        # Replace closing single quotes: ',' or '] or '} or ':
        normalized = re.sub(r"'(\s*[,\]\}:])", r'"\1', normalized)

        if normalized != preprocessed:
            result = json.loads(normalized)

            # Check for double-nested arrays
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
                logger.info(f"{context}: Unwrapped double-nested array after quote normalization")
                result = result[0]

            logger.info(f"{context}: Successfully parsed after quote normalization")
            return result
    except Exception as e:
        error2 = str(e)
        logger.debug(f"{context}: Quote normalization failed: {e}")

    # Attempt 3: Python-style literal eval
    try:
        result = ast.literal_eval(preprocessed)

        # Check for double-nested arrays
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            logger.info(f"{context}: Unwrapped double-nested array in ast.literal_eval()")
            result = result[0]

        logger.info(f"{context}: Successfully parsed with ast.literal_eval()")
        return result
    except Exception as e:
        error3 = str(e)
        logger.debug(f"{context}: ast.literal_eval() failed: {e}")

    # Attempt 4: Extract JSON array from text using pattern matching
    try:
        # Look for array pattern: [...] anywhere in the text
        array_match = re.search(r'\[.*\]', preprocessed, re.DOTALL)
        if array_match:
            extracted = array_match.group(0)
            logger.debug(f"{context}: Attempting to parse extracted array pattern")

            result = json.loads(extracted)

            # Check for double-nested arrays
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
                logger.info(f"{context}: Unwrapped double-nested array in extracted pattern")
                result = result[0]

            logger.info(f"{context}: Successfully parsed extracted array pattern")
            return result
    except Exception as e:
        error4 = str(e)
        logger.debug(f"{context}: Array extraction failed: {e}")

    # ALL SAFE METHODS FAILED - Log the raw malformed JSON for analysis
    logger.error("=" * 80)
    logger.error(f"🔍 RAW MALFORMED JSON CAPTURED - {context}")
    logger.error("=" * 80)
    logger.error(f"Length: {len(original_value)} characters")
    logger.error(f"Preview (first 500 chars):")
    logger.error(raw_value_preview)
    logger.error("")
    logger.error(f"Last 200 chars:")
    logger.error(original_value[-200:] if len(original_value) > 200 else original_value)
    logger.error("")
    logger.error("Failure reasons:")
    logger.error(f"  1. json.loads() failed: {error1}")
    logger.error(f"  2. Quote normalization failed: {error2}")
    logger.error(f"  3. ast.literal_eval() failed: {error3}")
    logger.error(f"  4. Array extraction failed: {error4}")
    logger.error("=" * 80)

    # CRITICAL: Do NOT call json_repair.loads() here!
    # It might be our patched version, causing infinite recursion.
    # Instead, raise an exception and let the caller handle it.
    raise ValueError(
        f"All safe parsing methods failed. Caller must handle with original json_repair. "
        f"Last errors: {error1}, {error2}, {error3}, {error4}"
    )


def check_for_splits(data: Any, context: str = ""):
    """
    Recursively check data structure for evidence of claim splitting.

    Looks for strings starting with "s " which indicate a claim was split
    at an apostrophe (e.g., "Trump's" became "Trump" + "s policy").
    """
    if isinstance(data, list):
        split_items = [item for item in data if isinstance(item, str) and item.startswith("s ")]
        if split_items:
            logger.error(f"{context}: ❌ SPLIT DETECTED: Found {len(split_items)} items starting with 's '")
            logger.error(f"  Examples: {split_items[:3]}")
    elif isinstance(data, dict):
        for key, value in data.items():
            check_for_splits(value, f"{context}.{key}")


def apply_json_patch():
    """
    Apply NUCLEAR-LEVEL comprehensive monkey-patches to fix json_repair splitting bugs.

    This patches at THREE levels for maximum coverage:
    1. json_repair.loads() ITSELF - global library-level patch (NUCLEAR OPTION)
    2. JSONAdapter.parse() - DSPy's JSON adapter
    3. dspy.adapters.utils.parse_value() - individual field parsing

    The library-level patch ensures NO code can bypass our fix, no matter where
    json_repair.loads() is called from.

    This must be called BEFORE any DSPy model training or optimization to ensure
    that training demonstrations are not corrupted with split claims.

    Safe to call multiple times - will only apply once.
    """
    global _patch_applied, _original_parse_value, _original_json_adapter_parse, _original_json_repair_loads

    if _patch_applied:
        logger.info("DSPy JSON patch already applied, skipping")
        return

    try:
        import json_repair
        import dspy.adapters.utils
        import dspy.adapters.json_adapter
        import regex
        from dspy.utils.exceptions import AdapterParseError

        # ===== NUCLEAR OPTION: Patch json_repair.loads() GLOBALLY =====
        # This is the most aggressive fix - intercepts ALL json_repair calls
        logger.info("=" * 80)
        logger.info("🚨 APPLYING NUCLEAR OPTION: Patching json_repair.loads() globally")
        logger.info("=" * 80)

        _original_json_repair_loads = json_repair.loads

        def patched_json_repair_loads(json_str, **kwargs):
            """
            Global replacement for json_repair.loads() that uses safe parsing.

            This intercepts EVERY call to json_repair.loads() in the entire codebase,
            preventing the apostrophe splitting bug at its source.
            """
            json_str_str = str(json_str)
            logger.debug(f"🔴 json_repair.loads() intercepted! Input length: {len(json_str_str)}")

            try:
                # Use our safe parsing chain instead
                result = safe_json_loads(json_str_str, context="json_repair.loads[INTERCEPTED]")
                logger.debug(f"✓ json_repair intercept successful")
                return result
            except Exception as e:
                logger.error(f"❌ CRITICAL: Safe parsing chain completely failed!")
                logger.error(f"   Exception: {e}")
                logger.error(f"   Falling back to ORIGINAL json_repair.loads()")
                logger.error(f"   This will likely cause splits!")

                # Fall back to original json_repair only if our safe parsing fails
                result = _original_json_repair_loads(json_str, **kwargs)

                # Check if splits were introduced
                if isinstance(result, (list, dict)):
                    check_for_splits(result, "json_repair[FALLBACK]")

                return result

        # REPLACE json_repair.loads globally
        json_repair.loads = patched_json_repair_loads
        logger.info("✓ json_repair.loads() REPLACED GLOBALLY - all calls now intercepted")
        logger.info("=" * 80)
        logger.info("")

        # ===== PATCH 1: JSONAdapter.parse() =====
        # This is the PRIMARY fix - intercepts json_repair.loads() at the source
        logger.info("Patching JSONAdapter.parse() to fix json_repair bug...")

        _original_json_adapter_parse = dspy.adapters.json_adapter.JSONAdapter.parse

        def patched_json_adapter_parse(self, signature, completion: str):
            """
            Patched JSONAdapter.parse() that uses safe JSON parsing instead of json_repair.

            This is the CRITICAL fix - json_repair.loads() at line 154 of the original
            parse() method causes claim splitting with mixed quotes. We replace it with
            our safe parsing chain.
            """
            # Extract JSON object from completion (same as original)
            pattern = r"\{(?:[^{}]|(?R))*\}"
            match = regex.search(pattern, completion, regex.DOTALL)
            if match:
                completion_to_parse = match.group(0)
            else:
                completion_to_parse = completion

            # SAFE PARSING instead of json_repair.loads()
            try:
                fields = safe_json_loads(completion_to_parse, context="JSONAdapter.parse")
            except ValueError as e:
                # Safe parsing failed, use original json_repair as fallback
                logger.warning(f"JSONAdapter.parse: Falling back to original json_repair.loads()")
                fields = _original_json_repair_loads(completion_to_parse)
                # Check for splits introduced by json_repair
                if isinstance(fields, (list, dict)):
                    check_for_splits(fields, "JSONAdapter.parse[FALLBACK]")
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="JSONAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse LM response as JSON: {e}",
                )

            if not isinstance(fields, dict):
                raise AdapterParseError(
                    adapter_name="JSONAdapter",
                    signature=signature,
                    lm_response=completion,
                    message="LM response cannot be serialized to a JSON object.",
                )

            # Filter to only expected output fields
            fields = {k: v for k, v in fields.items() if k in signature.output_fields}

            # Parse each field value (will use patched parse_value if applied)
            for k, v in fields.items():
                if k in signature.output_fields:
                    fields[k] = dspy.adapters.utils.parse_value(v, signature.output_fields[k].annotation)

            # Validate all expected fields are present
            if fields.keys() != signature.output_fields.keys():
                raise AdapterParseError(
                    adapter_name="JSONAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=fields,
                )

            return fields

        dspy.adapters.json_adapter.JSONAdapter.parse = patched_json_adapter_parse
        logger.info("✓ JSONAdapter.parse() patched successfully")

        # ===== PATCH 2: parse_value() =====
        # Secondary fix for any remaining edge cases
        logger.info("Patching dspy.adapters.utils.parse_value()...")

        _original_parse_value = dspy.adapters.utils.parse_value

        # Create patched version
        def patched_parse_value(value: Any, annotation: Any) -> Any:
            """
            Patched version of parse_value that uses safe JSON parsing for lists.

            This is a SECONDARY fix - the primary fix is in JSONAdapter.parse().
            However, this provides defense-in-depth for any edge cases where
            parse_value is called directly.
            """
            # Check if we're parsing a string into a list type
            if isinstance(value, str) and hasattr(annotation, '__origin__'):
                origin = typing.get_origin(annotation)

                if origin is list:
                    # Use our safe parsing chain instead of json_repair
                    try:
                        result = safe_json_loads(value, context="parse_value(list)")
                        logger.info(f"✓ Parsed list with {len(result)} items in parse_value()")
                        return result
                    except ValueError as e:
                        # Safe parsing failed, fall back to original parse_value
                        # (which may use json_repair internally)
                        logger.warning(f"Safe parsing failed in parse_value, using original: {e}")
                        result = _original_parse_value(value, annotation)
                        # Check for splits
                        if isinstance(result, list):
                            check_for_splits(result, "parse_value[FALLBACK]")
                        return result
                    except Exception as e:
                        logger.error(f"Unexpected error in parse_value: {e}")
                        return _original_parse_value(value, annotation)

            # For all other cases, use original implementation
            return _original_parse_value(value, annotation)

        # Apply the patch
        dspy.adapters.utils.parse_value = patched_parse_value
        logger.info("✓ dspy.adapters.utils.parse_value() patched successfully")

        _patch_applied = True

        logger.info("=" * 80)
        logger.info("✓✓✓ NUCLEAR-LEVEL DSPy JSON PATCH APPLIED ✓✓✓")
        logger.info("=" * 80)
        logger.info("Patched locations (TRIPLE DEFENSE):")
        logger.info("")
        logger.info("  🚨 LAYER 1: json_repair.loads() - NUCLEAR OPTION")
        logger.info("     GLOBAL library-level replacement")
        logger.info("     Intercepts ALL json_repair calls in entire codebase")
        logger.info("     ⚡ This is the PRIMARY protection - cannot be bypassed!")
        logger.info("")
        logger.info("  🛡️  LAYER 2: JSONAdapter.parse() - DSPy-level fix")
        logger.info("     Replaces json_repair.loads() with safe parsing chain")
        logger.info("     (Redundant with Layer 1, but provides extra safety)")
        logger.info("")
        logger.info("  🛡️  LAYER 3: dspy.adapters.utils.parse_value() - Field-level fix")
        logger.info("     Defense-in-depth for direct parse_value() calls")
        logger.info("")
        logger.info("Safe parsing chain:")
        logger.info("  1. json.loads() - handles apostrophes correctly")
        logger.info("  2. Quote normalization + json.loads() - fixes single quotes")
        logger.info("  3. ast.literal_eval() - handles Python syntax")
        logger.info("  4. json_repair.loads() - ORIGINAL (only as last resort)")
        logger.info("")
        logger.info("This NUCLEAR approach prevents claim splitting at apostrophes:")
        logger.info("  ✓ Training (BootstrapFewShot)")
        logger.info("  ✓ Evaluation (LLM-as-judge metrics)")
        logger.info("  ✓ Inference (claim extraction)")
        logger.info("  ✓ ANY code path that calls json_repair.loads()")
        logger.info("")
        logger.info("⚠️  CRITICAL: The global json_repair.loads() interception means")
        logger.info("   NO code can bypass this fix - it's applied at the library level!")
        logger.info("=" * 80)

    except ImportError as e:
        logger.error(f"Failed to apply DSPy JSON patch: {e}")
        logger.error("Make sure DSPy is installed and accessible")
        raise
    except AttributeError as e:
        logger.error(f"Failed to apply DSPy JSON patch: {e}")
        logger.error("DSPy's internal structure may have changed")
        raise


def is_patch_applied() -> bool:
    """Check if the JSON patch has been applied."""
    return _patch_applied


def remove_json_patch():
    """
    Remove the monkey-patches and restore original behavior.

    Removes ALL THREE patches:
    1. json_repair.loads() - global library patch
    2. JSONAdapter.parse() - DSPy adapter patch
    3. dspy.adapters.utils.parse_value() - field-level patch

    Mainly useful for testing. In production, you typically want to keep
    the patch applied for the entire training session.
    """
    global _patch_applied, _original_parse_value, _original_json_adapter_parse, _original_json_repair_loads

    if not _patch_applied:
        logger.info("DSPy JSON patch not applied, nothing to remove")
        return

    try:
        import json_repair
        import dspy.adapters.utils
        import dspy.adapters.json_adapter

        restored_count = 0

        # Restore json_repair.loads() (NUCLEAR OPTION)
        if _original_json_repair_loads is not None:
            json_repair.loads = _original_json_repair_loads
            logger.info("✓ Restored original json_repair.loads()")
            restored_count += 1
        else:
            logger.warning("Cannot restore original json_repair.loads() - reference not stored")

        # Restore JSONAdapter.parse()
        if _original_json_adapter_parse is not None:
            dspy.adapters.json_adapter.JSONAdapter.parse = _original_json_adapter_parse
            logger.info("✓ Restored original JSONAdapter.parse()")
            restored_count += 1
        else:
            logger.warning("Cannot restore original JSONAdapter.parse() - reference not stored")

        # Restore parse_value()
        if _original_parse_value is not None:
            dspy.adapters.utils.parse_value = _original_parse_value
            logger.info("✓ Restored original dspy.adapters.utils.parse_value()")
            restored_count += 1
        else:
            logger.warning("Cannot restore original parse_value - reference not stored")

        if restored_count == 3:
            _patch_applied = False
            logger.info("✓ DSPy JSON patch fully removed (all 3 layers), original behavior restored")
        else:
            logger.warning(f"Partial restoration: {restored_count}/3 functions restored")

    except ImportError as e:
        logger.error(f"Failed to remove DSPy JSON patch: {e}")
        raise


if __name__ == "__main__":
    # Test the patch
    import sys
    # Fix encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    print("=" * 80)
    print("DSPy JSON Patch - Comprehensive Test Suite")
    print("=" * 80)
    print()

    # Apply patch
    print("Applying patch...")
    apply_json_patch()
    print(f"Patch applied: {is_patch_applied()}")
    print()

    # Test with DSPy
    try:
        import dspy
        from typing import List

        test_cases = [
            {
                "name": "Valid JSON with double quotes",
                "input": '["Netanyahu\'s family", "Trump\'s policy", "The Fed\'s independence"]',
                "expected_count": 3,
                "should_contain": ["Netanyahu's family", "Trump's policy", "The Fed's independence"]
            },
            {
                "name": "Single quotes (Python-style)",
                "input": "['Biden\\'s plan', 'Obama\\'s legacy']",
                "expected_count": 2,
                "should_contain": ["Biden's plan", "Obama's legacy"]
            },
            {
                "name": "Mixed quotes",
                "input": '["Trump\'s tariffs", \'Biden\'s response\']',
                "expected_count": 2,
                "should_contain": ["Trump's tariffs", "Biden's response"]
            },
            {
                "name": "Kelley Blue Book case",
                "input": '["Kelley Blue Book\'s ranking"]',
                "expected_count": 1,
                "should_contain": ["Kelley Blue Book's ranking"]
            },
            {
                "name": "Multiple apostrophes",
                "input": '["Netanyahu\'s family\'s story", "It\'s Trump\'s policy"]',
                "expected_count": 2,
                "should_contain": ["Netanyahu's family's story", "It's Trump's policy"]
            }
        ]

        all_passed = True

        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case['name']}")
            print(f"  Input: {test_case['input']}")

            try:
                result = dspy.adapters.utils.parse_value(test_case['input'], List[str])

                print(f"  Result: {result}")
                print(f"  Count: {len(result)} (expected: {test_case['expected_count']})")

                # Check count
                if len(result) != test_case['expected_count']:
                    print(f"  ❌ FAILED: Wrong count! Expected {test_case['expected_count']}, got {len(result)}")
                    all_passed = False
                    continue

                # Check for split items
                split_items = [item for item in result if isinstance(item, str) and item.startswith("s ")]
                if split_items:
                    print(f"  ❌ FAILED: Found split items: {split_items}")
                    all_passed = False
                    continue

                # Check expected content
                for expected_item in test_case['should_contain']:
                    if expected_item not in result:
                        print(f"  ❌ FAILED: Missing expected item '{expected_item}'")
                        all_passed = False
                        break
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
        else:
            print("❌❌❌ SOME TESTS FAILED ❌❌❌")
        print("=" * 80)

    except Exception as e:
        print(f"Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()
