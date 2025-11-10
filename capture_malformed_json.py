"""
Helper script to extract malformed JSON examples from training logs.

Usage:
    1. Run training and capture output:
       PowerShell: uv run python -m src.training.train_claim_extractor --fresh-start 2>&1 | Tee-Object -FilePath training_nuclear.log
       CMD: uv run python -m src.training.train_claim_extractor --fresh-start > training_nuclear.log 2>&1

    2. Extract malformed JSON examples:
       python capture_malformed_json.py training_nuclear.log
"""

import sys
import re

def extract_malformed_json_examples(log_file, max_examples=3):
    """Extract the first N malformed JSON examples from log file."""

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all malformed JSON captures
    pattern = r'🔍 RAW MALFORMED JSON CAPTURED.*?={80}'
    matches = re.findall(pattern, content, re.DOTALL)

    print(f"Found {len(matches)} malformed JSON examples in log file")
    print("=" * 80)

    for i, match in enumerate(matches[:max_examples], 1):
        print(f"\nExample {i}:")
        print("=" * 80)
        print(match)
        print("=" * 80)

    if len(matches) > max_examples:
        print(f"\n... and {len(matches) - max_examples} more examples (showing first {max_examples})")

    # Also save to file
    output_file = log_file.replace('.log', '_malformed_json_examples.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, match in enumerate(matches[:max_examples], 1):
            f.write(f"Example {i}:\n")
            f.write("=" * 80 + "\n")
            f.write(match)
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python capture_malformed_json.py <training_log_file>")
        print("Example: python capture_malformed_json.py training_nuclear.log")
        sys.exit(1)

    log_file = sys.argv[1]
    extract_malformed_json_examples(log_file)
