#!/usr/bin/env python3
"""
Process claims_from_chunks_reviewed.csv and generate training examples.

This script:
1. Parses the CSV handling different claim formats from different editors
2. Extracts individual claims from each row
3. Generates training examples in the required format
4. Saves to evaluation/claims_manual_review.json

Different editors use different formats:
- Ahmed: JSON with {"claims": [{"claim": "..."}]}
- Catalin: JSON with {"claims": ["...", "..."]}
- Dovile: Semicolon-separated plain text
- Simas: Semicolon-separated format
"""

import csv
import json
import re
import sys
from pathlib import Path


def parse_claims(claims_text: str, maintainer: str) -> list[str]:
    """
    Parse claims from different formats depending on the editor.

    Returns a list of individual claim strings.
    """
    claims_text = claims_text.strip()

    if not claims_text:
        return []

    # Check for special cases first
    if "only ads" in claims_text.lower() and "no claims" in claims_text.lower():
        return []

    # Try to parse as JSON first (Ahmed and Catalin formats)
    if claims_text.startswith('{') or claims_text.startswith('['):
        data = None

        # Try parsing as-is first
        try:
            data = json.loads(claims_text)
        except json.JSONDecodeError:
            # Try fixing common JSON errors
            fixed_text = claims_text

            # Fix 1: Remove trailing commas
            fixed_text = re.sub(r',(\s*[\]}])', r'\1', fixed_text)

            # Fix 2: Add missing commas between objects (e.g., } { -> }, {)
            fixed_text = re.sub(r'}\s*{', r'}, {', fixed_text)

            # Fix 3: Remove control characters (tabs, newlines) from inside strings
            # This is a heuristic: replace any literal control chars with spaces
            import codecs
            fixed_text = ''.join(c if c.isprintable() or c in '\n\r\t ' else ' ' for c in fixed_text)

            try:
                data = json.loads(fixed_text)
            except json.JSONDecodeError:
                # Last resort: try to manually extract claims using regex
                # Look for "claim": "..." patterns
                claim_pattern = r'"claim"\s*:\s*"([^"]*(?:\\"[^"]*)*)"'
                matches = re.findall(claim_pattern, fixed_text)
                if matches:
                    return [m.replace('\\"', '"') for m in matches]
                pass  # Fall through to other parsing methods

        # If JSON parsing succeeded, extract claims
        if data is not None:
            # Handle direct array format: [{"claim": "..."}, {"claim": "..."}]
            if isinstance(data, list):
                # Array of claim objects with 'claim' key
                if data and isinstance(data[0], dict) and 'claim' in data[0]:
                    return [item['claim'] for item in data if 'claim' in item]
                # Array of strings
                elif data and isinstance(data[0], str):
                    return data

            # Handle object format: {"claims": [...]}
            if isinstance(data, dict) and 'claims' in data and isinstance(data['claims'], list):
                claims_list = data['claims']

                # Ahmed format: [{"claim": "..."}, {"claim": "..."}]
                if claims_list and isinstance(claims_list[0], dict):
                    return [item['claim'] for item in claims_list if 'claim' in item]

                # Catalin format: ["claim1", "claim2"]
                elif claims_list and isinstance(claims_list[0], str):
                    return claims_list

    # Dovile/Simas format: semicolon-separated
    if ';' in claims_text:
        claims = [claim.strip() for claim in claims_text.split(';')]
        # Filter out empty strings and obvious placeholders
        claims = [c for c in claims if c and c.lower() not in ['etc', 'etc.', '...']]
        return claims

    # If we get here, treat the whole text as a single claim
    # (might be a single claim without any separators)
    if claims_text:
        return [claims_text]

    return []


def process_csv(csv_file: str, output_file: str):
    """Process the CSV and generate training examples."""

    print("Processing claims from chunks reviewed CSV...")
    print("=" * 80)
    print()

    # Group claims by chunk using a dict
    chunks_to_claims = {}
    stats = {
        'total_rows': 0,
        'rows_with_claims': 0,
        'total_claims': 0,
        'by_maintainer': {}
    }

    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            stats['total_rows'] += 1

            chunk_text = row.get('chunk_text', '').strip()
            claims_text = row.get('Claims', '').strip()
            maintainer = row.get('Maintainer', '').strip()

            # Initialize maintainer stats
            if maintainer and maintainer not in stats['by_maintainer']:
                stats['by_maintainer'][maintainer] = {
                    'rows': 0,
                    'claims': 0
                }

            # Skip if no claims or no chunk text
            if not claims_text or not chunk_text:
                continue

            # Parse claims based on format
            claims = parse_claims(claims_text, maintainer)

            if not claims:
                continue

            stats['rows_with_claims'] += 1
            if maintainer:
                stats['by_maintainer'][maintainer]['rows'] += 1

            # Group claims by chunk (multiple claims per chunk)
            if chunk_text not in chunks_to_claims:
                chunks_to_claims[chunk_text] = []

            for claim in claims:
                claim = claim.strip()
                if not claim:
                    continue

                chunks_to_claims[chunk_text].append(claim)
                stats['total_claims'] += 1

                if maintainer:
                    stats['by_maintainer'][maintainer]['claims'] += 1

    # Convert grouped data to examples list
    # Each example has ONE chunk with ALL its claims
    examples = []
    for chunk_text, claims_list in chunks_to_claims.items():
        example = {
            "transcript_chunk": chunk_text,
            "claims": claims_list,  # Array of ALL claims from this chunk
            "quality": "good"
        }
        examples.append(example)

    # Save to JSON file
    output_data = {
        "examples": examples
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print statistics
    print(f"Processed {stats['total_rows']:,} total rows")
    print(f"Found {stats['rows_with_claims']:,} rows with claims")
    print(f"Extracted {stats['total_claims']:,} individual claims")
    print(f"Created {len(examples):,} unique chunk examples")
    print()

    # Show statistics about claims per chunk
    if examples:
        claims_per_chunk = [len(ex['claims']) for ex in examples]
        print("Example Statistics:")
        print(f"  Avg claims per chunk: {sum(claims_per_chunk) / len(examples):.1f}")
        print(f"  Min claims in a chunk: {min(claims_per_chunk)}")
        print(f"  Max claims in a chunk: {max(claims_per_chunk)}")
        print()

    print("By Maintainer:")
    for maintainer, counts in sorted(stats['by_maintainer'].items()):
        print(f"  {maintainer}:")
        print(f"    Rows: {counts['rows']:,}")
        print(f"    Claims: {counts['claims']:,}")
    print()

    print(f"Saved to: {output_file}")
    print()

    # Show sample examples
    if examples:
        print("Sample examples (first 3):")
        print("-" * 80)
        for i, example in enumerate(examples[:3], 1):
            print(f"\nExample {i}:")
            print(f"  Chunk: {example['transcript_chunk'][:100]}...")
            print(f"  Claims ({len(example['claims'])}):")
            for j, claim in enumerate(example['claims'][:3], 1):  # Show first 3 claims
                claim_display = claim[:80] + "..." if len(claim) > 80 else claim
                print(f"    {j}. {claim_display}")
            if len(example['claims']) > 3:
                print(f"    ... and {len(example['claims']) - 3} more")
            print(f"  Quality: {example['quality']}")
    else:
        print("No examples to display.")

    print()
    print("=" * 80)
    print("Processing complete!")
    print("=" * 80)

    return len(examples)


if __name__ == "__main__":
    csv_file = "data/claims_from_chunks_reviewed.csv"
    output_file = "evaluation/claims_manual_review.json"

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    try:
        num_examples = process_csv(csv_file, output_file)
        print(f"\nSuccessfully created {num_examples:,} training examples")
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing CSV: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
