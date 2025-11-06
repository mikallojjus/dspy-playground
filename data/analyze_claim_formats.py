#!/usr/bin/env python3
"""
Script to analyze how different editors are storing claims in the CSV.
"""

import csv
from collections import defaultdict


def analyze_claim_formats(csv_file: str):
    """Analyze claim formats by different maintainers."""

    print("Analyzing claim formats by editor...\n")
    print("=" * 80)

    editors_samples = defaultdict(list)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            maintainer = row.get('Maintainer', '').strip()
            claims = row.get('Claims', '').strip()

            if claims and maintainer:  # Only if both are non-empty
                # Store first 5 samples per editor
                if len(editors_samples[maintainer]) < 5:
                    editors_samples[maintainer].append(claims)

    # Display samples for each editor
    for editor, samples in sorted(editors_samples.items()):
        print(f"\nEditor: {editor}")
        print("-" * 80)
        print(f"Total samples found: {len(samples)}")
        print("\nSample claims:")
        for i, sample in enumerate(samples, 1):
            # Show first 200 chars of each sample
            display = sample[:300].replace('\n', '\\n')
            if len(sample) > 300:
                display += "..."
            print(f"\n  Sample {i}:")
            print(f"  {display}")
            print()


if __name__ == "__main__":
    csv_file = "data/claims_from_chunks_reviewed.csv"
    analyze_claim_formats(csv_file)
