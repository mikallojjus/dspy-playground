"""
Regenerate train/val splits from manual review datasets.

This script:
1. Loads entailment_manual_review.json and claims_manual_review.json
2. Shuffles examples randomly
3. Splits 70% train / 30% validation
4. Saves to *_train.json and *_val.json

Usage:
    python -m src.training.regenerate_splits
"""

import json
import random
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def regenerate_entailment_splits():
    """Regenerate entailment train/val splits."""
    print("=" * 80)
    print("Regenerating Entailment Validation Splits")
    print("=" * 80)
    print()

    # Load manual review
    with open('evaluation/entailment_manual_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = data['examples']

    print(f"Total examples: {len(examples)}")

    # Count distribution
    supports = sum(1 for ex in examples if ex['relationship'] == 'SUPPORTS')
    related = sum(1 for ex in examples if ex['relationship'] == 'RELATED')
    neutral = sum(1 for ex in examples if ex['relationship'] == 'NEUTRAL')
    contradicts = sum(1 for ex in examples if ex['relationship'] == 'CONTRADICTS')

    print(f"Distribution: SUPPORTS={supports}, RELATED={related}, NEUTRAL={neutral}, CONTRADICTS={contradicts}")
    print()

    # Shuffle and split 70/30
    random.seed(42)  # For reproducibility
    shuffled = examples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.7)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    # Save splits
    train_data = {"examples": train}
    val_data = {"examples": val}

    with open('evaluation/entailment_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)

    with open('evaluation/entailment_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)

    print(f"✓ Train set: {len(train)} examples → evaluation/entailment_train.json")
    print(f"✓ Val set: {len(val)} examples → evaluation/entailment_val.json")
    print()


def regenerate_claims_splits():
    """Regenerate claims train/val splits."""
    print("=" * 80)
    print("Regenerating Claim Extraction Splits")
    print("=" * 80)
    print()

    # Load manual review
    with open('evaluation/claims_manual_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = data['examples']

    print(f"Total examples: {len(examples)}")

    # Count distribution
    good = sum(1 for ex in examples if ex['quality'] == 'good')
    bad = sum(1 for ex in examples if ex['quality'] == 'bad')

    print(f"Distribution: GOOD={good}, BAD={bad}")
    print()

    # Shuffle and split 70/30
    random.seed(42)  # For reproducibility
    shuffled = examples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.7)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]

    # Save splits
    train_data = {"examples": train}
    val_data = {"examples": val}

    with open('evaluation/claims_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)

    with open('evaluation/claims_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)

    print(f"✓ Train set: {len(train)} examples → evaluation/claims_train.json")
    print(f"✓ Val set: {len(val)} examples → evaluation/claims_val.json")
    print()

    # Also create positive-only dataset for claim training
    positive_only = [ex for ex in examples if ex['quality'] == 'good']
    positive_data = {"examples": positive_only}

    with open('evaluation/claims_positive_only.json', 'w', encoding='utf-8') as f:
        json.dump(positive_data, f, indent=2)

    print(f"✓ Positive-only set: {len(positive_only)} examples → evaluation/claims_positive_only.json")
    print()


if __name__ == '__main__':
    regenerate_entailment_splits()
    regenerate_claims_splits()

    print("=" * 80)
    print("✓ All splits regenerated successfully!")
    print("=" * 80)
