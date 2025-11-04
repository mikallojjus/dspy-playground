# Ad Classification Dataset

## Overview

This directory contains training and validation datasets for the ad classification model. The model learns to distinguish between advertisement/promotional claims and genuine content claims from podcast transcripts.

## Dataset Files

### Source of Truth

- **`ad_manual_review.json`** - Master dataset with all manually labeled examples (40 examples)

### Generated Splits (Auto-generated, DO NOT edit directly)

- **`ad_train.json`** - Training dataset (28 examples, 70% split)
- **`ad_val.json`** - Validation dataset (12 examples, 30% split)

**Important:** Only edit `ad_manual_review.json`. The train/val splits are auto-generated from it.

## Dataset Structure

### Manual Review Format

```json
{
  "examples": [
    {
      "claim_text": "The claim text to classify",
      "is_advertisement": true,
      "notes": "Explanation of why this is/isn't an ad"
    }
  ]
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `claim_text` | string | Yes | The claim to classify |
| `is_advertisement` | boolean | Yes | `true` if advertisement, `false` if content |
| `notes` | string | Optional | Human explanation for labeling decision |

**Note:** The `notes` field is for documentation only and not used during training.

## Guidelines for Labeling

### Advertisement Claims (is_advertisement: true)

Claims that include:
- **Explicit promotional codes** - "Use code BANKLESS for 20% off"
- **Calls to action with commercial intent** - "Visit athleticgreens.com/bankless"
- **Sponsor mentions** - "Today's episode is sponsored by Ledger"
- **Product features in promotional context** - "Athletic Greens contains 75 vitamins" (when part of sponsor read)
- **Discount offers** - "Get 50% off your subscription"
- **Affiliate links** - "Check out the link in the description for a special deal"

### Content Claims (is_advertisement: false)

Claims that include:
- **Factual statements** - "Ethereum's merge reduced energy consumption by 99%"
- **Historical facts** - "Bitcoin reached $69,000 in November 2021"
- **Technical explanations** - "Layer 2 solutions improve transaction throughput"
- **Guest opinions** - "Mike Neuder believes the Ethereum roadmap is on track"
- **Industry news** - "Decentralized finance protocols locked over $100 billion"
- **Educational content** - "Smart contracts are executed by the Ethereum Virtual Machine"

### Edge Cases

**Product mentions without promotion:**
- "Ledger makes hardware wallets" → **CONTENT** (neutral statement of fact)
- "Ledger hardware wallets are available at ledger.com/bankless" → **ADVERTISEMENT** (promotional link)

**Statistics about products:**
- "Athletic Greens contains 75 vitamins" → **ADVERTISEMENT** (promotional feature claim)
- "Athletic Greens was founded in 2010" → **CONTENT** (historical fact, not promotional)

**General rule:** If the claim is trying to sell something, get you to click a link, or use a promo code → **ADVERTISEMENT**. Otherwise → **CONTENT**.

## Workflow: Adding New Examples

### 1. Edit Manual Review Dataset

Add new examples to **`ad_manual_review.json`** only:

```json
{
  "claim_text": "Your new claim here",
  "is_advertisement": true,
  "notes": "Why you labeled it this way"
}
```

### 2. Regenerate Train/Val Splits

After adding/editing examples in `ad_manual_review.json`, regenerate the splits:

```bash
python -m src.training.regenerate_splits
```

This will:
- Load `ad_manual_review.json`
- Shuffle examples with fixed seed (reproducible splits)
- Split 70% train / 30% validation
- Overwrite `ad_train.json` and `ad_val.json`

### 3. Verify Splits

Check the output for distribution:

```
================================================================================
Regenerating Ad Classification Splits
================================================================================

Total examples: 40
Distribution: ADVERTISEMENT=15, CONTENT=25

✓ Train set: 28 examples → evaluation/ad_train.json
✓ Val set: 12 examples → evaluation/ad_val.json
```

## Expanding the Dataset

The provided 40 examples are a starting point. For better model performance:

1. **Collect real examples** from your podcast transcripts
2. **Manually label** and add to `ad_manual_review.json`
3. **Balance the dataset** - aim for roughly 40-60% advertisements
4. **Include edge cases** - borderline examples help the model learn boundaries
5. **Vary the language** - different phrasings of ads and content
6. **Run regenerate_splits** after each batch of additions

### Recommended Dataset Sizes

| Size | Total Examples | Train | Val | Use Case |
|------|----------------|-------|-----|----------|
| **Current** | 40 | 28 | 12 | Starting point |
| **Minimum** | 70 | 49 | 21 | Basic model |
| **Good** | 140 | 98 | 42 | Production-ready |
| **Ideal** | 280+ | 196+ | 84+ | High accuracy |

## Training the Model

### Prerequisites

1. **Add examples to `ad_manual_review.json`** (at least 40-70 examples recommended)
2. **Regenerate splits:** `python -m src.training.regenerate_splits`
3. **Verify splits** look balanced

### Train the Model

```bash
# Train the ad classifier (uses ad_train.json and ad_val.json)
python -m src.training.train_ad_classifier
```

The training script will:
1. Load train/val splits (auto-generated from manual_review)
2. Create a baseline (zero-shot) model
3. Optimize using BootstrapFewShot with LLM-as-judge metric
4. Save the optimized model to `models/ad_classifier_v1.json`
5. Report accuracy improvements

**Note:** The training script reads from `ad_train.json` and `ad_val.json`, which are generated from `ad_manual_review.json`. Always edit the manual_review file, not the splits directly.

## Using the Trained Model

### Standalone Usage

```python
from src.dspy_models.ad_classifier import AdClassifierModel

classifier = AdClassifierModel()

# Classify single claim
result = classifier.classify("Use code BANKLESS for 20% off")
print(result)  # {"is_advertisement": True, "confidence": 0.95}

# Filter ads from list
claims = [
    "Use code BANKLESS for 20% off",
    "Ethereum's merge reduced energy consumption by 99%",
    "Visit athleticgreens.com for special offer"
]
content_only = classifier.filter_ads(claims, threshold=0.7)
# Returns: ["Ethereum's merge reduced energy consumption by 99%"]
```

### Pipeline Integration

To enable ad filtering in the extraction pipeline, set in your `.env` or `settings.py`:

```bash
FILTER_ADVERTISEMENT_CLAIMS=true
AD_CLASSIFICATION_THRESHOLD=0.7
```

This will automatically filter advertisement claims during claim extraction.

## Quality Metrics

After training, evaluate your model:

- **Accuracy > 0.90** - Excellent (90%+ classifications correct)
- **Accuracy 0.80-0.90** - Good (usable for filtering)
- **Accuracy < 0.80** - Needs more training data or better labels

Check for:
- **False positives** - Content claims marked as ads (filtered incorrectly)
- **False negatives** - Ad claims marked as content (not filtered)

Adjust `ad_classification_threshold` based on your priorities:
- **Higher threshold (0.8-0.9)** - More conservative, fewer false positives
- **Lower threshold (0.5-0.7)** - More aggressive, fewer false negatives

## Current Dataset Statistics

**Last updated:** After initial manual review creation

```
Total examples: 40
Distribution:
  - Advertisements: 15 (37.5%)
  - Content: 25 (62.5%)

Split (70/30):
  - Train: 28 examples
  - Validation: 12 examples
```

## Notes

- **Source of truth:** Only edit `ad_manual_review.json`
- **Auto-generated:** `ad_train.json` and `ad_val.json` are generated by `regenerate_splits.py`
- **Reproducible splits:** Uses `random.seed(42)` for consistent train/val splits
- **Pattern consistency:** Follows same workflow as `claims` and `entailment` datasets
- The provided examples use cryptocurrency podcast themes (Bankless-style content)
- Adapt examples to match your podcast domain
- The model will generalize to new ad patterns not in training data
- LLM-as-judge metric helps the model understand semantic meaning, not just keywords
