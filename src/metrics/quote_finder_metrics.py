"""
Evaluation metric for DSPy QuoteFinder optimization.

Measures quote quality using three components:
1. Verification rate: % of quotes that pass substring verification (catch hallucinations)
2. Entailment rate: % of verified quotes that SUPPORT the claim (not just RELATED)
3. Recall: % of ground truth quotes found (if available)

Usage:
    from src.metrics.quote_finder_metrics import QuoteFinderMetric

    metric = QuoteFinderMetric(entailment_validator)

    # Evaluate a prediction
    example = dspy.Example(
        claim="Bitcoin reached $69,000",
        transcript_chunks="...",
        gold_quotes=["Bitcoin hit $69k in November 2021"]
    ).with_inputs("claim", "transcript_chunks")

    prediction = quote_finder(claim=example.claim, transcript_chunks=example.transcript_chunks)

    score = metric(example, prediction)
    print(f"Score: {score:.2f}")  # 0.0 to 1.0
"""

import dspy
from typing import Optional, Any
from difflib import SequenceMatcher

from src.search.quote_verification import QuoteVerifier
from src.dspy_models.entailment_validator import EntailmentValidatorModel
from src.infrastructure.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class QuoteFinderMetric:
    """
    Metric for evaluating quote finding quality.

    Composite score (0.0-1.0):
    - 40% verification rate (quotes must actually exist in transcript)
    - 40% entailment rate (quotes must SUPPORT claim, not just RELATED)
    - 20% recall (finds ground truth quotes, if available)

    Higher scores indicate better quote finding:
    - >0.80: Excellent (low hallucinations, high precision)
    - 0.60-0.80: Good
    - 0.40-0.60: Acceptable
    - <0.40: Poor (high hallucination rate or low precision)

    Example:
        ```python
        metric = QuoteFinderMetric(entailment_validator)

        # Example with ground truth
        example = dspy.Example(
            claim="Bitcoin reached $69,000 in November 2021",
            transcript_chunks="Speaker: Bitcoin hit $69,000 in late 2021...",
            gold_quotes=["Bitcoin hit $69,000 in late 2021"]
        ).with_inputs("claim", "transcript_chunks")

        prediction = dspy.Prediction(quotes=[
            {"text": "Bitcoin hit $69,000 in late 2021", "reasoning": "..."}
        ])

        score = metric(example, prediction)
        # Returns: 1.0 (perfect score: verified, supports, matches ground truth)
        ```
    """

    def __init__(
        self,
        entailment_validator: Optional[EntailmentValidatorModel] = None,
        verifier: Optional[QuoteVerifier] = None,
        verification_weight: float = 0.4,
        entailment_weight: float = 0.4,
        recall_weight: float = 0.2
    ):
        """
        Initialize the quote finder metric.

        Args:
            entailment_validator: Entailment validator (default: creates new one)
            verifier: Quote verifier (default: creates new one)
            verification_weight: Weight for verification rate (default: 0.4)
            entailment_weight: Weight for entailment rate (default: 0.4)
            recall_weight: Weight for recall (default: 0.2)
        """
        self.entailment_validator = entailment_validator or EntailmentValidatorModel()
        self.verifier = verifier or QuoteVerifier()

        self.verification_weight = verification_weight
        self.entailment_weight = entailment_weight
        self.recall_weight = recall_weight

        logger.info(
            f"Initialized QuoteFinderMetric: "
            f"weights=[verification={verification_weight}, "
            f"entailment={entailment_weight}, recall={recall_weight}]"
        )

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[Any] = None
    ) -> float:
        """
        Evaluate predicted quotes against example.

        Args:
            example: Example containing claim, transcript_chunks, and optionally gold_quotes
            prediction: Prediction containing quotes list
            trace: Optional DSPy trace (not used)

        Returns:
            Score from 0.0 to 1.0 (higher is better)

        Example:
            ```python
            metric = QuoteFinderMetric(entailment_validator)

            example = dspy.Example(
                claim="Bitcoin reached $69,000",
                transcript_chunks="Bitcoin hit $69k in 2021...",
                gold_quotes=["Bitcoin hit $69k in 2021"]
            ).with_inputs("claim", "transcript_chunks")

            prediction = dspy.Prediction(quotes=[
                {"text": "Bitcoin hit $69k in 2021", "reasoning": "..."}
            ])

            score = metric(example, prediction)
            print(f"Score: {score:.3f}")
            ```
        """
        claim = example.claim
        transcript_chunks = example.transcript_chunks
        predicted_quotes = prediction.quotes if hasattr(prediction, 'quotes') else []

        # Handle empty predictions
        if not predicted_quotes or not isinstance(predicted_quotes, list):
            logger.debug(f"No quotes predicted for claim: {claim[:50]}...")
            return 0.0

        logger.debug(
            f"Evaluating {len(predicted_quotes)} predicted quotes for claim: {claim[:50]}..."
        )

        # Metric 1: Verification rate (catch hallucinations)
        verified_quotes = []
        for quote_data in predicted_quotes:
            if not isinstance(quote_data, dict):
                continue

            quote_text = quote_data.get("text", "")
            if not quote_text:
                continue

            verification_result = self.verifier.verify(
                quote_text,
                transcript_chunks,
                claim
            )

            if verification_result.is_valid and verification_result.confidence >= settings.quote_verification_min_confidence:
                verified_quotes.append(verification_result.corrected_text)

        verification_rate = len(verified_quotes) / len(predicted_quotes) if predicted_quotes else 0.0

        if not verified_quotes:
            # All quotes were hallucinations
            logger.debug(
                f"All {len(predicted_quotes)} quotes were hallucinations "
                f"(verification rate: 0.0)"
            )
            return 0.0

        logger.debug(
            f"Verification: {len(verified_quotes)}/{len(predicted_quotes)} passed "
            f"(rate: {verification_rate:.2%})"
        )

        # Metric 2: Entailment rate (SUPPORTS vs RELATED)
        supports_count = 0
        for quote_text in verified_quotes:
            try:
                entailment_result = self.entailment_validator.validate(claim, quote_text)

                if entailment_result["relationship"] == "SUPPORTS":
                    supports_count += 1

                logger.debug(
                    f"Entailment: {entailment_result['relationship']} "
                    f"(confidence: {entailment_result['confidence']:.2f})"
                )
            except Exception as e:
                logger.error(f"Error in entailment validation: {e}")
                # Don't count as SUPPORTS on error
                continue

        entailment_rate = supports_count / len(verified_quotes) if verified_quotes else 0.0

        logger.debug(
            f"Entailment: {supports_count}/{len(verified_quotes)} SUPPORTS "
            f"(rate: {entailment_rate:.2%})"
        )

        # Metric 3: Recall (if ground truth available)
        recall = 1.0  # Default if no ground truth
        if hasattr(example, 'gold_quotes') and example.gold_quotes:
            # Calculate recall: what % of ground truth quotes were found?
            gold_quotes = example.gold_quotes
            matches = 0

            for gold_quote in gold_quotes:
                # Check if any verified quote is similar to this gold quote
                for verified_quote in verified_quotes:
                    similarity = self._calculate_similarity(gold_quote, verified_quote)
                    if similarity >= 0.8:  # 80% similarity threshold
                        matches += 1
                        break  # Count each gold quote only once

            recall = matches / len(gold_quotes) if gold_quotes else 1.0

            logger.debug(
                f"Recall: {matches}/{len(gold_quotes)} gold quotes found "
                f"(rate: {recall:.2%})"
            )

        # Composite score (weighted)
        # Heavily penalize hallucinations and non-supporting quotes
        score = (
            self.verification_weight * verification_rate +
            self.entailment_weight * entailment_rate +
            self.recall_weight * recall
        )

        logger.debug(
            f"Final score: {score:.3f} "
            f"(verification={verification_rate:.2f}, "
            f"entailment={entailment_rate:.2f}, "
            f"recall={recall:.2f})"
        )

        return score

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using normalized token overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score from 0.0 to 1.0
        """
        # Normalize texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()

        return similarity

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def evaluate_batch(
        self,
        examples: list[dspy.Example],
        predictions: list[dspy.Prediction]
    ) -> dict:
        """
        Evaluate multiple examples and return aggregate statistics.

        Args:
            examples: List of examples
            predictions: List of predictions (must match examples length)

        Returns:
            Dict with statistics:
                - mean_score: Average score across all examples
                - scores: List of individual scores
                - verification_rate: Average verification rate
                - entailment_rate: Average entailment rate
                - recall: Average recall (if ground truth available)

        Example:
            ```python
            metric = QuoteFinderMetric(entailment_validator)

            examples = [...]  # List of examples
            predictions = [...]  # List of predictions

            stats = metric.evaluate_batch(examples, predictions)
            print(f"Mean score: {stats['mean_score']:.3f}")
            print(f"Verification rate: {stats['verification_rate']:.2%}")
            print(f"Entailment rate: {stats['entailment_rate']:.2%}")
            ```
        """
        if len(examples) != len(predictions):
            raise ValueError(
                f"Examples ({len(examples)}) and predictions ({len(predictions)}) "
                "must have same length"
            )

        logger.info(f"Evaluating batch of {len(examples)} examples...")

        scores = []
        total_verification_rate = 0.0
        total_entailment_rate = 0.0
        total_recall = 0.0

        for i, (example, prediction) in enumerate(zip(examples, predictions), 1):
            score = self(example, prediction)
            scores.append(score)

            logger.debug(f"Example {i}/{len(examples)}: score={score:.3f}")

        mean_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            f"Batch evaluation complete: "
            f"mean_score={mean_score:.3f}, "
            f"min={min(scores) if scores else 0:.3f}, "
            f"max={max(scores) if scores else 0:.3f}"
        )

        return {
            "mean_score": mean_score,
            "scores": scores,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "count": len(scores)
        }
