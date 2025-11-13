"""
DSPy Ad Classification Model.

Classifies claims as advertisement/promotional content or genuine content.
Uses optimized model trained with LLM-as-judge metric.

Usage:
    from src.dspy_models.ad_classifier import AdClassifierModel

    classifier = AdClassifierModel()
    result = classifier.classify("Use code BANKLESS for 20% off")
    print(result)  # {"is_advertisement": True, "confidence": 0.95}
"""

import dspy
import asyncio
from typing import List, Dict
from pathlib import Path

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class AdClassification(dspy.Signature):
    """
    Determine if a claim is advertisement/promotional content.

    Advertisement claims include:
    - Product or service promotions
    - Discount codes or special offers
    - Sponsor mentions or endorsements
    - Calls to action for commercial products
    - Affiliate links or referral codes

    Content claims include:
    - Factual statements about topics discussed
    - Guest opinions or expert insights
    - Historical facts or data points
    - Industry news or analysis
    - Technical explanations or tutorials

    Examples of ADVERTISEMENT claims:
    - "Use code BANKLESS for 20% off Athletic Greens"
    - "Athletic Greens contains 75 vitamins and minerals"
    - "Visit athleticgreens.com/bankless for a special offer"
    - "Today's episode is sponsored by Ledger"

    Examples of CONTENT claims:
    - "Ethereum's merge reduced energy consumption by 99%"
    - "Bitcoin reached $69,000 in November 2021"
    - "Layer 2 solutions improve transaction throughput"
    - "Mike Neuder thinks the Ethereum roadmap is on track"

    Output format instructions:
    - For is_advertisement: respond with exactly "True" or "False" (case-sensitive)
    - For confidence: respond with a decimal number between 0.0 and 1.0
    """

    claim_text: str = dspy.InputField(desc="The claim to classify")
    is_advertisement: str = dspy.OutputField(desc="'True' if claim is promotional content, 'False' otherwise")
    confidence: str = dspy.OutputField(desc="Classification confidence as string decimal (0.0-1.0)")


class BatchAdClassification(dspy.Signature):
    """
    Classify multiple claims as advertisement/promotional content in a single call.

    Same classification criteria as AdClassification, but processes multiple claims
    at once for better efficiency.

    Output format:
    - Return a JSON array of objects
    - Each object: {"is_advertisement": "True" or "False", "confidence": "0.0-1.0"}
    - Must return exactly the same number of results as input claims
    - Results must be in the same order as input claims

    Example:
    Input: ["Use code BANKLESS for 20% off", "Bitcoin reached $69,000"]
    Output: [
        {"is_advertisement": "True", "confidence": "0.95"},
        {"is_advertisement": "False", "confidence": "0.92"}
    ]
    """

    claims: List[str] = dspy.InputField(desc="List of claims to classify")
    results: str = dspy.OutputField(
        desc='JSON array of classification results: [{"is_advertisement": "True"/"False", "confidence": "0.0-1.0"}, ...]'
    )


class AdClassifier(dspy.Module):
    """
    Wrapper module that handles string-to-typed-value parsing.

    This matches the training structure in train_ad_classifier.py,
    allowing the model to be loaded correctly.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(AdClassification)

    def forward(self, claim_text):
        # Get raw prediction with string outputs
        raw_pred = self.predictor(claim_text=claim_text)

        # Parse string outputs to proper types
        is_advertisement = self._parse_bool(raw_pred.is_advertisement)
        confidence = self._parse_float(raw_pred.confidence)

        # Return new prediction with parsed values
        return dspy.Prediction(
            is_advertisement=is_advertisement,
            confidence=confidence,
            reasoning=raw_pred.reasoning if hasattr(raw_pred, 'reasoning') else None
        )

    def _parse_bool(self, value):
        """Parse string to bool, with fallback."""
        if value is None:
            return False

        value_str = str(value).strip().lower()
        if value_str in ['true', '1', 'yes']:
            return True
        elif value_str in ['false', '0', 'no']:
            return False
        else:
            return False

    def _parse_float(self, value):
        """Parse string to float, with fallback."""
        if value is None:
            return 0.5

        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.5


class BatchAdClassifier(dspy.Module):
    """
    Batch wrapper module for classifying multiple claims in a single LLM call.

    Processes multiple claims at once for better efficiency (fewer API calls).
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(BatchAdClassification)

    def forward(self, claims):
        import json

        # Get raw prediction with JSON string output
        raw_pred = self.predictor(claims=claims)

        # Parse JSON results
        try:
            results_json = json.loads(raw_pred.results)

            # Parse each result
            parsed_results = []
            for result in results_json:
                is_ad = self._parse_bool(result.get('is_advertisement', 'False'))
                conf = self._parse_float(result.get('confidence', '0.5'))
                parsed_results.append({
                    'is_advertisement': is_ad,
                    'confidence': conf
                })

            return dspy.Prediction(results=parsed_results)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error parsing batch classification results: {e}")
            # Return conservative defaults for all claims
            return dspy.Prediction(
                results=[
                    {'is_advertisement': False, 'confidence': 0.0}
                    for _ in claims
                ]
            )

    def _parse_bool(self, value):
        """Parse string to bool, with fallback."""
        if value is None:
            return False

        value_str = str(value).strip().lower()
        if value_str in ['true', '1', 'yes', True]:
            return True
        elif value_str in ['false', '0', 'no', False]:
            return False
        else:
            return False

    def _parse_float(self, value):
        """Parse string to float, with fallback."""
        if value is None:
            return 0.5

        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.5


class AdClassifierModel:
    """
    DSPy-based ad classifier using optimized model.

    Loads the optimized model from models/ad_classifier_v1.json
    which was trained using BootstrapFewShot with LLM-as-judge metric.

    Attributes:
        model: DSPy ChainOfThought module with few-shot examples
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the ad classifier model.

        Args:
            model_path: Path to the optimized DSPy model file (default from settings)

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If DSPy configuration fails
        """
        if model_path is None:
            model_path = settings.ad_classifier_model_path

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Optimized model not found at {model_path}. "
                "Run src/training/train_ad_classifier.py to create optimized model."
            )

        # Configure DSPy with Ollama
        logger.info(f"Configuring DSPy with Ollama at {settings.ollama_url}")
        lm = dspy.LM(
            f"ollama/{settings.ollama_model}",
            api_base=settings.ollama_url,
            num_ctx=32768
        )
        dspy.configure(lm=lm)

        # Load optimized model (must use AdClassifier wrapper to match training structure)
        logger.info(f"Loading optimized ad classifier from {model_path}")
        self.model = AdClassifier()
        self.model.load(str(self.model_path))

        # Log few-shot examples count
        if hasattr(self.model.predictor, 'demos') and self.model.predictor.demos:
            logger.info(f"Loaded model with {len(self.model.predictor.demos)} few-shot examples")
        else:
            logger.info("Loaded model (zero-shot)")

        # Create async wrapper using dspy.asyncify for parallel processing
        # This captures the current DSPy configuration context
        self._async_model = dspy.asyncify(self.model)
        logger.debug("Created async wrapper for parallel ad classification")

        # Create zero-shot batch classifier for true batching
        # (No trained model needed - uses zero-shot reasoning)
        self.batch_model = BatchAdClassifier()
        self._async_batch_model = dspy.asyncify(self.batch_model)
        logger.info("Created zero-shot batch classifier for multi-claim processing")

    def classify(self, claim_text: str) -> Dict[str, any]:
        """
        Classify a claim as ad or content (synchronous).

        Args:
            claim_text: The claim to classify

        Returns:
            Dict with keys: is_advertisement (bool), confidence (float)

        Example:
            ```python
            classifier = AdClassifierModel()
            result = classifier.classify("Use code BANKLESS for 20% off")
            # Returns: {
            #     "is_advertisement": True,
            #     "confidence": 0.95
            # }
            ```
        """
        try:
            result = self.model(claim_text=claim_text)

            is_ad = result.is_advertisement if hasattr(result, 'is_advertisement') else False
            confidence = float(result.confidence) if hasattr(result, 'confidence') else 0.0

            logger.debug(
                f"Classified claim ({len(claim_text)} chars): "
                f"is_ad={is_ad}, confidence={confidence:.2f}"
            )

            return {
                "is_advertisement": is_ad,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error classifying claim: {e}", exc_info=True)
            # Return conservative default (not ad) on error
            return {
                "is_advertisement": False,
                "confidence": 0.0
            }

    async def classify_async(self, claim_text: str) -> Dict[str, any]:
        """
        Classify a claim as ad or content (asynchronous using dspy.asyncify).

        This method uses dspy.asyncify to run the DSPy model in a worker thread
        while properly propagating the DSPy configuration context. This enables
        true parallel processing of multiple claims.

        Args:
            claim_text: The claim to classify

        Returns:
            Dict with keys: is_advertisement (bool), confidence (float)

        Example:
            ```python
            classifier = AdClassifierModel()
            result = await classifier.classify_async("Use code BANKLESS for 20% off")
            # Returns: {
            #     "is_advertisement": True,
            #     "confidence": 0.95
            # }
            ```

        Note:
            This method is preferred over classify() when processing multiple
            claims in parallel, as it properly handles DSPy's thread-local
            configuration.
        """
        try:
            result = await self._async_model(claim_text=claim_text)

            is_ad = result.is_advertisement if hasattr(result, 'is_advertisement') else False
            confidence = float(result.confidence) if hasattr(result, 'confidence') else 0.0

            logger.debug(
                f"Classified claim ({len(claim_text)} chars): "
                f"is_ad={is_ad}, confidence={confidence:.2f}"
            )

            return {
                "is_advertisement": is_ad,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error in async ad classification: {e}", exc_info=True)
            # Return conservative default (not ad) on error
            return {
                "is_advertisement": False,
                "confidence": 0.0
            }

    async def classify_batch_parallel(
        self,
        claim_texts: List[str],
        max_concurrency: int = None
    ) -> List[Dict[str, any]]:
        """
        Classify multiple claims in parallel using dspy.asyncify.

        This method processes classification calls concurrently using a semaphore
        to limit GPU load. It properly propagates DSPy configuration to worker threads.

        Args:
            claim_texts: List of claims to classify
            max_concurrency: Maximum concurrent LLM calls (default from settings)

        Returns:
            List of classification result dicts

        Example:
            ```python
            classifier = AdClassifierModel()
            claims = [
                "Use code BANKLESS for 20% off",
                "Ethereum's merge reduced energy consumption by 99%",
                "Visit athleticgreens.com for special offer"
            ]
            results = await classifier.classify_batch_parallel(claims)
            # Returns: [
            #     {"is_advertisement": True, "confidence": 0.95},
            #     {"is_advertisement": False, "confidence": 0.92},
            #     {"is_advertisement": True, "confidence": 0.88}
            # ]
            ```

        Note:
            - Uses semaphore to prevent GPU overload
            - Default concurrency from settings.max_ad_classification_concurrency
            - Falls back gracefully on errors
            - Significantly faster than sequential processing for large batches
        """
        if not claim_texts:
            return []

        # Get max concurrency from settings if not specified
        if max_concurrency is None:
            max_concurrency = settings.max_ad_classification_concurrency

        logger.info(
            f"Classifying {len(claim_texts)} claims in parallel "
            f"(max_concurrency={max_concurrency})"
        )

        # Semaphore to limit concurrent GPU calls
        semaphore = asyncio.Semaphore(max_concurrency)

        async def classify_with_limit(claim_text: str) -> Dict[str, any]:
            """Classify with semaphore to control concurrency."""
            async with semaphore:
                return await self.classify_async(claim_text)

        # Execute all classifications in parallel (controlled by semaphore)
        results = await asyncio.gather(
            *[classify_with_limit(claim) for claim in claim_texts],
            return_exceptions=False  # Let errors propagate to classify_async
        )

        # Log summary
        ad_count = sum(1 for r in results if r["is_advertisement"])
        logger.info(
            f"Parallel classification complete: {ad_count}/{len(results)} advertisements "
            f"(processed {len(results)} claims with concurrency={max_concurrency})"
        )

        return results

    async def classify_batch(
        self,
        claim_texts: List[str],
        batch_size: int = None
    ) -> List[Dict[str, any]]:
        """
        Classify multiple claims using TRUE batching (multiple claims per LLM call).

        This method groups claims into batches and makes fewer LLM calls compared
        to classify_batch_parallel() which makes one call per claim.

        Example: 125 claims → ~12 LLM calls (10 claims per call) instead of 125 calls

        Args:
            claim_texts: List of claims to classify
            batch_size: Number of claims per LLM call (default from settings)

        Returns:
            List of classification result dicts

        Example:
            ```python
            classifier = AdClassifierModel()
            claims = [... 125 claims ...]
            results = await classifier.classify_batch(claims, batch_size=10)
            # Makes ~12 LLM calls instead of 125
            ```

        Note:
            - Uses zero-shot batch classifier (no training needed)
            - Much more efficient for large batches (90% fewer API calls)
            - Falls back gracefully on errors
        """
        if not claim_texts:
            return []

        # Get batch size from settings if not specified
        if batch_size is None:
            batch_size = settings.ad_classification_batch_size

        logger.info(
            f"Classifying {len(claim_texts)} claims using TRUE batching "
            f"(batch_size={batch_size}, ~{(len(claim_texts) + batch_size - 1) // batch_size} LLM calls)"
        )

        all_results = []

        # Process in batches
        for i in range(0, len(claim_texts), batch_size):
            batch = claim_texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(claim_texts) + batch_size - 1) // batch_size

            logger.debug(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} claims)"
            )

            try:
                # Call batch classifier with multiple claims
                result = await self._async_batch_model(claims=batch)

                # Extract results list
                batch_results = result.results if hasattr(result, 'results') else []

                # Verify we got the right number of results
                if len(batch_results) != len(batch):
                    logger.warning(
                        f"Batch {batch_num}: Expected {len(batch)} results, got {len(batch_results)}. "
                        "Padding with defaults."
                    )
                    # Pad with defaults if needed
                    while len(batch_results) < len(batch):
                        batch_results.append({'is_advertisement': False, 'confidence': 0.0})

                all_results.extend(batch_results[:len(batch)])  # Truncate if too many

            except Exception as e:
                logger.error(
                    f"Error in batch {batch_num} classification: {e}",
                    exc_info=True
                )
                # Return conservative defaults for this batch
                all_results.extend([
                    {'is_advertisement': False, 'confidence': 0.0}
                    for _ in batch
                ])

        # Log summary
        ad_count = sum(1 for r in all_results if r["is_advertisement"])
        logger.info(
            f"Batch classification complete: {ad_count}/{len(all_results)} advertisements "
            f"(processed {len(all_results)} claims in "
            f"{(len(claim_texts) + batch_size - 1) // batch_size} batches)"
        )

        return all_results

    def filter_ads(
        self,
        claim_texts: List[str],
        threshold: float = 0.7
    ) -> List[str]:
        """
        Filter out advertisement claims from a list.

        This is the primary use case for the ad classifier in the pipeline.

        Args:
            claim_texts: List of claim texts
            threshold: Minimum confidence to filter (default: 0.7)

        Returns:
            List of non-ad claims only

        Example:
            ```python
            classifier = AdClassifierModel()
            claims = [
                "Use code BANKLESS for 20% off",
                "Ethereum's merge reduced energy consumption by 99%",
                "Visit athleticgreens.com for special offer"
            ]

            content_claims = classifier.filter_ads(claims, threshold=0.7)
            # Returns: [
            #     "Ethereum's merge reduced energy consumption by 99%"
            # ]
            ```
        """
        if not claim_texts:
            logger.debug("No claims to filter")
            return []

        logger.info(f"Filtering {len(claim_texts)} claims (threshold={threshold})")

        # Classify all claims using async parallel processing
        results = asyncio.run(self.classify_batch_parallel(claim_texts))

        # Filter out ads (keep only content)
        content_claims = [
            claim_text
            for claim_text, result in zip(claim_texts, results)
            if not (result["is_advertisement"] and result["confidence"] >= threshold)
        ]

        filtered_count = len(claim_texts) - len(content_claims)
        if filtered_count > 0:
            logger.info(
                f"Filtered {filtered_count} advertisement claims "
                f"({len(content_claims)} content claims remaining)"
            )

        return content_claims
