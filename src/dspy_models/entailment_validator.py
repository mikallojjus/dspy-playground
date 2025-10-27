"""
DSPy Entailment Validator Model.

Validates whether quotes support claims (SUPPORTS/RELATED/NEUTRAL/CONTRADICTS).
Uses optimized DSPy model trained with LLM-as-judge metric.

Usage:
    from src.dspy_models.entailment_validator import EntailmentValidatorModel

    validator = EntailmentValidatorModel()
    result = validator.validate(
        claim="Bitcoin reached $69,000 in November 2021",
        quote="Bitcoin hit its all-time high of $69,000..."
    )
    print(result["relationship"])  # "SUPPORTS"
    print(result["confidence"])    # 0.95
"""

import dspy
import time
import asyncio
from typing import Literal, List, Dict
from pathlib import Path

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class EntailmentValidation(dspy.Signature):
    """
    Validate whether a quote supports a claim.

    Relationship types:
    - SUPPORTS: Quote directly asserts the claim or provides clear evidence
    - RELATED: Quote is topically related but doesn't validate the claim
    - NEUTRAL: Quote is unrelated or provides no evidence
    - CONTRADICTS: Quote contradicts or undermines the claim
    """

    claim: str = dspy.InputField(desc="The claim to validate")
    quote: str = dspy.InputField(desc="The quote to check for support")
    relationship: Literal["SUPPORTS", "RELATED", "NEUTRAL", "CONTRADICTS"] = (
        dspy.OutputField(desc="The relationship between quote and claim")
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the relationship")
    confidence: float = dspy.OutputField(desc="Confidence score (0.0-1.0)")


class EntailmentValidatorModel:
    """
    DSPy-based entailment validator using optimized model.

    Loads optimized model from models/entailment_validator_v1.json.
    Falls back to baseline (zero-shot) if optimized model not found.

    Attributes:
        model: DSPy ChainOfThought module (with few-shot examples if optimized)
        optimized: Whether using optimized model (True) or baseline (False)
    """

    def __init__(self, model_path: str = "models/entailment_validator_v1.json"):
        """
        Initialize the entailment validator.

        Args:
            model_path: Path to the optimized DSPy model

        Example:
            ```python
            validator = EntailmentValidatorModel()
            result = validator.validate("Bitcoin reached $69,000", "BTC hit $69k")
            print(result["relationship"])  # "SUPPORTS"
            ```
        """
        self.model_path = Path(model_path)

        # Check if optimized model exists
        if self.model_path.exists():
            logger.info(f"Loading optimized entailment validator from {model_path}")

            # Configure DSPy
            lm = dspy.LM(
                f"ollama/{settings.ollama_model}", api_base=settings.ollama_url
            )
            dspy.configure(lm=lm)

            # Load optimized model
            self.model = dspy.ChainOfThought(EntailmentValidation)
            self.model.load(str(self.model_path))
            self.optimized = True

            # Log few-shot examples
            if hasattr(self.model, "demos") and self.model.demos:
                logger.info(
                    f"Loaded model with {len(self.model.demos)} few-shot examples"
                )
            else:
                logger.info("Loaded model (zero-shot)")
        else:
            logger.warning(
                f"Optimized model not found at {model_path}. "
                "Using baseline (zero-shot) validation. "
                "Run src/experiments/exp_4_1_optimize_entailment.py to create optimized model."
            )

            # Configure DSPy for baseline
            lm = dspy.LM(
                f"ollama/{settings.ollama_model}", api_base=settings.ollama_url
            )
            dspy.configure(lm=lm)

            # Create baseline model
            self.model = dspy.ChainOfThought(EntailmentValidation)
            self.optimized = False

        # Create async wrapper using dspy.asyncify for parallel processing
        # This captures the current DSPy configuration context
        self._async_model = dspy.asyncify(self.model)
        logger.debug("Created async wrapper for parallel entailment validation")

    def validate(
        self, claim: str, quote: str, retry_count: int = 2, retry_delay: float = 2.0
    ) -> Dict[str, any]:
        """
        Validate whether a quote supports a claim with retry logic for GPU errors.

        Args:
            claim: The claim to validate
            quote: The quote to check
            retry_count: Number of retries on GPU/CUDA errors (default: 2)
            retry_delay: Delay before retry in seconds (default: 2.0s for GPU cooldown)

        Returns:
            Dict with keys: relationship, reasoning, confidence

        Example:
            ```python
            validator = EntailmentValidatorModel()
            result = validator.validate(
                claim="Bitcoin reached $69,000 in November 2021",
                quote="Bitcoin hit its all-time high of $69,000 in November"
            )
            # Returns: {
            #     "relationship": "SUPPORTS",
            #     "reasoning": "Quote directly states the claim",
            #     "confidence": 0.95
            # }
            ```
        """
        for attempt in range(retry_count + 1):
            try:
                result = self.model(claim=claim, quote=quote)

                return {
                    "relationship": result.relationship,
                    "reasoning": result.reasoning,
                    "confidence": (
                        float(result.confidence)
                        if hasattr(result, "confidence")
                        else 0.8
                    ),
                }
            except Exception as e:
                error_msg = str(e).lower()
                is_gpu_error = any(
                    keyword in error_msg
                    for keyword in [
                        "cuda",
                        "gpu",
                        "memory",
                        "llama runner",
                        "exit status 2",
                    ]
                )

                if is_gpu_error and attempt < retry_count:
                    logger.warning(
                        f"GPU error during entailment validation (attempt {attempt + 1}/{retry_count + 1}): {e}"
                    )
                    logger.info(
                        f"Waiting {retry_delay}s for GPU cooldown before retry..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error in entailment validation: {e}", exc_info=True)
                    # Return neutral on error
                    return {
                        "relationship": "NEUTRAL",
                        "reasoning": f"Error during validation: {str(e)[:100]}",
                        "confidence": 0.0,
                    }

        # Should not reach here, but just in case
        return {
            "relationship": "NEUTRAL",
            "reasoning": "All retries exhausted",
            "confidence": 0.0,
        }

    async def validate_async(
        self, claim: str, quote: str
    ) -> Dict[str, any]:
        """
        Validate whether a quote supports a claim (asynchronous using dspy.asyncify).

        This method uses dspy.asyncify to run the DSPy model in a worker thread
        while properly propagating the DSPy configuration context. This enables
        true parallel processing of multiple validation calls.

        Args:
            claim: The claim to validate
            quote: The quote to check

        Returns:
            Dict with keys: relationship, reasoning, confidence

        Example:
            ```python
            validator = EntailmentValidatorModel()
            result = await validator.validate_async(
                claim="Bitcoin reached $69,000 in November 2021",
                quote="Bitcoin hit its all-time high of $69,000 in November"
            )
            # Returns: {
            #     "relationship": "SUPPORTS",
            #     "reasoning": "Quote directly states the claim",
            #     "confidence": 0.95
            # }
            ```

        Note:
            This method is preferred over validate() when processing multiple
            claim-quote pairs in parallel, as it properly handles DSPy's
            thread-local configuration.
        """
        try:
            result = await self._async_model(claim=claim, quote=quote)

            return {
                "relationship": result.relationship,
                "reasoning": result.reasoning,
                "confidence": (
                    float(result.confidence)
                    if hasattr(result, "confidence")
                    else 0.8
                ),
            }
        except Exception as e:
            logger.error(f"Error in async entailment validation: {e}", exc_info=True)
            # Return neutral on error
            return {
                "relationship": "NEUTRAL",
                "reasoning": f"Error during validation: {str(e)[:100]}",
                "confidence": 0.0,
            }

    def validate_batch(
        self, claim_quote_pairs: List[tuple[str, str]]
    ) -> List[Dict[str, any]]:
        """
        Validate multiple claim-quote pairs sequentially.

        Note: Sequential processing is used because DSPy is not thread-safe.
        The old TypeScript system achieved parallelism through JavaScript's
        event loop (true async I/O), which Python cannot replicate with
        synchronous DSPy calls without causing thread contention and PC freezing.

        Args:
            claim_quote_pairs: List of (claim, quote) tuples

        Returns:
            List of validation result dicts

        Example:
            ```python
            validator = EntailmentValidatorModel()
            pairs = [
                ("Bitcoin reached $69,000", "BTC hit $69k in Nov 2021"),
                ("Ethereum uses proof-of-stake", "ETH migrated to PoS in 2022"),
            ]
            results = validator.validate_batch(pairs)
            # Returns: [
            #     {"relationship": "SUPPORTS", "reasoning": "...", "confidence": 0.95},
            #     {"relationship": "SUPPORTS", "reasoning": "...", "confidence": 0.92}
            # ]
            ```
        """
        logger.info(f"Validating {len(claim_quote_pairs)} claim-quote pairs")

        results = []
        for i, (claim, quote) in enumerate(claim_quote_pairs, 1):
            result = self.validate(claim, quote)
            results.append(result)

            logger.debug(
                f"Pair {i}/{len(claim_quote_pairs)}: {result['relationship']} "
                f"(confidence: {result['confidence']:.2f})"
            )

        # Log summary
        supports_count = sum(1 for r in results if r["relationship"] == "SUPPORTS")
        logger.info(
            f"Batch validation complete: {supports_count}/{len(results)} SUPPORTS"
        )

        return results

    def filter_supporting_quotes(
        self, claim: str, quotes: List[str]
    ) -> List[tuple[str, Dict[str, any]]]:
        """
        Filter quotes to keep only those that SUPPORT the claim.

        This is the primary use case for the entailment validator in the pipeline.

        Args:
            claim: The claim to validate
            quotes: List of quote texts

        Returns:
            List of (quote, validation_result) tuples for SUPPORTS relationships only

        Example:
            ```python
            validator = EntailmentValidatorModel()
            claim = "Bitcoin reached $69,000"
            quotes = [
                "BTC hit $69k in November 2021",  # SUPPORTS
                "Crypto was volatile in 2021",    # RELATED
                "Weather was nice yesterday"      # NEUTRAL
            ]

            supporting = validator.filter_supporting_quotes(claim, quotes)
            # Returns: [
            #     ("BTC hit $69k in November 2021", {"relationship": "SUPPORTS", ...})
            # ]
            ```
        """
        if not quotes:
            logger.debug(f"No quotes to filter for claim: {claim[:50]}...")
            return []

        logger.info(f"Filtering {len(quotes)} quotes for claim: {claim[:60]}...")

        # Validate all quotes
        pairs = [(claim, quote) for quote in quotes]
        results = self.validate_batch(pairs)

        # Filter to SUPPORTS only
        supporting_quotes = [
            (quote, result)
            for quote, result in zip(quotes, results)
            if result["relationship"] == "SUPPORTS"
        ]

        filtered_count = len(quotes) - len(supporting_quotes)
        if filtered_count > 0:
            logger.info(
                f"Filtered {filtered_count} non-supporting quotes "
                f"({len(supporting_quotes)} remaining)"
            )

        return supporting_quotes

    async def validate_batch_parallel(
        self,
        claim_quote_pairs: List[tuple[str, str]],
        max_concurrency: int = None
    ) -> List[Dict[str, any]]:
        """
        Validate multiple claim-quote pairs in parallel using dspy.asyncify.

        This method processes validation calls concurrently using a semaphore to
        limit GPU load. It properly propagates DSPy configuration to worker threads.

        Args:
            claim_quote_pairs: List of (claim, quote) tuples
            max_concurrency: Maximum concurrent LLM calls (default from settings)

        Returns:
            List of validation result dicts

        Example:
            ```python
            validator = EntailmentValidatorModel()
            pairs = [
                ("Bitcoin reached $69,000", "BTC hit $69k in Nov 2021"),
                ("Ethereum uses proof-of-stake", "ETH migrated to PoS in 2022"),
            ]
            results = await validator.validate_batch_parallel(pairs)
            # Returns: [
            #     {"relationship": "SUPPORTS", "reasoning": "...", "confidence": 0.95},
            #     {"relationship": "SUPPORTS", "reasoning": "...", "confidence": 0.92}
            # ]
            ```

        Note:
            - Uses semaphore to prevent GPU overload
            - Default concurrency from settings.max_entailment_concurrency
            - Falls back gracefully on errors
            - Significantly faster than validate_batch() for large batches
        """
        if not claim_quote_pairs:
            return []

        # Get max concurrency from settings if not specified
        if max_concurrency is None:
            from src.config.settings import settings
            max_concurrency = settings.max_entailment_concurrency

        logger.info(
            f"Validating {len(claim_quote_pairs)} claim-quote pairs in parallel "
            f"(max_concurrency={max_concurrency})"
        )

        # Semaphore to limit concurrent GPU calls
        semaphore = asyncio.Semaphore(max_concurrency)

        async def validate_with_limit(claim: str, quote: str) -> Dict[str, any]:
            """Validate with semaphore to control concurrency."""
            async with semaphore:
                return await self.validate_async(claim, quote)

        # Execute all validations in parallel (controlled by semaphore)
        results = await asyncio.gather(
            *[validate_with_limit(claim, quote) for claim, quote in claim_quote_pairs],
            return_exceptions=False  # Let errors propagate to validate_async
        )

        # Log summary
        supports_count = sum(1 for r in results if r["relationship"] == "SUPPORTS")
        logger.info(
            f"Parallel validation complete: {supports_count}/{len(results)} SUPPORTS "
            f"(processed {len(results)} pairs with concurrency={max_concurrency})"
        )

        return results
