"""
DSPy Claim Extractor Model.

Loads the optimized claim extraction model trained with LLM-as-judge metric.

Usage:
    from src.dspy_models.claim_extractor import ClaimExtractorModel

    extractor = ClaimExtractorModel()
    claims = extractor.extract_claims("Bitcoin reached $69,000 in November 2021.")
    print(claims)  # ["Bitcoin reached $69,000 in November 2021"]
"""

import dspy
from typing import List
from pathlib import Path

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ClaimExtraction(dspy.Signature):
    """
    Extract factual, verifiable claims from podcast transcript text.

    Claims should be:
    - Factual (not opinions)
    - Self-contained (no pronouns like he/she/they)
    - Specific (include names, numbers, dates)
    - Concise (5-40 words)
    """

    transcript_chunk: str = dspy.InputField(desc="The podcast transcript text to analyze")
    claims: List[str] = dspy.OutputField(desc="List of factual claims extracted from the transcript")


class ClaimExtractorModel:
    """
    DSPy-based claim extractor using optimized model.

    Loads the optimized model from models/claim_extractor_llm_judge_v1.json
    which was trained using BootstrapFewShot with LLM-as-judge metric.

    Attributes:
        model: DSPy ChainOfThought module with few-shot examples
    """

    def __init__(self, model_path: str = "models/claim_extractor_llm_judge_v1.json"):
        """
        Initialize the claim extractor model.

        Args:
            model_path: Path to the optimized DSPy model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If DSPy configuration fails
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Optimized model not found at {model_path}. "
                "Run claim extraction optimization experiment first."
            )

        # Configure DSPy with Ollama
        logger.info(f"Configuring DSPy with Ollama at {settings.ollama_url}")
        lm = dspy.LM(
            f"ollama/{settings.ollama_model}",
            api_base=settings.ollama_url
        )
        dspy.configure(lm=lm)

        # Load optimized model
        logger.info(f"Loading optimized claim extractor from {model_path}")
        self.model = dspy.ChainOfThought(ClaimExtraction)
        self.model.load(str(self.model_path))

        # Log few-shot examples count
        if hasattr(self.model, 'demos') and self.model.demos:
            logger.info(f"Loaded model with {len(self.model.demos)} few-shot examples")
        else:
            logger.info("Loaded model (zero-shot)")

        # Create async wrapper using dspy.asyncify for parallel processing
        # This captures the current DSPy configuration context
        self._async_model = dspy.asyncify(self.model)
        logger.debug("Created async wrapper for parallel claim extraction")

    def extract_claims(self, transcript_chunk: str) -> List[str]:
        """
        Extract claims from a transcript chunk (synchronous).

        Args:
            transcript_chunk: Text from podcast transcript

        Returns:
            List of extracted claim strings

        Example:
            ```python
            extractor = ClaimExtractorModel()
            claims = extractor.extract_claims(
                "Bitcoin reached $69,000 in November 2021. "
                "This was the all-time high for the cryptocurrency."
            )
            # Returns: [
            #     "Bitcoin reached $69,000 in November 2021",
            #     "Bitcoin's all-time high was $69,000"
            # ]
            ```
        """
        try:
            result = self.model(transcript_chunk=transcript_chunk)
            claims = getattr(result, 'claims', None) or []

            logger.debug(f"Extracted {len(claims)} claims from chunk ({len(transcript_chunk)} chars)")

            return claims

        except Exception as e:
            logger.error(f"Error extracting claims: {e}", exc_info=True)
            return []

    async def extract_claims_async(self, transcript_chunk: str) -> List[str]:
        """
        Extract claims from a transcript chunk (asynchronous using dspy.asyncify).

        This method uses dspy.asyncify to run the DSPy model in a worker thread
        while properly propagating the DSPy configuration context. This enables
        true parallel processing of multiple chunks.

        Args:
            transcript_chunk: Text from podcast transcript

        Returns:
            List of extracted claim strings

        Example:
            ```python
            extractor = ClaimExtractorModel()
            claims = await extractor.extract_claims_async(
                "Bitcoin reached $69,000 in November 2021."
            )
            # Returns: ["Bitcoin reached $69,000 in November 2021"]
            ```

        Note:
            This method is preferred over extract_claims() when processing
            multiple chunks in parallel, as it properly handles DSPy's
            thread-local configuration.
        """
        try:
            result = await self._async_model(transcript_chunk=transcript_chunk)
            claims = getattr(result, 'claims', None) or []

            logger.debug(f"Extracted {len(claims)} claims from chunk ({len(transcript_chunk)} chars)")

            return claims

        except Exception as e:
            logger.error(f"Error in async claim extraction: {e}", exc_info=True)
            return []

    def extract_claims_batch(self, transcript_chunks: List[str]) -> List[List[str]]:
        """
        Extract claims from multiple transcript chunks.

        Args:
            transcript_chunks: List of transcript text chunks

        Returns:
            List of claim lists (one per chunk)

        Example:
            ```python
            extractor = ClaimExtractorModel()
            results = extractor.extract_claims_batch([chunk1, chunk2, chunk3])
            # Returns: [
            #     ["claim1 from chunk1", "claim2 from chunk1"],
            #     ["claim1 from chunk2"],
            #     ["claim1 from chunk3", "claim2 from chunk3", "claim3 from chunk3"]
            # ]
            ```
        """
        logger.info(f"Extracting claims from {len(transcript_chunks)} chunks")

        results = []
        for i, chunk in enumerate(transcript_chunks, 1):
            claims = self.extract_claims(chunk)
            results.append(claims)
            logger.debug(f"Chunk {i}/{len(transcript_chunks)}: {len(claims)} claims")

        total_claims = sum(len(claims) for claims in results)
        logger.info(f"Extracted {total_claims} total claims from {len(transcript_chunks)} chunks")

        return results
